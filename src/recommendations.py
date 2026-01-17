"""Core recommendation engine for Bandcamp based on supporter purchases."""

import json
import re
import subprocess
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from threading import Lock
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from seleniumwire import webdriver as wire_webdriver
from webdriver_manager.chrome import ChromeDriverManager

# Suppress selenium-wire RuntimeWarnings (harmless coroutine warnings)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="seleniumwire")


class SupporterRecommender:
    """Generates Bandcamp recommendations based on what supporters purchased."""

    def __init__(self, headless: bool = True):
        """Initialize the recommender.

        Args:
            headless: Ignored - Selenium always runs headless to prevent popup windows.
                     Kept for API compatibility.
        """
        # headless parameter is ignored - always runs headless
        self.driver = None  # Lazy initialization - only created when needed
        self.item_cache: Dict[str, Dict[str, str]] = {}  # Cache item_id -> item_info
        self._cache_lock = Lock()  # Thread-safe cache updates
        self._driver_pool: Queue = None  # Driver pool for parallel processing
        self._driver_pool_lock = Lock()  # Lock for driver pool initialization
        self._chrome_service = None  # Reuse ChromeDriver service

    def _init_driver(self):
        """Initialize the Selenium webdriver with appropriate options.
        
        Only initialized when needed (for collection pages that require cookies).
        """
        options = Options()
        # Always run headless to avoid popup windows
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-images")  # Don't load images - faster page loads
        options.page_load_strategy = "eager"  # Don't wait for all resources to load
        options.add_experimental_option("excludeSwitches", ["enable-logging", "enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)

        # Auto-detect Chrome binary
        chrome_paths = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
            "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
            "/Applications/Arc.app/Contents/MacOS/Arc",
        ]
        for path in chrome_paths:
            try:
                options.binary_location = path
                break
            except Exception:
                continue

        service = Service(ChromeDriverManager().install())
        self.driver = wire_webdriver.Chrome(service=service, options=options)
    
    def _ensure_driver(self):
        """Ensure driver is initialized (lazy initialization)."""
        if self.driver is None:
            self._init_driver()
    
    def _get_driver_pool(self, pool_size: int = 15):
        """Get or create a driver pool for parallel processing."""
        with self._driver_pool_lock:
            if self._driver_pool is None:
                self._driver_pool = Queue(maxsize=pool_size)
                
                # Pre-create ChromeDriver service (expensive operation, do once)
                if self._chrome_service is None:
                    self._chrome_service = Service(ChromeDriverManager().install())
                
                # Pre-create drivers
                options = self._get_driver_options()
                for _ in range(pool_size):
                    driver = wire_webdriver.Chrome(service=self._chrome_service, options=options)
                    self._driver_pool.put(driver)
        
        return self._driver_pool
    
    def _get_driver_options(self):
        """Get optimized driver options (reusable)."""
        options = Options()
        # Always run headless to avoid popup windows
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-images")  # Don't load images - faster page loads
        options.page_load_strategy = "eager"  # Don't wait for all resources to load
        options.add_experimental_option("excludeSwitches", ["enable-logging", "enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)

        # Auto-detect Chrome binary
        chrome_paths = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
            "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
            "/Applications/Arc.app/Contents/MacOS/Arc",
        ]
        for path in chrome_paths:
            try:
                options.binary_location = path
                break
            except Exception:
                continue
        
        return options
    
    def _create_driver(self):
        """Create a new driver instance (for parallel processing).
        
        Note: Prefer using driver pool for better performance.
        """
        options = self._get_driver_options()
        if self._chrome_service is None:
            self._chrome_service = Service(ChromeDriverManager().install())
        return wire_webdriver.Chrome(service=self._chrome_service, options=options)

    def get_recommendations(
        self,
        wishlist_item_url: str,
        max_recommendations: int = 10,
        min_supporters: int = 2,
        progress_callback=None,
    ) -> List[Dict[str, Any]]:
        """Get recommendations based on supporter purchases.

        Args:
            wishlist_item_url: URL of the Bandcamp item to get recommendations for
            max_recommendations: Maximum number of recommendations to return
            min_supporters: Minimum number of supporters who must have purchased an item
            progress_callback: Optional callback function(status, current, total, estimated_seconds)

        Returns:
            List of recommendation dictionaries with item_title, band_name, item_url, supporters_count
        """
        import time as time_module
        
        # Get supporters of the wishlist item
        if progress_callback:
            progress_callback("Extracting supporters from album page...", 0, 0, 0)
        supporters = self._get_supporters(wishlist_item_url)
        if not supporters:
            if progress_callback:
                progress_callback("No supporters found.", 0, 0, 0)
            return []

        if progress_callback:
            progress_callback(f"Found {len(supporters)} supporters", len(supporters), len(supporters), 0)

        # Get the original item ID to exclude it from recommendations
        if progress_callback:
            progress_callback("Extracting item ID...", 0, 0, 0)
        original_item_id = self._extract_item_id(wishlist_item_url)

        # Get purchases from all supporters (with metadata) - parallel processing
        all_purchases = []
        start_time = time_module.time()
        total_supporters = len(supporters)
        completed_count = 0
        completed_lock = Lock()
        
        # Initialize driver pool
        pool_size = min(10, total_supporters)
        driver_pool = self._get_driver_pool(pool_size)
        
        def fetch_supporter_purchases(supporter):
            """Fetch purchases for a single supporter (thread-safe).
            
            Uses driver pool to avoid expensive driver creation overhead.
            """
            # Get driver from pool (blocks if none available)
            driver = driver_pool.get()
            try:
                purchases = self._get_supporter_purchases_with_driver(supporter, driver)
                return purchases, supporter
            finally:
                # Return driver to pool for reuse
                driver_pool.put(driver)
        
        # Use ThreadPoolExecutor for parallel processing
        # Increased to 15 workers for better throughput (driver pool handles reuse)
        max_workers = min(15, total_supporters)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_supporter = {
                executor.submit(fetch_supporter_purchases, supporter): supporter
                for supporter in supporters
            }
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_supporter):
                try:
                    purchases, supporter = future.result()
                    with completed_lock:
                        all_purchases.extend(purchases)
                        completed_count += 1
                        
                        if progress_callback:
                            elapsed = time_module.time() - start_time
                            avg_time_per_supporter = elapsed / completed_count if completed_count > 0 else 2.0
                            remaining = total_supporters - completed_count
                            estimated_seconds = avg_time_per_supporter * remaining
                            progress_callback(
                                f"Fetching purchases from supporter {completed_count}/{total_supporters} ({supporter})...",
                                completed_count,
                                total_supporters,
                                int(estimated_seconds)
                            )
                except Exception as e:
                    supporter = future_to_supporter[future]
                    print(f"Error processing {supporter}: {e}")
                    with completed_lock:
                        completed_count += 1

        # Count purchases and filter
        purchase_counter = Counter(all_purchases)
        # Remove the original item from recommendations
        if original_item_id:
            purchase_counter.pop(original_item_id, None)

        if progress_callback:
            progress_callback(f"Processing {len(all_purchases)} purchases from {completed_count} supporters...", total_supporters, total_supporters, 0)
        
        if len(all_purchases) == 0:
            if progress_callback:
                progress_callback(f"Note: No purchases found. Collections are likely private and require authentication.", total_supporters, total_supporters, 0)
        
        # Filter by minimum supporters and get top items
        filtered_items = {
            item_id: count
            for item_id, count in purchase_counter.items()
            if count >= min_supporters
        }
        top_items = sorted(
            filtered_items.items(), key=lambda x: x[1], reverse=True
        )[:max_recommendations]

        if progress_callback:
            progress_callback("Building recommendations...", total_supporters, total_supporters, 0)

        # Get item info and build recommendations
        recommendations = []
        for item_id, supporters_count in top_items:
            # Try to get item info from URL if we stored it
            # Otherwise, we'd need to fetch it
            item_info = self._get_item_info_from_id(item_id)
            if item_info:
                item_info["supporters_count"] = supporters_count
                recommendations.append(item_info)

        if progress_callback:
            progress_callback(f"Complete! Found {len(recommendations)} recommendations.", total_supporters, total_supporters, 0)

        return recommendations

    def _get_supporters(self, item_url: str) -> List[str]:
        """Get list of supporter usernames from an item page.
        
        Uses curl instead of Selenium for better performance and no popups.

        Args:
            item_url: URL of the Bandcamp item

        Returns:
            List of supporter usernames
        """
        # Use curl instead of Selenium - no browser needed, more reliable than requests
        curl_cmd = [
            "curl",
            "-s",  # Silent mode
            "-L",  # Follow redirects
            "--compressed",  # Automatically decompress gzip/deflate
            "-H", "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "-H", "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "-H", "Accept-Language: en-US,en;q=0.5",
            "-H", "Connection: keep-alive",
            item_url,
        ]
        
        try:
            result = subprocess.run(
                curl_cmd, capture_output=True, text=True, timeout=15
            )
            if result.returncode != 0:
                print(f"Error fetching page with curl: {result.stderr}")
                return []
        except Exception as e:
            print(f"Error running curl: {e}")
            return []

        soup = BeautifulSoup(result.stdout, features="html.parser")

        supporters = []
        
        # Extract from collectors-data JSON blob (most reliable)
        collectors_data = soup.find("div", id="collectors-data")
        if collectors_data:
            data_blob = collectors_data.get("data-blob")
            if data_blob:
                try:
                    collectors_json = json.loads(data_blob)
                    # Extract usernames from thumbs array
                    thumbs = collectors_json.get("thumbs", [])
                    for thumb in thumbs:
                        username = thumb.get("username")
                        if username:
                            supporters.append(username)
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error parsing collectors-data: {e}")
        
        # Fallback - look for links with class "fan pic"
        if not supporters:
            fan_links = soup.find_all("a", class_=re.compile("fan.*pic|pic.*fan"))
            for link in fan_links:
                href = link.get("href", "")
                # Extract username from href like https://bandcamp.com/username?from=...
                match = re.search(r"bandcamp\.com/([^/?]+)", href)
                if match:
                    username = match.group(1)
                    if username and username != "compliments":  # Exclude special accounts
                        supporters.append(username)

        # Remove duplicates while preserving order
        seen = set()
        unique_supporters = []
        for supporter in supporters:
            if supporter not in seen:
                seen.add(supporter)
                unique_supporters.append(supporter)

        return unique_supporters

    def _get_supporter_purchases(self, username: str) -> List[str]:
        """Get purchases for a supporter using the Bandcamp API.
        
        Uses Selenium only when needed to get cookies for API authentication.

        Args:
            username: Supporter username

        Returns:
            List of item IDs (tralbum_id) that the supporter purchased
        """
        # Ensure driver is initialized (lazy)
        self._ensure_driver()
        return self._get_supporter_purchases_with_driver(username, self.driver)
    
    def _get_supporter_purchases_with_driver(self, username: str, driver) -> List[str]:
        """Get purchases for a supporter using a specific driver instance.
        
        Args:
            username: Supporter username
            driver: Selenium WebDriver instance to use

        Returns:
            List of item IDs (tralbum_id) that the supporter purchased
        """
        try:
            # Navigate to supporter's wishlist page to get fan_id and cookies
            # (wishlist/profile pages have fan_data, /music page doesn't)
            wishlist_url = f"https://bandcamp.com/{username}/wishlist"
            driver.get(wishlist_url)
            
            # Wait for pagedata element instead of fixed sleep (much faster!)
            # Reduced timeout to 3 seconds - page usually loads faster
            try:
                WebDriverWait(driver, 3).until(
                    EC.presence_of_element_located((By.ID, "pagedata"))
                )
            except:
                # If wishlist page doesn't work, try profile page
                profile_url = f"https://bandcamp.com/{username}"
                driver.get(profile_url)
                try:
                    WebDriverWait(driver, 3).until(
                        EC.presence_of_element_located((By.ID, "pagedata"))
                    )
                except:
                    return []

            soup = BeautifulSoup(driver.page_source, features="html.parser")
            pagedata_elem = soup.find(id="pagedata")
            if not pagedata_elem:
                return []

            pagedata = json.loads(pagedata_elem.get("data-blob", "{}"))
            
            # Get fan_id for API call
            fan_data = pagedata.get("fan_data", {})
            fan_id = fan_data.get("fan_id")
            if not fan_id:
                return []
            
            # Extract first page from pagedata
            collection_data = pagedata.get("collection_data", {})
            item_cache = pagedata.get("item_cache", {}).get("collection", {})
            
            # Get item IDs from sequence and pending_sequence (first page)
            sequence = collection_data.get("sequence", [])
            pending_sequence = collection_data.get("pending_sequence", [])
            first_page_item_ids = []
            
            for item_key in sequence + pending_sequence:
                item_data = item_cache.get(item_key)
                if item_data:
                    tralbum_id = item_data.get("tralbum_id")
                    if tralbum_id:
                        first_page_item_ids.append(str(tralbum_id))
                        # Store metadata from first page
                        with self._cache_lock:
                            item_id_str = str(tralbum_id)
                            if item_id_str not in self.item_cache:
                                item_url = item_data.get("item_url", "")
                                self.item_cache[item_id_str] = {
                                    "item_title": item_data.get("item_title", "Unknown Title"),
                                    "band_name": item_data.get("band_name", "Unknown Artist"),
                                    "item_url": item_url or f"https://bandcamp.com/album/{tralbum_id}",
                                }
            
            # Get remaining items via API using last_token
            all_item_ids = list(first_page_item_ids)
            last_token = collection_data.get("last_token", "")
            item_count = collection_data.get("item_count", 0)
            first_page_count = len(first_page_item_ids)
            
            # Skip API call if first page has all items (common for small collections)
            if last_token and first_page_count < item_count:
                # Get cookies from Selenium session
                cookies = {}
                for cookie in driver.get_cookies():
                    cookies[cookie["name"]] = cookie["value"]
                
                # Call API to get remaining items
                api_url = "https://bandcamp.com/api/fancollection/1/collection_items"
                payload = {
                    "fan_id": fan_id,
                    "older_than_token": last_token,
                    "count": 10000,
                }
                
                cookie_string = "; ".join([f"{k}={v}" for k, v in cookies.items()])
                curl_cmd = [
                    "curl",
                    "-X",
                    "POST",
                    "-H",
                    "Content-Type: application/json",
                    "-H",
                    f"Cookie: {cookie_string}",
                    "-H",
                    "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                    "-H",
                    f"Referer: {wishlist_url}",
                    "-d",
                    json.dumps(payload),
                    api_url,
                ]
                
                result = subprocess.run(
                    curl_cmd, capture_output=True, text=True, timeout=30
                )
                
                if result.returncode == 0:
                    try:
                        data = json.loads(result.stdout)
                        items = data.get("items", [])
                        
                        # Extract tralbum_id from API response and store metadata
                        for item in items:
                            tralbum_id = item.get("tralbum_id")
                            if tralbum_id:
                                item_id_str = str(tralbum_id)
                                if item_id_str not in all_item_ids:  # Avoid duplicates
                                    all_item_ids.append(item_id_str)
                                    
                                    # Store item metadata in cache if not already present (thread-safe)
                                    with self._cache_lock:
                                        if item_id_str not in self.item_cache:
                                            item_url = item.get("item_url", "")
                                            self.item_cache[item_id_str] = {
                                                "item_title": item.get("item_title", "Unknown Title"),
                                                "band_name": item.get("band_name", "Unknown Artist"),
                                                "item_url": item_url or f"https://bandcamp.com/album/{tralbum_id}",
                                            }
                    except (json.JSONDecodeError, KeyError):
                        pass  # API might have failed, but we still have first page items
            
            return all_item_ids

        except Exception as e:
            print(f"Error getting purchases for {username}: {e}")
            return []

    def _extract_item_id(self, item_url: str) -> Optional[str]:
        """Extract tralbum_id from an item URL or page.
        
        Uses curl instead of Selenium.

        Args:
            item_url: URL of the Bandcamp item

        Returns:
            tralbum_id as string, or None if not found
        """
        try:
            curl_cmd = [
                "curl",
                "-s",
                "-L",
                "--compressed",
                "-H", "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                item_url,
            ]
            result = subprocess.run(
                curl_cmd, capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                return None
            
            soup = BeautifulSoup(result.stdout, features="html.parser")
            pagedata_elem = soup.find(id="pagedata")
            if pagedata_elem:
                pagedata = json.loads(pagedata_elem.get("data-blob", "{}"))
                # Try multiple possible locations for tralbum_id
                tralbum_id = None
                
                # Try tralbum_data first
                tralbum_data = pagedata.get("tralbum_data")
                if isinstance(tralbum_data, dict):
                    tralbum_id = tralbum_data.get("tralbum_id")
                
                # Try fan_tralbum_data
                if not tralbum_id:
                    fan_tralbum_data = pagedata.get("fan_tralbum_data")
                    if isinstance(fan_tralbum_data, dict):
                        tralbum_id = fan_tralbum_data.get("tralbum_id")
                
                # Try album_id as fallback
                if not tralbum_id:
                    tralbum_id = pagedata.get("album_id")
                
                if tralbum_id:
                    return str(tralbum_id)
        except Exception as e:
            print(f"Error extracting item ID: {e}")
            pass

        return None

    def _get_item_info_from_id(self, item_id: str) -> Optional[Dict[str, str]]:
        """Get item info from tralbum_id using cache.

        Args:
            item_id: tralbum_id

        Returns:
            Dict with item_title, band_name, item_url, or None if not in cache
        """
        return self.item_cache.get(item_id)


    def close(self):
        """Close the webdriver and cleanup driver pool."""
        if self.driver:
            self.driver.quit()
        
        # Clean up driver pool
        if self._driver_pool:
            while not self._driver_pool.empty():
                try:
                    driver = self._driver_pool.get_nowait()
                    driver.quit()
                except:
                    pass
            self._driver_pool = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

