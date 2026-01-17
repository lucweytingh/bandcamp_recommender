"""Core recommendation engine for Bandcamp based on supporter purchases."""

import json
import re
import subprocess
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import log
from queue import Queue
from threading import Lock
from typing import Any, Dict, List, Optional, Set

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
                
                # Pre-create drivers (this can take a while, but we do it once)
                options = self._get_driver_options()
                for i in range(pool_size):
                    try:
                        driver = wire_webdriver.Chrome(service=self._chrome_service, options=options)
                        self._driver_pool.put(driver)
                    except Exception as e:
                        # If driver creation fails, continue with fewer drivers
                        print(f"Warning: Failed to create driver {i+1}/{pool_size}: {e}")
                        break
        
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
        
        # Fallback - look for links with class "fan pic" or near supporter thumbnails
        if not supporters:
            # Try fan pic links first
            fan_links = soup.find_all("a", class_=re.compile("fan.*pic|pic.*fan"))
            for link in fan_links:
                href = link.get("href", "")
                # Extract username from href like https://bandcamp.com/username?from=...
                match = re.search(r"bandcamp\.com/([^/?]+)", href)
                if match:
                    username = match.group(1)
                    if username and username != "compliments":  # Exclude special accounts
                        supporters.append(username)
            
            # If still no supporters, look for "supported by" section (track pages)
            if not supporters:
                # Find the section containing "supported by" text
                for elem in soup.find_all(["div", "section", "span", "p"]):
                    text = elem.get_text()
                    if "supported by" in text.lower():
                        # Find all links within this section
                        links = elem.find_all("a", href=re.compile(r"bandcamp\.com/[^/?]+"))
                        for link in links:
                            href = link.get("href", "")
                            match = re.search(r"bandcamp\.com/([^/?]+)", href)
                            if match:
                                username = match.group(1)
                                # Exclude common non-supporter links
                                excluded = ["artists", "music", "merch", "community", "partner", 
                                           "sign", "log", "help", "settings", "compliments", 
                                           "album", "track", "EmbeddedPlayer"]
                                if username and username not in excluded:
                                    supporters.append(username)
                        break  # Found the section, no need to continue
            
            # Final fallback - look for links near thumbnail images (works for track pages)
            if not supporters:
                # Find thumbnail images and get their parent links
                thumbnails = soup.find_all("img", alt=re.compile(".*thumbnail"))
                for thumb in thumbnails:
                    # Check parent link
                    parent = thumb.parent
                    if parent and parent.name == "a":
                        href = parent.get("href", "")
                        match = re.search(r"bandcamp\.com/([^/?]+)", href)
                        if match:
                            username = match.group(1)
                            # Exclude common non-supporter links
                            excluded = ["artists", "music", "merch", "community", "partner", 
                                       "sign", "log", "help", "settings", "compliments",
                                       "album", "track", "EmbeddedPlayer", "discover"]
                            if username and username not in excluded:
                                supporters.append(username)
                    # Also check if thumbnail is in a link itself
                    elif thumb.parent and thumb.parent.parent:
                        grandparent = thumb.parent.parent
                        if grandparent.name == "a":
                            href = grandparent.get("href", "")
                            match = re.search(r"bandcamp\.com/([^/?]+)", href)
                            if match:
                                username = match.group(1)
                                excluded = ["artists", "music", "merch", "community", "partner", 
                                           "sign", "log", "help", "settings", "compliments",
                                           "album", "track", "EmbeddedPlayer", "discover"]
                                if username and username not in excluded:
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
    
    def _get_supporter_purchases_with_driver(self, username: str, driver, first_page_only: bool = False, extract_tags: bool = True) -> List[str]:
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
                                # Extract tags only if extract_tags is True
                                tags = []
                                if extract_tags and item_url:
                                    tags = self._extract_tags(item_url)
                                
                                self.item_cache[item_id_str] = {
                                    "item_title": item_data.get("item_title", "Unknown Title"),
                                    "band_name": item_data.get("band_name", "Unknown Artist"),
                                    "item_url": item_url or f"https://bandcamp.com/album/{tralbum_id}",
                                    "tags": tags,
                                }
            
            # Get remaining items via API using last_token
            all_item_ids = list(first_page_item_ids)
            
            # Skip API call if first_page_only is True (for speed in random mode)
            if first_page_only:
                return all_item_ids
            
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
                                            # Extract tags only if extract_tags is True
                                            tags = []
                                            if extract_tags and item_url:
                                                tags = self._extract_tags(item_url)
                                            
                                            self.item_cache[item_id_str] = {
                                                "item_title": item.get("item_title", "Unknown Title"),
                                                "band_name": item.get("band_name", "Unknown Artist"),
                                                "item_url": item_url or f"https://bandcamp.com/album/{tralbum_id}",
                                                "tags": tags,
                                            }
                    except (json.JSONDecodeError, KeyError):
                        pass  # API might have failed, but we still have first page items
            
            return all_item_ids

        except Exception as e:
            print(f"Error getting purchases for {username}: {e}")
            return []
    
    def _get_supporter_wishlist_with_driver(self, username: str, driver, first_page_only: bool = False, extract_tags: bool = True) -> List[str]:
        """Get wishlist items for a supporter using a specific driver instance.
        
        Args:
            username: Supporter username
            driver: Selenium WebDriver instance to use
            first_page_only: If True, only get first page items (skip API call for speed)
            extract_tags: If False, skip tag extraction (faster but no tag data)

        Returns:
            List of item IDs (tralbum_id) that the supporter has in their wishlist
        """
        try:
            # Navigate to supporter's wishlist page
            wishlist_url = f"https://bandcamp.com/{username}/wishlist"
            driver.get(wishlist_url)
            
            # Reduced timeout for first_page_only mode
            wait_timeout = 2 if first_page_only else 3
            try:
                WebDriverWait(driver, wait_timeout).until(
                    EC.presence_of_element_located((By.ID, "pagedata"))
                )
            except:
                return []

            soup = BeautifulSoup(driver.page_source, features="html.parser")
            pagedata_elem = soup.find(id="pagedata")
            if not pagedata_elem:
                return []

            pagedata = json.loads(pagedata_elem.get("data-blob", "{}"))
            
            # Extract wishlist from pagedata
            wishlist_data = pagedata.get("wishlist_data", {})
            item_cache = pagedata.get("item_cache", {}).get("wishlist", {})
            
            # Get item IDs from sequence and pending_sequence (first page)
            sequence = wishlist_data.get("sequence", [])
            pending_sequence = wishlist_data.get("pending_sequence", [])
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
                                # Extract tags only if extract_tags is True
                                tags = []
                                if extract_tags and item_url:
                                    tags = self._extract_tags(item_url)
                                
                                self.item_cache[item_id_str] = {
                                    "item_title": item_data.get("item_title", "Unknown Title"),
                                    "band_name": item_data.get("band_name", "Unknown Artist"),
                                    "item_url": item_url or f"https://bandcamp.com/album/{tralbum_id}",
                                    "tags": tags,
                                }
            
            # Get remaining items via API using last_token
            all_item_ids = list(first_page_item_ids)
            
            # Skip API call if first_page_only is True (for speed in random mode)
            if first_page_only:
                return all_item_ids
            
            last_token = wishlist_data.get("last_token", "")
            item_count = wishlist_data.get("item_count", 0)
            first_page_count = len(first_page_item_ids)
            
            # Skip API call if first page has all items (common for small wishlists)
            if last_token and first_page_count < item_count:
                # Get fan_id for API call
                fan_data = pagedata.get("fan_data", {})
                fan_id = fan_data.get("fan_id")
                
                if fan_id:
                    # Get cookies from Selenium session
                    cookies = {}
                    for cookie in driver.get_cookies():
                        cookies[cookie["name"]] = cookie["value"]
                    
                    # Call API to get remaining items (wishlist uses same endpoint with different token)
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
                                                # Extract tags only if extract_tags is True
                                                tags = []
                                                if extract_tags and item_url:
                                                    tags = self._extract_tags(item_url)
                                                
                                                self.item_cache[item_id_str] = {
                                                    "item_title": item.get("item_title", "Unknown Title"),
                                                    "band_name": item.get("band_name", "Unknown Artist"),
                                                    "item_url": item_url or f"https://bandcamp.com/album/{tralbum_id}",
                                                    "tags": tags,
                                                }
                        except (json.JSONDecodeError, KeyError):
                            pass  # API might have failed, but we still have first page items
            
            return all_item_ids

        except Exception as e:
            print(f"Error getting wishlist for {username}: {e}")
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
            Dict with item_title, band_name, item_url, tags, or None if not in cache
        """
        return self.item_cache.get(item_id)
    
    def _extract_tags(self, item_url: str) -> List[str]:
        """Extract tags from a Bandcamp item page.
        
        Tags are extracted from DOM elements with class 'tag'.
        
        Args:
            item_url: URL of the Bandcamp item
            
        Returns:
            List of tag strings, or empty list if not found
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
                return []
            
            soup = BeautifulSoup(result.stdout, features="html.parser")
            
            # Extract tags from DOM elements with class 'tag'
            tag_links = soup.find_all("a", class_=re.compile("tag"))
            tags = [tag.get_text(strip=True) for tag in tag_links if tag.get_text(strip=True)]
            
            return tags
        except Exception as e:
            # Log error for debugging but don't fail silently
            print(f"Error extracting tags from {item_url}: {e}")
            return []
    
    def _normalize_tag(self, tag: str) -> str:
        """Normalize a tag for comparison (lowercase, strip, handle variations).
        
        Args:
            tag: Tag string to normalize
            
        Returns:
            Normalized tag string
        """
        # Lowercase and strip
        normalized = tag.lower().strip()
        
        # Handle common variations
        variations = {
            'uk': 'united kingdom',
            'u.k.': 'united kingdom',
            'usa': 'united states',
            'u.s.a.': 'united states',
        }
        
        return variations.get(normalized, normalized)
    
    def _calculate_tag_similarity(
        self,
        original_tags: List[str],
        candidate_tags: List[str],
        tag_frequencies: Optional[Dict[str, int]] = None,
        total_items: int = 1
    ) -> float:
        """Calculate sophisticated tag similarity score between two tag sets.
        
        Uses TF-IDF weighted Jaccard similarity with tag normalization.
        
        Args:
            original_tags: Tags from the original item
            candidate_tags: Tags from the candidate item
            tag_frequencies: Optional dict of tag -> frequency across all items (for TF-IDF)
            total_items: Total number of items (for TF-IDF calculation)
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not original_tags or not candidate_tags:
            return 0.0
        
        # Normalize tags
        original_set = {self._normalize_tag(t) for t in original_tags}
        candidate_set = {self._normalize_tag(t) for t in candidate_tags}
        
        # Calculate intersection and union
        intersection = original_set & candidate_set
        union = original_set | candidate_set
        
        if not union:
            return 0.0
        
        # Basic Jaccard similarity
        jaccard = len(intersection) / len(union)
        
        # If we have tag frequencies, use TF-IDF weighting
        if tag_frequencies and total_items > 1:
            # Calculate weighted similarity
            # Weight each matching tag by its inverse document frequency (IDF)
            # Rare tags that match are more significant
            weighted_score = 0.0
            total_weight = 0.0
            
            for tag in intersection:
                # IDF = log(total_items / (tag_frequency + 1))
                # +1 to avoid division by zero
                tag_freq = tag_frequencies.get(tag, 0)
                idf = log(total_items / (tag_freq + 1))
                weighted_score += idf
                total_weight += idf
            
            # Also weight non-matching tags (penalty for dissimilarity)
            for tag in union - intersection:
                tag_freq = tag_frequencies.get(tag, 0)
                idf = log(total_items / (tag_freq + 1))
                total_weight += idf
            
            # Normalize weighted score
            if total_weight > 0:
                weighted_jaccard = weighted_score / total_weight
                # Combine basic Jaccard with weighted score (weighted average)
                return 0.6 * jaccard + 0.4 * weighted_jaccard
        
        return jaccard
    
    def get_tag_similar_recommendations(
        self,
        item_url: str,
        max_recommendations: int = 10,
        min_similarity: float = 0.1,
        max_supporters: Optional[int] = None,
        progress_callback=None,
    ) -> List[Dict[str, Any]]:
        """Get recommendations based on tag similarity.
        
        Explores supporters' collections and ranks items by tag similarity to the original.
        
        Args:
            item_url: URL of the Bandcamp item to get recommendations for
            max_recommendations: Maximum number of recommendations to return
            min_similarity: Minimum tag similarity score (0.0 to 1.0)
            max_supporters: Maximum number of supporters to fetch items from (None = all)
            progress_callback: Optional callback function(status, current, total, estimated_seconds)
            
        Returns:
            List of recommendation dictionaries with item_title, band_name, item_url, 
            tags, similarity_score, and supporters_count
        """
        import time as time_module
        
        # Get original item tags
        if progress_callback:
            progress_callback("Extracting tags from original item...", 0, 0, 0)
        original_tags = self._extract_tags(item_url)
        if not original_tags:
            # Try one more time in case of transient error
            original_tags = self._extract_tags(item_url)
            if not original_tags:
                if progress_callback:
                    progress_callback("No tags found for original item.", 0, 0, 0)
                return []
        
        if progress_callback:
            progress_callback(f"Found tags: {', '.join(original_tags)}", 0, 0, 0)
        
        original_item_id = self._extract_item_id(item_url)
        
        # Get supporters
        if progress_callback:
            progress_callback("Extracting supporters from page...", 0, 0, 0)
        supporters = self._get_supporters(item_url)
        if not supporters:
            if progress_callback:
                progress_callback("No supporters found.", 0, 0, 0)
            return []
        
        if progress_callback:
            progress_callback(f"Found {len(supporters)} supporters", len(supporters), len(supporters), 0)
        
        # Limit number of supporters if specified
        if max_supporters and max_supporters < len(supporters):
            import random
            supporters = random.sample(supporters, max_supporters)
            if progress_callback:
                progress_callback(f"Using {len(supporters)} random supporters", len(supporters), len(supporters), 0)
        
        # Get all items from supporters' collections
        all_items = []
        start_time = time_module.time()
        total_supporters = len(supporters)
        completed_count = 0
        completed_lock = Lock()
        
        # Initialize driver pool
        pool_size = min(15, total_supporters)
        if progress_callback:
            progress_callback("Initializing driver pool (this may take a moment)...", 0, total_supporters, 0)
        
        # Driver pool initialization can take time - do it before starting threads
        try:
            driver_pool = self._get_driver_pool(pool_size)
        except Exception as e:
            if progress_callback:
                progress_callback(f"Error initializing driver pool: {e}", 0, total_supporters, 0)
            return []
        
        if progress_callback:
            progress_callback(f"Driver pool ready. Fetching items from {total_supporters} supporters...", 0, total_supporters, 0)
        
        def fetch_supporter_items(supporter):
            """Fetch items for a single supporter (thread-safe)."""
            driver = None
            try:
                # Get driver with timeout to prevent indefinite blocking
                try:
                    driver = driver_pool.get(timeout=30)
                except Exception as e:
                    return [], supporter, f"Timeout getting driver: {str(e)[:50]}"
                
                # Fetch items with the driver
                try:
                    items = self._get_supporter_purchases_with_driver(supporter, driver)
                    return items, supporter, None
                except Exception as e:
                    return [], supporter, f"Error fetching items: {str(e)[:50]}"
            finally:
                # Always return driver to pool, even on error
                if driver:
                    try:
                        driver_pool.put_nowait(driver)  # Use put_nowait to avoid blocking
                    except:
                        # If pool is full (shouldn't happen), try with timeout
                        try:
                            driver_pool.put(driver, timeout=2)
                        except:
                            # Last resort: just continue without returning driver
                            # This shouldn't happen in normal operation
                            pass
        
        max_workers = min(15, total_supporters)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_supporter = {
                executor.submit(fetch_supporter_items, supporter): supporter
                for supporter in supporters
            }
            
            # Process futures with manual polling to avoid indefinite blocking
            pending_futures = dict(future_to_supporter)
            future_start_times = {f: time_module.time() for f in pending_futures.keys()}
            max_future_time = 30  # Max seconds per future
            
            while pending_futures:
                completed_this_round = []
                
                for future, supporter in list(pending_futures.items()):
                    # Check if future is done
                    if future.done():
                        completed_this_round.append(future)
                        try:
                            items, supporter, error = future.result(timeout=1)
                            with completed_lock:
                                if error:
                                    if progress_callback:
                                        progress_callback(
                                            f"Error from {supporter}: {error[:30]}... ({completed_count + 1}/{total_supporters})",
                                            completed_count + 1,
                                            total_supporters,
                                            0
                                        )
                                else:
                                    all_items.extend(items)
                                    if progress_callback:
                                        elapsed = time_module.time() - start_time
                                        avg_time = elapsed / completed_count if completed_count > 0 else 2.0
                                        remaining = total_supporters - completed_count
                                        estimated_seconds = avg_time * remaining
                                        progress_callback(
                                            f"Fetched {len(items)} items from {supporter} ({completed_count + 1}/{total_supporters})...",
                                            completed_count + 1,
                                            total_supporters,
                                            int(estimated_seconds)
                                        )
                                completed_count += 1
                        except Exception as e:
                            with completed_lock:
                                completed_count += 1
                                if progress_callback:
                                    error_msg = str(e)[:50] if str(e) else "Unknown error"
                                    progress_callback(
                                        f"Error from {supporter}: {error_msg}... ({completed_count}/{total_supporters})",
                                        completed_count,
                                        total_supporters,
                                        0
                                    )
                    # Check for timeout
                    elif time_module.time() - future_start_times[future] > max_future_time:
                        # Future is taking too long, cancel it
                        completed_this_round.append(future)
                        future.cancel()
                        with completed_lock:
                            completed_count += 1
                            if progress_callback:
                                progress_callback(
                                    f"Timeout from {supporter} ({completed_count}/{total_supporters})...",
                                    completed_count,
                                    total_supporters,
                                    0
                                )
                
                # Remove completed futures
                for future in completed_this_round:
                    pending_futures.pop(future, None)
                    future_start_times.pop(future, None)
                
                # If no futures completed, wait a bit before checking again
                if not completed_this_round and pending_futures:
                    time_module.sleep(0.5)  # Small sleep to avoid busy-waiting
        
        if progress_callback:
            progress_callback("Calculating tag similarities...", total_supporters, total_supporters, 0)
        
        # Remove duplicates and original item
        unique_items = list(set(all_items))
        if original_item_id and original_item_id in unique_items:
            unique_items.remove(original_item_id)
        
        # Build tag frequency map for TF-IDF weighting
        tag_frequencies: Dict[str, int] = Counter()
        items_with_tags: Dict[str, List[str]] = {}
        
        for item_id in unique_items:
            item_info = self._get_item_info_from_id(item_id)
            if item_info and item_info.get('tags'):
                tags = item_info['tags']
                items_with_tags[item_id] = tags
                for tag in tags:
                    normalized = self._normalize_tag(tag)
                    tag_frequencies[normalized] += 1
        
        total_items = len(items_with_tags) if items_with_tags else 1
        
        # Calculate similarity scores
        item_similarities: Dict[str, float] = {}
        for item_id, candidate_tags in items_with_tags.items():
            similarity = self._calculate_tag_similarity(
                original_tags,
                candidate_tags,
                tag_frequencies,
                total_items
            )
            if similarity >= min_similarity:
                item_similarities[item_id] = similarity
        
        # Sort by similarity (descending)
        sorted_items = sorted(
            item_similarities.items(),
            key=lambda x: x[1],
            reverse=True
        )[:max_recommendations]
        
        # Build recommendations
        recommendations = []
        for item_id, similarity_score in sorted_items:
            item_info = self._get_item_info_from_id(item_id)
            if item_info:
                item_info['similarity_score'] = similarity_score
                # Count how many supporters have this item
                supporters_count = all_items.count(item_id)
                item_info['supporters_count'] = supporters_count
                recommendations.append(item_info)
        
        if progress_callback:
            progress_callback(
                f"Complete! Found {len(recommendations)} tag-similar recommendations.",
                total_supporters,
                total_supporters,
                0
            )
        
        return recommendations


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

