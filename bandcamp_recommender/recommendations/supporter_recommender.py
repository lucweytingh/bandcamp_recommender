"""Main recommendation engine for Bandcamp based on supporter purchases."""

import json
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any, Callable, Dict, List, Optional

from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from bandcamp_recommender.recommendations.api import (
    fetch_collection_items_api,
    get_cookies_from_driver,
    get_fan_id_from_page,
)
from bandcamp_recommender.recommendations.driver_manager import DriverManager
from bandcamp_recommender.recommendations.scraper import extract_item_id, extract_supporters, extract_tags
from bandcamp_recommender.recommendations.tags import calculate_tag_similarity, normalize_tag


class SupporterRecommender:
    """Generates Bandcamp recommendations based on what supporters purchased."""

    def __init__(self, headless: bool = True):
        """Initialize the recommender.

        Args:
            headless: Ignored - Selenium always runs headless to prevent popup windows.
                     Kept for API compatibility.
        """
        self._driver_manager = DriverManager()
        self.item_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = Lock()

    def get_recommendations(
        self,
        wishlist_item_url: str,
        max_recommendations: int = 10,
        min_supporters: int = 2,
        progress_callback: Optional[Callable] = None,
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
        # Get supporters of the wishlist item
        if progress_callback:
            progress_callback("Extracting supporters from album page...", 0, 0, 0)
        supporters = extract_supporters(wishlist_item_url)
        if not supporters:
            if progress_callback:
                progress_callback("No supporters found.", 0, 0, 0)
            return []

        if progress_callback:
            progress_callback(f"Found {len(supporters)} supporters", len(supporters), len(supporters), 0)

        # Get the original item ID to exclude it from recommendations
        if progress_callback:
            progress_callback("Extracting item ID...", 0, 0, 0)
        original_item_id = extract_item_id(wishlist_item_url)

        # Get purchases from all supporters (with metadata) - parallel processing
        all_purchases = []
        start_time = time.time()
        total_supporters = len(supporters)
        completed_count = 0
        completed_lock = Lock()

        # Initialize driver pool
        pool_size = min(10, total_supporters)
        driver_pool = self._driver_manager.get_driver_pool(pool_size)

        def fetch_supporter_purchases(supporter):
            """Fetch purchases for a single supporter (thread-safe)."""
            driver = driver_pool.get()
            try:
                purchases = self._get_supporter_purchases_with_driver(supporter, driver)
                return purchases, supporter
            finally:
                driver_pool.put(driver)

        # Use ThreadPoolExecutor for parallel processing
        max_workers = min(15, total_supporters)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_supporter = {
                executor.submit(fetch_supporter_purchases, supporter): supporter
                for supporter in supporters
            }

            # Process completed tasks as they finish
            for future in future_to_supporter:
                try:
                    purchases, supporter = future.result()
                    with completed_lock:
                        all_purchases.extend(purchases)
                        completed_count += 1

                        if progress_callback:
                            elapsed = time.time() - start_time
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
            progress_callback(
                f"Processing {len(all_purchases)} purchases from {completed_count} supporters...",
                total_supporters,
                total_supporters,
                0
            )

        if len(all_purchases) == 0:
            if progress_callback:
                progress_callback(
                    "Note: No purchases found. Collections are likely private and require authentication.",
                    total_supporters,
                    total_supporters,
                    0
                )

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
            item_info = self._get_item_info_from_id(item_id)
            if item_info:
                item_info["supporters_count"] = supporters_count
                recommendations.append(item_info)

        if progress_callback:
            progress_callback(
                f"Complete! Found {len(recommendations)} recommendations.",
                total_supporters,
                total_supporters,
                0
            )

        return recommendations

    def _get_supporter_purchases_with_driver(
        self,
        username: str,
        driver: WebDriver,
        first_page_only: bool = False,
        extract_tags_flag: bool = True
    ) -> List[str]:
        """Get purchases for a supporter using a specific driver instance.

        Args:
            username: Supporter username
            driver: Selenium WebDriver instance to use
            first_page_only: If True, only get first page items (skip API call for speed)
            extract_tags_flag: If False, skip tag extraction (faster but no tag data)

        Returns:
            List of item IDs (tralbum_id) that the supporter purchased
        """
        try:
            # Get fan_id from wishlist/profile page (which also has collection_data)
            fan_id = get_fan_id_from_page(driver, username)
            if not fan_id:
                return []

            # Get pagedata from current page (wishlist page has collection_data)
            soup = BeautifulSoup(driver.page_source, features="html.parser")
            pagedata_elem = soup.find(id="pagedata")
            if not pagedata_elem:
                return []

            pagedata = json.loads(pagedata_elem.get("data-blob", "{}"))

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
                        self._store_item_metadata(
                            str(tralbum_id),
                            item_data,
                            extract_tags_flag
                        )

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
                cookies = get_cookies_from_driver(driver)
                wishlist_url = f"https://bandcamp.com/{username}/wishlist"
                items = fetch_collection_items_api(fan_id, last_token, cookies, wishlist_url)

                # Extract tralbum_id from API response and store metadata
                for item in items:
                    tralbum_id = item.get("tralbum_id")
                    if tralbum_id:
                        item_id_str = str(tralbum_id)
                        if item_id_str not in all_item_ids:  # Avoid duplicates
                            all_item_ids.append(item_id_str)
                            self._store_item_metadata(item_id_str, item, extract_tags_flag)

            return all_item_ids

        except Exception as e:
            # Silently handle errors (timeouts, network issues, etc.)
            return []

    def _get_supporter_wishlist_with_driver(
        self,
        username: str,
        driver: WebDriver,
        first_page_only: bool = False,
        extract_tags_flag: bool = True
    ) -> List[str]:
        """Get wishlist items for a supporter using a specific driver instance.

        Args:
            username: Supporter username
            driver: Selenium WebDriver instance to use
            first_page_only: If True, only get first page items (skip API call for speed)
            extract_tags_flag: If False, skip tag extraction (faster but no tag data)

        Returns:
            List of item IDs (tralbum_id) that the supporter has in their wishlist
        """
        try:
            wishlist_url = f"https://bandcamp.com/{username}/wishlist"
            driver.get(wishlist_url)

            # Reduced timeout for first_page_only mode
            wait_timeout = 2 if first_page_only else 3
            try:
                WebDriverWait(driver, wait_timeout).until(
                    EC.presence_of_element_located((By.ID, "pagedata"))
                )
            except Exception:
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
                        self._store_item_metadata(
                            str(tralbum_id),
                            item_data,
                            extract_tags_flag
                        )

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
                fan_id = get_fan_id_from_page(driver, username)
                if fan_id:
                    cookies = get_cookies_from_driver(driver)
                    items = fetch_collection_items_api(fan_id, last_token, cookies, wishlist_url)

                    # Extract tralbum_id from API response and store metadata
                    for item in items:
                        tralbum_id = item.get("tralbum_id")
                        if tralbum_id:
                            item_id_str = str(tralbum_id)
                            if item_id_str not in all_item_ids:  # Avoid duplicates
                                all_item_ids.append(item_id_str)
                                self._store_item_metadata(item_id_str, item, extract_tags_flag)

            return all_item_ids

        except Exception as e:
            # Silently handle errors (timeouts, network issues, etc.)
            return []

    def _store_item_metadata(
        self,
        item_id_str: str,
        item_data: Dict[str, Any],
        extract_tags_flag: bool
    ):
        """Store item metadata in cache (thread-safe).

        Args:
            item_id_str: Item ID as string
            item_data: Item data dictionary
            extract_tags_flag: Whether to extract tags
        """
        with self._cache_lock:
            if item_id_str not in self.item_cache:
                item_url = item_data.get("item_url", "")
                # Extract tags only if extract_tags_flag is True
                tags = []
                if extract_tags_flag and item_url:
                    tags = extract_tags(item_url)

                self.item_cache[item_id_str] = {
                    "item_title": item_data.get("item_title", "Unknown Title"),
                    "band_name": item_data.get("band_name", "Unknown Artist"),
                    "item_url": item_url or f"https://bandcamp.com/album/{item_id_str}",
                    "tags": tags,
                }

    def _get_item_info_from_id(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Get item info from tralbum_id using cache.

        Args:
            item_id: tralbum_id

        Returns:
            Dict with item_title, band_name, item_url, tags, or None if not in cache
        """
        return self.item_cache.get(item_id)

    def get_tag_similar_recommendations(
        self,
        item_url: str,
        max_recommendations: int = 10,
        min_similarity: float = 0.1,
        max_supporters: Optional[int] = None,
        progress_callback: Optional[Callable] = None,
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
        # Get original item tags
        if progress_callback:
            progress_callback("Extracting tags from original item...", 0, 0, 0)
        original_tags = extract_tags(item_url)
        if not original_tags:
            # Try one more time in case of transient error
            original_tags = extract_tags(item_url)
            if not original_tags:
                if progress_callback:
                    progress_callback("No tags found for original item.", 0, 0, 0)
                return []

        if progress_callback:
            progress_callback(f"Found tags: {', '.join(original_tags)}", 0, 0, 0)

        original_item_id = extract_item_id(item_url)

        # Get supporters
        if progress_callback:
            progress_callback("Extracting supporters from page...", 0, 0, 0)
        supporters = extract_supporters(item_url)
        if not supporters:
            if progress_callback:
                progress_callback("No supporters found.", 0, 0, 0)
            return []

        if progress_callback:
            progress_callback(f"Found {len(supporters)} supporters", len(supporters), len(supporters), 0)

        # Limit number of supporters if specified
        if max_supporters and max_supporters < len(supporters):
            supporters = random.sample(supporters, max_supporters)
            if progress_callback:
                progress_callback(f"Using {len(supporters)} random supporters", len(supporters), len(supporters), 0)

        # Get all items from supporters' collections
        all_items = []
        start_time = time.time()
        total_supporters = len(supporters)
        completed_count = 0
        completed_lock = Lock()

        # Initialize driver pool
        pool_size = min(10, total_supporters)
        if progress_callback:
            progress_callback("Initializing driver pool (this may take a moment)...", 0, total_supporters, 0)

        try:
            driver_pool = self._driver_manager.get_driver_pool(pool_size)
        except Exception as e:
            if progress_callback:
                progress_callback(f"Error initializing driver pool: {e}", 0, total_supporters, 0)
            return []

        if progress_callback:
            progress_callback(
                f"Driver pool ready. Fetching items from {total_supporters} supporters...",
                0,
                total_supporters,
                0
            )

        def fetch_supporter_items(supporter):
            """Fetch items for a single supporter (thread-safe)."""
            driver = None
            try:
                try:
                    driver = driver_pool.get(timeout=30)
                except Exception as e:
                    return [], supporter, f"Timeout getting driver: {str(e)[:50]}"

                try:
                    items = self._get_supporter_purchases_with_driver(supporter, driver)
                    return items, supporter, None
                except Exception as e:
                    return [], supporter, f"Error fetching items: {str(e)[:50]}"
            finally:
                if driver:
                    try:
                        driver_pool.put_nowait(driver)
                    except Exception:
                        try:
                            driver_pool.put(driver, timeout=2)
                        except Exception:
                            pass

        max_workers = min(15, total_supporters)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_supporter = {
                executor.submit(fetch_supporter_items, supporter): supporter
                for supporter in supporters
            }

            # Process futures with manual polling to avoid indefinite blocking
            pending_futures = dict(future_to_supporter)
            future_start_times = {f: time.time() for f in pending_futures.keys()}
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
                                        elapsed = time.time() - start_time
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
                    elif time.time() - future_start_times[future] > max_future_time:
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
                    time.sleep(0.5)  # Small sleep to avoid busy-waiting

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
                    normalized = normalize_tag(tag)
                    tag_frequencies[normalized] += 1

        total_items = len(items_with_tags) if items_with_tags else 1

        # Calculate similarity scores
        item_similarities: Dict[str, float] = {}
        for item_id, candidate_tags in items_with_tags.items():
            similarity = calculate_tag_similarity(
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

    def _get_supporters(self, item_url: str) -> List[str]:
        """Get list of supporter usernames from an item page.
        
        Wrapper method for backward compatibility with scripts.
        
        Args:
            item_url: URL of the Bandcamp item
            
        Returns:
            List of supporter usernames
        """
        return extract_supporters(item_url)

    def _get_driver_pool(self, pool_size: int = 10):
        """Get or create a driver pool for parallel processing.
        
        Wrapper method for backward compatibility with scripts.
        
        Args:
            pool_size: Number of drivers to create in the pool
            
        Returns:
            Queue of driver instances
        """
        return self._driver_manager.get_driver_pool(pool_size)

    def get_random_items(
        self,
        item_url: str,
        num_items: int,
        num_supporters: int = 20,
        use_wishlist: bool = False,
        min_overlap: Optional[int] = None,
        use_fallback: bool = False,
        progress_callback: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        """Get random items from random supporters' collections.
        
        Args:
            item_url: URL of the Bandcamp item to get supporters from
            num_items: Number of random items to return
            num_supporters: Number of random supporters to check (default: 20)
            use_wishlist: If True, use wishlist items instead of purchases (default: False)
            min_overlap: Only select items found in at least N supporters' collections (default: None, i.e., any item)
            use_fallback: If True and min_overlap is set, automatically reduce min_overlap if not enough items found
            progress_callback: Optional callback function(status, current, total, estimated_seconds)
            
        Returns:
            List of item dictionaries with item_title, band_name, item_url, tags, and overlap_count
        """
        import threading
        from concurrent.futures import ThreadPoolExecutor
        
        # Get supporters from the album
        if progress_callback:
            progress_callback("Extracting supporters from album page...", 0, 0, 0)
        supporters = extract_supporters(item_url)
        
        if not supporters:
            if progress_callback:
                progress_callback("No supporters found.", 0, 0, 0)
            return []
        
        if progress_callback:
            progress_callback(f"Found {len(supporters)} supporters", len(supporters), len(supporters), 0)
        
        # Select random supporters
        if len(supporters) > num_supporters:
            selected_supporters = random.sample(supporters, num_supporters)
        else:
            selected_supporters = supporters
        
        if progress_callback:
            progress_callback(f"Checking {len(selected_supporters)} random supporters...", len(selected_supporters), len(selected_supporters), 0)
        
        # Get items from selected supporters
        all_items = []
        start_time = time.time()
        total_supporters = len(selected_supporters)
        completed_count = 0
        completed_lock = Lock()
        
        # Initialize driver pool
        pool_size = min(10, total_supporters)
        if progress_callback:
            progress_callback("Initializing driver pool (this may take a moment)...", 0, total_supporters, 0)
        driver_pool = self._driver_manager.get_driver_pool(pool_size)
        if progress_callback:
            progress_callback(f"Driver pool ready. Fetching items from {total_supporters} supporters...", 0, total_supporters, 0)
        
        def fetch_supporter_items(supporter):
            """Fetch items (purchases or wishlist) for a single supporter (thread-safe)."""
            driver = None
            try:
                driver = driver_pool.get(timeout=30)
                if use_wishlist:
                    items = self._get_supporter_wishlist_with_driver(supporter, driver, extract_tags_flag=False)
                else:
                    items = self._get_supporter_purchases_with_driver(supporter, driver, extract_tags_flag=False)
                return items, supporter
            except Exception:
                return [], supporter
            finally:
                if driver:
                    try:
                        driver_pool.put_nowait(driver)
                    except Exception:
                        try:
                            driver_pool.put(driver, timeout=2)
                        except Exception:
                            pass
        
        # Use ThreadPoolExecutor for parallel processing
        max_workers = min(15, total_supporters)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_supporter = {
                executor.submit(fetch_supporter_items, supporter): supporter
                for supporter in selected_supporters
            }
            
            # Use manual polling to avoid indefinite blocking
            pending_futures = dict(future_to_supporter)
            future_start_times = {f: time.time() for f in pending_futures.keys()}
            max_future_time = 30  # Max seconds per future
            
            while pending_futures:
                completed_this_round = []
                
                for future, supporter in list(pending_futures.items()):
                    # Check if future is done
                    if future.done():
                        completed_this_round.append(future)
                        try:
                            items, supporter = future.result(timeout=1)
                            with completed_lock:
                                all_items.extend(items)
                                completed_count += 1
                                
                                if progress_callback:
                                    elapsed = time.time() - start_time
                                    avg_time = elapsed / completed_count if completed_count > 0 else 2.0
                                    remaining = total_supporters - completed_count
                                    estimated_seconds = avg_time * remaining
                                    item_type = "wishlist items" if use_wishlist else "purchases"
                                    progress_callback(
                                        f"Fetched {len(items)} {item_type} from {supporter} ({completed_count}/{total_supporters})...",
                                        completed_count,
                                        total_supporters,
                                        int(estimated_seconds)
                                    )
                        except Exception:
                            with completed_lock:
                                completed_count += 1
                                if progress_callback:
                                    progress_callback(
                                        f"Error from {supporter} ({completed_count}/{total_supporters})...",
                                        completed_count,
                                        total_supporters,
                                        0
                                    )
                    # Check for timeout
                    elif time.time() - future_start_times[future] > max_future_time:
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
                    time.sleep(0.5)  # Small sleep to avoid busy-waiting
        
        if not all_items:
            if progress_callback:
                progress_callback("No items found.", total_supporters, total_supporters, 0)
            return []
        
        # Get original item ID to exclude it
        original_item_id = extract_item_id(item_url)
        
        # Count item occurrences (for min_overlap filtering)
        item_counts = Counter(all_items)
        
        # Remove the original item from counts
        if original_item_id and original_item_id in item_counts:
            item_counts.pop(original_item_id)
        
        # Filter by min_overlap if specified, with fallback if enabled
        final_overlap = None
        if min_overlap is not None and min_overlap > 1:
            current_overlap = min_overlap
            filtered_items = {}
            
            # Try progressively lower overlap requirements if fallback is enabled
            while current_overlap >= 1:
                filtered_items = {
                    item_id: count
                    for item_id, count in item_counts.items()
                    if count >= current_overlap
                }
                
                # Check if we have enough items at this overlap level
                if filtered_items and len(filtered_items) >= num_items:
                    # Found enough items with current overlap requirement
                    final_overlap = current_overlap
                    if current_overlap < min_overlap and progress_callback:
                        progress_callback(
                            f"Found {len(filtered_items)} items with overlap >= {current_overlap} (fallback from {min_overlap})",
                            total_supporters,
                            total_supporters,
                            0
                        )
                    break
                
                # Not enough items found, try lower overlap if fallback enabled
                if use_fallback and current_overlap > 1:
                    if filtered_items:
                        # Some items found but not enough
                        if progress_callback:
                            progress_callback(
                                f"Found {len(filtered_items)} items with overlap >= {current_overlap} (need {num_items}), trying overlap >= {current_overlap - 1}...",
                                total_supporters,
                                total_supporters,
                                0
                            )
                    else:
                        # No items found at this level
                        if progress_callback:
                            progress_callback(
                                f"No items with overlap >= {current_overlap}, trying overlap >= {current_overlap - 1}...",
                                total_supporters,
                                total_supporters,
                                0
                            )
                    current_overlap -= 1
                else:
                    # No fallback or reached minimum
                    final_overlap = current_overlap
                    if filtered_items:
                        # Some items found but not enough and fallback disabled
                        if progress_callback:
                            progress_callback(
                                f"Found {len(filtered_items)} items with overlap >= {min_overlap} (need {num_items}).",
                                total_supporters,
                                total_supporters,
                                0
                            )
                        # Use what we have (will return fewer items than requested)
                        break
                    else:
                        # No items found
                        if progress_callback:
                            progress_callback(
                                f"No items found in at least {min_overlap} collections.",
                                total_supporters,
                                total_supporters,
                                0
                            )
                        return []
            
            item_counts = filtered_items
        elif min_overlap == 1:
            final_overlap = 1
        
        # Select random items
        unique_items = list(item_counts.keys())
        if len(unique_items) > num_items:
            selected_item_ids = random.sample(unique_items, num_items)
        else:
            selected_item_ids = unique_items
        
        if progress_callback:
            if final_overlap is not None and final_overlap != min_overlap:
                progress_callback(
                    f"Selected {len(selected_item_ids)} random items (using overlap >= {final_overlap}, requested >= {min_overlap}).",
                    total_supporters,
                    total_supporters,
                    0
                )
            else:
                progress_callback(f"Selected {len(selected_item_ids)} random items.", total_supporters, total_supporters, 0)
        
        # Build result list with metadata
        results = []
        for item_id in selected_item_ids:
            item_info = self._get_item_info_from_id(item_id)
            if item_info:
                item_info['overlap_count'] = item_counts.get(item_id, 0)
                if final_overlap is not None:
                    item_info['final_overlap'] = final_overlap
                results.append(item_info)
            else:
                # Fallback if metadata not in cache
                result_item = {
                    'item_id': item_id,
                    'item_title': 'Unknown Title',
                    'band_name': 'Unknown Artist',
                    'item_url': f"https://bandcamp.com/album/{item_id}",
                    'tags': [],
                    'overlap_count': item_counts.get(item_id, 0),
                }
                if final_overlap is not None:
                    result_item['final_overlap'] = final_overlap
                results.append(result_item)
        
        return results

    def close(self):
        """Close the webdriver and cleanup driver pool."""
        self._driver_manager.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

