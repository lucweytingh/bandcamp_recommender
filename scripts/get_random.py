#!/usr/bin/env python3
"""Get random items (purchases or wishlist) from random supporters."""

import sys
import random
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.recommendations import SupporterRecommender


def format_time(seconds):
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"


def progress_callback(status, current, total, estimated_seconds):
    """Display progress to user."""
    if total > 0:
        percentage = (current / total) * 100
        if estimated_seconds > 0:
            time_str = format_time(estimated_seconds)
            message = f"[{percentage:5.1f}%] {status} (~{time_str} remaining)"
        else:
            message = f"[{percentage:5.1f}%] {status}"
    else:
        message = status
    
    # Use ANSI escape code to clear to end of line (\033[K)
    print(f"\r\033[K{message}", end="", flush=True)


def main():
    """Main function to get random items from supporters."""
    # Parse arguments
    use_wishlist = False
    args = []
    
    for arg in sys.argv[1:]:
        if arg == "--wishlist":
            use_wishlist = True
        else:
            args.append(arg)
    
    if len(args) < 2:
        print("Usage: python get_random.py <bandcamp_item_url> <num_items> [num_supporters] [--wishlist]")
        print("  bandcamp_item_url: URL to get supporters from")
        print("  num_items: Number of random items to return")
        print("  num_supporters: Number of random supporters to check (default: 20)")
        print("  --wishlist: Use wishlist items instead of purchases (default: purchases)")
        sys.exit(1)

    item_url = args[0]
    num_items = int(args[1])
    num_supporters = int(args[2]) if len(args) > 2 else 20
    
    item_type = "wishlist" if use_wishlist else "purchase"
    item_type_plural = "wishlist items" if use_wishlist else "purchases"

    print(f"Getting {num_items} random {item_type_plural} from {num_supporters} random supporters")
    print(f"Source album: {item_url}")
    print("-" * 60)

    with SupporterRecommender() as recommender:
        # Get supporters from the album
        if progress_callback:
            progress_callback("Extracting supporters from album page...", 0, 0, 0)
        supporters = recommender._get_supporters(item_url)
        
        if not supporters:
            print("\nNo supporters found.")
            return
        
        print(f"\nFound {len(supporters)} supporters")
        
        # Select random supporters
        if len(supporters) > num_supporters:
            selected_supporters = random.sample(supporters, num_supporters)
        else:
            selected_supporters = supporters
        
        print(f"Checking {len(selected_supporters)} random supporters...\n")
        
        # Get items from selected supporters
        all_items = []
        start_time = time.time()
        total_supporters = len(selected_supporters)
        completed_count = 0
        completed_lock = threading.Lock()
        
        # Initialize driver pool (this can take a few seconds)
        pool_size = min(15, total_supporters)
        if progress_callback:
            progress_callback("Initializing driver pool (this may take a moment)...", 0, total_supporters, 0)
        driver_pool = recommender._get_driver_pool(pool_size)
        if progress_callback:
            progress_callback(f"Driver pool ready. Fetching items from {total_supporters} supporters...", 0, total_supporters, 0)
        
        def fetch_supporter_items(supporter):
            """Fetch items (purchases or wishlist) for a single supporter (thread-safe)."""
            driver = None
            try:
                driver = driver_pool.get(timeout=30)
                if use_wishlist:
                    items = recommender._get_supporter_wishlist_with_driver(supporter, driver, extract_tags_flag=False)
                else:
                    items = recommender._get_supporter_purchases_with_driver(supporter, driver, extract_tags_flag=False)
                return items, supporter
            except Exception as e:
                return [], supporter
            finally:
                if driver:
                    try:
                        driver_pool.put_nowait(driver)
                    except:
                        try:
                            driver_pool.put(driver, timeout=2)
                        except:
                            pass
        
        # Use ThreadPoolExecutor for parallel processing
        max_workers = min(15, total_supporters)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_supporter = {
                executor.submit(fetch_supporter_items, supporter): supporter
                for supporter in selected_supporters
            }
            
            # Use manual polling to avoid indefinite blocking (same fix as get_similar.py)
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
                                    progress_callback(
                                        f"Fetched {len(items)} {item_type_plural} from {supporter} ({completed_count}/{total_supporters})...",
                                        completed_count,
                                        total_supporters,
                                        int(estimated_seconds)
                                    )
                        except Exception as e:
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
        
        # Print newline after progress updates
        print()
        
        if not all_items:
            print(f"\nNo {item_type_plural} found.")
            return
        
        # Remove duplicates while preserving some randomness
        unique_items = list(set(all_items))
        print(f"\nFound {len(unique_items)} unique {item_type_plural}")
        
        # Select random items
        if len(unique_items) > num_items:
            selected_items = random.sample(unique_items, num_items)
        else:
            selected_items = unique_items
        
        print(f"\nSelected {len(selected_items)} random items:\n")
        
        for i, item_id in enumerate(selected_items, 1):
            item_info = recommender._get_item_info_from_id(item_id)
            if item_info:
                print(f"{i}. {item_info['band_name']} - {item_info['item_title']}")
                print(f"   URL: {item_info['item_url']}")
                if item_info.get('tags'):
                    print(f"   Tags: {', '.join(item_info['tags'])}")
            else:
                print(f"{i}. Item ID: {item_id} (metadata not available)")
            print()


if __name__ == "__main__":
    main()

