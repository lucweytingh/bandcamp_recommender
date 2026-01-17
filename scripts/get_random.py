#!/usr/bin/env python3
"""Get random items (purchases or wishlist) from random supporters."""

import argparse
import random
import sys
import time
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.recommendations import SupporterRecommender
from src.recommendations.scraper import extract_item_id


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
    parser = argparse.ArgumentParser(
        description="Get random items (purchases or wishlist) from random supporters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s https://artist.bandcamp.com/album/name 10
  %(prog)s https://artist.bandcamp.com/album/name 10 --num-supporters 30
  %(prog)s https://artist.bandcamp.com/album/name 5 --wishlist
  %(prog)s https://artist.bandcamp.com/album/name 10 --min-overlap 2
        """
    )
    parser.add_argument(
        "url",
        help="Bandcamp item URL (album or track) to get supporters from"
    )
    parser.add_argument(
        "num_items",
        type=int,
        help="Number of random items to return"
    )
    parser.add_argument(
        "--num-supporters",
        type=int,
        default=20,
        help="Number of random supporters to check (default: 20)"
    )
    parser.add_argument(
        "--wishlist",
        action="store_true",
        help="Use wishlist items instead of purchases (default: purchases)"
    )
    parser.add_argument(
        "--min-overlap",
        type=int,
        default=None,
        metavar="N",
        help="Only select items found in at least N supporters' collections (default: 1, i.e., any item)"
    )
    
    args = parser.parse_args()
    
    item_url = args.url
    num_items = args.num_items
    num_supporters = args.num_supporters
    use_wishlist = args.wishlist
    min_overlap = args.min_overlap
    
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
        
        # Get original item ID to exclude it
        original_item_id = extract_item_id(item_url)
        
        # Count item occurrences (for min_overlap filtering)
        item_counts = Counter(all_items)
        
        # Remove the original item from counts
        if original_item_id and original_item_id in item_counts:
            item_counts.pop(original_item_id)
            print(f"\nFound {len(item_counts)} unique {item_type_plural}")
        else:
            print(f"\nFound {len(item_counts)} unique {item_type_plural}")
        
        # Filter by min_overlap if specified
        if min_overlap is not None and min_overlap > 1:
            filtered_items = [
                item_id for item_id, count in item_counts.items()
                if count >= min_overlap
            ]
            print(f"Items found in at least {min_overlap} collections: {len(filtered_items)}")
            
            if not filtered_items:
                print(f"\nNo items found in at least {min_overlap} collections.")
                return
            
            # Select random items from filtered list
            if len(filtered_items) > num_items:
                selected_items = random.sample(filtered_items, num_items)
            else:
                selected_items = filtered_items
        else:
            # Select random items from all unique items
            unique_items = list(item_counts.keys())
            if len(unique_items) > num_items:
                selected_items = random.sample(unique_items, num_items)
            else:
                selected_items = unique_items
        
        print(f"\nSelected {len(selected_items)} random items:\n")
        
        for i, item_id in enumerate(selected_items, 1):
            item_info = recommender._get_item_info_from_id(item_id)
            if item_info:
                overlap_count = item_counts.get(item_id, 0)
                print(f"{i}. {item_info['band_name']} - {item_info['item_title']}")
                print(f"   URL: {item_info['item_url']}")
                if min_overlap is not None and min_overlap > 1:
                    print(f"   Found in {overlap_count} collection(s)")
                if item_info.get('tags'):
                    print(f"   Tags: {', '.join(item_info['tags'])}")
            else:
                overlap_count = item_counts.get(item_id, 0)
                print(f"{i}. Item ID: {item_id} (metadata not available)")
                if min_overlap is not None and min_overlap > 1:
                    print(f"   Found in {overlap_count} collection(s)")
            print()


if __name__ == "__main__":
    main()

