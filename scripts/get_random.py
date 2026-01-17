#!/usr/bin/env python3
"""Get random items (purchases or wishlist) from random supporters."""

import argparse
import sys
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
    parser = argparse.ArgumentParser(
        description="Get random items (purchases or wishlist) from random supporters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s https://artist.bandcamp.com/album/name 10
  %(prog)s https://artist.bandcamp.com/album/name 10 --num-supporters 30
  %(prog)s https://artist.bandcamp.com/album/name 5 --wishlist
  %(prog)s https://artist.bandcamp.com/album/name 10 --min-overlap 2
  %(prog)s https://artist.bandcamp.com/album/name 10 --min-overlap 5 --use-fallback
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
    parser.add_argument(
        "--use-fallback",
        action="store_true",
        help="If min-overlap is not met, automatically try lower overlap values (N-1, N-2, etc.) until items are found"
    )
    
    args = parser.parse_args()
    
    item_url = args.url
    num_items = args.num_items
    num_supporters = args.num_supporters
    use_wishlist = args.wishlist
    min_overlap = args.min_overlap
    use_fallback = args.use_fallback
    
    item_type = "wishlist" if use_wishlist else "purchase"
    item_type_plural = "wishlist items" if use_wishlist else "purchases"

    print(f"Getting {num_items} random {item_type_plural} from {num_supporters} random supporters")
    print(f"Source album: {item_url}")
    print("-" * 60)

    with SupporterRecommender() as recommender:
        results = recommender.get_random_items(
            item_url=item_url,
            num_items=num_items,
            num_supporters=num_supporters,
            use_wishlist=use_wishlist,
            min_overlap=min_overlap,
            use_fallback=use_fallback,
            progress_callback=progress_callback,
        )
        
        # Print newline after progress updates
        print()
        
        if not results:
            print(f"\nNo {item_type_plural} found.")
            return
        
        # Check if fallback was used
        final_overlap = results[0].get('final_overlap') if results else None
        if final_overlap is not None and min_overlap is not None and final_overlap != min_overlap:
            print(f"\nSelected {len(results)} random items (using overlap >= {final_overlap}, requested >= {min_overlap}):\n")
        else:
            print(f"\nSelected {len(results)} random items:\n")
        
        for i, item_info in enumerate(results, 1):
            print(f"{i}. {item_info['band_name']} - {item_info['item_title']}")
            print(f"   URL: {item_info['item_url']}")
            if min_overlap is not None and min_overlap > 1:
                print(f"   Found in {item_info.get('overlap_count', 0)} collection(s)")
            if item_info.get('tags'):
                print(f"   Tags: {', '.join(item_info['tags'])}")
            print()


if __name__ == "__main__":
    main()

