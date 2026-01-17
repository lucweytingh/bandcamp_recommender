#!/usr/bin/env python3
"""Get Bandcamp recommendations based on supporter overlap (collaborative filtering)."""

import argparse
from pathlib import Path
import sys

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
    """Main function to demonstrate usage."""
    parser = argparse.ArgumentParser(
        description="Get Bandcamp recommendations based on supporter overlap (collaborative filtering).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s https://artist.bandcamp.com/album/name
  %(prog)s https://artist.bandcamp.com/album/name --max-recommendations 20
  %(prog)s https://artist.bandcamp.com/album/name --max-recommendations 10 --min-supporters 3
        """
    )
    parser.add_argument(
        "url",
        help="Bandcamp item URL (album or track)"
    )
    parser.add_argument(
        "--max-recommendations",
        type=int,
        default=10,
        help="Maximum number of recommendations to return (default: 10)"
    )
    parser.add_argument(
        "--min-supporters",
        type=int,
        default=2,
        help="Minimum number of supporters who must have purchased an item (default: 2)"
    )
    
    args = parser.parse_args()
    
    item_url = args.url
    max_recommendations = args.max_recommendations
    min_supporters = args.min_supporters

    print(f"Getting recommendations for: {item_url}")
    print(f"Max recommendations: {max_recommendations}, Min supporters: {min_supporters}")
    print("-" * 60)

    with SupporterRecommender() as recommender:
        recommendations = recommender.get_recommendations(
            wishlist_item_url=item_url,
            max_recommendations=max_recommendations,
            min_supporters=min_supporters,
            progress_callback=progress_callback,
        )
        
        # Print newline after progress updates
        print()

        if not recommendations:
            print("\nNo recommendations found.")
            print("\nPossible reasons:")
            print("  - Supporter collections are private (most common)")
            print("  - Collections require authentication to access")
            print("  - No overlapping purchases found between supporters")
            print("  - Minimum supporter threshold not met")
            return

        print(f"\nFound {len(recommendations)} recommendations:\n")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['band_name']} - {rec['item_title']}")
            print(f"   URL: {rec['item_url']}")
            print(f"   Supported by {rec['supporters_count']} people who also bought the original")
            if rec.get('tags'):
                print(f"   Tags: {', '.join(rec['tags'])}")
            print()


if __name__ == "__main__":
    main()

