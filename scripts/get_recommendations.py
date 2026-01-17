#!/usr/bin/env python3
"""Example script to get Bandcamp recommendations."""

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
    """Main function to demonstrate usage."""
    if len(sys.argv) < 2:
        print("Usage: python get_recommendations.py <bandcamp_item_url> [max_recommendations] [min_supporters]")
        sys.exit(1)

    item_url = sys.argv[1]
    max_recommendations = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    min_supporters = int(sys.argv[3]) if len(sys.argv) > 3 else 2

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
            print()


if __name__ == "__main__":
    main()

