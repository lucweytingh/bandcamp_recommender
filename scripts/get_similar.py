#!/usr/bin/env python3
"""Get recommendations based on tag similarity."""

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
    """Main function to get tag-similar recommendations."""
    if len(sys.argv) < 2:
        print("Usage: python get_similar.py <bandcamp_item_url> [max_recommendations] [min_similarity] [max_supporters]")
        print("  bandcamp_item_url: URL to get recommendations for")
        print("  max_recommendations: Maximum number of recommendations (default: 10)")
        print("  min_similarity: Minimum similarity score 0.0-1.0 (default: 0.1)")
        print("  max_supporters: Maximum number of supporters to fetch from (default: all)")
        sys.exit(1)

    item_url = sys.argv[1]
    max_recommendations = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    min_similarity = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    max_supporters = int(sys.argv[4]) if len(sys.argv) > 4 else None

    print(f"Getting tag-similar recommendations for: {item_url}")
    print(f"Max recommendations: {max_recommendations}, Min similarity: {min_similarity}")
    if max_supporters:
        print(f"Max supporters: {max_supporters}")
    print("-" * 60)

    with SupporterRecommender() as recommender:
        recommendations = recommender.get_tag_similar_recommendations(
            item_url=item_url,
            max_recommendations=max_recommendations,
            min_similarity=min_similarity,
            max_supporters=max_supporters,
            progress_callback=progress_callback,
        )
        
        # Print newline after progress updates
        print()

        if not recommendations:
            print("\nNo similar recommendations found.")
            print("\nPossible reasons:")
            print("  - No tags found for the original item")
            print("  - No items with similar tags found")
            print("  - Minimum similarity threshold too high")
            return

        print(f"\nFound {len(recommendations)} tag-similar recommendations:\n")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['band_name']} - {rec['item_title']}")
            print(f"   URL: {rec['item_url']}")
            print(f"   Similarity: {rec['similarity_score']:.3f}")
            if rec.get('tags'):
                print(f"   Tags: {', '.join(rec['tags'])}")
            if rec.get('supporters_count'):
                print(f"   Also purchased by {rec['supporters_count']} supporters of the original")
            print()


if __name__ == "__main__":
    main()

