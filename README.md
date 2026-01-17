# Bandcamp Recommender

A Python package that generates Bandcamp recommendations using collaborative filtering. Finds items that multiple supporters of a given album also purchased.

## Installation

Requires Python >=3.10 and uses `uv` for package management:

```bash
uv sync
```

## Usage

### Command Line

```bash
export PYTHONPATH=$(pwd)
uv run python scripts/get_overlap.py <bandcamp_url> [max_recommendations] [min_supporters]

# Example
uv run python scripts/get_overlap.py "https://artist.bandcamp.com/album/name" 10 2
```

### Python Module

```python
from src.recommendations import SupporterRecommender

with SupporterRecommender() as recommender:
    recommendations = recommender.get_recommendations(
        wishlist_item_url="https://example.bandcamp.com/album/example",
        max_recommendations=10,
        min_supporters=2
    )
    
    for rec in recommendations:
        print(f"{rec['band_name']} - {rec['item_title']}")
        print(f"  Supported by {rec['supporters_count']} people")
```

## How It Works

1. Extracts supporter usernames from the album page
2. Fetches each supporter's collection (using pagedata + API)
3. Counts item occurrences and ranks by popularity
4. Returns top recommendations with metadata

## Technical Details

- Uses `curl` for HTTP requests (no browser popups)
- Selenium (headless) only for authenticated collection access
- Driver pool for efficient parallel processing (~7x faster)
- Automatically detects Chrome/Chromium/Brave/Arc browsers

## Requirements

- Python >=3.10
- Chrome/Chromium browser installed
- `uv` package manager
