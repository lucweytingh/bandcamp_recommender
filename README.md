# Bandcamp Recommender

A Python package that generates Bandcamp recommendations using collaborative filtering. Finds items that multiple supporters of a given album also purchased.

## Installation

Requires Python >=3.10 and uses `uv` for package management:

```bash
uv sync
```

## Usage

### Command Line Scripts

Two recommendation modes are available:

#### 1. Collaborative Filtering (Overlap)
Finds items purchased by multiple supporters of the original item:

```bash
export PYTHONPATH=$(pwd)
uv run python scripts/get_overlap.py <bandcamp_url> [max_recommendations] [min_supporters]

# Example
uv run python scripts/get_overlap.py "https://artist.bandcamp.com/album/name" 10 2
```

#### 2. Random Items
Gets random purchases/wishlist items from random supporters:

```bash
uv run python scripts/get_random.py <bandcamp_url> <num_items> [num_supporters] [--wishlist]

# Example - 10 random purchases from 20 random supporters
uv run python scripts/get_random.py "https://artist.bandcamp.com/album/name" 10 20

# Example - 5 random wishlist items
uv run python scripts/get_random.py "https://artist.bandcamp.com/album/name" 5 20 --wishlist
```

### Python Module

```python
from bandcamp_recommender import SupporterRecommender

with SupporterRecommender() as recommender:
    # Collaborative filtering
    recommendations = recommender.get_recommendations(
        wishlist_item_url="https://example.bandcamp.com/album/example",
        max_recommendations=10,
        min_supporters=2
    )
```

## Architecture

The codebase is organized into modular components:

- `bandcamp_recommender/recommendations/supporter_recommender.py` - Main recommendation engine
- `bandcamp_recommender/recommendations/driver_manager.py` - Selenium WebDriver management & pooling
- `bandcamp_recommender/recommendations/scraper.py` - Web scraping utilities (curl, BeautifulSoup)
- `bandcamp_recommender/recommendations/api.py` - Bandcamp API interaction utilities
- `bandcamp_recommender/recommendations/tags.py` - Tag extraction utilities

## How It Works

### Collaborative Filtering
1. Extracts supporter usernames from the album/track page
2. Fetches each supporter's collection (using pagedata + API)
3. Counts item occurrences and ranks by popularity
4. Returns top recommendations with metadata

## Technical Details

- Uses `curl` for HTTP requests (no browser popups for most operations)
- Selenium (headless) only for authenticated collection access
- Driver pool for efficient parallel processing (~7x faster)
- Thread-safe caching of item metadata
- Automatically detects Chrome/Chromium/Brave/Arc browsers
- Modular architecture for maintainability

## Using as a Package

This package can be imported and used in other Python projects on the same computer.

### Installation

Add to your project's `pyproject.toml`:
```toml
[project]
dependencies = [
    "bandcamp-recommender @ file:///path/to/bandcamp_recommender",
]
```

Or with `uv`:
```bash
uv add --editable /path/to/bandcamp_recommender
```

### Usage

```python
from bandcamp_recommender import SupporterRecommender

# Collaborative filtering
with SupporterRecommender() as recommender:
    recs = recommender.get_recommendations(
        wishlist_item_url="https://artist.bandcamp.com/album/name",
        max_recommendations=10
    )

# Random items with overlap filtering
with SupporterRecommender() as recommender:
    items = recommender.get_random_items(
        item_url="https://artist.bandcamp.com/album/name",
        num_items=10,
        num_supporters=15,
        min_overlap=3,
        use_fallback=True
    )
```

See `USAGE_AS_PACKAGE.md` for complete documentation.

## Requirements

- Python >=3.10
- Chrome/Chromium browser installed
- `uv` package manager
