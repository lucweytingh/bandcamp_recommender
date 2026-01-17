# Bandcamp Recommender

A standalone Python package that generates Bandcamp recommendations based on collaborative filtering. If multiple people who purchased item A also purchased item B, then B is recommended for someone interested in A.

## How It Works

1. **Get Supporters**: Scrapes the "supported by X fans" section from a Bandcamp item page to find users who purchased it
2. **Get Their Purchases**: For each supporter, fetches their collection using the Bandcamp API
3. **Count & Rank**: Counts how many supporters purchased each item and ranks them
4. **Return Recommendations**: Returns top items with metadata (title, artist, URL, supporter count)

## Installation

This project uses `uv` for package management and requires Python >=3.10.

```bash
# Install dependencies
uv sync

# The package will automatically install all required dependencies
```

## Usage

### As a Python Module

```python
from src.recommendations import SupporterRecommender

# Use context manager for automatic cleanup
with SupporterRecommender(headless=True) as recommender:
    recommendations = recommender.get_recommendations(
        wishlist_item_url="https://example.bandcamp.com/album/example",
        max_recommendations=10,
        min_supporters=2
    )

    for rec in recommendations:
        print(f"{rec['band_name']} - {rec['item_title']}")
        print(f"  Supported by {rec['supporters_count']} people who also bought the original")
        print(f"  URL: {rec['item_url']}")
```

### As a Script

```bash
# Set PYTHONPATH
export PYTHONPATH=$(pwd)

# Run the script
uv run python scripts/get_recommendations.py <bandcamp_item_url> [max_recommendations] [min_supporters]

# Example
uv run python scripts/get_recommendations.py "https://artist.bandcamp.com/album/name" 10 2
```

## API Reference

### `SupporterRecommender`

Main class for generating recommendations.

#### `__init__(headless: bool = True)`

Initialize the recommender with a Selenium webdriver.

- `headless`: Whether to run browser in headless mode (default: True)

#### `get_recommendations(wishlist_item_url: str, max_recommendations: int = 10, min_supporters: int = 2) -> List[Dict[str, Any]]`

Get recommendations based on supporter purchases.

**Parameters:**
- `wishlist_item_url`: URL of the Bandcamp item to get recommendations for
- `max_recommendations`: Maximum number of recommendations to return (default: 10)
- `min_supporters`: Minimum number of supporters who must have purchased an item (default: 2)

**Returns:**
List of recommendation dictionaries with:
- `item_title`: Title of the recommended item
- `band_name`: Name of the artist/band
- `item_url`: URL of the recommended item
- `supporters_count`: Number of supporters who also purchased this item

#### `close()`

Close the webdriver. Automatically called when using context manager.

## Project Structure

```
bandcamp_recommender/
├── src/
│   ├── __init__.py
│   └── recommendations.py    # Core recommendation engine
├── scripts/
│   └── get_recommendations.py  # Example script
├── pyproject.toml            # Package configuration
├── README.md                 # This file
└── context.md                # Implementation context
```

## Dependencies

- `beautifulsoup4>=4.14.3` - HTML parsing
- `selenium>=4.39.0` - Web automation (only for collection pages requiring cookies)
- `selenium-wire>=5.1.0` - Enhanced Selenium with request interception
- `webdriver-manager>=4.0.2` - Automatic ChromeDriver management
- `blinker<1.8` - Event signaling (compatibility requirement for selenium-wire)
- `setuptools>=80.9.0` - Package utilities (required by selenium-wire)

Note: The package uses `curl` (system command) for HTTP requests instead of Python libraries for better reliability and to avoid browser popups.

## Technical Details

### Architecture

The package minimizes browser usage to avoid popup windows:

1. **Supporter Extraction**: Uses `curl` to fetch HTML pages (no browser needed)
2. **Collection API Calls**: Uses `curl` with cookies from Selenium (browser only when needed)
3. **Selenium Usage**: Only initialized when accessing collection pages that require authentication cookies

### Bandcamp API

The package uses the Bandcamp collection API:
- **Endpoint**: `POST https://bandcamp.com/api/fancollection/1/collection_items`
- **Payload**: `{"fan_id": <int>, "older_than_token": "<token>", "count": 10000}`
- **Authentication**: Requires cookies from a Selenium session (only when accessing collections)

The API is much faster than scraping (1-2 seconds vs 30+ seconds per supporter).

### Browser Requirements

Selenium is only used when accessing collection pages. The package automatically detects and uses:
- Google Chrome
- Chromium
- Brave Browser
- Arc Browser

ChromeDriver is automatically managed via `webdriver-manager`. The browser always runs in headless mode to prevent popup windows.

## Limitations

- Requires a browser (Chrome/Chromium) to be installed
- API calls require cookies from a Selenium session
- Rate limiting may apply when fetching many supporters' collections
- Some Bandcamp pages may have different HTML structures

## Development

```bash
# Set up development environment
export PYTHONPATH=$(pwd)

# Run tests (if available)
uv run pytest

# Format code (if configured)
uv run black src/ scripts/
```

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

