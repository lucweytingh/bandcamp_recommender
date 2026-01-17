# Using bandcamp-recommender as a Package

Since both projects are on the same computer, importing as a package is the simplest approach.

## Installation

### Option 1: Install as Editable Package (Recommended)

In your other project's `pyproject.toml`, add:

```toml
[project]
dependencies = [
    "bandcamp-recommender @ file:///path/to/bandcamp_recommender",
]
```

Or if using `uv` directly:

```bash
cd /path/to/your/other/project
uv add --editable /path/to/bandcamp_recommender
```

### Option 2: Add to PYTHONPATH (Simplest)

In your other project, just add the path:

```python
import sys
from pathlib import Path

# Add bandcamp_recommender to path
sys.path.insert(0, str(Path("/path/to/bandcamp_recommender").resolve()))

from src.recommendations import SupporterRecommender
```

### Option 3: Install in Development Mode

From the bandcamp_recommender directory:

```bash
uv pip install -e .
```

## Basic Usage

```python
from src.recommendations import SupporterRecommender

# Always use as context manager to ensure proper cleanup
with SupporterRecommender() as recommender:
    # Your code here
    pass
```

## Available Methods

### 1. `get_recommendations()` - Collaborative Filtering

Finds items purchased by multiple supporters of the original item.

```python
from src.recommendations import SupporterRecommender

with SupporterRecommender() as recommender:
    recommendations = recommender.get_recommendations(
        wishlist_item_url="https://artist.bandcamp.com/album/name",
        max_recommendations=10,
        min_supporters=2,
        progress_callback=None  # Optional: function(status, current, total, estimated_seconds)
    )
    
    # Returns: List[Dict] with keys:
    # - 'item_title': str
    # - 'band_name': str
    # - 'item_url': str
    # - 'supporters_count': int
    # - 'tags': List[str] (if available)
    
    for rec in recommendations:
        print(f"{rec['band_name']} - {rec['item_title']}")
        print(f"  Supported by {rec['supporters_count']} people")
```

**Parameters:**
- `wishlist_item_url` (str): URL of the Bandcamp item (album or track)
- `max_recommendations` (int, default=10): Maximum number of recommendations
- `min_supporters` (int, default=2): Minimum number of supporters who must have purchased an item
- `progress_callback` (Callable, optional): Function(status, current, total, estimated_seconds)

**Returns:** `List[Dict[str, Any]]`

---

### 2. `get_tag_similar_recommendations()` - Tag-Based Similarity

Finds items with similar tags to the original item using TF-IDF weighted Jaccard similarity.

```python
from src.recommendations import SupporterRecommender

with SupporterRecommender() as recommender:
    similar = recommender.get_tag_similar_recommendations(
        item_url="https://artist.bandcamp.com/album/name",
        max_recommendations=10,
        min_similarity=0.1,
        max_supporters=None,  # None = all supporters
        progress_callback=None
    )
    
    # Returns: List[Dict] with keys:
    # - 'item_title': str
    # - 'band_name': str
    # - 'item_url': str
    # - 'tags': List[str]
    # - 'similarity_score': float (0.0 to 1.0)
    # - 'supporters_count': int
    
    for item in similar:
        print(f"{item['band_name']} - {item['item_title']}")
        print(f"  Similarity: {item['similarity_score']:.3f}")
        print(f"  Tags: {', '.join(item['tags'])}")
```

**Parameters:**
- `item_url` (str): URL of the Bandcamp item
- `max_recommendations` (int, default=10): Maximum number of recommendations
- `min_similarity` (float, default=0.1): Minimum similarity score (0.0 to 1.0)
- `max_supporters` (int, optional): Maximum number of supporters to fetch from (None = all)
- `progress_callback` (Callable, optional): Function(status, current, total, estimated_seconds)

**Returns:** `List[Dict[str, Any]]`

---

### 3. `get_random_items()` - Random Items from Supporters

Gets random items from random supporters' collections, with optional overlap filtering.

```python
from src.recommendations import SupporterRecommender

with SupporterRecommender() as recommender:
    random_items = recommender.get_random_items(
        item_url="https://artist.bandcamp.com/album/name",
        num_items=10,
        num_supporters=15,
        use_wishlist=False,  # True for wishlist items, False for purchases
        min_overlap=3,  # Only items found in at least N collections
        use_fallback=True,  # Automatically reduce overlap if not enough items
        progress_callback=None
    )
    
    # Returns: List[Dict] with keys:
    # - 'item_title': str
    # - 'band_name': str
    # - 'item_url': str
    # - 'tags': List[str] (empty if extract_tags=False)
    # - 'overlap_count': int (number of collections containing this item)
    # - 'final_overlap': int (actual overlap level used, if fallback was used)
    
    for item in random_items:
        print(f"{item['band_name']} - {item['item_title']}")
        print(f"  Found in {item['overlap_count']} collections")
        if 'final_overlap' in item:
            print(f"  (Used overlap >= {item['final_overlap']})")
```

**Parameters:**
- `item_url` (str): URL of the Bandcamp item to get supporters from
- `num_items` (int): Number of random items to return
- `num_supporters` (int, default=20): Number of random supporters to check
- `use_wishlist` (bool, default=False): Use wishlist items instead of purchases
- `min_overlap` (int, optional): Only select items found in at least N collections (None = any item)
- `use_fallback` (bool, default=False): If True and min_overlap is set, automatically reduce min_overlap if not enough items found
- `progress_callback` (Callable, optional): Function(status, current, total, estimated_seconds)

**Returns:** `List[Dict[str, Any]]`

**Fallback Behavior:**
When `use_fallback=True` and `min_overlap` is set:
- If not enough items found at overlap >= N, tries overlap >= N-1
- Continues until enough items found or reaches overlap >= 1
- Makes a new random selection from the larger pool at each level
- Returns items with the highest overlap level that has enough items

---

## Progress Callback

All methods support an optional `progress_callback` function for real-time progress updates:

```python
def my_progress_callback(status, current, total, estimated_seconds):
    """Progress callback function.
    
    Args:
        status: Status message string
        current: Current progress (number completed)
        total: Total number of items
        estimated_seconds: Estimated seconds remaining
    """
    if total > 0:
        percentage = (current / total) * 100
        print(f"[{percentage:.1f}%] {status}")

with SupporterRecommender() as recommender:
    results = recommender.get_recommendations(
        wishlist_item_url="https://...",
        progress_callback=my_progress_callback
    )
```

---

## Complete Example

```python
from src.recommendations import SupporterRecommender

# Example: Get random items with overlap filtering and fallback
with SupporterRecommender() as recommender:
    # Get 10 random purchases from 15 random supporters
    # Only items found in at least 3 collections
    # If not enough, automatically try lower overlap levels
    items = recommender.get_random_items(
        item_url="https://artist.bandcamp.com/album/name",
        num_items=10,
        num_supporters=15,
        min_overlap=3,
        use_fallback=True
    )
    
    print(f"Found {len(items)} items")
    for item in items:
        print(f"\n{item['band_name']} - {item['item_title']}")
        print(f"  URL: {item['item_url']}")
        print(f"  Found in {item['overlap_count']} collections")
        if 'final_overlap' in item:
            print(f"  (Used overlap >= {item['final_overlap']} due to fallback)")
```

---

## Notes

- Always use `SupporterRecommender` as a context manager (`with` statement) to ensure proper cleanup
- The original item is automatically excluded from all results
- Collections may be private and require authentication (most common reason for empty results)
- Tag extraction can be slow; it's automatically skipped in `get_random_items()` for performance
- Driver pool is automatically managed for parallel processing (up to 15 concurrent workers)
