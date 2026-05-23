# Bandcamp Recommender

A Python package for Bandcamp music discovery. Two complementary capabilities:

* **Collaborative filtering** — given an album, finds items that multiple
  supporters of that album also purchased / wishlisted.
* **Per-track feature vectors** — given any preview URL and its tags,
  computes a 9-feature vector covering tag mood, audio energy, and
  tempo. Use the bundled distance function for "more like this"
  matching, or project to a single chill ↔ party scalar for mood
  filtering / radio-style toggles.

## Installation

Requires Python >=3.10 and uses `uv` for package management:

```bash
# core: collaborative filtering only
uv sync

# add the audio stack for BPM + intensity feature extraction
uv sync --extra bpm
```

The `bpm` extra pulls in numpy + librosa. It's optional — the package
gracefully degrades to "audio feature unavailable" if the extra isn't
installed, so tag features still work standalone.

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

### Feature vectors (similarity matching, mood projection)

```python
from bandcamp_recommender.features import (
    extract_features, distance, project_mood,
)

# A track is a dict with at least item_url + optional tags / audio_url.
seed = {
    "item_url": "https://artist.bandcamp.com/track/seed",
    "tags": ["downtempo", "trip-hop", "Bristol"],
    "audio_url": "https://t4.bcbits.com/.../seed.mp3",
}
seed_vec = extract_features(seed)

# Pairwise similarity: weighted Euclidean over the intersection
# of features both vectors actually have (missing features are
# skipped, denominator renormalized).
cand_vec = extract_features(some_other_track)
d = distance(seed_vec, cand_vec)        # smaller = more similar

# Single chill (-1) ↔ party (+1) scalar derived from the same
# vector. Use this for a radio-style mood slider.
mood = project_mood(seed_vec)
```

The full feature universe (each in a documented normalized range):

| feature             | source        | range         |
|---------------------|---------------|---------------|
| `tag_mood`          | everynoise    | -1 chill, +1 party |
| `tag_spikiness`     | everynoise    | -1 dense, +1 spiky |
| `rms_mean`          | audio         | [0, 1]        |
| `rms_p95`           | audio         | [0, 1]        |
| `onset_rate`        | audio         | [0, 1]        |
| `spectral_centroid` | audio         | [0, 1]        |
| `crest_factor`      | audio         | [0, 1]        |
| `bpm_norm`          | audio         | [0, 1] over [60, 200] BPM |
| `bpm_folded_norm`   | audio         | [0, 1] over [80, 160) BPM octave |

Audio features need the `bpm` extra installed; tag features work on
their own.

## Architecture

The codebase is organized into modular components:

- `bandcamp_recommender/features.py` - Vector similarity API (`extract_features`, `distance`, `project_mood`)
- `bandcamp_recommender/recommendations/supporter_recommender.py` - Main recommendation engine
- `bandcamp_recommender/recommendations/mood_tags.py` - Everynoise-derived tag-mood + tag-spikiness lexicon
- `bandcamp_recommender/recommendations/intensity.py` - Audio energy / texture feature extraction (librosa)
- `bandcamp_recommender/recommendations/bpm.py` - BPM detection (Joe Sullivan + librosa backends) + `_load_audio_segment` shared decode
- `bandcamp_recommender/recommendations/driver_manager.py` - Selenium WebDriver management & pooling
- `bandcamp_recommender/recommendations/scraper.py` - Web scraping utilities (curl, BeautifulSoup)
- `bandcamp_recommender/recommendations/api.py` - Bandcamp API interaction utilities
- `bandcamp_recommender/recommendations/tags.py` - Tag extraction + normalization utilities

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
