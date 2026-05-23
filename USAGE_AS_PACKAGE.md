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

from bandcamp_recommender import SupporterRecommender
```

### Option 3: Install in Development Mode

From the bandcamp_recommender directory:

```bash
uv pip install -e .
```

## Basic Usage

```python
from bandcamp_recommender import SupporterRecommender

# Always use as context manager to ensure proper cleanup
with SupporterRecommender() as recommender:
    # Your code here
    pass
```

## Available Methods

### 1. `get_recommendations()` - Collaborative Filtering

Finds items purchased by multiple supporters of the original item.

```python
from bandcamp_recommender import SupporterRecommender

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

### 2. `get_random_items()` - Random Items from Supporters

Gets random items from random supporters' collections, with optional overlap filtering.

```python
from bandcamp_recommender import SupporterRecommender

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

### 3. `get_similar_recommendations()` - One-call "more like this"

End-to-end pipeline: from a source URL → supporter-overlap candidate
pool → feature extraction (shared decode per track) → distance ranking
→ enriched recommendation list. Use this when your downstream consumer
(e.g. a radio queue) needs full per-track features attached and
ordering by audio/tag similarity rather than supporter count.

```python
from bandcamp_recommender import SupporterRecommender

with SupporterRecommender() as r:
    similar = r.get_similar_recommendations(
        source_url="https://artist.bandcamp.com/track/seed",
        max_recommendations=10,
        candidate_pool_size=30,           # how many supporter-overlap candidates to score
        min_supporters=1,
        feature_weights=None,             # or override DEFAULT_WEIGHTS
        intensity_duration=60.0,
        bpm_duration=60.0,
        progress_callback=None,
    )

    for rec in similar:
        bpm = rec["features"]["bpm"]
        bpm_str = f"{bpm:.0f} BPM" if bpm is not None else "no BPM"
        print(f"d={rec['distance']:.3f}  {rec['band_name']} - {rec['item_title']}  ({bpm_str})")
```

**Returned dict per recommendation:**

```python
{
    # Standard recommendation metadata
    "item_title":      "...",
    "band_name":       "...",
    "item_url":        "https://...",
    "supporters_count": int,
    "tags":            ["..."],

    # Added by get_similar_recommendations
    "audio_url":       "https://t4.bcbits.com/...mp3" | None,
    "features": {
        "tag_mood":          float | None,
        "tag_spikiness":     float | None,
        "rms_mean":          float | None,
        "rms_p95":           float | None,
        "onset_rate":        float | None,
        "spectral_centroid": float | None,
        "crest_factor":      float | None,
        "bpm_folded_norm":   float | None,
        "bpm_norm":          float | None,
        "bpm":               float | None,   # raw BPM, not in DEFAULT_WEIGHTS
    },
    "distance":  float | None,    # ascending order; None sinks to bottom
}
```

**How it differs from `get_recommendations`:**

| | `get_recommendations` | `get_similar_recommendations` |
|---|---|---|
| Ordering | by `supporters_count` (popularity-among-overlap) | by feature distance (audio + tag similarity) |
| Per-rec features | only when you opt in via `include_bpm` / `include_intensity` / `include_mood_tag_score` | full vector + raw BPM always attached |
| Audio URL hydration | not done | done for every returned item |
| Cost | 1 supporter scrape + tag hydration | same + N audio decodes (~1–3 s each) |

Use `get_recommendations` for fast "what's popular among shared fans"
lists; use `get_similar_recommendations` when downstream wants to play
the result and needs the feature vector for filtering / display /
beat-matching.

---

### 4. Feature vectors (`bandcamp_recommender.features`)

Per-track feature vectors for "more like this" similarity matching and
mood projection. Works independently of `SupporterRecommender` — given
any track with tags and/or a preview URL, you get a normalized feature
dict that can be compared against any other track's vector.

```python
from bandcamp_recommender.features import (
    extract_features,
    distance,
    project_mood,
    DEFAULT_WEIGHTS,
    FEATURE_RANGES,
)

# Each "item" is a dict with at least item_url; tags / audio_url are
# optional. Tags must already be hydrated by the caller (extract_features
# does not fetch the Bandcamp page).
seed = {
    "item_url": "https://artist.bandcamp.com/track/seed",
    "tags": ["downtempo", "trip-hop", "Bristol"],
    "audio_url": "https://t4.bcbits.com/.../seed.mp3",
}

vec = extract_features(seed)
# vec is a Dict[str, float | None] with keys from DEFAULT_WEIGHTS.
# Any feature that couldn't be computed (no audio URL, no recognised
# tag, no librosa installed) is None — not missing from the dict.
```

**Feature universe and ranges**

```python
{
    "tag_mood":          (-1.0, 1.0),   # chill ↔ party (everynoise top axis)
    "tag_spikiness":     (-1.0, 1.0),   # dense ↔ spiky (everynoise left axis)
    "rms_mean":          (0.0, 1.0),    # mean RMS energy, normalized
    "rms_p95":           (0.0, 1.0),    # 95th-percentile RMS, normalized
    "onset_rate":        (0.0, 1.0),    # onsets/sec, normalized
    "spectral_centroid": (0.0, 1.0),    # brightness, normalized
    "crest_factor":      (0.0, 1.0),    # transient punchiness, normalized
    "bpm_folded_norm":   (0.0, 1.0),    # BPM folded to [80, 160), normalized
    "bpm_norm":          (0.0, 1.0),    # raw BPM normalized over [60, 200]
}
```

**Distance** — weighted Euclidean over the intersection of features both
vectors have. Missing features don't kill the score; the denominator is
the sum of weights of *present* features, so a track with 6 of 9 known
features still gets a distance on the same scale as a fully-featured
one.

```python
cand = extract_features(some_other_track)
d = distance(seed, cand)            # uses DEFAULT_WEIGHTS
# Or override:
d = distance(seed, cand, weights={
    **DEFAULT_WEIGHTS,
    "tag_mood": 2.0,                # double the mood axis
    "bpm_norm": 0.0,                # ignore raw BPM entirely
})
```

`distance` returns `None` when no feature is shared between the two
vectors (rare in practice but the right behavior — there's nothing to
say).

**`project_mood`** — collapse a vector back to a single chill ↔ party
scalar for UI use (e.g. a radio's mood slider).

```python
mood = project_mood(vec)             # in [-1, 1] (None if no signal)
```

The default projection blends `tag_mood` with the audio energy features
that correlate with intensity (`rms_p95`, `onset_rate`, `crest_factor`,
`spectral_centroid`, `bpm_folded_norm`). Pass `weights={…}` to
customise, same key set as `extract_features` output.

**Tag-only and audio-only modes**

If you don't have an audio URL (or don't want to pay the decode cost),
omit it — the audio features come back `None` and `distance` /
`project_mood` rely on whatever tag features resolved:

```python
tag_only = {"item_url": "...", "tags": ["ambient", "drone"]}
extract_features(tag_only)
# {'tag_mood': -0.41, 'tag_spikiness': -0.71,
#  'rms_mean': None, ..., 'bpm_norm': None}
```

Conversely, audio-only items (preview URL but no resolvable tags) get
None tag features and a 7-dim audio + BPM vector.

**Caching**

Both intensity and BPM extraction cache per `audio_url` in-process, so
a radio that calls `extract_features` lazily as tracks queue up only
pays the network + decode cost once per track. Cache survives the
lifetime of the Python process; clear via:

```python
from bandcamp_recommender.recommendations.intensity import clear_intensity_cache
from bandcamp_recommender.recommendations.bpm import clear_bpm_cache, clear_seed_bpm_cache
clear_intensity_cache()
clear_bpm_cache()
clear_seed_bpm_cache()
```

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
from bandcamp_recommender import SupporterRecommender

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

## Audio Intensity Score (for radio-style consumers)

When the caller passes `include_intensity=True`, each recommendation gets
an `intensity` key in `[0.0, 1.0]` (or `None` if no preview was available).
The score blends RMS energy, onset rate, spectral centroid, and crest
factor — see `bandcamp_recommender/recommendations/intensity.py` for the
normalisation constants and weights.

Typical use from a downstream radio that switches between "chill" and
"party" modes:

```python
from bandcamp_recommender import SupporterRecommender

with SupporterRecommender() as recommender:
    recs = recommender.get_recommendations(
        wishlist_item_url="https://artist.bandcamp.com/album/name",
        max_recommendations=30,
        include_bpm=True,
        include_intensity=True,
    )

# Sort low → high for a chill set, high → low for a party set.
chill = sorted(
    (r for r in recs if r.get("intensity") is not None),
    key=lambda r: r["intensity"],
)
party = list(reversed(chill))

# Or switch modes by threshold.
mode = "party" if user_mode == "party" else "chill"
target = 0.75 if mode == "party" else 0.25
recs.sort(key=lambda r: abs((r.get("intensity") or 0.5) - target))
```

When both `include_bpm` and `include_intensity` are True, each track's
preview audio is downloaded and decoded once and shared between the two
detectors, so enabling both costs roughly the same as enabling either.

---

## Notes

- Always use `SupporterRecommender` as a context manager (`with` statement) to ensure proper cleanup
- The original item is automatically excluded from all results
- Collections may be private and require authentication (most common reason for empty results)
- Tag extraction can be slow; it's automatically skipped in `get_random_items()` for performance
- Driver pool is automatically managed for parallel processing (up to 15 concurrent workers)

---

## Updating to a New Version (for agents in consumer repos)

**Current version: `0.2.0`** (was `0.1.1`). The public API is unchanged — `SupporterRecommender`, `get_recommendations`, `get_random_items`, and `get_tag_similar_recommendations` all keep the same signatures and return shapes. Internally, supporter fetches now go through `curl` first and only fall back to Selenium if blocked, which is ~40% faster on `get_random.py` and avoids spawning Chrome on healthy runs.

### Pick the right command based on how the package was installed

Check the consumer's `pyproject.toml` for a `bandcamp-recommender` entry to determine the install style.

**1. Editable install (`uv add --editable /path/...` or `bandcamp-recommender @ file:///...` with `[tool.uv.sources] bandcamp-recommender = { path = "...", editable = true }`):**

The consumer points at the source tree directly. Just pull the source:
```bash
cd /path/to/bandcamp_recommender   # the path the consumer points at
git pull --ff-only origin master
```
No `uv sync` needed — next Python process picks up the new code.

**2. Non-editable file-URL install (`bandcamp-recommender @ file:///path/to/bandcamp_recommender` with no editable flag):**

`uv` caches the built wheel against the version string, so a behavior change without a version bump would silently use the old code. Since this update bumps `0.1.1 → 0.2.0`, force a refresh in the consumer repo:
```bash
cd /path/to/consumer/project
uv lock --upgrade-package bandcamp-recommender
uv sync
```

**3. PYTHONPATH / sys.path insert (Option 2 in this doc):**

Just pull. No package metadata involved.
```bash
cd /path/to/bandcamp_recommender
git pull --ff-only origin master
```

### Verify the upgrade succeeded

From the consumer repo, after updating:
```bash
uv run python -c "import bandcamp_recommender, importlib.metadata as m; print(m.version('bandcamp-recommender'))"
```
Expected output: `0.2.0`.

Also confirm the new curl-first path is wired in (only present in 0.2.0+):
```bash
uv run python -c "from bandcamp_recommender import SupporterRecommender; assert hasattr(SupporterRecommender, '_get_supporter_items_curl_first'); print('OK')"
```

### What the consumer code needs to change

Nothing — this is a drop-in replacement. The two new env-var knobs are optional:

- `BANDCAMP_DRIVER_POOL=N` — cap the Selenium fallback pool size (default 5; lower if RAM is tight).
- `BANDCAMP_DISABLE_JS=1` — disable JavaScript in Chrome for extra speed in the fallback path. Off by default because it disables the in-browser `fetch()` used by the collection API; only safe if curl works for the collection API on your network.

### If the upgrade misbehaves

Pin back to `0.1.1` in the consumer's `pyproject.toml`:
```toml
dependencies = [
    "bandcamp-recommender @ file:///path/to/bandcamp_recommender",
]
```
And in `bandcamp_recommender`, check out the prior commit:
```bash
cd /path/to/bandcamp_recommender
git checkout <commit-before-curl-first>
```
Then `uv sync --reinstall-package bandcamp-recommender` in the consumer.
