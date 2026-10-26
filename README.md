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

### Similarity-ordered recommendations

One call: given a Bandcamp source song, get recs ordered by feature
similarity with full per-track feature vectors (incl. raw BPM)
attached.

```python
from bandcamp_recommender import SupporterRecommender

with SupporterRecommender() as r:
    recs = r.get_similar_recommendations(
        source_url="https://artist.bandcamp.com/track/seed",
        max_recommendations=10,
        candidate_pool_size=30,      # supporter-overlap pool size
    )

    for rec in recs:
        print(f"{rec['distance']:.2f}  {rec['band_name']} - {rec['item_title']}")
        print(f"   tempo: {rec['features']['bpm']} BPM")
        print(f"   audio: {rec['audio_url']}")
```

Each returned dict includes:
- standard recommender metadata (`item_title`, `band_name`,
  `item_url`, `supporters_count`, `tags`)
- `audio_url` — preview URL used for feature extraction
- `features` — the full feature dict from
  `bandcamp_recommender.features.extract_features` (incl. raw `bpm`)
- `distance` — weighted-Euclidean distance to the source
  (smaller = more similar)

#### In-vibe diversity (opt-in)

By default the top-k cluster in style. Pass `diversify` to instead get k recs
that are **diverse in style while keeping the source's vibe** (intensity within
`vibe_tau`) and staying related to it — a "more like this, but varied" mode:

```python
recs = r.get_similar_recommendations(
    source_url="https://artist.bandcamp.com/track/seed",
    max_recommendations=4,
    diversify="mmr",        # "mmr" | "maxmin" | "stratified" (None = plain ranking)
    diversity_lambda=0.5,   # mmr: 1.0 = relevance-only, 0.0 = pure spread
    vibe_tau=0.15,          # max |intensity - source intensity| to stay in-vibe
)
```

MMR is the recommended default (best diversity↔vibe-tightness↔anchoring balance);
see `bandcamp_recommender/recommendations/diversity.py` and the side-by-side study
in `docs/diversity-eval-results.md`. Backward-compatible: `diversify=None` (the
default) is the unchanged similarity ranking; the `BANDCAMP_DIVERSIFY` env var
sets a global default when the param is unset.

#### Candidate-pool diversity (opt-in)

A seed's supporters mostly own its artist's other releases, and on a small
scene the top of the overlap ranking is a few artists' back catalogues, so the
candidate pool can hold very few artists. Two options act on the ranked
candidates before the pool is cut (also on `get_recommendations`):

```python
recs = r.get_similar_recommendations(
    source_url="https://artist.bandcamp.com/track/seed",
    candidate_pool_size=30,
    exclude_seed_artist=True,   # no candidates by the seed's own artist
    max_per_artist=2,           # each artist's 2 best-ranked releases at most
)
r.last_call_stats["pool"]       # {"seed_artist", "seed_artist_dropped", "artist_capped", ...}
```

Artists are compared with `artists.artist_tokens` ("A & B", "A feat. B" count
for both A and B). A placeholder credit ("Various Artists", "Unknown Artist",
"V/A", blank) is identified by its release page's host instead: one label's
placeholder releases count as one act (capped and excluded together), while
comps from different pages stay distinct. The seed's artist is
read from the seed page, which the call has just read anyway: no extra
request. Defaults (`False` / `None`) give the same result as before.

### A fan's own lists (wishlist + collection)

```python
from bandcamp_recommender import fans

lib = fans.fan_library("someuser")            # lists=("wishlist", "collection")
lib["fan_id"], lib["username"]
lib["hidden"]                                 # e.g. ["wishlist"]: private, so []
for item in lib["collection"]:                # full list, newest first
    item["item_url"], item["item_title"], item["band_name"], item["item_type"]
    item["added_ts"]                          # epoch s it entered the list, or None
```

One fan page read plus one API call per non-empty visible list; the lists are
read fresh on every call. `added_ts` is the list entry time (bought, for the
collection; wishlisted, for the wishlist). Failures raise (the `http` errors,
or `fans.FanNotFound` for an unknown username when the page answers 404 over
plain HTTP; through the fallback browser, which can't see the status, an
unknown username raises `http.Unexpected`). A list is never silently empty.

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

For a Bandcamp item, `features_for_item` gives the vector of its featured
track, cached on disk per (item, track) for 90 days (the stream URL is not
part of the key, so the cache outlives Bandcamp's daily re-signing):

```python
from bandcamp_recommender import features, items

vec, preview = features.features_for_item(item_url)   # reads the page if needed

# Already read the page yourself? Hand over its preview: no page read here.
info = items.parse_item_html(html, item_url)
vec, preview = features.features_for_item(
    item_url, items.feature_tags(info), preview=info["preview"],
)
```

### Item pages (`bandcamp_recommender.items`)

One parse of an album or track page, so consumers don't scrape them:

```python
from bandcamp_recommender import http, items

try:
    info = items.item_info("https://artist.bandcamp.com/album/x")
except http.FetchError as e:  # e.g. Challenged / RateLimited / Timeout / HTTPStatus
    ...
info["artist"], info["title"], info["tags"], info["location"]
(info["preview"] or {}).get("audio_path")  # the track that plays; None on preorders
items.art_url(info["art_id"], size=10) or info["og_image_url"]  # 10 = 1200 px, 2 = 700 px
items.feature_tags(info)                   # tags + location: the tag-feature input
```

The `items.parse_item_html` docstring lists every key; it is the same parse
without I/O and never raises (missing pieces are None / empty). `item_info`
is one page read through `http.get_page` (validated, cached for a few
minutes, browser fallback on a challenge). A track page's own description is
usually None, since Bandcamp describes the album:
`follow_album_description=True` reads the album page for it (one more paced
read).

## Network behaviour (0.8.0)

- **Plain HTTP, honest User-Agent.** Every Bandcamp page and API call goes
  through `bandcamp_recommender.http` (requests, HTTP/1.1). Bandcamp's CDN
  serves its JavaScript bot challenge (as HTTP 200) to clients that claim to
  be a browser but don't look like one; the old curl transport spoofed a
  Chrome UA and was challenged from datacenter IPs. Payloads are validated
  (`get_page(url, expect=...)`), and failures are typed (`Challenged`,
  `RateLimited`, `Timeout`, `HTTPStatus`, `Unexpected`).
- **Paced to ~1 request/second sustained** (measured limit; `429
  Retry-After: 3` beyond it), with bursts of up to 20 back to back after
  idle time (Bandcamp's limiter let ~100 page reads through within a
  minute from idle), at most 3 in flight, AIMD backoff on 429 (which also
  uses up the burst). Interactive callers will feel this on cold calls: an
  80-supporter seed costs ~80-110 requests. Tune with
  `BANDCAMP_MIN_REQUEST_INTERVAL` / `BANDCAMP_BURST` (1 = evenly spaced) /
  `BANDCAMP_MAX_CONCURRENCY`. Read other Bandcamp pages through
  `http.get_page` too, so all of a process's traffic shares one budget.
- **Disk cache** (`~/.cache/bandcamp_recommender/cache.sqlite`,
  `BANDCAMP_CACHE_PATH`, `BANDCAMP_CACHE_DISABLED=1`): supporter
  collections 7 d, item metadata (featured track, tags) 30 d, audio features
  90 d. A repeated seed costs a couple of requests.
- **What the 0.8.0 reads cost**: `items.item_info` is one page read (two
  for a track page on an album with `follow_album_description=True`),
  shared with the recommender through the short-lived page cache.
  `fans.fan_library` is one fan page read plus one API call per non-empty
  visible list, never cached; when Bandcamp challenges plain HTTP the rest
  of the call runs in ONE browser session (one page load, the API calls
  from inside it). `exclude_seed_artist` reads the seed page from the page
  cache.
- **Selenium is a fallback only**: one shared driver per process
  (`bandcamp_recommender.browser`), used when plain HTTP is challenged, at
  most `BANDCAMP_SELENIUM_FALLBACK_BUDGET` (8) supporters per call. It quits
  itself after `BANDCAMP_BROWSER_IDLE_S` (120) seconds unused (0 = never);
  `browser.close()` / `close_if_idle()` still quit it on demand. Closing a
  `SupporterRecommender` leaves it running for the next caller.
- **The library cleans up after its own Chrome**: every driver runs on its
  own profile dir, named after the owning process, under
  `BANDCAMP_BROWSER_PROFILE_DIR` (default
  `~/snap/chromium/common/bandcamp_recommender/chrome-profiles` when snap
  Chromium is installed, since it can't write hidden dirs in `$HOME`, else
  `~/.cache/bandcamp_recommender/chrome-profiles`). Quitting a driver kills
  whatever still runs on its profile and deletes it. `chrome_procs.reap_orphans()`
  kills Chrome left by library processes that died and deletes their
  profiles; it never touches a live process's browser or a Chrome the
  library didn't start. It runs by itself before every driver start and on
  the first Bandcamp request of a process (a cache hit doesn't count),
  so a long-lived service should also call it once at startup: after a
  restart it may make no request for a long time. (Any call counts as the
  first-request sweep.) Processes on different hosts or in different pid
  namespaces must not share a profiles dir.
  `DriverManager.reap_orphans()` (the opt-in cold-start reaper) does the
  same and also kills what earlier versions leave behind: an unmarked
  chromedriver whose spawner died, or a Chrome on a chromedriver temp
  profile whose chromedriver died. That includes a chromedriver run as a
  service straight under init, systemd or launchd (e.g. a Selenium Grid
  node).
- `collection_page_size` (default 10000 = everything) limits how many of each
  supporter's newest items are read; `candidate_filter` drops items before
  the candidate pool is cut. Opt-in pool diversity, also before the cut:
  `exclude_seed_artist=True` drops the seed's own artist and
  `max_per_artist=N` keeps each artist's N best-ranked releases (artists
  compared by `artists.artist_tokens`, so collabs count for each member and
  placeholder credits such as "Various Artists" are grouped by release page
  host, not into one artist). Audio features describe each
  item's featured track (`preview_track_id`); `audio_url` is only set when
  the page was read during the call (stream URLs are signed and expire).
- `http.snapshot()` / `http.diff()` expose request counters (`requests`,
  `rate_limited`, `challenged`, `selenium_fallbacks`, `supporter_api`, ...);
  `SupporterRecommender.last_call_stats` has per-phase timings of the last
  similarity call, and under `pool` what the pool options did (`seed_artist`,
  `seed_artist_dropped`, `filtered`, `artist_capped`).
- A `SupporterRecommender` keeps per-call state: don't share one instance
  across threads for concurrent calls.

## Upgrading to 0.8.0

0.8.2 stops mining Bandcamp's own links as supporters: an item page whose
supporter data is empty has no supporters (the text fallbacks used to pick
up tag links under `/discover` and the artist's `/yum`).

0.8.1 adds `features_for_item(..., preview=...)` (see Feature vectors) and
lets the pacer burst (`BANDCAMP_BURST`, default 20; see Network behaviour).
Nothing else changes.

Pin `bandcamp-recommender[bpm] @ git+https://github.com/lucweytingh/bandcamp_recommender.git@v0.8.0`.
New: `items.item_info` / `parse_item_html` / `art_url` / `feature_tags`,
`fans.fan_library`, `artists.artist_tokens`, the `exclude_seed_artist` /
`max_per_artist` options, `chrome_procs.reap_orphans`. Every public 0.7.0
signature is unchanged, and so is every default result, except these
deliberate changes:

- The HTTP session uses Python's standard TLS handshake
  (`ssl.create_default_context()` with certifi's CA bundle, full certificate
  and hostname checks). urllib3's own SSL context is answered with
  Bandcamp's bot challenge from residential IPs, which sent every page read
  on a laptop to the fallback Chrome; datacenter IPs passed either way.
  `verify="<bundle>"` / `REQUESTS_CA_BUNDLE` get a context trusting exactly
  that bundle; `verify=False` or a client `cert` use urllib3's own context.
  Requires `requests>=2.32.3` (2.32.0-2.32.2 override a custom context) and
  declares `certifi`.

- The fallback Chrome quits itself after `BANDCAMP_BROWSER_IDLE_S` (120) s
  unused; it used to live until `browser.close()`, a recommender's
  `close()`, or exit. With `0` it lives until `browser.close()` or exit.
- `SupporterRecommender.close()` (and leaving its `with` block) no longer
  quits the shared browser (0.7.0 quit it if idle; 0.5 quit the
  recommender's own drivers): the idle timer does, so the page reads that
  follow a call reuse it. `browser.close()` quits it now.
- The library reaps Chrome by itself (first request, every driver start) and
  only Chrome of library processes that died. `DriverManager.reap_orphans()`
  keeps its signature but no longer kills every Chrome whose parent is init
  (on macOS that was the user's own Chrome, launched from the Dock).
- Chrome runs on a library-owned profile dir (`BANDCAMP_BROWSER_PROFILE_DIR`)
  that is deleted when the driver quits, instead of a chromedriver temp
  profile.
- Track dicts (`bpm.extract_track_info`, `get_preview_for_item`,
  `items.preview_from_tralbum`) carry an absolute `url` (was the
  page-relative link) and the new keys; a missing `track_num` is None (was
  0) and a missing `title` is None (was `""`).
- `http.get_page(url, fallback=False)` no longer remembers a bot challenge
  for 60 s, so a later call with the fallback still gets the page through
  the browser.
- Page tags are read from the tag block and de-duplicated ignoring case
  (identical output on every real page tested).
- `diversify="stratified"` compares artists with
  `artist_tokens(artist, item_url)`: collabs and case variants are one
  artist, and a placeholder credit ("Various Artists", "Unknown Artist",
  "V/A", blank) is its release page host (`{"@label.bandcamp.com"}`), so one
  label's placeholder releases block each other and other labels' comps no
  longer do. The same identity drives the new `exclude_seed_artist` and
  `max_per_artist`. `artist_tokens` without a URL gives placeholders no
  identity (an empty set).
- `items.normalize_item_url` also strips surrounding whitespace; every disk
  cache key stays valid.

## Architecture

The codebase is organized into modular components:

- `bandcamp_recommender/features.py` - Vector similarity API (`extract_features`, `distance`, `project_mood`)
- `bandcamp_recommender/items.py` - The one item-page parse (`parse_item_html`, `item_info`), cached item metadata, and `normalize_item_url`, the one canonical item URL for comparisons and cache keys
- `bandcamp_recommender/fans.py` - A fan's own wishlist + collection (`fan_library`), with list entry times
- `bandcamp_recommender/artists.py` - `artist_tokens`, artist identity: collaboration credits split into their members, case folded, placeholder credits ("Various Artists", "Unknown Artist") identified by their release page host (two credits are the same artist when their tokens intersect)
- `bandcamp_recommender/recommendations/supporter_recommender.py` - Main recommendation engine
- `bandcamp_recommender/recommendations/mood_tags.py` - Everynoise-derived tag-mood + tag-spikiness lexicon
- `bandcamp_recommender/recommendations/intensity.py` - Audio energy / texture feature extraction (librosa)
- `bandcamp_recommender/recommendations/bpm.py` - BPM detection (Joe Sullivan + librosa backends) + `_load_audio_segment` shared decode
- `bandcamp_recommender/http.py` - The paced, validated plain-HTTP client and page policy (`get_page`), request counters
- `bandcamp_recommender/browser.py` - The one shared fallback browser (idle close, `close`)
- `bandcamp_recommender/chrome_procs.py` - Owned Chrome profiles and the owner-aware reaper (`reap_orphans`)
- `bandcamp_recommender/cache.py` - The sqlite disk cache
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
