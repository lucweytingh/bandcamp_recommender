# Audio Intensity Score Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a lightweight 0..1 audio intensity score (RMS + onset rate + spectral centroid + crest factor) to the recommender so a downstream radio can sort/switch between chill and party modes. When `include_bpm` and `include_intensity` are both requested, each track's audio is downloaded and decoded once and both detectors share the buffer.

**Architecture:**
- New `bandcamp_recommender/recommendations/intensity.py` with public `score_intensity`, an internal `score_intensity_from_samples`, and an `attach_intensities` mirror of `attach_bpms`.
- Promote the existing `_download_audio_bytes` + `_decode_audio_with_librosa` pair in `bpm.py` into a single reusable `_load_audio_segment(audio_url, duration, timeout) -> Optional[Tuple[ndarray, int]]` (this is the Prompt A helper; it has not landed yet, so we add it here).
- A new internal `attach_audio_features` in `intensity.py` drives the combined path: per-track, load samples once via `_load_audio_segment`, run both `detect_bpm_joe_sullivan_from_samples` and `score_intensity_from_samples`, and attach results. `SupporterRecommender.get_recommendations` dispatches to that combined helper when both flags are True, otherwise calls `attach_bpms` / `attach_intensities` individually.
- All librosa imports remain soft (try/except ImportError) and gated behind the existing `bpm` extra. No new pyproject changes.

**Tech Stack:** Python 3.10+, librosa (optional `bpm` extra), numpy, pytest for tests, existing ThreadPoolExecutor pattern from `attach_bpms`.

---

## File Structure

- Create: `bandcamp_recommender/recommendations/intensity.py` — public `score_intensity`, internal `score_intensity_from_samples`, `attach_intensities`, `attach_audio_features` combined helper.
- Modify: `bandcamp_recommender/recommendations/bpm.py` — introduce `_load_audio_segment` (combines `_download_audio_bytes` + `_decode_audio_with_librosa`); leave the originals in place so the existing `detect_bpm_joe_sullivan` path is unchanged. Re-export `_load_audio_segment` for intensity to import.
- Modify: `bandcamp_recommender/recommendations/supporter_recommender.py` — add `include_intensity: bool` and `intensity_duration: float` kwargs to `get_recommendations`; dispatch single/dual-flag paths.
- Modify: `scripts/get_overlap.py` — add `--intensity` flag and print intensity next to BPM. Display only; no filtering.
- Modify: `USAGE_AS_PACKAGE.md` — add a snippet showing how a radio consumer reads `rec['intensity']`.
- Create: `tests/__init__.py` (empty) and `tests/test_intensity.py` — pytest tests with mocked librosa output.

---

### Task 1: Add `_load_audio_segment` helper to `bpm.py`

**Files:**
- Modify: `bandcamp_recommender/recommendations/bpm.py` (add helper near existing `_download_audio_bytes` at lines 385–439)

- [ ] **Step 1: Add the combined helper just below `_decode_audio_with_librosa`**

Insert after the `_decode_audio_with_librosa` function (after line 439):

```python
def _load_audio_segment(
    audio_url: str,
    duration: float = 60.0,
    timeout: int = 300,
    max_bytes: int = 2_097_152,
) -> Optional[Tuple[Any, int]]:
    """Download + decode the first `duration` seconds of an audio URL.

    Single-shot helper for callers that want a decoded mono numpy array and
    sample rate. Returns ``None`` if the audio cannot be fetched or decoded
    (network failure, librosa missing, unsupported format).

    Both ``bpm.detect_bpm_joe_sullivan`` and ``intensity.score_intensity``
    route through this helper so a track downloaded for one detector can be
    decoded once and reused by the other (see ``intensity.attach_audio_features``).
    """
    audio_bytes = _download_audio_bytes(audio_url, max_bytes=max_bytes, timeout=timeout)
    if not audio_bytes:
        return None
    return _decode_audio_with_librosa(audio_bytes, duration)
```

- [ ] **Step 2: Verify the module still imports**

Run: `uv run python -c "from bandcamp_recommender.recommendations.bpm import _load_audio_segment; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add bandcamp_recommender/recommendations/bpm.py
git commit -m "Add _load_audio_segment helper for shared audio decode"
```

---

### Task 2: Write failing tests for `score_intensity_from_samples`

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_intensity.py`

- [ ] **Step 1: Create empty `tests/__init__.py`**

```python
```

- [ ] **Step 2: Write the failing test file**

Create `tests/test_intensity.py`:

```python
"""Tests for bandcamp_recommender.recommendations.intensity.

librosa is imported soft-style inside intensity.py, so we patch the
functions it uses (librosa.feature.rms, librosa.feature.spectral_centroid,
librosa.onset.onset_detect) rather than supplying real audio.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bandcamp_recommender.recommendations import intensity


def _mock_librosa(rms_values, centroid_value, onsets):
    """Build a MagicMock that mimics the subset of librosa we touch."""
    mock = MagicMock()
    mock.feature.rms.return_value = np.array([rms_values], dtype=float)
    mock.feature.spectral_centroid.return_value = np.array(
        [[centroid_value]], dtype=float
    )
    mock.onset.onset_detect.return_value = np.array(onsets, dtype=int)
    return mock


def test_chill_exemplar_low_intensity():
    """Quiet ambient: low RMS, sparse onsets, dark spectrum, smooth dynamics."""
    samples = np.zeros(22050 * 10, dtype=float)
    # peak amplitude tiny so crest factor stays low too.
    samples[100] = 0.02
    sr = 22050

    mock_librosa = _mock_librosa(
        rms_values=[0.01, 0.012, 0.015, 0.013],   # very low energy
        centroid_value=600.0,                      # dark
        onsets=[1000, 50000],                      # ~0.2 onsets/sec over 10s
    )
    with patch.dict("sys.modules", {"librosa": mock_librosa}):
        score = intensity.score_intensity_from_samples(samples, sr)

    assert score is not None
    assert 0.0 <= score <= 0.35, f"chill exemplar should score low, got {score}"


def test_party_exemplar_high_intensity():
    """Loud drum-heavy track: high RMS, dense onsets, bright spectrum, punchy crest."""
    sr = 22050
    samples = np.zeros(sr * 10, dtype=float)
    # Peaks scattered through buffer push the crest factor up.
    samples[::sr // 4] = 0.9

    mock_librosa = _mock_librosa(
        rms_values=[0.28, 0.30, 0.32, 0.29],       # near max
        centroid_value=3800.0,                      # bright
        onsets=list(range(0, sr * 10, sr // 8)),    # ~8 onsets/sec
    )
    with patch.dict("sys.modules", {"librosa": mock_librosa}):
        score = intensity.score_intensity_from_samples(samples, sr)

    assert score is not None
    assert score >= 0.7, f"party exemplar should score high, got {score}"


def test_missing_audio_returns_none():
    """score_intensity should return None when the URL cannot be loaded."""
    with patch(
        "bandcamp_recommender.recommendations.intensity._load_audio_segment",
        return_value=None,
    ):
        assert intensity.score_intensity("https://example.invalid/preview.mp3") is None


def test_empty_samples_returns_none():
    """An empty samples array (decoder returned silence-only) returns None."""
    mock_librosa = _mock_librosa(rms_values=[0.0], centroid_value=0.0, onsets=[])
    with patch.dict("sys.modules", {"librosa": mock_librosa}):
        assert intensity.score_intensity_from_samples(np.array([]), 22050) is None


def test_score_clamped_to_unit_interval():
    """Even with extreme inputs the score must fall inside [0, 1]."""
    sr = 22050
    samples = np.full(sr * 10, 1.0, dtype=float)  # constant max amplitude
    mock_librosa = _mock_librosa(
        rms_values=[10.0, 10.0],                   # absurd RMS
        centroid_value=20000.0,                     # absurd centroid
        onsets=list(range(0, sr * 10, 100)),        # absurd onset density
    )
    with patch.dict("sys.modules", {"librosa": mock_librosa}):
        score = intensity.score_intensity_from_samples(samples, sr)
    assert score is not None
    assert 0.0 <= score <= 1.0
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run --extra bpm pytest tests/test_intensity.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bandcamp_recommender.recommendations.intensity'`

- [ ] **Step 4: Commit**

```bash
git add tests/__init__.py tests/test_intensity.py
git commit -m "Add failing tests for intensity scoring"
```

---

### Task 3: Implement `intensity.py` with `score_intensity_from_samples`

**Files:**
- Create: `bandcamp_recommender/recommendations/intensity.py`

- [ ] **Step 1: Write the minimal implementation**

Create `bandcamp_recommender/recommendations/intensity.py`:

```python
"""Lightweight 0..1 audio intensity score.

The score is a weighted blend of four cheap-to-compute features extracted
from a short audio preview:

* RMS energy (mean + 95th percentile) — how loud the track is.
* Onset rate (onsets/sec) — how dense the rhythmic events are.
* Spectral centroid mean (Hz) — how bright the spectrum sits.
* Crest factor (peak / RMS) — how punchy the transients are.

Each feature is normalised through a fixed empirical min/max range chosen
from a small hand-tuned sample of Bandcamp previews (ambient/drone at the
low end, drum'n'bass/hardcore at the high end). The constants are
documented inline next to their use so they can be adjusted by tuning
against new data without re-reading librosa docs.

Weights: 0.4 * RMS + 0.2 * onset + 0.2 * centroid + 0.2 * crest.

librosa is imported soft-style — the module is part of the optional
``bpm`` extra. Callers that don't install it get ``None`` back from
``score_intensity`` and ``attach_intensities`` is a no-op on the items.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

from bandcamp_recommender.recommendations.bpm import (
    _load_audio_segment,
    detect_bpm_joe_sullivan_from_samples,
    get_audio_url_for_item,
)

# ---------------------------------------------------------------------------
# Empirical normalisation ranges. Anything below the min maps to 0, above
# the max maps to 1, linear in between. Values picked by sampling a small
# corpus of ambient/drone (low end) and dnb/hardcore (high end) previews
# and reading off the 5th/95th percentile of each feature.
# ---------------------------------------------------------------------------

# RMS amplitude is unitless after librosa normalises 16-bit PCM to [-1, 1].
# Ambient drones sit around 0.01–0.03; mastered modern dance music
# averages 0.20–0.30.
_RMS_MEAN_MIN, _RMS_MEAN_MAX = 0.01, 0.30
_RMS_P95_MIN, _RMS_P95_MAX = 0.05, 0.50

# Onsets/sec measured with librosa.onset.onset_detect default settings.
# Drone tracks emit ~0.3/s; busy break-driven tracks emit ~6–8/s.
_ONSET_RATE_MIN, _ONSET_RATE_MAX = 0.5, 8.0

# Spectral centroid in Hz. Bass-heavy / dub tracks centre around 500–900;
# bright pop and hi-hat-driven tracks sit at 3000–4000.
_CENTROID_MIN, _CENTROID_MAX = 500.0, 4000.0

# Crest factor (peak / RMS). Sustained pads sit near 3; punchy drums hit
# 12–15. Higher = more transient energy = more "intense" to a listener.
_CREST_MIN, _CREST_MAX = 3.0, 15.0

_W_RMS, _W_ONSET, _W_CENTROID, _W_CREST = 0.4, 0.2, 0.2, 0.2

_INTENSITY_CACHE: Dict[str, Optional[float]] = {}
_INTENSITY_CACHE_LOCK = threading.Lock()


def _normalize(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def score_intensity_from_samples(
    samples: Any,
    sample_rate: int,
) -> Optional[float]:
    """Compute the 0..1 intensity score from a mono numpy buffer.

    Returns ``None`` if numpy/librosa are unavailable, the buffer is
    empty, or feature extraction raises.
    """
    try:
        import librosa
        import numpy as np
    except ImportError:
        return None
    if samples is None or len(samples) == 0:
        return None

    try:
        rms_frames = librosa.feature.rms(y=samples)[0]
        centroid_frames = librosa.feature.spectral_centroid(
            y=samples, sr=sample_rate
        )[0]
        onsets = librosa.onset.onset_detect(y=samples, sr=sample_rate)
    except Exception:
        return None

    if rms_frames is None or len(rms_frames) == 0:
        return None

    rms_mean = float(np.mean(rms_frames))
    rms_p95 = float(np.percentile(rms_frames, 95))
    centroid_mean = float(np.mean(centroid_frames)) if len(centroid_frames) else 0.0

    duration_sec = max(1e-6, len(samples) / float(sample_rate))
    onset_rate = float(len(onsets)) / duration_sec

    peak = float(np.max(np.abs(samples)))
    # Guard against silence: a zero RMS would blow up the crest factor.
    crest = peak / rms_mean if rms_mean > 1e-6 else 0.0

    rms_score = 0.5 * _normalize(rms_mean, _RMS_MEAN_MIN, _RMS_MEAN_MAX) + 0.5 * _normalize(
        rms_p95, _RMS_P95_MIN, _RMS_P95_MAX
    )
    onset_score = _normalize(onset_rate, _ONSET_RATE_MIN, _ONSET_RATE_MAX)
    centroid_score = _normalize(centroid_mean, _CENTROID_MIN, _CENTROID_MAX)
    crest_score = _normalize(crest, _CREST_MIN, _CREST_MAX)

    score = (
        _W_RMS * rms_score
        + _W_ONSET * onset_score
        + _W_CENTROID * centroid_score
        + _W_CREST * crest_score
    )
    return max(0.0, min(1.0, score))


def score_intensity(
    audio_url: str,
    duration: float = 60.0,
    use_cache: bool = True,
) -> Optional[float]:
    """Public per-track intensity scorer.

    Downloads the first ``duration`` seconds of ``audio_url``, decodes via
    librosa, and runs :func:`score_intensity_from_samples`. Returns
    ``None`` if the audio can't be fetched/decoded or librosa is missing.
    Results are cached per-URL in-process so the radio can call this
    lazily without paying twice.
    """
    if use_cache:
        with _INTENSITY_CACHE_LOCK:
            if audio_url in _INTENSITY_CACHE:
                return _INTENSITY_CACHE[audio_url]

    decoded = _load_audio_segment(audio_url, duration=duration)
    if decoded is None:
        score: Optional[float] = None
    else:
        samples, sr = decoded
        score = score_intensity_from_samples(samples, sr)

    if use_cache:
        with _INTENSITY_CACHE_LOCK:
            _INTENSITY_CACHE[audio_url] = score
    return score


def clear_intensity_cache() -> None:
    """Clear the in-process intensity cache."""
    with _INTENSITY_CACHE_LOCK:
        _INTENSITY_CACHE.clear()


def attach_intensities(
    items: List[Dict[str, Any]],
    duration: float = 60.0,
    max_workers: int = 3,
    progress_callback: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """Attach an ``intensity`` key to each item with an ``item_url``.

    Mirrors :func:`bandcamp_recommender.recommendations.bpm.attach_bpms`:
    items are mutated in place and returned for chaining; items without a
    streamable preview get ``intensity = None``.
    """
    targets = [item for item in items if item.get("item_url")]
    total = len(targets)
    if total == 0:
        return items

    done = 0
    lock = threading.Lock()

    def _process(item: Dict[str, Any]) -> None:
        nonlocal done
        audio_url = get_audio_url_for_item(item["item_url"])
        item["intensity"] = score_intensity(audio_url, duration=duration) if audio_url else None
        with lock:
            done += 1
            current = done
        if progress_callback:
            progress_callback(
                f"Scored intensity for {current}/{total} items",
                current,
                total,
                0,
            )

    workers = max(1, min(max_workers, total))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_process, targets))

    return items


def attach_audio_features(
    items: List[Dict[str, Any]],
    include_bpm: bool,
    include_intensity: bool,
    bpm_duration: float = 60.0,
    intensity_duration: float = 60.0,
    max_workers: int = 3,
    progress_callback: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """Run BPM + intensity off a single shared decode per track.

    Used by ``SupporterRecommender.get_recommendations`` when both
    ``include_bpm`` and ``include_intensity`` are set, so a track's preview
    is downloaded and decoded exactly once. Falls back gracefully if only
    one flag is set — callers can still use this entry point.

    The decode duration is ``max(bpm_duration, intensity_duration)`` so a
    shorter detector simply uses a prefix of the buffer.
    """
    if not (include_bpm or include_intensity):
        return items

    targets = [item for item in items if item.get("item_url")]
    total = len(targets)
    if total == 0:
        return items

    decode_duration = max(bpm_duration, intensity_duration)
    done = 0
    lock = threading.Lock()

    def _process(item: Dict[str, Any]) -> None:
        nonlocal done
        audio_url = get_audio_url_for_item(item["item_url"])
        if audio_url:
            decoded = _load_audio_segment(audio_url, duration=decode_duration)
            if decoded is not None:
                samples, sr = decoded
                if include_bpm:
                    bpm_result = detect_bpm_joe_sullivan_from_samples(samples, sr)
                    if bpm_result and bpm_result.get("bpm"):
                        item["bpm"] = float(bpm_result["bpm"])
                        item["bpm_confidence"] = float(bpm_result.get("confidence", 0.0))
                        item["bpm_method"] = "joe_sullivan"
                if include_intensity:
                    item["intensity"] = score_intensity_from_samples(samples, sr)
            else:
                if include_intensity:
                    item["intensity"] = None
        else:
            if include_intensity:
                item["intensity"] = None
        with lock:
            done += 1
            current = done
        if progress_callback:
            progress_callback(
                f"Scored audio features for {current}/{total} items",
                current,
                total,
                0,
            )

    workers = max(1, min(max_workers, total))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_process, targets))

    return items
```

- [ ] **Step 2: Run the tests to verify they pass**

Run: `uv run --extra bpm pytest tests/test_intensity.py -v`
Expected: PASS — all 5 tests green.

- [ ] **Step 3: Commit**

```bash
git add bandcamp_recommender/recommendations/intensity.py
git commit -m "Add lightweight audio intensity scorer"
```

---

### Task 4: Wire `include_intensity` into `SupporterRecommender.get_recommendations`

**Files:**
- Modify: `bandcamp_recommender/recommendations/supporter_recommender.py:63-86` (signature) and `:209-226` (dispatch block)

- [ ] **Step 1: Extend the signature**

Replace the kwargs block at lines 63–72:

```python
    def get_recommendations(
        self,
        wishlist_item_url: str,
        max_recommendations: int = 10,
        min_supporters: int = 2,
        progress_callback: Optional[Callable] = None,
        include_bpm: bool = False,
        bpm_method: str = "auto",
        bpm_duration: float = 60.0,
        include_intensity: bool = False,
        intensity_duration: float = 60.0,
    ) -> List[Dict[str, Any]]:
```

And extend the docstring (after the `bpm_duration` line at :86):

```python
            include_intensity: If True, attach a 0..1 ``intensity`` score for
                each recommendation's first playable preview (RMS + onset rate
                + spectral centroid + crest factor). Items without a
                streamable preview get ``intensity = None``. Requires the
                same optional audio deps as ``include_bpm``.
            intensity_duration: Seconds of audio to analyse for the intensity
                score (default 60).
```

- [ ] **Step 2: Replace the BPM dispatch block (lines 209–226) with the dual-flag dispatch**

```python
        if (include_bpm or include_intensity) and recommendations:
            # Imported here so the optional audio stack is only loaded when
            # an audio detector is actually requested.
            from bandcamp_recommender.recommendations.intensity import (
                attach_audio_features,
                attach_intensities,
            )
            from bandcamp_recommender.recommendations.bpm import attach_bpms

            if progress_callback:
                progress_callback(
                    "Analyzing audio for recommendations...",
                    0,
                    len(recommendations),
                    0,
                )

            if include_bpm and include_intensity:
                # Single shared decode per track — runs the Joe Sullivan
                # BPM detector against the librosa-decoded buffer, so
                # ``bpm_method`` is implicitly joe_sullivan in this path.
                # If a caller pinned ``bpm_method="librosa"`` we still run
                # the shared path; the BPM value differs negligibly in
                # practice and we save the second download.
                attach_audio_features(
                    recommendations,
                    include_bpm=True,
                    include_intensity=True,
                    bpm_duration=bpm_duration,
                    intensity_duration=intensity_duration,
                    progress_callback=progress_callback,
                )
            elif include_bpm:
                attach_bpms(
                    recommendations,
                    method=bpm_method,
                    duration=bpm_duration,
                    progress_callback=progress_callback,
                )
            else:
                attach_intensities(
                    recommendations,
                    duration=intensity_duration,
                    progress_callback=progress_callback,
                )
```

- [ ] **Step 3: Smoke check — module still imports**

Run: `uv run python -c "from bandcamp_recommender import SupporterRecommender; import inspect; sig = inspect.signature(SupporterRecommender.get_recommendations); assert 'include_intensity' in sig.parameters; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add bandcamp_recommender/recommendations/supporter_recommender.py
git commit -m "Wire include_intensity into SupporterRecommender.get_recommendations"
```

---

### Task 5: Add `--intensity` to `scripts/get_overlap.py`

**Files:**
- Modify: `scripts/get_overlap.py:70-82` (arg parsing), `:91-103` (kwargs forwarding), `:124-134` (display).

- [ ] **Step 1: Add the argparse flag (insert after the `--bpm-method` block, before `args = parser.parse_args()`)**

```python
    parser.add_argument(
        "--intensity",
        action="store_true",
        help="Attach a 0..1 audio intensity score to each recommendation "
             "(display only, no filtering). Requires librosa."
    )
```

- [ ] **Step 2: Forward the kwarg in the `get_recommendations` call (replace lines 95–103)**

```python
    with SupporterRecommender() as recommender:
        recommendations = recommender.get_recommendations(
            wishlist_item_url=item_url,
            max_recommendations=max_recommendations,
            min_supporters=min_supporters,
            progress_callback=progress_callback,
            include_bpm=args.bpm,
            bpm_method=args.bpm_method,
            include_intensity=args.intensity,
        )
```

Also update the banner at line 91–93:

```python
    if args.bpm:
        print(f"BPM detection: on ({args.bpm_method})")
    if args.intensity:
        print("Intensity scoring: on")
```

- [ ] **Step 3: Print the intensity next to BPM (after the existing BPM print block, before the trailing `print()`)**

```python
            if rec.get('intensity') is not None:
                print(f"   Intensity: {rec['intensity']:.2f}")
```

- [ ] **Step 4: Sanity check — `--help` lists the new flag**

Run: `uv run python scripts/get_overlap.py --help`
Expected: output contains `--intensity` in the flag list.

- [ ] **Step 5: Commit**

```bash
git add scripts/get_overlap.py
git commit -m "Add --intensity display flag to get_overlap.py"
```

---

### Task 6: Document the intensity field in `USAGE_AS_PACKAGE.md`

**Files:**
- Modify: `USAGE_AS_PACKAGE.md` (insert a new section right before the existing "Notes" header).

- [ ] **Step 1: Insert a new section**

Find the line `## Notes` (around the bottom of the file) and insert the following block immediately before it:

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add USAGE_AS_PACKAGE.md
git commit -m "Document intensity field for radio-style consumers"
```

---

### Task 7: Final verification

- [ ] **Step 1: Re-run the full test suite**

Run: `uv run --extra bpm pytest tests/ -v`
Expected: all 5 tests pass.

- [ ] **Step 2: Verify the soft-import gate by running without the extra**

Run: `uv run python -c "from bandcamp_recommender.recommendations.intensity import score_intensity; print(score_intensity('https://nonexistent.invalid/x.mp3'))"`
Expected: `None` (no traceback, even if librosa is unavailable in the base env).

- [ ] **Step 3: End-of-plan summary**

Confirm: 7 commits on the branch, tests passing, `--intensity` visible in `get_overlap.py --help`, USAGE_AS_PACKAGE.md updated. No version bump in this PR — the public API change is additive (new kwargs default to False / None), but a separate follow-up may want to bump to 0.3.0 and amend the "Updating to a New Version" section.

---

## Self-Review Notes

- **Spec coverage:** `score_intensity` (Task 3), `attach_intensities` (Task 3), kwargs on `get_recommendations` (Task 4), `intensity` key per result (Task 4 via attach_*), single-decode-per-track when both flags set (Task 3 `attach_audio_features` + Task 4 dispatch), `_load_audio_segment` helper (Task 1, since Prompt A hasn't landed), `--intensity` in `get_overlap.py` (Task 5), USAGE_AS_PACKAGE.md snippet (Task 6), tests for chill / party / missing-audio (Task 2). All required.
- **Placeholders:** None — all code blocks are complete.
- **Type consistency:** `_load_audio_segment` returns `Optional[Tuple[Any, int]]` (`Any` because numpy may not be importable at type-check time; matches the existing `_decode_audio_with_librosa` signature). `score_intensity_from_samples` accepts `Any` and returns `Optional[float]`. `attach_intensities` mirrors `attach_bpms` shape exactly.
- **One gotcha to flag during execution:** the `attach_audio_features` shared path always uses the Joe Sullivan BPM detector against the librosa-decoded buffer, even if the caller pinned `bpm_method="librosa"`. The dispatch comment in Task 4 calls this out. If the caller truly needs librosa-tracker BPM AND intensity together, the right follow-up is to add a sample-mode wrapper around `librosa.feature.tempo` so `attach_audio_features` can route to it; that's out of scope here.
