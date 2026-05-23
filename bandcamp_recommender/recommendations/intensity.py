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
from typing import Any, Callable, Dict, List, Optional

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
_FEATURE_CACHE: Dict[str, Dict[str, Optional[float]]] = {}
_INTENSITY_CACHE_LOCK = threading.Lock()


_FEATURE_KEYS = ("rms_mean", "rms_p95", "onset_rate", "spectral_centroid", "crest_factor")


def _empty_features() -> Dict[str, Optional[float]]:
    return {k: None for k in _FEATURE_KEYS}


def _normalize(value: float, lo: float, hi: float) -> float:
    """Clamp (value - lo) / (hi - lo) to [0, 1]; returns 0.0 when hi <= lo."""
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def extract_features_from_samples(
    samples: Any,
    sample_rate: int,
) -> Dict[str, Optional[float]]:
    """Compute the per-feature normalized scores from a mono numpy buffer.

    Returns a dict with keys :data:`_FEATURE_KEYS`, each normalized to
    ``[0, 1]`` against the documented empirical min/max ranges. Any
    failure (no librosa, empty buffer, librosa raises) returns a dict
    with all keys set to ``None`` — callers can still ``score`` such
    items, they just contribute nothing to vector similarity.
    """
    try:
        import librosa
        import numpy as np
    except ImportError:
        return _empty_features()
    if samples is None or len(samples) == 0:
        return _empty_features()

    try:
        rms_frames = librosa.feature.rms(y=samples)[0]
        centroid_frames = librosa.feature.spectral_centroid(
            y=samples, sr=sample_rate
        )[0]
        onsets = librosa.onset.onset_detect(y=samples, sr=sample_rate)
    except Exception:
        return _empty_features()

    if rms_frames is None or len(rms_frames) == 0:
        return _empty_features()

    rms_mean_val = float(np.mean(rms_frames))
    rms_p95_val = float(np.percentile(rms_frames, 95))
    centroid_mean_val = float(np.mean(centroid_frames)) if len(centroid_frames) else 0.0

    duration_sec = max(1e-6, len(samples) / float(sample_rate))
    onset_rate_val = float(len(onsets)) / duration_sec

    peak = float(np.max(np.abs(samples)))
    # Guard against silence: a zero RMS would blow up the crest factor.
    crest_val = peak / rms_mean_val if rms_mean_val > 1e-6 else 0.0

    return {
        "rms_mean": _normalize(rms_mean_val, _RMS_MEAN_MIN, _RMS_MEAN_MAX),
        "rms_p95": _normalize(rms_p95_val, _RMS_P95_MIN, _RMS_P95_MAX),
        "onset_rate": _normalize(onset_rate_val, _ONSET_RATE_MIN, _ONSET_RATE_MAX),
        "spectral_centroid": _normalize(centroid_mean_val, _CENTROID_MIN, _CENTROID_MAX),
        "crest_factor": _normalize(crest_val, _CREST_MIN, _CREST_MAX),
    }


def score_intensity_from_features(features: Dict[str, Optional[float]]) -> Optional[float]:
    """Collapse a feature dict to the legacy 0..1 intensity scalar.

    Uses the historical weights: RMS dominates (mean+p95 averaged),
    onset rate / centroid / crest each contribute one-fifth. Returns
    ``None`` when no feature has a value (caller should fall back).
    """
    rms_mean = features.get("rms_mean")
    rms_p95 = features.get("rms_p95")
    onset = features.get("onset_rate")
    centroid = features.get("spectral_centroid")
    crest = features.get("crest_factor")

    if all(v is None for v in (rms_mean, rms_p95, onset, centroid, crest)):
        return None

    rms_score = 0.5 * (rms_mean or 0.0) + 0.5 * (rms_p95 or 0.0)
    score = (
        _W_RMS * rms_score
        + _W_ONSET * (onset or 0.0)
        + _W_CENTROID * (centroid or 0.0)
        + _W_CREST * (crest or 0.0)
    )
    return max(0.0, min(1.0, score))


def score_intensity_from_samples(
    samples: Any,
    sample_rate: int,
) -> Optional[float]:
    """Legacy entrypoint: extract features from samples, then blend to one score."""
    features = extract_features_from_samples(samples, sample_rate)
    return score_intensity_from_features(features)


def extract_features(
    audio_url: str,
    duration: float = 60.0,
    use_cache: bool = True,
) -> Dict[str, Optional[float]]:
    """Return the per-feature dict for a single preview URL.

    Downloads + decodes the first ``duration`` seconds, then runs
    :func:`extract_features_from_samples`. On any failure (network,
    decode, missing librosa) returns a dict where every feature is
    ``None`` — vector callers should treat that as "no audio signal
    for this track" and rely on whatever non-audio features exist.

    Results are cached per-URL so the radio can call this lazily
    without re-downloading on repeated hits.
    """
    if use_cache:
        with _INTENSITY_CACHE_LOCK:
            cached = _FEATURE_CACHE.get(audio_url)
            if cached is not None:
                return dict(cached)

    decoded = _load_audio_segment(audio_url, duration=duration)
    if decoded is None:
        features = _empty_features()
    else:
        samples, sr = decoded
        features = extract_features_from_samples(samples, sr)

    if use_cache:
        with _INTENSITY_CACHE_LOCK:
            _FEATURE_CACHE[audio_url] = dict(features)
    return features


def score_intensity(
    audio_url: str,
    duration: float = 60.0,
    use_cache: bool = True,
) -> Optional[float]:
    """Public per-track intensity scorer (legacy 0..1 scalar).

    Now a thin wrapper over :func:`extract_features` + the documented
    feature weights. Preserved for the radio's chill/party slider and
    existing callers that want a single number.

    ``audio_url`` must be a direct CDN URL to the audio file (e.g. a
    ``https://*.bcbits.com/...`` preview), not a Bandcamp album/track page.
    Use ``bpm.get_audio_url_for_item`` to resolve a page URL first.
    """
    if use_cache:
        with _INTENSITY_CACHE_LOCK:
            if audio_url in _INTENSITY_CACHE:
                return _INTENSITY_CACHE[audio_url]

    features = extract_features(audio_url, duration=duration, use_cache=use_cache)
    score = score_intensity_from_features(features)

    if use_cache:
        with _INTENSITY_CACHE_LOCK:
            _INTENSITY_CACHE[audio_url] = score
    return score


def clear_intensity_cache() -> None:
    """Clear the in-process intensity + feature caches."""
    with _INTENSITY_CACHE_LOCK:
        _INTENSITY_CACHE.clear()
        _FEATURE_CACHE.clear()


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
