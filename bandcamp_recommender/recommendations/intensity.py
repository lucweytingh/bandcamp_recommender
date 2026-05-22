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
