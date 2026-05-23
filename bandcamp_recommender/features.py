"""Vector-similarity feature representation for tracks.

Each track is represented as a dict of normalized scalar features. The
feature universe is the union of what every signal source produces:

* From everynoise tags (``recommendations.mood_tags``):
    - ``tag_mood``       chill (-1) ↔ party (+1)
    - ``tag_spikiness``  dense/atmospheric (-1) ↔ spiky/bouncy (+1)

* From the audio preview (``recommendations.intensity``):
    - ``rms_mean``         normalized RMS energy mean
    - ``rms_p95``          normalized RMS 95th percentile
    - ``onset_rate``       normalized onsets per second
    - ``spectral_centroid`` normalized brightness
    - ``crest_factor``     normalized transient punchiness

* From the audio preview (``recommendations.bpm``):
    - ``bpm_norm``         raw BPM normalized over [60, 200]
    - ``bpm_folded_norm``  octave-folded BPM normalized over [80, 160)

A feature value is either a float (in its documented normalized range)
or ``None`` if the signal couldn't be computed for that track. Missing
features don't kill the vector — they're skipped in distance
calculations and renormalized so a track with 5 known features and
2 unknowns still has a meaningful similarity to other tracks.

Use this module's :func:`distance` for "more like this" ranking, and
:func:`project_mood` to derive the radio's existing chill/party scalar
from the same vector.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

# Default per-feature weights for the distance function. Tuned by hand:
# audio features dominate because they're more reliable than tags; the
# tag features punch above their weight on coverage but can be noisy.
# The raw-bpm dimension gets less weight than the folded one because
# octave equivalence is usually what listeners care about — but raw is
# kept around so a 90 BPM doom track and a 180 BPM dnb track aren't
# completely indistinguishable.
DEFAULT_WEIGHTS: Dict[str, float] = {
    # Tag axes
    "tag_mood":           1.0,
    "tag_spikiness":      0.6,
    # Audio energy / texture
    "rms_mean":           0.6,
    "rms_p95":            0.6,
    "onset_rate":         0.8,
    "spectral_centroid":  0.5,
    "crest_factor":       0.4,
    # Tempo
    "bpm_folded_norm":    1.0,
    "bpm_norm":           0.3,
}


# Documented feature ranges. Mostly for downstream consumers that want
# to know whether [0,1] or [-1,1] is the natural axis for a feature.
FEATURE_RANGES: Dict[str, tuple[float, float]] = {
    "tag_mood":          (-1.0, 1.0),
    "tag_spikiness":     (-1.0, 1.0),
    "rms_mean":          (0.0, 1.0),
    "rms_p95":           (0.0, 1.0),
    "onset_rate":        (0.0, 1.0),
    "spectral_centroid": (0.0, 1.0),
    "crest_factor":      (0.0, 1.0),
    "bpm_folded_norm":   (0.0, 1.0),
    "bpm_norm":          (0.0, 1.0),
}


def _empty_vector() -> Dict[str, Optional[float]]:
    return {k: None for k in DEFAULT_WEIGHTS}


def extract_features(
    item: Dict[str, Any],
    *,
    intensity_duration: float = 60.0,
    bpm_duration: float = 60.0,
) -> Dict[str, Optional[float]]:
    """Compute the full feature vector for one track.

    Pulls together the three feature sources:

    * Tag features come from ``item['tags']`` (no extra fetch — tags
      must already be hydrated by the caller).
    * Audio features (intensity + BPM) come from ``item['audio_url']``.
      The audio is downloaded and decoded **once** here; both
      intensity feature extraction and Joe Sullivan BPM detection run
      against the shared buffer. That halves wall-clock per track
      versus calling the two sub-modules independently.

    Returned dict always contains every key from ``DEFAULT_WEIGHTS``
    (unavailable features are ``None`` rather than missing). It also
    carries an additional ``bpm`` field — the **raw BPM as a float**
    when one was detected, or ``None`` otherwise. That value is not
    part of the similarity vector (not in ``DEFAULT_WEIGHTS``) but
    callers that need to display or filter on absolute tempo can read
    it directly without re-running BPM detection.
    """
    # Local imports keep the optional audio stack out of bare cold starts
    # (e.g. when only tag features are needed).
    from bandcamp_recommender.recommendations.mood_tags import (
        extract_features as extract_tag_features,
    )

    vector: Dict[str, Optional[float]] = _empty_vector()
    vector["bpm"] = None  # Raw tempo, populated when audio is available.
    vector.update(extract_tag_features(item.get("tags") or []))

    audio_url = item.get("audio_url")
    if not audio_url:
        return vector

    # One download + decode, two feature paths. Match
    # ``intensity.attach_audio_features`` and use the longer of the two
    # configured durations so a shorter detector sees a prefix.
    from bandcamp_recommender.recommendations.bpm import (
        _load_audio_segment,
        bpm_to_features,
        detect_bpm_joe_sullivan_from_samples,
        detect_bpm_librosa_from_samples,
    )
    from bandcamp_recommender.recommendations.intensity import (
        extract_features_from_samples as intensity_features_from_samples,
    )

    decode_duration = max(intensity_duration, bpm_duration)
    decoded = _load_audio_segment(audio_url, duration=decode_duration)
    if decoded is None:
        return vector
    samples, sr = decoded

    vector.update(intensity_features_from_samples(samples, sr))

    # Joe Sullivan first (fast, kick-driven music). Fall back to
    # librosa on the same buffer for non-kick tracks (ambient, jazz,
    # solo piano, …) so we don't lose tempo coverage just because we
    # consolidated the decode. Both paths use the same samples — no
    # second download.
    raw_bpm: Optional[float] = None
    js_result = detect_bpm_joe_sullivan_from_samples(samples, sr)
    if js_result and js_result.get("bpm"):
        raw_bpm = float(js_result["bpm"])
    else:
        raw_bpm = detect_bpm_librosa_from_samples(samples, sr)

    vector.update(bpm_to_features(raw_bpm))
    vector["bpm"] = raw_bpm
    return vector


def distance(
    a: Dict[str, Optional[float]],
    b: Dict[str, Optional[float]],
    weights: Optional[Dict[str, float]] = None,
) -> Optional[float]:
    """Weighted Euclidean over the intersection of features both vectors have.

    A feature is "shared" when both ``a`` and ``b`` have non-None values
    for it. Weights default to :data:`DEFAULT_WEIGHTS`. The sum of
    squared weighted differences is normalized by the sum of weights of
    *present* dimensions so a track missing 2 of 9 features still gets a
    distance in the same scale as a fully-featured track. Returns
    ``None`` when no dimension is shared.
    """
    w = weights or DEFAULT_WEIGHTS
    sq = 0.0
    total_w = 0.0
    for key, weight in w.items():
        va = a.get(key)
        vb = b.get(key)
        if va is None or vb is None:
            continue
        sq += weight * (va - vb) ** 2
        total_w += weight
    if total_w <= 0:
        return None
    return math.sqrt(sq / total_w)


def project_mood(
    features: Dict[str, Optional[float]],
    *,
    weights: Optional[Dict[str, float]] = None,
) -> Optional[float]:
    """Collapse a feature vector to a single chill (-1) ↔ party (+1) scalar.

    Preserves the radio's slider semantics. The default projection is a
    linear combination of features that should correlate with perceived
    intensity. Customise via ``weights`` (same keys as the feature
    vector). Returns ``None`` when no contributing feature is present.

    Tag features are mapped through their natural ``[-1, 1]`` range;
    audio ``[0, 1]`` features are remapped to ``[-1, 1]`` via
    ``2 * value - 1`` so they push the score in either direction.
    """
    default = {
        "tag_mood":           1.0,
        "rms_p95":            0.6,
        "onset_rate":         0.8,
        "spectral_centroid":  0.3,
        "crest_factor":       0.3,
        "bpm_folded_norm":    0.4,
    }
    weights = weights or default

    contribution = 0.0
    total = 0.0
    for key, weight in weights.items():
        v = features.get(key)
        if v is None:
            continue
        lo, hi = FEATURE_RANGES.get(key, (-1.0, 1.0))
        # Map [0, 1] features into [-1, 1]; tag features already there.
        centered = v if lo < 0 else (2.0 * v - 1.0)
        contribution += weight * centered
        total += abs(weight)
    if total <= 0:
        return None
    return max(-1.0, min(1.0, contribution / total))


__all__ = [
    "DEFAULT_WEIGHTS",
    "FEATURE_RANGES",
    "extract_features",
    "distance",
    "project_mood",
]
