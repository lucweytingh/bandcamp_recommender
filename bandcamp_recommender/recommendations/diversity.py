"""Vibe-gated diversity re-ranking for similarity recommendations.

``get_similar_recommendations`` ranks candidates by feature-vector distance to a
source, so the top-*k* tend to cluster in style. This module re-ranks that
candidate pool to instead surface *k* items that are **diverse in style but
consistent in vibe (intensity) and still related to the source** — an opt-in
"more like this, but varied" mode that sits on the package's own feature space.

Two orthogonal axes drive everything (see ``bandcamp_recommender.features``):

* **vibe / intensity** — the 0..1 scalar blended from
  ``rms_mean, rms_p95, onset_rate, spectral_centroid, crest_factor``. We keep
  this *similar* across the set (the gate-keeper).
* **style** — ``tag_mood`` (chill↔party), ``tag_spikiness`` (dense↔spiky) and
  ``bpm_folded_norm`` (octave-folded tempo). We *spread* picks along these.

Everything here is a pure function over plain :class:`Track` objects, so it runs
offline in the evaluation harness and in unit tests with no Selenium/network.
"""
from __future__ import annotations

import math
import statistics
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from bandcamp_recommender.features import distance as _feature_distance
from bandcamp_recommender.recommendations.intensity import (
    score_intensity_from_features,
)

NEG_INF = float("-inf")

# Axes that define "style" diversity — intensity/energy axes are deliberately
# excluded (they're the vibe we hold constant). Tempo's octave-folded form is a
# style axis: two same-energy tracks can sit at very different tempos.
STYLE_WEIGHTS: Dict[str, float] = {
    "tag_mood": 1.0,
    "tag_spikiness": 0.6,
    "bpm_folded_norm": 1.0,
}


@dataclass
class Track:
    """A candidate (or seed). ``features`` is the bandcamp_recommender vector
    (values may be ``None``). ``distance_to_seed`` is the recommender's own
    relevance distance when available — preferred over recomputing it so the
    re-ranker stays faithful to the recommender's notion of "more like this"."""

    features: Dict[str, Optional[float]] = field(default_factory=dict)
    artist: str = ""
    genre: str = ""
    distance_to_seed: Optional[float] = None
    id: str = ""


@dataclass
class GateConfig:
    """Vibe/realm gate parameters.

    ``tau_int``    max |intensity − seed_intensity| to stay in the same vibe.
    ``tau_bpm``    fallback band on raw BPM when intensity is unknown.
    ``realm_mult`` drop candidates whose seed-distance exceeds
                   ``realm_mult × median(seed-distance)`` (outlier removal so
                   picks stay related to the seed). ``None`` disables the realm
                   gate (the recommender pool is already realm-constrained).
    """

    tau_int: float = 0.15
    tau_bpm: float = 20.0
    realm_mult: Optional[float] = 1.5


@dataclass
class ScoreWeights:
    w_div: float = 1.0   # reward style diversity
    w_disp: float = 1.0  # penalize intensity dispersion (vibe tightness)
    w_rel: float = 0.5   # penalize mean distance to seed (stay in realm)
    w_cat: float = 0.3   # weight of the categorical-variety bonus within diversity


# --- building blocks -------------------------------------------------------

def track_intensity(track: Track) -> Optional[float]:
    """The 0..1 vibe scalar, or ``None`` when no intensity feature is present."""
    return score_intensity_from_features(track.features or {})


def _bpm(track: Track) -> Optional[float]:
    v = (track.features or {}).get("bpm")
    return float(v) if isinstance(v, (int, float)) else None


def style_distance(a: Track, b: Track) -> float:
    """Weighted-Euclidean distance over the style axes only (intensity ignored).

    Returns 0.0 when no style axis is shared, so identical-style tracks (or
    tracks with no style features) collapse together rather than erroring.
    """
    d = _feature_distance(a.features or {}, b.features or {}, STYLE_WEIGHTS)
    return float(d) if d is not None else 0.0


def full_distance(a: Track, b: Track) -> float:
    """Distance over the recommender's full default feature space."""
    d = _feature_distance(a.features or {}, b.features or {})
    return float(d) if d is not None else 0.0


def relevance(track: Track, seed: Track) -> float:
    """Distance to the seed — the recommender's value if present, else computed."""
    if track.distance_to_seed is not None:
        return float(track.distance_to_seed)
    return full_distance(track, seed)


# --- vibe gate -------------------------------------------------------------

def _passes_intensity_band(track: Track, seed: Track, cfg: GateConfig) -> bool:
    ti = track_intensity(track)
    si = track_intensity(seed)
    if ti is not None and si is not None:
        return abs(ti - si) <= cfg.tau_int
    # Intensity unknown → fall back to a raw-BPM band if both have BPM.
    tb, sb = _bpm(track), _bpm(seed)
    if tb is not None and sb is not None:
        return abs(tb - sb) <= cfg.tau_bpm
    # Nothing to judge on → fail open (never silence a slice on missing data).
    return True


def vibe_gate(candidates: Sequence[Track], seed: Track, cfg: Optional[GateConfig] = None) -> List[Track]:
    """Filter ``candidates`` to those that share the seed's vibe and realm."""
    cfg = cfg or GateConfig()
    in_band = [c for c in candidates if _passes_intensity_band(c, seed, cfg)]
    if cfg.realm_mult is None or not in_band:
        return in_band
    rels = [relevance(c, seed) for c in in_band]
    median = statistics.median(rels)
    if median <= 0:
        return in_band
    limit = cfg.realm_mult * median
    return [c for c, r in zip(in_band, rels) if r <= limit]


# --- selection strategies --------------------------------------------------
# Each returns up to k tracks from the (already gated, in production) pool.

def select_topk(pool: Sequence[Track], seed: Track, k: int) -> List[Track]:
    """Baseline: the k most relevant (lowest seed-distance). No diversity."""
    return sorted(pool, key=lambda t: relevance(t, seed))[:k]


def select_mmr(pool: Sequence[Track], seed: Track, k: int, lam: float = 0.5) -> List[Track]:
    """Maximal Marginal Relevance, expressed in distance terms.

    Maximizes ``-lam·relevance(c) + (1-lam)·min_style_dist(c, selected)``: small
    seed-distance (relevant) and large style-distance to what's already chosen
    (diverse). ``lam=1`` reduces to the baseline; ``lam=0`` is pure spread.
    """
    remaining = list(pool)
    if not remaining:
        return []
    # First pick is the most relevant (the diversity term is undefined for an
    # empty selection); MMR proper governs the rest.
    remaining.sort(key=lambda t: relevance(t, seed))
    selected = [remaining.pop(0)]
    while len(selected) < k and remaining:
        best, best_key = None, NEG_INF
        for c in remaining:
            min_div = min(style_distance(c, s) for s in selected)
            key = -lam * relevance(c, seed) + (1.0 - lam) * min_div
            if key > best_key:
                best, best_key = c, key
        selected.append(best)
        remaining.remove(best)
    return selected


def select_maxmin(pool: Sequence[Track], seed: Track, k: int,
                  relevance_top_frac: float = 1.0) -> List[Track]:
    """Farthest-point / facility-location spread within the relevant subset.

    Seeds with the single most relevant track, then greedily adds whoever
    maximizes the minimum style-distance to the chosen set. ``relevance_top_frac``
    optionally restricts to the most-relevant fraction first; it defaults to 1.0
    because in production the vibe gate already removes out-of-realm outliers, so
    the strategy's job is purely spread. Lower it for standalone (ungated) use.
    """
    if not pool:
        return []
    ranked = sorted(pool, key=lambda t: relevance(t, seed))
    cutoff = max(k, math.ceil(len(ranked) * relevance_top_frac))
    cand = ranked[:cutoff]
    selected = [cand.pop(0)]
    while len(selected) < k and cand:
        best, best_key = None, NEG_INF
        for c in cand:
            min_div = min(style_distance(c, s) for s in selected)
            if min_div > best_key:
                best, best_key = c, min_div
        selected.append(best)
        cand.remove(best)
    return selected


def _primary_bucket(track: Track) -> str:
    """A categorical key for stratification: the genre tag, else a coarse
    discretization of style space so the strategy still works tag-free."""
    if track.genre:
        return track.genre
    mood = (track.features or {}).get("tag_mood") or 0.0
    spik = (track.features or {}).get("tag_spikiness") or 0.0
    return f"m{round(mood, 1)}_s{round(spik, 1)}"


def select_stratified(pool: Sequence[Track], seed: Track, k: int) -> List[Track]:
    """Genre/artist-stratified coverage.

    Groups the pool by primary bucket (genre), ranks buckets by their most
    relevant member, then takes one track per bucket (most relevant first),
    never repeating an artist. Falls back to filling from leftovers if there are
    fewer buckets than k.
    """
    if not pool:
        return []
    buckets: Dict[str, List[Track]] = {}
    for t in pool:
        buckets.setdefault(_primary_bucket(t), []).append(t)
    for b in buckets.values():
        b.sort(key=lambda t: relevance(t, seed))
    # Order buckets by their best (most relevant) member.
    ordered = sorted(buckets.values(), key=lambda b: relevance(b[0], seed))

    selected: List[Track] = []
    used_artists: set = set()
    # Round-robin one per bucket, skipping already-used artists.
    progressed = True
    bucket_cursors = {id(b): 0 for b in ordered}
    while len(selected) < k and progressed:
        progressed = False
        for b in ordered:
            if len(selected) >= k:
                break
            i = bucket_cursors[id(b)]
            while i < len(b):
                t = b[i]
                i += 1
                if t.artist and t.artist in used_artists:
                    continue
                selected.append(t)
                if t.artist:
                    used_artists.add(t.artist)
                progressed = True
                break
            bucket_cursors[id(b)] = i

    # Never return a short slice: if preferring distinct artists left us under k
    # (a pool with fewer distinct artists than k), fill the remaining slots with
    # the most-relevant not-yet-picked tracks (allowing an artist to recur).
    # Keeps the slice the same size as the other strategies' for a fair compare.
    if len(selected) < k:
        chosen_ids = {id(t) for t in selected}
        leftover = sorted((t for t in pool if id(t) not in chosen_ids),
                          key=lambda t: relevance(t, seed))
        for t in leftover:
            if len(selected) >= k:
                break
            selected.append(t)
    return selected[:k]


# --- heuristic objective ---------------------------------------------------

def _mean_pairwise_style_distance(tracks: Sequence[Track]) -> float:
    if len(tracks) < 2:
        return 0.0
    dists = [
        style_distance(tracks[i], tracks[j])
        for i in range(len(tracks))
        for j in range(i + 1, len(tracks))
    ]
    return sum(dists) / len(dists)


def categorical_variety(tracks: Sequence[Track]) -> float:
    """0..1: how many distinct artists & genres, averaged. 1 = all distinct."""
    n = len(tracks)
    if n < 2:
        return 0.0
    artists = len({t.artist for t in tracks if t.artist})
    genres = len({t.genre for t in tracks if t.genre})
    a = (artists - 1) / (n - 1) if artists else 0.0
    g = (genres - 1) / (n - 1) if genres else 0.0
    # Average only the dimensions that carry information.
    parts = [v for v, present in ((a, artists), (g, genres)) if present]
    return sum(parts) / len(parts) if parts else 0.0


def set_diversity(tracks: Sequence[Track], weights: Optional[ScoreWeights] = None) -> float:
    """Style spread (mean pairwise style-distance) plus a categorical bonus."""
    w = weights or ScoreWeights()
    return _mean_pairwise_style_distance(tracks) + w.w_cat * categorical_variety(tracks)


def intensity_dispersion(tracks: Sequence[Track]) -> float:
    """Population stddev of intensity over the set (0 = identical vibe)."""
    vals = [v for v in (track_intensity(t) for t in tracks) if v is not None]
    if len(vals) < 2:
        return 0.0
    return statistics.pstdev(vals)


def realm_relevance(tracks: Sequence[Track], seed: Track) -> float:
    """Mean seed-distance (lower = more anchored to the player)."""
    if not tracks:
        return 0.0
    return sum(relevance(t, seed) for t in tracks) / len(tracks)


def vibe_diversity_score(
    seed: Track,
    tracks: Sequence[Track],
    gate: Optional[GateConfig] = None,
    weights: Optional[ScoreWeights] = None,
) -> float:
    """The diversity-optimizing heuristic.

    Returns ``NEG_INF`` when any member breaks the vibe/realm gate (the gate is
    a hard constraint — diversity is never bought at the cost of vibe). Otherwise
    ``w_div·diversity − w_disp·dispersion − w_rel·relevance``.
    """
    cfg = gate or GateConfig()
    w = weights or ScoreWeights()
    if not tracks:
        return NEG_INF
    # Hard gate = the vibe (intensity) band only: every member must share the
    # seed's vibe. The realm constraint is enforced upstream by vibe_gate
    # (a stable pool-level median, not a self-referential per-set one) and
    # reflected softly here by the w_rel penalty — so a set that passed the gate
    # always has a finite score.
    if any(not _passes_intensity_band(t, seed, cfg) for t in tracks):
        return NEG_INF
    return (
        w.w_div * set_diversity(tracks, w)
        - w.w_disp * intensity_dispersion(tracks)
        - w.w_rel * realm_relevance(tracks, seed)
    )


def select_oracle(
    pool: Sequence[Track],
    seed: Track,
    k: int,
    gate: Optional[GateConfig] = None,
    weights: Optional[ScoreWeights] = None,
    beam_width: int = 12,
) -> List[Track]:
    """Reference selector: (beam) search for the set maximizing the heuristic.

    Used as an upper bound in the side-by-side eval — what a strategy *could*
    achieve if it optimized the objective directly. Beam search keeps it cheap
    while staying close to the true argmax on the small pools we evaluate.
    """
    cfg = gate or GateConfig()
    gated = vibe_gate(pool, seed, cfg)
    if not gated:
        return []
    k = min(k, len(gated))

    def score(sel: Sequence[Track]) -> float:
        return vibe_diversity_score(seed, sel, cfg, weights)

    # Beam over growing subsets; score partial sets by the same objective.
    beams: List[List[Track]] = [[]]
    for _ in range(k):
        scored = []
        for sel in beams:
            sel_ids = {id(t) for t in sel}
            for c in gated:
                if id(c) in sel_ids:
                    continue
                cand = sel + [c]
                scored.append((score(cand), cand))
        # Dedup by frozenset of ids; keep best beam_width.
        seen = set()
        scored.sort(key=lambda x: x[0], reverse=True)
        beams = []
        for sc, cand in scored:
            key = frozenset(id(t) for t in cand)
            if key in seen:
                continue
            seen.add(key)
            beams.append(cand)
            if len(beams) >= beam_width:
                break
        if not beams:
            break
    best = max(beams, key=score)
    return best


# Registry for the eval harness (name -> selector with default params).
STRATEGIES = {
    "baseline_topk": select_topk,
    "mmr": select_mmr,
    "maxmin": select_maxmin,
    "stratified": select_stratified,
}


def diversify_items(
    items: Sequence[Dict],
    source_features: Dict[str, Optional[float]],
    mode: Optional[str] = "mmr",
    lam: float = 0.5,
    vibe_tau: float = 0.15,
    realm_mult: Optional[float] = 1.5,
    k: Optional[int] = None,
) -> List[Dict]:
    """Reorder recommendation item-dicts for in-vibe diversity.

    The item-dict adapter used by ``get_similar_recommendations``: each item is a
    dict with ``features`` (the feature vector), ``band_name`` (artist),
    ``distance`` (to the source, optional) and ``item_url``. ``source_features``
    is the source track's feature vector.

    Returns a permutation of ``items``: the candidates that share the source's
    vibe (intensity band, ``vibe_tau``) come first, ordered by ``mode`` (``mmr``
    / ``maxmin`` / ``stratified``) so the set spreads across styles while staying
    anchored to the source; out-of-vibe candidates are appended as backfill (so a
    caller asking for k items is never starved). ``k`` truncates the result.

    ``mode`` is normalized (trimmed + lower-cased): ``None``/""/"baseline"/"none"
    is a no-op (plain ranking, just truncates to k); ``"mmr"``/``"maxmin"``/
    ``"stratified"`` select a strategy. An **unknown** mode (an operator typo in a
    param or the ``BANDCAMP_DIVERSIFY`` env) does NOT raise — it warns and falls
    back to the plain ranking, so a config typo can never crash recommendation
    generation.
    """
    out = list(items)
    norm = (mode or "").strip().lower()
    if norm in ("", "baseline", "none") or len(out) < 2:
        return out[:k] if k else out

    selector_factories = {
        "mmr": lambda g, seed: select_mmr(g, seed, len(g), lam=lam),
        "maxmin": lambda g, seed: select_maxmin(g, seed, len(g)),
        "stratified": lambda g, seed: select_stratified(g, seed, len(g)),
    }
    factory = selector_factories.get(norm)
    if factory is None:
        warnings.warn(
            f"unknown diversify mode {mode!r}; returning plain similarity ranking "
            f"(expected one of {sorted(selector_factories)})",
            stacklevel=2,
        )
        return out[:k] if k else out

    seed = Track(features=source_features or {}, id="__source__")
    cand = [
        Track(
            features=it.get("features") or {},
            artist=it.get("band_name", ""),
            distance_to_seed=it.get("distance"),
            id=str(i),
        )
        for i, it in enumerate(out)
    ]
    gate = GateConfig(tau_int=vibe_tau, realm_mult=realm_mult if realm_mult else None)
    gated = vibe_gate(cand, seed, gate)
    gated_ids = {t.id for t in gated}
    selector = lambda g: factory(g, seed)

    ranked = selector(gated) if gated else []
    order = [int(t.id) for t in ranked]
    order += [i for i in range(len(cand)) if str(i) not in gated_ids]
    result = [out[i] for i in order]
    return result[:k] if k else result
