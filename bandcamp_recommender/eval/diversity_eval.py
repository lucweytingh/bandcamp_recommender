"""Offline side-by-side evaluation of diversity selection strategies.

The recommender's live path needs Selenium + a Bandcamp account, so this harness
evaluates the strategies on a **synthetic corpus** with known latent structure.
Fairness is the priority: the corpus is seeded/reproducible, candidate
distances-to-seed are computed honestly from the feature space (never assigned),
and every strategy is scored on the *same* pools with the *same* heuristic. The
gate effect is separated from the strategy effect by including a ``baseline_gated``
variant (top-k on the gated pool) alongside the ungated ``baseline``.

Synthetic model
---------------
``n_genres`` latent genres, each with a style centre (tag_mood, tag_spikiness,
bpm_folded) and — when ``intensity_coupled`` — its own mean intensity. A seed is
a track sampled near a random genre's centre. Its candidate pool mixes tracks
from the seed's genre (probability ``p_same``) and from other genres, each
jittered by ``style_std``. Intensity is sampled per-track from its genre's mean
(coupled) or globally (independent). ``distance_to_seed`` is the honest feature
distance, mirroring what the recommender supplies.
"""
from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional

from bandcamp_recommender.recommendations.diversity import (
    Track,
    GateConfig,
    ScoreWeights,
    full_distance,
    vibe_gate,
    select_topk,
    select_mmr,
    select_maxmin,
    select_stratified,
    select_oracle,
    vibe_diversity_score,
    set_diversity,
    intensity_dispersion,
    realm_relevance,
    track_intensity,
    NEG_INF,
)

# Order matters only for readable reports.
VARIANTS = [
    "baseline",        # top-k by distance, NO vibe gate (production today)
    "baseline_gated",  # top-k by distance on the gated pool (isolates gate effect)
    "mmr",
    "maxmin",
    "stratified",
    "oracle",          # heuristic argmax on the gated pool (upper bound)
]


@dataclass
class RegimeConfig:
    name: str
    n_genres: int = 6
    style_std: float = 0.12          # within-genre style jitter
    intensity_coupled: bool = True   # intensity tied to genre vs global
    intensity_std: float = 0.05
    pool_size: int = 24
    p_same: float = 0.5              # share of pool drawn from the seed's genre
    k: int = 4
    artists_per_genre: int = 3       # >1 so the same artist can recur in a pool
    gate: GateConfig = None          # type: ignore[assignment]
    weights: ScoreWeights = None     # type: ignore[assignment]

    def __post_init__(self):
        if self.gate is None:
            self.gate = GateConfig(tau_int=0.15, realm_mult=1.5)
        if self.weights is None:
            self.weights = ScoreWeights()


@dataclass
class Genre:
    name: str
    mood: float
    spikiness: float
    bpm_folded: float
    intensity_mean: float


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def make_genres(cfg: RegimeConfig, rng: random.Random) -> List[Genre]:
    genres = []
    for i in range(cfg.n_genres):
        genres.append(Genre(
            name=f"g{i}",
            mood=rng.uniform(-1.0, 1.0),
            spikiness=rng.uniform(-1.0, 1.0),
            bpm_folded=rng.uniform(0.0, 1.0),
            # Genres overlap substantially in intensity (realistic: chill ambient,
            # chill dub and chill folk all sit low-energy). A narrower spread than
            # the full 0..1 means a same-vibe band can legitimately span several
            # genres, so "diverse genres at one vibe" is achievable rather than
            # the gate collapsing every coupled pool to a single genre.
            intensity_mean=rng.uniform(0.35, 0.7),
        ))
    return genres


def _features(intensity: float, mood: float, spikiness: float, bpm_folded: float) -> Dict:
    intensity = _clip(intensity, 0.0, 1.0)
    return {
        # Five intensity features set equal -> blended intensity == `intensity`.
        "rms_mean": intensity,
        "rms_p95": intensity,
        "onset_rate": intensity,
        "spectral_centroid": intensity,
        "crest_factor": intensity,
        "tag_mood": _clip(mood, -1.0, 1.0),
        "tag_spikiness": _clip(spikiness, -1.0, 1.0),
        "bpm_folded_norm": _clip(bpm_folded, 0.0, 1.0),
        "bpm_norm": _clip(bpm_folded, 0.0, 1.0),
        "bpm": 60.0 + 140.0 * _clip(bpm_folded, 0.0, 1.0),
    }


def _sample_track(cfg, genre, rng, global_intensity, id_) -> Track:
    if cfg.intensity_coupled:
        intensity = rng.gauss(genre.intensity_mean, cfg.intensity_std)
    else:
        intensity = global_intensity
    # Draw the artist from a small per-genre roster so the same artist recurs
    # within a pool — this exercises stratified's artist-dedup and keeps the
    # categorical-variety metric non-degenerate.
    artist_n = rng.randrange(max(1, cfg.artists_per_genre))
    return Track(
        features=_features(
            intensity,
            rng.gauss(genre.mood, cfg.style_std),
            rng.gauss(genre.spikiness, cfg.style_std),
            rng.gauss(genre.bpm_folded, cfg.style_std),
        ),
        artist=f"{genre.name}-art{artist_n}",
        genre=genre.name,
        id=f"{genre.name}-{id_}",
    )


def make_seed(cfg: RegimeConfig, genres: List[Genre], rng: random.Random) -> Track:
    g = rng.choice(genres)
    gi = rng.uniform(0.3, 0.8)
    t = _sample_track(cfg, g, rng, gi, "seed")
    t.distance_to_seed = 0.0
    return t


def make_pool(cfg: RegimeConfig, genres: List[Genre], seed: Track,
              rng: random.Random) -> List[Track]:
    seed_genre = next((g for g in genres if g.name == seed.genre), genres[0])
    others = [g for g in genres if g.name != seed_genre.name] or genres
    global_intensity_seed = track_intensity(seed) or 0.5
    pool: List[Track] = []
    for i in range(cfg.pool_size):
        if rng.random() < cfg.p_same:
            g = seed_genre
        else:
            g = rng.choice(others)
        gi = (rng.gauss(global_intensity_seed, 0.18)
              if not cfg.intensity_coupled else 0.0)
        t = _sample_track(cfg, g, rng, gi, i)
        t.distance_to_seed = full_distance(t, seed)  # honest, recommender-style
        pool.append(t)
    return pool


def _selector(name: str):
    return {
        "baseline": lambda pool, seed, k, cfg: select_topk(pool, seed, k),
        "baseline_gated": lambda pool, seed, k, cfg: select_topk(
            vibe_gate(pool, seed, cfg.gate), seed, k),
        "mmr": lambda pool, seed, k, cfg: select_mmr(
            vibe_gate(pool, seed, cfg.gate), seed, k),
        "maxmin": lambda pool, seed, k, cfg: select_maxmin(
            vibe_gate(pool, seed, cfg.gate), seed, k),
        "stratified": lambda pool, seed, k, cfg: select_stratified(
            vibe_gate(pool, seed, cfg.gate), seed, k),
        "oracle": lambda pool, seed, k, cfg: select_oracle(
            pool, seed, k, cfg.gate, cfg.weights, beam_width=32),
    }[name]


def _distinct(seq):
    return len(set(seq))


def run_regime(cfg: RegimeConfig, n_seeds: int, rng: random.Random) -> Dict[str, Dict]:
    """Run every variant over ``n_seeds`` seeds; return per-variant aggregate
    metrics. Each seed gets its own genres+pool so the corpus isn't a single
    lucky layout.

    Metric denominators are kept consistent and explicit:

    * ``vibe_pass_rate`` / ``fill_rate`` — over ALL seeds (constraint
      satisfaction: does the slice keep the seed's vibe, and could it fill k?).
    * every quality metric (diversity, dispersion, relevance, the independent
      genre/artist coverage, Q, oracle gap) — over the **feasible** slices only
      (those that pass the vibe gate). This compares quality on equal footing:
      only where the vibe constraint is actually met. (Previously diversity used
      a wider denominator than Q, which flattered the ungated baseline.)

    ``mean_genre_coverage`` / ``mean_artist_coverage`` are *independent* of what
    any strategy optimizes (mmr/maxmin maximize continuous style-distance, not a
    distinct count), so the side-by-side doesn't rest solely on the optimizers'
    own objective.
    """
    keys = ["div", "disp", "rel", "genre_cov", "artist_cov", "q", "oracle_gap"]
    acc: Dict[str, Dict[str, list]] = {
        v: {k: [] for k in keys + ["vibe_pass", "fill"]} for v in VARIANTS
    }
    for _ in range(n_seeds):
        genres = make_genres(cfg, rng)
        seed = make_seed(cfg, genres, rng)
        pool = make_pool(cfg, genres, seed, rng)

        oracle_set = _selector("oracle")(pool, seed, cfg.k, cfg)
        q_oracle = vibe_diversity_score(seed, oracle_set, cfg.gate, cfg.weights)

        for v in VARIANTS:
            sel = _selector(v)(pool, seed, cfg.k, cfg)
            a = acc[v]
            a["fill"].append(1.0 if len(sel) == cfg.k else 0.0)
            if not sel:
                a["vibe_pass"].append(0.0)
                continue
            q = vibe_diversity_score(seed, sel, cfg.gate, cfg.weights)
            passes = q != NEG_INF
            a["vibe_pass"].append(1.0 if passes else 0.0)
            # Quality metrics only over feasible (vibe-passing) slices.
            if not passes:
                continue
            a["div"].append(set_diversity(sel, cfg.weights))
            a["disp"].append(intensity_dispersion(sel))
            a["rel"].append(realm_relevance(sel, seed))
            a["genre_cov"].append(float(_distinct(t.genre for t in sel)))
            a["artist_cov"].append(float(_distinct(t.artist for t in sel)))
            a["q"].append(q)
            if q_oracle != NEG_INF:
                a["oracle_gap"].append(q_oracle - q)

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    out: Dict[str, Dict] = {}
    for v in VARIANTS:
        a = acc[v]
        out[v] = {
            "mean_diversity": _mean(a["div"]),
            "mean_dispersion": _mean(a["disp"]),
            "mean_relevance": _mean(a["rel"]),
            "mean_genre_coverage": _mean(a["genre_cov"]),
            "mean_artist_coverage": _mean(a["artist_cov"]),
            "vibe_pass_rate": _mean(a["vibe_pass"]),
            "fill_rate": _mean(a["fill"]),
            "mean_Q_feasible": _mean(a["q"]),
            "mean_oracle_gap": _mean(a["oracle_gap"]),
        }
    return out


def default_regimes() -> List[RegimeConfig]:
    """A spread of regimes so the conclusion is robust, not setup-specific."""
    return [
        RegimeConfig("tight_coupled_fewgenre", n_genres=3, style_std=0.08,
                     intensity_coupled=True, pool_size=24, p_same=0.6),
        RegimeConfig("diffuse_coupled_manygenre", n_genres=12, style_std=0.18,
                     intensity_coupled=True, pool_size=36, p_same=0.4),
        RegimeConfig("independent_intensity", n_genres=8, style_std=0.14,
                     intensity_coupled=False, pool_size=30, p_same=0.5),
        RegimeConfig("scarce_pool", n_genres=6, style_std=0.12,
                     intensity_coupled=True, pool_size=14, p_same=0.5),
        RegimeConfig("seed_dominated", n_genres=8, style_std=0.12,
                     intensity_coupled=True, pool_size=30, p_same=0.75),
    ]


def evaluate(regimes: Optional[List[RegimeConfig]] = None, n_seeds: int = 200,
             base_seed: int = 1234) -> Dict[str, Dict[str, Dict]]:
    regimes = regimes if regimes is not None else default_regimes()
    results: Dict[str, Dict[str, Dict]] = {}
    for i, cfg in enumerate(regimes):
        results[cfg.name] = run_regime(cfg, n_seeds, random.Random(base_seed + i))
    return results
