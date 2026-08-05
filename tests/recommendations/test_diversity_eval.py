"""Tests for the offline side-by-side evaluation harness (src/diversity_eval.py).

These pin the harness's *fairness/correctness invariants* — reproducibility,
honest distances, every variant scored on a common corpus — not the experiment's
numeric outcome (that lives in the generated results doc).
"""
import random

import pytest

from bandcamp_recommender.recommendations.diversity import full_distance, GateConfig
from bandcamp_recommender.eval.diversity_eval import (
    RegimeConfig,
    make_genres,
    make_seed,
    make_pool,
    run_regime,
    VARIANTS,
)


def _cfg(**kw):
    base = dict(
        name="t", n_genres=6, style_std=0.12, intensity_coupled=True,
        intensity_std=0.05, pool_size=24, p_same=0.5, k=4, artists_per_genre=3,
        gate=GateConfig(tau_int=0.15, realm_mult=1.5),
    )
    base.update(kw)
    return RegimeConfig(**base)


def test_corpus_reproducible_for_same_seed():
    cfg = _cfg()
    g1 = make_genres(cfg, random.Random(7))
    g2 = make_genres(cfg, random.Random(7))
    rng1, rng2 = random.Random(99), random.Random(99)
    s1 = make_seed(cfg, g1, rng1)
    s2 = make_seed(cfg, g2, rng2)
    p1 = make_pool(cfg, g1, s1, rng1)
    p2 = make_pool(cfg, g2, s2, rng2)
    assert [t.id for t in p1] == [t.id for t in p2]
    assert [t.features for t in p1] == [t.features for t in p2]


def test_pool_has_requested_size():
    cfg = _cfg(pool_size=30)
    genres = make_genres(cfg, random.Random(1))
    rng = random.Random(2)
    seed = make_seed(cfg, genres, rng)
    pool = make_pool(cfg, genres, seed, rng)
    assert len(pool) == 30


def test_distance_to_seed_is_honest():
    # The harness must compute each candidate's distance_to_seed exactly as the
    # feature-space distance — no hand-assigned numbers that could bias ranking.
    cfg = _cfg()
    genres = make_genres(cfg, random.Random(3))
    rng = random.Random(4)
    seed = make_seed(cfg, genres, rng)
    pool = make_pool(cfg, genres, seed, rng)
    for t in pool:
        assert t.distance_to_seed == pytest.approx(full_distance(t, seed))


def test_run_regime_scores_every_variant():
    cfg = _cfg()
    res = run_regime(cfg, n_seeds=20, rng=random.Random(5))
    assert set(res.keys()) == set(VARIANTS)
    for name, m in res.items():
        assert 0.0 <= m["vibe_pass_rate"] <= 1.0
        assert 0.0 <= m["fill_rate"] <= 1.0
        # quality metrics + independent (no-strategy-optimizes-it) coverage
        for key in ("mean_diversity", "mean_dispersion", "mean_relevance",
                    "mean_genre_coverage", "mean_artist_coverage"):
            assert key in m


def test_corpus_has_artist_duplicates_within_genre():
    # The dedup path in stratified must actually be exercised: a genre must be
    # able to contribute several tracks by the same artist.
    cfg = _cfg(artists_per_genre=2, pool_size=40, p_same=0.9)
    genres = make_genres(cfg, random.Random(1))
    rng = random.Random(2)
    seed = make_seed(cfg, genres, rng)
    pool = make_pool(cfg, genres, seed, rng)
    from collections import Counter
    counts = Counter(t.artist for t in pool)
    assert max(counts.values()) >= 2  # at least one artist appears twice


def test_strategies_beat_baseline_on_independent_genre_coverage():
    # Distinct-genre coverage is NOT what any strategy directly maximizes
    # (mmr/maxmin optimize continuous style-distance; stratified groups by
    # bucket but is judged here on realised distinct genres). Strategies should
    # still cover more genres per slice than the plain gated top-k baseline.
    cfg = _cfg(style_std=0.16, n_genres=10, pool_size=32, p_same=0.4)
    res = run_regime(cfg, n_seeds=40, rng=random.Random(13))
    base = res["baseline_gated"]["mean_genre_coverage"]
    for name in ("mmr", "maxmin", "stratified"):
        assert res[name]["mean_genre_coverage"] >= base - 1e-9


def test_gated_strategies_never_break_vibe():
    # Strategies selecting from the gated pool must always pass the vibe gate.
    cfg = _cfg(intensity_coupled=True)
    res = run_regime(cfg, n_seeds=30, rng=random.Random(6))
    for name in ("mmr", "maxmin", "stratified", "baseline_gated"):
        assert res[name]["vibe_pass_rate"] == pytest.approx(1.0)


def test_ungated_baseline_can_break_vibe():
    # When intensity is independent of style, a low-distance (similar-style)
    # candidate can still sit at a very different intensity, so top-k-by-distance
    # (ungated) reaches out of the vibe band at least sometimes. This is exactly
    # what the gate exists to prevent — and the gated variants must not.
    cfg = _cfg(intensity_coupled=False, p_same=0.4, n_genres=8, pool_size=30)
    res = run_regime(cfg, n_seeds=80, rng=random.Random(8))
    assert res["baseline"]["vibe_pass_rate"] < 1.0
    assert res["baseline_gated"]["vibe_pass_rate"] == pytest.approx(1.0)


def test_diversity_strategies_beat_gated_baseline_on_diversity():
    # On a diffuse, many-genre regime there is diversity to be had; the explicit
    # diversifiers should achieve >= the plain gated top-k's diversity.
    cfg = _cfg(style_std=0.18, n_genres=10, pool_size=32, p_same=0.4)
    res = run_regime(cfg, n_seeds=40, rng=random.Random(11))
    base = res["baseline_gated"]["mean_diversity"]
    for name in ("mmr", "maxmin", "stratified"):
        assert res[name]["mean_diversity"] >= base - 1e-9


def test_oracle_gap_nonnegative():
    cfg = _cfg()
    res = run_regime(cfg, n_seeds=30, rng=random.Random(12))
    for name in ("mmr", "maxmin", "stratified", "baseline_gated"):
        assert res[name]["mean_oracle_gap"] >= -1e-9
