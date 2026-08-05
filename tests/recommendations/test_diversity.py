"""Tests for src/diversity.py — vibe-gated diversity re-rankers + the
diversity-optimizing heuristic objective.

The feature vocabulary is bandcamp_recommender's: intensity is the 0..1 blend
of (rms_mean, rms_p95, onset_rate, spectral_centroid, crest_factor); style lives
in (tag_mood, tag_spikiness, bpm_folded_norm). These tests construct tracks with
known intensity by setting the five intensity features equal to the target value
(the blend of equal values is the value itself), which keeps intent obvious.
"""
import math

import pytest

from bandcamp_recommender.recommendations.diversity import (
    Track,
    track_intensity,
    style_distance,
    relevance,
    vibe_gate,
    select_topk,
    select_mmr,
    select_maxmin,
    select_stratified,
    select_oracle,
    set_diversity,
    intensity_dispersion,
    realm_relevance,
    vibe_diversity_score,
    GateConfig,
    NEG_INF,
)


def feat(intensity=0.5, mood=0.0, spikiness=0.0, bpm_folded=0.5, bpm=120.0):
    """A full feature vector with a known intensity and style coordinates."""
    return {
        "rms_mean": intensity,
        "rms_p95": intensity,
        "onset_rate": intensity,
        "spectral_centroid": intensity,
        "crest_factor": intensity,
        "tag_mood": mood,
        "tag_spikiness": spikiness,
        "bpm_folded_norm": bpm_folded,
        "bpm_norm": bpm_folded,
        "bpm": bpm,
    }


def mk(intensity=0.5, mood=0.0, spikiness=0.0, bpm_folded=0.5, bpm=120.0,
       artist="a", genre="g", dist=None, id="t"):
    return Track(
        features=feat(intensity, mood, spikiness, bpm_folded, bpm),
        artist=artist, genre=genre, distance_to_seed=dist, id=id,
    )


# --- building blocks -------------------------------------------------------

def test_track_intensity_matches_blend():
    # all five intensity features equal to 0.7 -> blend is 0.7
    assert track_intensity(mk(intensity=0.7)) == pytest.approx(0.7)


def test_track_intensity_none_when_unknown():
    t = Track(features={"tag_mood": 0.2})  # no intensity features
    assert track_intensity(t) is None


def test_style_distance_ignores_intensity():
    # same style coords, very different intensity -> zero style distance
    a = mk(intensity=0.1, mood=0.3, spikiness=0.2, bpm_folded=0.4)
    b = mk(intensity=0.9, mood=0.3, spikiness=0.2, bpm_folded=0.4)
    assert style_distance(a, b) == pytest.approx(0.0, abs=1e-9)


def test_style_distance_grows_with_mood_gap():
    a = mk(mood=-1.0)
    b = mk(mood=0.0)
    c = mk(mood=1.0)
    assert style_distance(a, c) > style_distance(a, b) > 0


def test_relevance_prefers_precomputed_distance():
    t = mk(dist=0.42)
    seed = mk(mood=0.9)  # would be far if computed from features
    assert relevance(t, seed) == pytest.approx(0.42)


def test_relevance_falls_back_to_full_distance():
    t = mk(mood=0.0, dist=None)
    seed = mk(mood=0.0)
    assert relevance(t, seed) == pytest.approx(0.0, abs=1e-9)


# --- vibe gate -------------------------------------------------------------

def test_vibe_gate_keeps_within_intensity_band():
    seed = mk(intensity=0.5)
    pool = [mk(intensity=0.5, id="ok1"), mk(intensity=0.6, id="ok2"),
            mk(intensity=0.9, id="loud"), mk(intensity=0.2, id="quiet")]
    cfg = GateConfig(tau_int=0.15, realm_mult=None)
    kept = {t.id for t in vibe_gate(pool, seed, cfg)}
    assert kept == {"ok1", "ok2"}


def test_vibe_gate_fail_open_when_intensity_unknown():
    seed = mk(intensity=0.5)
    t = Track(features={"tag_mood": 0.1}, id="unknown")  # no intensity, no bpm-only band
    cfg = GateConfig(tau_int=0.15, realm_mult=None)
    kept = [x.id for x in vibe_gate([t], seed, cfg)]
    assert kept == ["unknown"]


def test_vibe_gate_realm_drops_far_outliers():
    seed = mk(mood=0.0)
    pool = [mk(mood=0.0, dist=0.1, id="near1"), mk(mood=0.0, dist=0.12, id="near2"),
            mk(mood=0.0, dist=0.11, id="near3"), mk(mood=0.0, dist=5.0, id="outlier")]
    cfg = GateConfig(tau_int=1.0, realm_mult=1.5)  # intensity gate effectively off
    kept = {t.id for t in vibe_gate(pool, seed, cfg)}
    assert "outlier" not in kept
    assert {"near1", "near2", "near3"} <= kept


# --- baseline + strategies -------------------------------------------------

def _pool_two_clusters():
    # cluster A: mood ~ -0.8 (three near-duplicates), cluster B: mood ~ +0.8 (one)
    return [
        mk(mood=-0.80, dist=0.10, artist="A1", genre="dub", id="a1"),
        mk(mood=-0.82, dist=0.12, artist="A2", genre="dub", id="a2"),
        mk(mood=-0.78, dist=0.14, artist="A3", genre="dub", id="a3"),
        mk(mood=+0.80, dist=0.20, artist="B1", genre="rave", id="b1"),
    ]


def test_select_topk_returns_k_lowest_distance():
    pool = _pool_two_clusters()
    picked = select_topk(pool, mk(), k=2)
    assert [t.id for t in picked] == ["a1", "a2"]


def test_select_respects_k_and_no_dupes():
    pool = _pool_two_clusters()
    for fn in (select_topk, select_mmr, select_maxmin, select_stratified):
        picked = fn(pool, mk(), k=3)
        assert len(picked) == 3
        assert len({t.id for t in picked}) == 3


def test_select_handles_pool_smaller_than_k():
    pool = _pool_two_clusters()[:2]
    for fn in (select_topk, select_mmr, select_maxmin, select_stratified):
        picked = fn(pool, mk(), k=5)
        assert len(picked) == 2


def test_mmr_lambda_one_equals_baseline():
    pool = _pool_two_clusters()
    seed = mk()
    a = [t.id for t in select_mmr(pool, seed, k=2, lam=1.0)]
    b = [t.id for t in select_topk(pool, seed, k=2)]
    assert a == b


def test_mmr_diversifies_more_than_baseline():
    # With diversity weight, the +0.8 cluster member should be reached before
    # piling up three near-duplicates from the -0.8 cluster.
    pool = _pool_two_clusters()
    seed = mk()
    picked = select_mmr(pool, seed, k=2, lam=0.3)
    assert "b1" in {t.id for t in picked}


def test_maxmin_picks_spread_set():
    pool = _pool_two_clusters()
    picked = select_maxmin(pool, mk(), k=2)
    ids = {t.id for t in picked}
    # one from each cluster, not two near-duplicates
    assert "b1" in ids
    assert len(ids & {"a1", "a2", "a3"}) == 1


def test_stratified_one_per_artist_and_covers_genres():
    pool = _pool_two_clusters()
    picked = select_stratified(pool, mk(), k=2)
    artists = [t.artist for t in picked]
    genres = {t.genre for t in picked}
    assert len(artists) == len(set(artists))  # no artist twice
    assert genres == {"dub", "rave"}          # both genres represented


def test_stratified_never_repeats_artist_with_duplicates():
    # A genre with several tracks by ONE artist plus another genre: stratified
    # must not fill the slice with that one artist — it dedups by artist.
    pool = [
        mk(mood=-0.8, artist="A", genre="dub", dist=0.10, id="d1"),
        mk(mood=-0.8, artist="A", genre="dub", dist=0.11, id="d2"),
        mk(mood=-0.8, artist="A", genre="dub", dist=0.12, id="d3"),
        mk(mood=0.8, artist="B", genre="rave", dist=0.30, id="r1"),
    ]
    picked = select_stratified(pool, mk(), k=2)
    artists = [t.artist for t in picked]
    assert artists.count("A") <= 1
    assert "B" in artists  # the other artist is reached rather than repeating A


def test_stratified_fills_k_when_distinct_artists_short():
    # Only TWO distinct artists but k=3: prefer distinct, but never return a
    # short slice — fall back to filling the last slot (so its set is the same
    # size as the other strategies', keeping the side-by-side comparison fair).
    pool = [
        mk(mood=-0.8, artist="A", genre="dub", dist=0.10, id="a1"),
        mk(mood=-0.7, artist="A", genre="dub", dist=0.11, id="a2"),
        mk(mood=0.8, artist="B", genre="dub", dist=0.20, id="b1"),
    ]
    picked = select_stratified(pool, mk(), k=3)
    assert len(picked) == 3
    assert len({t.id for t in picked}) == 3  # distinct tracks, no track twice


# --- heuristic objective ---------------------------------------------------

def test_set_diversity_higher_for_varied_set():
    seed = mk()
    clustered = [mk(mood=-0.8, id="x1"), mk(mood=-0.79, id="x2")]
    varied = [mk(mood=-0.8, id="y1"), mk(mood=0.8, id="y2")]
    assert set_diversity(varied) > set_diversity(clustered)


def test_intensity_dispersion_zero_for_equal_intensity():
    s = [mk(intensity=0.5), mk(intensity=0.5), mk(intensity=0.5)]
    assert intensity_dispersion(s) == pytest.approx(0.0, abs=1e-9)


def test_intensity_dispersion_positive_when_spread():
    s = [mk(intensity=0.2), mk(intensity=0.8)]
    assert intensity_dispersion(s) > 0


def test_vibe_score_rejects_out_of_band_member():
    seed = mk(intensity=0.5)
    bad = [mk(intensity=0.5, mood=-0.8), mk(intensity=0.95, mood=0.8)]  # 2nd too loud
    cfg = GateConfig(tau_int=0.15, realm_mult=None)
    assert vibe_diversity_score(seed, bad, gate=cfg) == NEG_INF


def test_vibe_score_prefers_diverse_same_vibe_over_clustered():
    seed = mk(intensity=0.5)
    cfg = GateConfig(tau_int=0.2, realm_mult=None)
    diverse = [mk(intensity=0.5, mood=-0.7, artist="A", genre="dub", id="d1"),
               mk(intensity=0.55, mood=0.7, artist="B", genre="rave", id="d2")]
    clustered = [mk(intensity=0.5, mood=-0.7, artist="A", genre="dub", id="c1"),
                 mk(intensity=0.5, mood=-0.69, artist="A", genre="dub", id="c2")]
    assert vibe_diversity_score(seed, diverse, gate=cfg) > vibe_diversity_score(seed, clustered, gate=cfg)


def test_oracle_beats_or_ties_every_strategy():
    # On the same gated pool, the oracle maximizes the heuristic, so its score
    # is >= each strategy's score.
    seed = mk(intensity=0.5)
    cfg = GateConfig(tau_int=0.5, realm_mult=None)
    pool = [
        mk(intensity=0.5, mood=-0.9, spikiness=-0.5, artist="A", genre="dub", dist=0.10, id="p1"),
        mk(intensity=0.52, mood=-0.85, spikiness=-0.4, artist="A", genre="dub", dist=0.11, id="p2"),
        mk(intensity=0.5, mood=0.0, spikiness=0.0, artist="C", genre="house", dist=0.20, id="p3"),
        mk(intensity=0.55, mood=0.9, spikiness=0.6, artist="D", genre="rave", dist=0.30, id="p4"),
        mk(intensity=0.48, mood=0.5, spikiness=0.3, artist="E", genre="techno", dist=0.25, id="p5"),
    ]
    k = 3
    oracle = select_oracle(pool, seed, k=k, gate=cfg)
    q_oracle = vibe_diversity_score(seed, oracle, gate=cfg)
    for fn in (select_topk, select_mmr, select_maxmin, select_stratified):
        s = fn(pool, seed, k=k)
        q = vibe_diversity_score(seed, s, gate=cfg)
        assert q_oracle >= q - 1e-9, f"{fn.__name__} scored {q} > oracle {q_oracle}"


# --- diversify_items: the item-dict adapter used by get_similar_recommendations

from bandcamp_recommender.recommendations.diversity import diversify_items


def _item(mood=0.0, intensity=0.5, dist=0.1, artist="b", url="u"):
    return {
        "item_url": url, "band_name": artist, "distance": dist,
        "features": feat(intensity=intensity, mood=mood),
    }


def test_diversify_items_is_permutation():
    src = feat(intensity=0.5)
    items = [_item(mood=-0.8, url="a"), _item(mood=0.8, url="b"), _item(mood=0.0, url="c")]
    out = diversify_items(items, src, mode="mmr")
    assert {i["item_url"] for i in out} == {"a", "b", "c"}
    assert len(out) == 3


def test_diversify_items_none_mode_is_noop():
    src = feat(intensity=0.5)
    items = [_item(mood=-0.8, dist=0.1, url="a"), _item(mood=0.8, dist=0.2, url="b")]
    assert [i["item_url"] for i in diversify_items(items, src, mode=None)] == ["a", "b"]


def test_diversify_items_reorders_for_spread():
    src = feat(intensity=0.5)
    items = [
        _item(mood=-0.9, dist=0.10, url="a"),
        _item(mood=-0.88, dist=0.11, url="a2"),
        _item(mood=0.9, dist=0.13, url="b"),
    ]
    out = [i["item_url"] for i in diversify_items(items, src, mode="mmr", lam=0.5)]
    assert out[0] == "a"
    assert out[1] == "b"  # diverse pick before the near-duplicate


def test_diversify_items_pushes_out_of_vibe_to_tail():
    src = feat(intensity=0.5)
    items = [
        _item(mood=0.0, intensity=0.95, dist=0.01, url="loud"),  # out of vibe band
        _item(mood=-0.5, intensity=0.5, dist=0.20, url="ok1"),
        _item(mood=0.5, intensity=0.5, dist=0.25, url="ok2"),
    ]
    out = [i["item_url"] for i in diversify_items(items, src, mode="mmr", vibe_tau=0.15)]
    assert out[-1] == "loud"


def test_diversify_items_truncates_to_k():
    src = feat(intensity=0.5)
    items = [_item(mood=m / 10.0, url=f"u{m}") for m in range(8)]
    out = diversify_items(items, src, mode="maxmin", k=4)
    assert len(out) == 4


def test_diversify_items_handles_missing_features_and_distance():
    src = feat(intensity=0.5)
    items = [{"item_url": "a", "band_name": "x"}, {"item_url": "b", "band_name": "y"}]
    out = diversify_items(items, src, mode="mmr")
    assert {i["item_url"] for i in out} == {"a", "b"}


def test_diversify_items_normalizes_mode_case_and_whitespace():
    src = feat(intensity=0.5)
    items = [_item(mood=-0.9, dist=0.10, url="a"), _item(mood=-0.88, dist=0.11, url="a2"),
             _item(mood=0.9, dist=0.13, url="b")]
    canonical = [i["item_url"] for i in diversify_items(items, src, mode="mmr")]
    for variant in ("MMR", " mmr ", "Mmr"):
        assert [i["item_url"] for i in diversify_items(items, src, mode=variant)] == canonical


def test_diversify_items_unknown_mode_is_noop_not_crash():
    src = feat(intensity=0.5)
    items = [_item(mood=-0.8, dist=0.1, url="a"), _item(mood=0.8, dist=0.2, url="b")]
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = diversify_items(items, src, mode="max-min")  # operator typo
    # Degrades gracefully to the plain ranking instead of raising.
    assert [i["item_url"] for i in out] == ["a", "b"]
    assert any("unknown diversify mode" in str(x.message).lower() for x in w)
