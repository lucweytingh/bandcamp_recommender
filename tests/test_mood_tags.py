"""Tests for the everynoise-backed chill/party mood prior."""

from __future__ import annotations

import pytest

from bandcamp_recommender.recommendations.mood_tags import (
    _GENRE_ENTRIES,
    genre_score,
    genre_weight,
    tag_mood_score,
)


# ---------------------------------------------------------------------------
# Snapshot sanity
# ---------------------------------------------------------------------------


def test_snapshot_loaded():
    # The vendored CSV should have parsed thousands of genres.
    assert len(_GENRE_ENTRIES) > 5000


def test_scores_are_bounded():
    for tag, (mood, spikiness, weight) in _GENRE_ENTRIES.items():
        assert -1.0 <= mood <= 1.0, (tag, mood)
        assert -1.0 <= spikiness <= 1.0, (tag, spikiness)
        assert weight >= 1.0, (tag, weight)


def test_snapshot_has_party_and_chill_extremes():
    # Sanity that both ends of the spectrum are represented.
    scores = [m for m, _, _ in _GENRE_ENTRIES.values()]
    assert any(s > 0.9 for s in scores)
    assert any(s < -0.9 for s in scores)


# ---------------------------------------------------------------------------
# IDF weighting
# ---------------------------------------------------------------------------


def test_popular_genres_get_smallest_weight():
    # "pop" is the single most popular genre on everynoise (font 160%),
    # so it should sit at the floor weight of 1.0.
    assert genre_weight("pop") == 1.0


def test_niche_genres_get_higher_weight():
    # Anything that isn't the top-1 popular genre should weigh more.
    w_pop = genre_weight("pop")
    w_speedcore = genre_weight("speedcore")
    assert w_speedcore is not None and w_pop is not None
    assert w_speedcore > w_pop


def test_idf_pulls_score_toward_rare_tag():
    # "rock" is very popular and roughly mid-mood; "speedcore" is rare
    # and strongly party. Weighted average should land much closer to
    # speedcore than an unweighted average would.
    tags = ["rock", "speedcore"]
    unweighted = tag_mood_score(tags, weighted=False)
    weighted = tag_mood_score(tags, weighted=True)
    rock = genre_score("rock")
    speed = genre_score("speedcore")
    assert None not in (unweighted, weighted, rock, speed)
    # Unweighted is the plain midpoint.
    assert abs(unweighted - (rock + speed) / 2) < 1e-9
    # Weighted should be strictly closer to the rare tag.
    assert abs(weighted - speed) < abs(unweighted - speed)


def test_weighted_default_is_on():
    # No keyword == weighted behavior.
    tags = ["rock", "speedcore"]
    assert tag_mood_score(tags) == tag_mood_score(tags, weighted=True)


def test_genre_weight_unknown_is_none():
    assert genre_weight("definitely-not-a-genre-xyz") is None
    assert genre_weight("") is None


# ---------------------------------------------------------------------------
# Direction checks — known-party tags should score > 0, known-chill < 0.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tag",
    ["techno", "hard techno", "gabber", "drum and bass", "jungle", "hardstyle"],
)
def test_party_tags_score_positive(tag):
    score = genre_score(tag)
    assert score is not None, f"{tag!r} missing from snapshot"
    assert score > 0, f"{tag!r} expected party-leaning, got {score}"


@pytest.mark.parametrize(
    "tag",
    ["ambient", "drone", "classical", "contemporary classical"],
)
def test_chill_tags_score_negative(tag):
    score = genre_score(tag)
    assert score is not None, f"{tag!r} missing from snapshot"
    assert score < 0, f"{tag!r} expected chill-leaning, got {score}"


# ---------------------------------------------------------------------------
# tag_mood_score behavior — synthetic mixes.
# ---------------------------------------------------------------------------


def test_chill_only_is_negative():
    score = tag_mood_score(["ambient", "drone"])
    assert score is not None and score < 0


def test_party_only_is_positive():
    score = tag_mood_score(["techno", "gabber", "drum and bass"])
    assert score is not None and score > 0


def test_balanced_mix_near_zero():
    # Avg of "techno" (~+0.9) and "ambient" (~-0.34) lands between them
    # and the chill end pulls less, so we only assert it's between them.
    techno = genre_score("techno")
    ambient = genre_score("ambient")
    score = tag_mood_score(["techno", "ambient"])
    assert score is not None
    assert min(techno, ambient) < score < max(techno, ambient)


def test_no_relevant_tags_returns_none():
    assert tag_mood_score(["definitely-not-a-genre-xyz", "luc-was-here"]) is None


def test_empty_input_returns_none():
    assert tag_mood_score([]) is None
    assert tag_mood_score(None) is None  # type: ignore[arg-type]


def test_irrelevant_tags_ignored_when_mixed_with_relevant():
    # An unknown tag should not move the score; result equals the lone
    # known tag's score.
    expected = genre_score("ambient")
    assert tag_mood_score(["ambient", "made-up-tag-1234"]) == pytest.approx(expected)


def test_duplicate_tags_counted_once():
    # Three copies of "ambient" plus one "techno" should equal one of each.
    a = tag_mood_score(["ambient", "ambient", "ambient", "techno"])
    b = tag_mood_score(["ambient", "techno"])
    assert a == b


def test_normalization_is_case_insensitive():
    assert tag_mood_score(["AMBIENT"]) == pytest.approx(genre_score("ambient"))


def test_whitespace_is_stripped():
    assert tag_mood_score(["  techno  "]) == pytest.approx(genre_score("techno"))


def test_empty_strings_skipped():
    expected = genre_score("ambient")
    assert tag_mood_score(["", "ambient", ""]) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Aliases — Bandcamp-side spellings that don't appear verbatim on everynoise.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "alias, canonical",
    [
        ("lofi", "lo-fi"),
        ("lo fi", "lo-fi"),
        ("dnb", "drum and bass"),
        ("d&b", "drum and bass"),
        ("drum & bass", "drum and bass"),
        ("psytrance", "psychedelic trance"),
    ],
)
def test_alias_maps_to_canonical_score(alias, canonical):
    canonical_score = genre_score(canonical)
    assert canonical_score is not None
    assert genre_score(alias) == canonical_score
