"""Tag-based features sourced from everynoise.com.

Glenn McDonald's everynoise.com places ~6000 music genres on a 2D plane.
We treat the two axes as independent musical features:

* **Vertical** (``top:`` pixel value): chill ↔ party / mellow ↔ intense.
  Party end (top ~0) — hard techno, gabber, speedcore, dnb, jungle.
  Chill end (top ~22000) — ambient, drone, classical, modern composition.

* **Horizontal** (``left:`` pixel value): dense/atmospheric ↔ spiky/bouncy.
  Dense end (left ~0)   — drone, ambient, modern classical, dark ambient.
  Spiky end (left ~max) — hard techno, drum and bass, jazz, bossa nova,
                          breakcore. Captures rhythmic density rather
                          than tempo: jazz can be slow but is rhythmically
                          "spiky", drone can be fast but reads dense.

We vendor a snapshot of those coordinates as
``data/everynoise_genres.csv`` (refresh with ``scripts/fetch_everynoise.py``)
and map each axis to ``[-1.0, 1.0]``:

    tag_mood      = 1.0 - 2.0 * (top  / max_top)    # +1 party, -1 chill
    tag_spikiness = 2.0 * (left / max_left) - 1.0   # -1 dense, +1 spiky

These two features are exposed via :func:`extract_features`. The legacy
single-axis :func:`tag_mood_score` is preserved for the radio's
chill/party slider and returns just the ``tag_mood`` axis.

**IDF weighting.** The snapshot also carries each genre's font-size
(100%..160%), which is everynoise's visual proxy for popularity. The
visual scale is already perceptually log(popularity), so a *linear*
function of font-size gives us classic ``-log(df)`` IDF without a
second log. We weight each tag's contribution by

    weight(g) = (font_max + 1) - font_pct(g)

so the rarest genres (``font_pct=100``) contribute ~61x more than the
single most popular one (``font_pct=160``) — niche tags like "speedcore"
swing the score harder than catch-alls like "rock" or "pop". The final
score is the weight-normalized mean of per-tag scores. Pass
``weighted=False`` to fall back to an unweighted mean.

Tags that don't appear on everynoise contribute nothing (so geographic
or hyper-specific Bandcamp tags don't muddy the score), and the whole
call returns ``None`` when *no* tag has a match — callers can then fall
back to BPM or another signal rather than treating absence as "neutral".

Matching uses :func:`bandcamp_recommender.recommendations.tags.normalize_tag`
on both sides. A tag that survives normalization but still doesn't match
(e.g. "lofi" vs everynoise's "lo-fi") is unscored — extend the snapshot
or add aliases in :data:`_TAG_ALIASES` below if you hit a real-world miss
that matters.
"""

from __future__ import annotations

import csv
from importlib import resources
from typing import Iterable, Optional

from bandcamp_recommender.recommendations.tags import normalize_tag


# Bandcamp-side spellings that don't appear verbatim on everynoise but
# refer to the same genre. Keep this list short — prefer to leave a tag
# unscored over a wrong-direction match.
_TAG_ALIASES: dict[str, str] = {
    "lofi": "lo-fi",
    "lo fi": "lo-fi",
    "dnb": "drum and bass",
    "d&b": "drum and bass",
    "drum & bass": "drum and bass",
    "trip-hop": "trip hop",
    "neo soul": "neo-soul",
    "dream pop": "dream-pop",
    "post rock": "post-rock",
    "psytrance": "psychedelic trance",
}


def _load_everynoise() -> tuple[dict[str, tuple[float, float, float]], int, int, int]:
    """Load the snapshot and pre-compute ``(mood_score, spikiness_score, idf_weight)`` per genre."""
    data_file = resources.files(
        "bandcamp_recommender.recommendations"
    ).joinpath("data/everynoise_genres.csv")
    rows: list[tuple[str, int, int, int]] = []
    with data_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((
                row["genre"],
                int(row["top"]),
                int(row["left"]),
                int(row["font_pct"]),
            ))

    if not rows:
        return {}, 0, 0, 0

    max_top = max(top for _, top, _, _ in rows)
    max_left = max(left for _, _, left, _ in rows)
    max_font = max(font for _, _, _, font in rows)

    entries: dict[str, tuple[float, float, float]] = {}
    for name, top, left, font in rows:
        norm = normalize_tag(name)
        mood = 1.0 - 2.0 * top / max_top
        # left=0 is the leftmost (dense/atmospheric); max_left is the
        # rightmost (spiky/bouncy) — map to [-1, 1].
        spikiness = 2.0 * left / max_left - 1.0 if max_left else 0.0
        # font is perceptually log(popularity), so a linear (max-font)+1
        # gap recovers classic -log(df) IDF weight. +1 keeps the most
        # popular genre at weight 1 rather than 0.
        weight = float(max_font + 1 - font)
        # First occurrence wins on normalization collisions.
        entries.setdefault(norm, (mood, spikiness, weight))
    return entries, max_top, max_left, max_font


_GENRE_ENTRIES, _MAX_TOP, _MAX_LEFT, _MAX_FONT = _load_everynoise()


def _resolve(norm_tag: str) -> Optional[tuple[float, float, float]]:
    entry = _GENRE_ENTRIES.get(norm_tag)
    if entry is not None:
        return entry
    alias = _TAG_ALIASES.get(norm_tag)
    if alias is not None:
        return _GENRE_ENTRIES.get(normalize_tag(alias))
    return None


def _weighted_axis_mean(
    tags: Iterable[str],
    axis_index: int,
    *,
    weighted: bool,
) -> Optional[float]:
    """Internal: weighted mean of a single axis (0=mood, 1=organic)."""
    if not tags:
        return None
    seen: set[str] = set()
    total = 0.0
    weight_sum = 0.0
    for raw in tags:
        if not raw:
            continue
        norm = normalize_tag(raw)
        if norm in seen:
            continue
        seen.add(norm)
        entry = _resolve(norm)
        if entry is None:
            continue
        axis_value = entry[axis_index]
        weight = entry[2]
        w = weight if weighted else 1.0
        total += w * axis_value
        weight_sum += w
    if weight_sum == 0:
        return None
    return total / weight_sum


def tag_mood_score(
    tags: Iterable[str], *, weighted: bool = True
) -> Optional[float]:
    """Score a list of tags on a chill (-1) to party (+1) axis.

    Preserved for the radio's chill/party slider — returns just the
    ``tag_mood`` axis. For the full feature vector use
    :func:`extract_features`.
    """
    return _weighted_axis_mean(tags, axis_index=0, weighted=weighted)


def tag_spikiness_score(
    tags: Iterable[str], *, weighted: bool = True
) -> Optional[float]:
    """Score a list of tags on a dense/atmospheric (-1) to spiky/bouncy (+1) axis."""
    return _weighted_axis_mean(tags, axis_index=1, weighted=weighted)


def extract_features(
    tags: Iterable[str], *, weighted: bool = True
) -> dict[str, Optional[float]]:
    """Return all tag-derived features for an item as a dict.

    Both features are independent IDF-weighted means over the subset of
    tags that have an everynoise entry. ``None`` for a feature means
    no recognised tag contributed to that axis.

    Keys:
        ``tag_mood``       — chill (-1) ↔ party (+1)
        ``tag_spikiness``  — dense/atmospheric (-1) ↔ spiky/bouncy (+1)
    """
    # Walking the tag list once for both axes avoids re-resolving each
    # genre lookup twice.
    if not tags:
        return {"tag_mood": None, "tag_spikiness": None}
    seen: set[str] = set()
    mood_total = 0.0
    spiky_total = 0.0
    weight_sum = 0.0
    for raw in tags:
        if not raw:
            continue
        norm = normalize_tag(raw)
        if norm in seen:
            continue
        seen.add(norm)
        entry = _resolve(norm)
        if entry is None:
            continue
        mood, spikiness, weight = entry
        w = weight if weighted else 1.0
        mood_total += w * mood
        spiky_total += w * spikiness
        weight_sum += w
    if weight_sum == 0:
        return {"tag_mood": None, "tag_spikiness": None}
    return {
        "tag_mood": mood_total / weight_sum,
        "tag_spikiness": spiky_total / weight_sum,
    }


def genre_score(tag: str) -> Optional[float]:
    """Look up a single tag's mood score, or ``None`` if not on everynoise."""
    if not tag:
        return None
    entry = _resolve(normalize_tag(tag))
    return None if entry is None else entry[0]


def genre_spikiness(tag: str) -> Optional[float]:
    """Look up a single tag's dense↔spiky score, or ``None`` if not on everynoise."""
    if not tag:
        return None
    entry = _resolve(normalize_tag(tag))
    return None if entry is None else entry[1]


def genre_weight(tag: str) -> Optional[float]:
    """Look up a single tag's IDF weight, or ``None`` if not on everynoise.

    The weight is a linear function of ``(max_font + 1) - font_pct`` (see
    module docstring). Useful for callers who want to combine the mood
    prior with other signals at the same rarity weighting.
    """
    if not tag:
        return None
    entry = _resolve(normalize_tag(tag))
    return None if entry is None else entry[2]


__all__ = [
    "extract_features",
    "tag_mood_score",
    "tag_spikiness_score",
    "genre_score",
    "genre_spikiness",
    "genre_weight",
]
