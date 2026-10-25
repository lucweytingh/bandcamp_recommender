"""Opt-in candidate-pool diversity: exclude_seed_artist + max_per_artist.

A seed's supporters mostly own its artist's other releases, and a small
scene's ranking is a few artists' back catalogues, so the pool cut to N
candidates can hold very few artists. Both options act on the ranked
candidates BEFORE the cut (seed artist before candidate_filter, the cap
after it), comparing artists by artist_tokens. Network is fully stubbed.
"""

from __future__ import annotations

import html as htmllib
import json
import unittest
from typing import List, Tuple
from unittest.mock import patch

from bandcamp_recommender.recommendations import scraper
from bandcamp_recommender.recommendations import supporter_recommender as sr
from bandcamp_recommender.recommendations.supporter_recommender import SupporterRecommender

SEED = "https://siumata.bandcamp.com/album/seed"


def _run(ranked: List[Tuple[str, str]], seed_artist="Siu Mata", **kwargs):
    """get_recommendations over ``ranked`` (``(item_id, band_name)``, best
    first: item i is owned by len(ranked) - i supporters, so the ranking is
    strict and deterministic). Returns the item ids in result order."""
    rec = SupporterRecommender()
    n = len(ranked)
    supporters = [f"s{i}" for i in range(n)]
    owns = {s: [iid for r, (iid, _) in enumerate(ranked) if i < n - r]
            for i, s in enumerate(supporters)}
    bands = dict(ranked)

    def fetch(username, data_key, first_page_only=False):
        for iid in owns[username]:
            rec._store_item_metadata(iid, {"item_title": iid, "band_name": bands[iid],
                                           "item_url": f"https://x.bandcamp.com/album/{iid}"}, True)
        return owns[username]

    with patch.object(sr, "extract_supporters", return_value=supporters), \
         patch.object(sr, "extract_item_id", return_value=None), \
         patch.object(sr, "extract_artist", return_value=seed_artist), \
         patch.object(rec, "_get_supporter_items_curl_first", side_effect=fetch):
        out = rec.get_recommendations(SEED, min_supporters=1, hydrate_tags=False, **kwargs)
    return [o["item_title"] for o in out]


RANKED = [
    ("seed2", "Siu Mata"),
    ("collab", "Amor Satyr & SIU MATA"),
    ("a1", "Artist A"),
    ("a2", "Artist A"),
    ("seed3", "siu mata"),
    ("a3", "Artist A"),
    ("b1", "Artist B"),
    ("a4", "Artist A"),
    ("c1", "Artist C"),
]


class DefaultsTests(unittest.TestCase):
    def test_defaults_keep_the_plain_ranking(self):
        # The app calls get_recommendations without the options: unchanged.
        self.assertEqual(_run(RANKED, max_recommendations=5),
                         ["seed2", "collab", "a1", "a2", "seed3"])

    def test_max_per_artist_must_be_positive(self):
        with self.assertRaises(ValueError):
            _run(RANKED, max_recommendations=5, max_per_artist=0)


class ExcludeSeedArtistTests(unittest.TestCase):
    def test_drops_the_seed_artist_and_its_collabs_before_the_cut(self):
        self.assertEqual(_run(RANKED, max_recommendations=5, exclude_seed_artist=True),
                         ["a1", "a2", "a3", "b1", "a4"])

    def test_collab_seed_excludes_each_member(self):
        ranked = [("x", "Amor Satyr"), ("y", "Siu Mata"), ("z", "Other")]
        self.assertEqual(
            _run(ranked, seed_artist="Siu Mata x Amor Satyr", max_recommendations=3,
                 exclude_seed_artist=True),
            ["z"],
        )

    def test_compilation_seed_excludes_nothing(self):
        ranked = [("comp", "Various Artists"), ("x", "Siu Mata"), ("y", "V/A")]
        self.assertEqual(
            _run(ranked, seed_artist="Various Artists", max_recommendations=3,
                 exclude_seed_artist=True),
            ["comp", "x", "y"],
        )

    def test_unreadable_seed_artist_excludes_nothing(self):
        self.assertEqual(
            _run(RANKED, seed_artist=None, max_recommendations=2, exclude_seed_artist=True),
            ["seed2", "collab"],
        )

    def test_candidate_filter_never_sees_seed_artist_releases(self):
        seen = []
        _run(RANKED, max_recommendations=5, exclude_seed_artist=True,
             candidate_filter=lambda meta: seen.append(meta["item_title"]) or True)
        self.assertEqual(seen, ["a1", "a2", "a3", "b1", "a4", "c1"])


class MaxPerArtistTests(unittest.TestCase):
    def test_keeps_each_artists_best_ranked_releases_in_order(self):
        self.assertEqual(
            _run(RANKED, max_recommendations=6, exclude_seed_artist=True, max_per_artist=2),
            ["a1", "a2", "b1", "c1"],
        )

    def test_cap_fills_the_pool_with_more_artists(self):
        # Without the cap the 4-slot pool is three releases by A.
        self.assertEqual(
            _run(RANKED, max_recommendations=4, exclude_seed_artist=True),
            ["a1", "a2", "a3", "b1"],
        )
        self.assertEqual(
            _run(RANKED, max_recommendations=4, exclude_seed_artist=True, max_per_artist=1),
            ["a1", "b1", "c1"],
        )

    def test_runs_after_candidate_filter(self):
        # A release the caller rejects doesn't use up its artist's slot.
        out = _run(RANKED, max_recommendations=6, exclude_seed_artist=True, max_per_artist=2,
                   candidate_filter=lambda meta: meta["item_title"] != "a1")
        self.assertEqual(out, ["a2", "a3", "b1", "c1"])

    def test_collab_counts_for_each_artist(self):
        ranked = [
            ("solo_a", "Artist A"),
            ("a_and_b", "Artist A & Artist B"),  # A is at the cap: skipped
            ("solo_b", "artist b"),              # B's slot is still free
            ("b_x_c", "Artist B x Artist C"),    # B is at the cap: skipped
            ("solo_c", "Artist C"),
        ]
        self.assertEqual(
            _run(ranked, seed_artist="Seed", max_recommendations=5, max_per_artist=1),
            ["solo_a", "solo_b", "solo_c"],
        )

    def test_compilations_are_never_capped(self):
        ranked = [("v1", "Various Artists"), ("v2", "Various Artists"), ("v3", "V/A"),
                  ("e1", ""), ("e2", "")]
        self.assertEqual(
            _run(ranked, seed_artist="Seed", max_recommendations=5, max_per_artist=1),
            ["v1", "v2", "v3", "e1", "e2"],
        )


class SimilarPassThroughTests(unittest.TestCase):
    def test_get_similar_recommendations_forwards_both_options(self):
        rec = SupporterRecommender()
        with patch.object(SupporterRecommender, "get_recommendations", return_value=[]) as spy:
            rec.get_similar_recommendations(SEED, candidate_pool_size=30,
                                            exclude_seed_artist=True, max_per_artist=2)
        kwargs = spy.call_args.kwargs
        self.assertEqual(kwargs["max_recommendations"], 30)
        self.assertIs(kwargs["exclude_seed_artist"], True)
        self.assertEqual(kwargs["max_per_artist"], 2)

    def test_get_similar_recommendations_defaults_are_off(self):
        rec = SupporterRecommender()
        with patch.object(SupporterRecommender, "get_recommendations", return_value=[]) as spy:
            rec.get_similar_recommendations(SEED)
        kwargs = spy.call_args.kwargs
        self.assertIs(kwargs["exclude_seed_artist"], False)
        self.assertIsNone(kwargs["max_per_artist"])


def _item_page(tralbum: dict, band: dict) -> str:
    return (
        f'<script data-tralbum="{htmllib.escape(json.dumps(tralbum))}" '
        f'data-band="{htmllib.escape(json.dumps(band))}"></script>'
    )


class ExtractArtistTests(unittest.TestCase):
    def test_label_page_gives_the_release_artist_not_the_account(self):
        # The fan-collection API's band_name is the release artist; on a
        # label's page data-band is the label (measured on dbh-music).
        page = _item_page({"artist": "Paul Johnson"}, {"name": "dbh-music"})
        with patch.object(scraper, "fetch_page_html", return_value=page):
            self.assertEqual(scraper.extract_artist("https://dbh-music.bandcamp.com/track/t"),
                             "Paul Johnson")

    def test_falls_back_to_the_account_name(self):
        page = _item_page({"current": {}}, {"name": "Siu Mata"})
        with patch.object(scraper, "fetch_page_html", return_value=page):
            self.assertEqual(scraper.extract_artist(SEED), "Siu Mata")

    def test_unreadable_page_is_none(self):
        with patch.object(scraper, "fetch_page_html", return_value=None):
            self.assertIsNone(scraper.extract_artist(SEED))
        with patch.object(scraper, "fetch_page_html", return_value="<html></html>"):
            self.assertIsNone(scraper.extract_artist(SEED))


if __name__ == "__main__":
    unittest.main()
