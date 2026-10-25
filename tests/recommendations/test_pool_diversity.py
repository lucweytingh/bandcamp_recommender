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


def _recommend(ranked: List[Tuple[str, str]], seed_artist="Siu Mata", calls=None, **kwargs):
    """get_recommendations over ``ranked`` (``(item_id, band_name)``, best
    first: item i is owned by len(ranked) - i supporters, so the ranking is
    strict and deterministic). Returns the item ids in result order and the
    recommender. ``calls`` (a list) records the order of seed-artist and
    supporter reads."""
    rec = SupporterRecommender()
    n = len(ranked)
    supporters = [f"s{i}" for i in range(n)]
    owns = {s: [t[0] for r, t in enumerate(ranked) if i < n - r]
            for i, s in enumerate(supporters)}
    bands = {t[0]: t[1] for t in ranked}
    # Optional third element: the release's page URL (default: one page).
    urls = {t[0]: (t[2] if len(t) > 2 else f"https://x.bandcamp.com/album/{t[0]}") for t in ranked}
    calls = [] if calls is None else calls

    def fetch(username, data_key, first_page_only=False):
        calls.append("supporter")
        for iid in owns[username]:
            rec._store_item_metadata(iid, {"item_title": iid, "band_name": bands[iid],
                                           "item_url": urls[iid]}, True)
        return owns[username]

    def artist(url):
        if not kwargs.get("exclude_seed_artist"):
            raise AssertionError("seed artist read without exclude_seed_artist")
        calls.append("seed_artist")
        return seed_artist

    with patch.object(sr, "extract_supporters", return_value=supporters), \
         patch.object(sr, "extract_item_id", return_value=None), \
         patch.object(sr, "extract_artist", side_effect=artist), \
         patch.object(rec, "_get_supporter_items_curl_first", side_effect=fetch):
        out = rec.get_recommendations(SEED, min_supporters=1, hydrate_tags=False, **kwargs)
    return [o["item_title"] for o in out], rec


def _run(ranked: List[Tuple[str, str]], seed_artist="Siu Mata", **kwargs):
    return _recommend(ranked, seed_artist, **kwargs)[0]


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
        # The app calls get_recommendations without the options: unchanged,
        # and the seed artist is never read (_recommend raises if it is).
        ids, rec = _recommend(RANKED, max_recommendations=5)
        self.assertEqual(ids, ["seed2", "collab", "a1", "a2", "seed3"])
        self.assertEqual(rec._pool_stats, {})

    def test_max_per_artist_must_be_positive(self):
        with self.assertRaises(ValueError):
            _run(RANKED, max_recommendations=5, max_per_artist=0)


class ExcludeSeedArtistTests(unittest.TestCase):
    def test_drops_the_seed_artist_and_its_collabs_before_the_cut(self):
        self.assertEqual(_run(RANKED, max_recommendations=5, exclude_seed_artist=True),
                         ["a1", "a2", "a3", "b1", "a4"])

    def test_seed_artist_is_read_before_the_supporter_phase(self):
        # Right after the seed page was fetched, so it's a page-cache hit;
        # a slow supporter phase could otherwise outlive the cached page.
        calls = []
        _recommend(RANKED, calls=calls, max_recommendations=5, exclude_seed_artist=True)
        self.assertEqual(calls[0], "seed_artist")
        self.assertEqual(calls.count("seed_artist"), 1)

    def test_collab_seed_excludes_each_member(self):
        ranked = [("x", "Amor Satyr"), ("y", "Siu Mata"), ("z", "Other")]
        self.assertEqual(
            _run(ranked, seed_artist="Siu Mata x Amor Satyr", max_recommendations=3,
                 exclude_seed_artist=True),
            ["z"],
        )

    def test_compilation_seed_excludes_nothing(self):
        ranked = [("comp", "Various Artists"), ("x", "Siu Mata"), ("y", "V/A")]
        ids, rec = _recommend(ranked, seed_artist="Various Artists", max_recommendations=3,
                              exclude_seed_artist=True)
        self.assertEqual(ids, ["comp", "x", "y"])
        self.assertEqual(rec._pool_stats, {"seed_artist": "Various Artists",
                                           "seed_artist_dropped": 0})

    def test_unreadable_seed_artist_excludes_nothing_and_says_so(self):
        # Fail open, but visibly: a markup change must not silently turn the
        # exclusion off.
        with self.assertLogs(sr.logger, level="WARNING") as logs:
            ids, rec = _recommend(RANKED, seed_artist=None, max_recommendations=2,
                                  exclude_seed_artist=True)
        self.assertEqual(ids, ["seed2", "collab"])
        self.assertIn("no artist on the seed page", logs.output[0])
        self.assertEqual(rec._pool_stats, {"seed_artist": None, "seed_artist_dropped": 0})

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

    def test_compilations_from_different_pages_are_not_capped(self):
        ranked = [("v1", "Various Artists", "https://label-one.bandcamp.com/album/v1"),
                  ("v2", "Various Artists", "https://label-two.bandcamp.com/album/v2"),
                  ("v3", "V/A", "https://label-three.bandcamp.com/album/v3"),
                  ("e1", "", "https://four.bandcamp.com/album/e1"),
                  ("e2", "", "https://five.bandcamp.com/album/e2")]
        self.assertEqual(
            _run(ranked, seed_artist="Seed", max_recommendations=5, max_per_artist=1),
            ["v1", "v2", "v3", "e1", "e2"],
        )

    def test_placeholder_credits_from_one_page_are_one_act(self):
        # Measured: a label crediting every release to "Unknown Artist" aired
        # three of them back to back; the page is their identity.
        ranked = [("k18", "Unknown Artist", "https://kcik.bandcamp.com/track/kcik-18"),
                  ("k32", "Unknown Artist", "https://kcik.bandcamp.com/track/kcik-32"),
                  ("k26", "Unknown Artist", "https://kcik.bandcamp.com/track/kcik-26"),
                  ("b1", "Artist B", "https://b.bandcamp.com/album/b1")]
        self.assertEqual(
            _run(ranked, seed_artist="Seed", max_recommendations=4, max_per_artist=1),
            ["k18", "b1"],
        )

    def test_placeholder_seed_drops_its_own_pages_releases(self):
        ranked = [("k32", "Unknown Artist", "https://siumata.bandcamp.com/track/k32"),
                  ("x", "Artist X", "https://x.bandcamp.com/album/x")]
        ids, _ = _recommend(ranked, seed_artist="Unknown Artist", max_recommendations=2,
                            exclude_seed_artist=True)
        self.assertEqual(ids, ["x"])

    def test_bpm_match_caps_the_expanded_pool(self):
        # bpm_match cuts the pool at max(50, 3 * n) instead of n; the cap
        # fills that pool too. The ranked event shows the pool itself.
        ranked = [(f"a{i}", "Artist A") for i in range(5)] + \
                 [(f"x{i}", f"Artist {i}") for i in range(60)]
        events = []
        with patch("bandcamp_recommender.recommendations.bpm.get_seed_bpm", return_value=None):
            ids, rec = _recommend(ranked, max_recommendations=2, max_per_artist=2,
                                  bpm_match=True, event_callback=events.append)
        pool = [t["title"] for e in events if e["type"] == "ranked" for t in e["top"]]
        self.assertEqual(len(pool), 50)
        self.assertEqual(pool[:3], ["a0", "a1", "x0"])
        self.assertEqual(rec._pool_stats, {"artist_capped": 3})
        self.assertEqual(ids, ["a0", "a1"])  # no seed BPM: trimmed to n


class PoolStatsTests(unittest.TestCase):
    def test_counts_what_each_step_dropped(self):
        ids, rec = _recommend(RANKED, max_recommendations=6, exclude_seed_artist=True,
                              max_per_artist=2,
                              candidate_filter=lambda meta: meta["item_title"] != "c1")
        self.assertEqual(ids, ["a1", "a2", "b1"])
        self.assertEqual(rec._pool_stats, {
            "seed_artist": "Siu Mata",
            "seed_artist_dropped": 3,  # seed2, collab, seed3
            "filtered": 1,             # c1
            "artist_capped": 2,        # a3, a4
        })

    def test_cap_stops_counting_once_the_pool_is_full(self):
        _, rec = _recommend(RANKED, max_recommendations=2, exclude_seed_artist=True,
                            max_per_artist=1)
        # a1 kept, a2 and a3 skipped, b1 fills the pool: a4 is never looked at.
        self.assertEqual(rec._pool_stats["artist_capped"], 2)

    def test_get_similar_recommendations_reports_them(self):
        rec = SupporterRecommender()
        cand = {"item_url": "https://x.bandcamp.com/album/a", "band_name": "A", "tags": []}

        def pool(**kwargs):
            rec._pool_stats = {"seed_artist": "Siu Mata", "seed_artist_dropped": 4}
            return [dict(cand)]

        with patch.object(rec, "get_recommendations", side_effect=pool), \
             patch.object(sr, "extract_tags", return_value=[]), \
             patch("bandcamp_recommender.features.features_for_item",
                   return_value=({"rms_mean": 0.5}, None)):
            rec.get_similar_recommendations(SEED, exclude_seed_artist=True)
        self.assertEqual(rec.last_call_stats["pool"],
                         {"seed_artist": "Siu Mata", "seed_artist_dropped": 4})


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

    def test_reads_the_one_item_page_parse(self):
        # The same artist items.parse_item_html gives, e.g. from ld+json on a
        # page shape without data-tralbum.
        ld = {"@type": "MusicAlbum", "byArtist": {"@type": "MusicGroup", "name": "Rene Wise"}}
        page = f'<script type="application/ld+json">{json.dumps(ld)}</script>'
        with patch.object(scraper, "fetch_page_html", return_value=page):
            self.assertEqual(scraper.extract_artist(SEED), "Rene Wise")

    def test_unreadable_page_is_none(self):
        with patch.object(scraper, "fetch_page_html", return_value=None):
            self.assertIsNone(scraper.extract_artist(SEED))
        with patch.object(scraper, "fetch_page_html", return_value="<html></html>"):
            self.assertIsNone(scraper.extract_artist(SEED))


if __name__ == "__main__":
    unittest.main()
