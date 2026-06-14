"""Tests for the fast-path knobs on get_recommendations.

boemketel's progressive /recommend returns a candidate pool *fast*:
it reads only each supporter's first collection page (no whole-collection
API pagination) and skips per-item tag curls (tags arrive out-of-band
from its own page scrape). These tests pin that contract:

* first_page_only is threaded down to the supporter fetch
* hydrate_tags=False skips the per-item tag fetch entirely
* the defaults reproduce the old behavior (full pages + tag hydration)
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from bandcamp_recommender.recommendations import supporter_recommender as sr
from bandcamp_recommender.recommendations.supporter_recommender import (
    SupporterRecommender,
)


class FastPathKnobsTests(unittest.TestCase):
    def _fake_fetch(self, rec, seen):
        """A _get_supporter_items_curl_first stand-in that records the
        first_page_only flag and registers one owned item per supporter."""

        def fetch(username, data_key, first_page_only=False):
            seen.append(first_page_only)
            item_id = f"id_{username}"
            rec._store_item_metadata(
                item_id,
                {
                    "item_title": f"t_{username}",
                    "band_name": "b",
                    "item_url": f"https://x/{username}",
                    "tralbum_id": item_id,
                },
                True,
            )
            return [item_id]

        return fetch

    def test_first_page_only_and_skip_tags(self):
        rec = SupporterRecommender()
        seen = []
        with patch.object(sr, "extract_supporters", return_value=["alice", "bob"]), \
             patch.object(sr, "extract_item_id", return_value=None), \
             patch.object(rec, "_get_supporter_items_curl_first",
                          side_effect=self._fake_fetch(rec, seen)), \
             patch.object(rec, "_hydrate_tags_for_items") as hyd:
            recs = rec.get_recommendations(
                "https://seed",
                max_recommendations=10,
                min_supporters=1,
                first_page_only=True,
                hydrate_tags=False,
            )
        self.assertTrue(seen and all(v is True for v in seen),
                        f"first_page_only not threaded through: {seen}")
        hyd.assert_not_called()
        self.assertEqual(len(recs), 2)
        # bare candidates: title/band/url present, tags empty
        self.assertEqual(recs[0]["tags"], [])
        self.assertTrue(recs[0]["item_url"].startswith("https://x/"))

    def test_defaults_preserve_old_behavior(self):
        rec = SupporterRecommender()
        seen = []
        with patch.object(sr, "extract_supporters", return_value=["alice"]), \
             patch.object(sr, "extract_item_id", return_value=None), \
             patch.object(rec, "_get_supporter_items_curl_first",
                          side_effect=self._fake_fetch(rec, seen)), \
             patch.object(rec, "_hydrate_tags_for_items") as hyd:
            rec.get_recommendations(
                "https://seed", max_recommendations=10, min_supporters=1,
            )
        self.assertEqual(seen, [False])
        hyd.assert_called_once()

    def _union_fetch(self, rec, calls):
        """Stand-in returning distinct items per data_key plus one shared
        item that appears in BOTH the collection and wishlist of a supporter."""

        def fetch(username, data_key, first_page_only=False):
            calls.append((username, data_key))
            if data_key == "collection_data":
                ids = ["c_" + username, "shared_" + username]
            else:
                ids = ["w_" + username, "shared_" + username]
            for i in ids:
                rec._store_item_metadata(
                    i, {"item_title": i, "band_name": "b",
                        "item_url": "https://x/" + i, "tralbum_id": i}, True)
            return ids

        return fetch

    def test_use_wishlist_unions_collection_and_wishlist(self):
        rec = SupporterRecommender()
        calls = []
        with patch.object(sr, "extract_supporters", return_value=["alice"]), \
             patch.object(sr, "extract_item_id", return_value=None), \
             patch.object(rec, "_get_supporter_items_curl_first",
                          side_effect=self._union_fetch(rec, calls)), \
             patch.object(rec, "_hydrate_tags_for_items"):
            recs = rec.get_recommendations(
                "https://seed", max_recommendations=10, min_supporters=1,
                first_page_only=True, hydrate_tags=False, use_wishlist=True,
            )
        self.assertEqual({dk for _, dk in calls}, {"collection_data", "wishlist_data"})
        urls = {r["item_url"] for r in recs}
        self.assertEqual(
            urls, {"https://x/c_alice", "https://x/w_alice", "https://x/shared_alice"})

    def test_default_does_not_fetch_wishlist(self):
        rec = SupporterRecommender()
        calls = []
        with patch.object(sr, "extract_supporters", return_value=["alice"]), \
             patch.object(sr, "extract_item_id", return_value=None), \
             patch.object(rec, "_get_supporter_items_curl_first",
                          side_effect=self._union_fetch(rec, calls)), \
             patch.object(rec, "_hydrate_tags_for_items"):
            rec.get_recommendations(
                "https://seed", max_recommendations=10, min_supporters=1,
                first_page_only=True, hydrate_tags=False,  # use_wishlist defaults False
            )
        self.assertEqual({dk for _, dk in calls}, {"collection_data"})

    def test_wishlist_union_dedupes_per_supporter(self):
        """An item in BOTH a supporter's collection and wishlist counts as ONE
        supporter — so min_supporters keeps counting people, not occurrences."""
        rec = SupporterRecommender()

        def fetch(username, data_key, first_page_only=False):
            rec._store_item_metadata(
                "dup", {"item_title": "dup", "band_name": "b",
                        "item_url": "https://x/dup", "tralbum_id": "dup"}, True)
            return ["dup"]  # same item from both collection and wishlist

        with patch.object(sr, "extract_supporters", return_value=["alice"]), \
             patch.object(sr, "extract_item_id", return_value=None), \
             patch.object(rec, "_get_supporter_items_curl_first", side_effect=fetch), \
             patch.object(rec, "_hydrate_tags_for_items"):
            recs = rec.get_recommendations(
                "https://seed", max_recommendations=10, min_supporters=2,
                first_page_only=True, hydrate_tags=False, use_wishlist=True,
            )
        # Without per-supporter dedup, 'dup' would count 2 (≥2) and survive.
        self.assertEqual(recs, [])


if __name__ == "__main__":
    unittest.main()
