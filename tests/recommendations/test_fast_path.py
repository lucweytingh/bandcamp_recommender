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


if __name__ == "__main__":
    unittest.main()
