"""Structured event_callback on the supporter miners.

Pins the graph-animation contract: get_recommendations emits a
`supporters` event, one `supporter_done` per supporter, then a final
`ranked` event ordered by overlap count — and is a no-op when
event_callback is None (legacy progress_callback path unaffected).
Network is fully stubbed (no curl, no Selenium).
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from bandcamp_recommender.recommendations import supporter_recommender as sr
from bandcamp_recommender.recommendations.supporter_recommender import (
    SupporterRecommender,
)


def _fake_fetch(rec):
    """Stub _get_supporter_items_curl_first.

    u1 owns A,B · u2 owns A,C · u3 owns A  → A shared by all three.
    """
    owns = {"u1": ["A", "B"], "u2": ["A", "C"], "u3": ["A"]}

    def fetch(username, data_key, first_page_only=False):
        ids = owns.get(username, [])
        for iid in ids:
            rec._store_item_metadata(
                iid,
                {
                    "item_title": f"title_{iid}",
                    "band_name": f"band_{iid}",
                    "item_url": f"https://bc/{iid}",
                    "tralbum_id": iid,
                },
                True,
            )
        return list(ids)

    return fetch


class EventCallbackTests(unittest.TestCase):
    def test_emits_supporters_then_per_supporter_then_ranked(self):
        rec = SupporterRecommender()
        events = []
        with patch.object(sr, "extract_supporters", return_value=["u1", "u2", "u3"]), \
             patch.object(sr, "extract_item_id", return_value="SEED"), \
             patch.object(
                 rec, "_get_supporter_items_curl_first", side_effect=_fake_fetch(rec)
             ):
            rec.get_recommendations(
                wishlist_item_url="https://bc/seed",
                max_recommendations=10,
                min_supporters=1,
                first_page_only=True,
                hydrate_tags=False,
                event_callback=events.append,
            )

        types = [e["type"] for e in events]
        self.assertEqual(types[0], "supporters")
        self.assertEqual(events[0]["total"], 3)
        self.assertEqual(types.count("supporter_done"), 3)
        self.assertEqual(types[-1], "ranked")
        top = events[-1]["top"]
        self.assertEqual(top[0]["id"], "A")
        self.assertEqual(top[0]["supporters_count"], 3)
        self.assertEqual(top[0]["item_url"], "https://bc/A")

    def test_none_callback_is_noop(self):
        rec = SupporterRecommender()
        with patch.object(sr, "extract_supporters", return_value=["u1"]), \
             patch.object(sr, "extract_item_id", return_value="SEED"), \
             patch.object(
                 rec, "_get_supporter_items_curl_first", side_effect=_fake_fetch(rec)
             ):
            out = rec.get_recommendations(
                wishlist_item_url="https://bc/seed",
                max_recommendations=10,
                min_supporters=1,
                first_page_only=True,
                hydrate_tags=False,
                event_callback=None,
            )
        self.assertTrue(any(r["item_url"] == "https://bc/A" for r in out))


    def test_no_supporters_emits_empty_supporters(self):
        rec = SupporterRecommender()
        events = []
        with patch.object(sr, "extract_supporters", return_value=[]):
            out = rec.get_recommendations(
                wishlist_item_url="https://bc/seed",
                max_recommendations=10,
                min_supporters=1,
                first_page_only=True,
                hydrate_tags=False,
                event_callback=events.append,
            )
        self.assertEqual(out, [])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0], {"type": "supporters", "supporters": [], "total": 0})


if __name__ == "__main__":
    unittest.main()
