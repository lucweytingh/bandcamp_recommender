"""Never recommend the seed itself — URL-normalized backstop.

get_recommendations removes the seed by its tralbum id
(extract_item_id → purchase_counter.pop). But extract_item_id can fail
(curl 403, page-structure shift) and return None, in which case the
id-based removal is a no-op and the seed leaks into its own
recommendations. These tests pin a second, URL-based backstop: even when
extract_item_id returns None, no returned rec may have an item_url that
normalizes to the seed url. The same backstop applies to
get_random_items. Network is fully stubbed (no curl, no Selenium).
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from bandcamp_recommender.recommendations import supporter_recommender as sr
from bandcamp_recommender.recommendations.supporter_recommender import (
    SupporterRecommender,
    _normalize_item_url,
)


SEED = "https://artist.bandcamp.com/album/Seed-Album"


class ExcludeSeedTests(unittest.TestCase):
    def _fetch_including_seed(self, rec):
        """Stub _get_supporter_items_curl_first so each supporter's
        collection includes an item whose stored item_url equals the seed
        url (a different spelling of it), plus two other items."""

        # The seed item, stored under a different-looking tralbum id and a
        # trailing-slash/uppercase-query spelling that still normalizes to SEED.
        seed_spelling = SEED + "/?from=fanpub#x"

        owns = {
            "alice": [
                ("seed_item", seed_spelling),
                ("a1", "https://artist.bandcamp.com/album/other-one"),
            ],
            "bob": [
                ("seed_item", seed_spelling),
                ("a2", "https://artist.bandcamp.com/album/other-two"),
            ],
        }

        def fetch(username, data_key, first_page_only=False):
            ids = []
            for iid, url in owns.get(username, []):
                rec._store_item_metadata(
                    iid,
                    {
                        "item_title": f"title_{iid}",
                        "band_name": "band",
                        "item_url": url,
                        "tralbum_id": iid,
                    },
                    True,
                )
                ids.append(iid)
            return ids

        return fetch

    def test_seed_excluded_when_extract_item_id_is_none(self):
        rec = SupporterRecommender()
        seed_key = _normalize_item_url(SEED)
        with patch.object(sr, "extract_supporters", return_value=["alice", "bob"]), \
             patch.object(sr, "extract_item_id", return_value=None), \
             patch.object(
                 rec, "_get_supporter_items_curl_first",
                 side_effect=self._fetch_including_seed(rec),
             ), \
             patch.object(rec, "_hydrate_tags_for_items"):
            recs = rec.get_recommendations(
                wishlist_item_url=SEED,
                max_recommendations=10,
                min_supporters=1,
                first_page_only=True,
                hydrate_tags=False,
            )
        # The seed item is owned by both supporters, so without the URL
        # backstop it would rank #1. Assert it is gone.
        self.assertTrue(recs, "expected some non-seed recommendations")
        for r in recs:
            self.assertNotEqual(
                _normalize_item_url(r.get("item_url", "")),
                seed_key,
                f"seed leaked into recommendations: {r.get('item_url')!r}",
            )
        # The two genuine items survive.
        self.assertEqual(
            {_normalize_item_url(r["item_url"]) for r in recs},
            {
                "https://artist.bandcamp.com/album/other-one",
                "https://artist.bandcamp.com/album/other-two",
            },
        )

    def _assert_no_seed_in_supporter_done(self, events):
        """Across every supporter_done event, the seed item (stored under id
        ``seed_item`` with an item_url that normalizes to SEED) is never
        emitted, but the genuine items DO appear. supporter_done items carry
        no item_url, so the seed shows up only as its id ``seed_item`` — its
        absence proves both the id-check and the url-check paths worked."""
        emitted_ids = set()
        saw_event = False
        for ev in events:
            if ev.get("type") != "supporter_done":
                continue
            saw_event = True
            for it in ev.get("items", []):
                emitted_ids.add(it["id"])
        self.assertTrue(saw_event, "expected at least one supporter_done event")
        # The seed item id must never be emitted into the cloud.
        self.assertNotIn(
            "seed_item",
            emitted_ids,
            "seed item leaked into a supporter_done event",
        )
        # The two genuine items must be present across the cloud.
        self.assertIn("a1", emitted_ids)
        self.assertIn("a2", emitted_ids)

    def test_seed_excluded_from_supporter_done_via_id(self):
        """extract_item_id returns the seed's real tralbum id — the seed must
        be skipped from the supporter_done cloud by the id check."""
        rec = SupporterRecommender()
        events: list[dict] = []
        with patch.object(sr, "extract_supporters", return_value=["alice", "bob"]), \
             patch.object(sr, "extract_item_id", return_value="seed_item"), \
             patch.object(
                 rec, "_get_supporter_items_curl_first",
                 side_effect=self._fetch_including_seed(rec),
             ), \
             patch.object(rec, "_hydrate_tags_for_items"):
            rec.get_recommendations(
                wishlist_item_url=SEED,
                max_recommendations=10,
                min_supporters=1,
                first_page_only=True,
                hydrate_tags=False,
                event_callback=events.append,
            )
        self._assert_no_seed_in_supporter_done(events)

    def test_seed_excluded_from_supporter_done_via_url(self):
        """extract_item_id returns None — the seed must still be skipped from
        the supporter_done cloud by the url-normalization check."""
        rec = SupporterRecommender()
        events: list[dict] = []
        with patch.object(sr, "extract_supporters", return_value=["alice", "bob"]), \
             patch.object(sr, "extract_item_id", return_value=None), \
             patch.object(
                 rec, "_get_supporter_items_curl_first",
                 side_effect=self._fetch_including_seed(rec),
             ), \
             patch.object(rec, "_hydrate_tags_for_items"):
            rec.get_recommendations(
                wishlist_item_url=SEED,
                max_recommendations=10,
                min_supporters=1,
                first_page_only=True,
                hydrate_tags=False,
                event_callback=events.append,
            )
        self._assert_no_seed_in_supporter_done(events)

    def test_random_items_excludes_seed_from_supporter_done_via_url(self):
        """get_random_items must also keep the seed out of the
        supporter_done cloud (url path, extract_item_id → None)."""
        rec = SupporterRecommender()
        events: list[dict] = []
        with patch.object(sr, "extract_supporters", return_value=["alice", "bob"]), \
             patch.object(sr, "extract_item_id", return_value=None), \
             patch.object(
                 rec, "_get_supporter_items_curl_first",
                 side_effect=self._fetch_including_seed(rec),
             ):
            rec.get_random_items(
                item_url=SEED,
                num_items=10,
                num_supporters=20,
                event_callback=events.append,
            )
        self._assert_no_seed_in_supporter_done(events)

    def test_random_items_excludes_seed_when_extract_item_id_is_none(self):
        rec = SupporterRecommender()
        seed_key = _normalize_item_url(SEED)
        with patch.object(sr, "extract_supporters", return_value=["alice", "bob"]), \
             patch.object(sr, "extract_item_id", return_value=None), \
             patch.object(
                 rec, "_get_supporter_items_curl_first",
                 side_effect=self._fetch_including_seed(rec),
             ):
            results = rec.get_random_items(
                item_url=SEED,
                num_items=10,
                num_supporters=20,
            )
        self.assertTrue(results, "expected some non-seed random items")
        for r in results:
            self.assertNotEqual(
                _normalize_item_url(r.get("item_url", "")),
                seed_key,
                f"seed leaked into random items: {r.get('item_url')!r}",
            )


if __name__ == "__main__":
    unittest.main()
