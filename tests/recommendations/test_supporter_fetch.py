"""Supporter collection reads: plain HTTP first, typed failures, a bounded
browser fallback, and the on-disk cache."""

from __future__ import annotations

import contextlib
import html as htmllib
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from bandcamp_recommender import browser, cache, http
from bandcamp_recommender.recommendations.supporter_recommender import SupporterRecommender


def _fan_page(first_ids, item_count, last_token="tok", fan_id=7):
    item_cache = {
        f"k{i}": {"tralbum_id": i, "item_title": f"t{i}", "band_name": f"b{i}",
                  "item_url": f"https://a.bandcamp.com/album/{i}"}
        for i in first_ids
    }
    blob = {
        "fan_data": {"fan_id": fan_id},
        "collection_data": {
            "sequence": [], "pending_sequence": [f"k{i}" for i in first_ids],
            "last_token": last_token, "item_count": item_count,
        },
        "item_cache": {"collection": item_cache},
    }
    return f'<div id="pagedata" data-blob="{htmllib.escape(json.dumps(blob))}"></div>'


def _api_items(ids):
    return {"items": [{"tralbum_id": i, "item_title": f"t{i}", "band_name": f"b{i}",
                       "item_url": f"https://a.bandcamp.com/album/{i}"} for i in ids]}


class SupporterFetchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = {
            "BANDCAMP_CACHE_PATH": os.path.join(self.tmp.name, "c.sqlite"),
            "BANDCAMP_CURL_BREAKER_STATE": os.path.join(self.tmp.name, "breaker.json"),
        }
        patcher = patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        cache.reset()
        self.addCleanup(cache.reset)
        self.rec = SupporterRecommender()
        self.rec._begin_call(collection_page_size=500)

    def test_full_read_is_cached_and_hit_restores_metadata(self):
        with patch.object(http, "get_html", return_value=_fan_page([1, 2], 5)) as g, \
             patch.object(http, "post_json", return_value=_api_items([3, 4, 5])) as pj:
            ids = self.rec._get_supporter_items_curl_first("alice", "collection_data")
        self.assertEqual(ids, ["1", "2", "3", "4", "5"])
        self.assertEqual(pj.call_args[0][1]["count"], 500)

        fresh = SupporterRecommender()
        fresh._begin_call(collection_page_size=500)
        with patch.object(http, "get_html") as g2:
            ids2 = fresh._get_supporter_items_curl_first("alice", "collection_data")
        g2.assert_not_called()
        self.assertEqual(ids2, ids)
        self.assertEqual(fresh.item_cache["3"]["item_url"], "https://a.bandcamp.com/album/3")

    def test_cache_key_includes_depth(self):
        with patch.object(http, "get_html", return_value=_fan_page([1], 2)), \
             patch.object(http, "post_json", return_value=_api_items([2])):
            self.rec._get_supporter_items_curl_first("alice", "collection_data")
        other = SupporterRecommender()
        other._begin_call(collection_page_size=10000)
        with patch.object(http, "get_html", return_value=_fan_page([1], 2)) as g, \
             patch.object(http, "post_json", return_value=_api_items([2])):
            other._get_supporter_items_curl_first("alice", "collection_data")
        g.assert_called_once()

    def test_truncated_page_counts_as_complete(self):
        # 2000-item fan read with a 500 page: 20 + 500 is the whole request.
        with patch.object(http, "get_html", return_value=_fan_page(list(range(1, 21)), 2000)), \
             patch.object(http, "post_json", return_value=_api_items(list(range(21, 521)))):
            ids = self.rec._get_supporter_items_curl_first("whale", "collection_data")
        self.assertEqual(len(ids), 520)
        self.assertIsNotNone(cache.get_supporter_items("whale", "collection_data", "n500"))

    def test_partial_read_is_returned_but_not_cached(self):
        with patch.object(http, "get_html", return_value=_fan_page([1, 2], 100)), \
             patch.object(http, "post_json", return_value=_api_items([3])):
            ids = self.rec._get_supporter_items_curl_first("bob", "collection_data")
        self.assertEqual(ids, ["1", "2", "3"])
        self.assertIsNone(cache.get_supporter_items("bob", "collection_data", "n500"))

    def test_api_failure_keeps_first_page_uncached(self):
        with patch.object(http, "get_html", return_value=_fan_page([1, 2], 50)), \
             patch.object(http, "post_json", side_effect=http.RateLimited("429")):
            ids = self.rec._get_supporter_items_curl_first("carol", "collection_data")
        self.assertEqual(ids, ["1", "2"])
        self.assertIsNone(cache.get_supporter_items("carol", "collection_data", "n500"))

    def test_private_fan_is_cached_empty(self):
        with patch.object(http, "get_html", return_value="<html>no data here</html>"):
            self.assertEqual(self.rec._get_supporter_items_curl_first("ghost", "collection_data"), [])
        self.assertEqual(cache.get_supporter_items("ghost", "collection_data", "n500"), [])

    def test_404_fan_is_cached_empty(self):
        with patch.object(http, "get_html", side_effect=http.HTTPStatus(404, "u")):
            self.assertEqual(self.rec._get_supporter_items_curl_first("gone", "collection_data"), [])
        self.assertEqual(cache.get_supporter_items("gone", "collection_data", "n500"), [])

    def test_transient_failure_skips_without_browser_or_cache(self):
        with patch.object(http, "get_html", side_effect=http.Timeout("slow")), \
             patch.object(browser, "session") as sess:
            self.assertEqual(self.rec._get_supporter_items_curl_first("dave", "collection_data"), [])
        sess.assert_not_called()
        self.assertIsNone(cache.get_supporter_items("dave", "collection_data", "n500"))

    def test_challenge_uses_browser_within_budget_only(self):
        driver = MagicMock()

        @contextlib.contextmanager
        def fake_session():
            yield driver

        with patch.dict(os.environ, {"BANDCAMP_SELENIUM_FALLBACK_BUDGET": "1"}):
            self.rec._begin_call(collection_page_size=500)
        with patch.object(http, "get_html", side_effect=http.Challenged("x")), \
             patch.object(browser, "session", side_effect=fake_session) as sess, \
             patch.object(browser, "load", return_value=_fan_page([1], 1)):
            first = self.rec._get_supporter_items_curl_first("eve", "collection_data")
            second = self.rec._get_supporter_items_curl_first("frank", "collection_data")
        self.assertEqual(first, ["1"])
        self.assertEqual(second, [])
        self.assertEqual(sess.call_count, 1)


class SupporterApiPathTests(unittest.TestCase):
    """The one-call path: fan_id known from the seed page."""

    def setUp(self):
        SupporterFetchTests.setUp(self)
        self.rec._fan_ids = {"alice": 77}

    def test_one_api_call_with_future_token_no_fan_page(self):
        before = http.snapshot()
        with patch.object(http, "get_html") as g, \
             patch.object(http, "post_json", return_value=_api_items([1, 2, 3])) as pj:
            ids = self.rec._get_supporter_items_curl_first("alice", "collection_data")
        g.assert_not_called()
        self.assertEqual(ids, ["1", "2", "3"])
        payload = pj.call_args[0][1]
        self.assertEqual(payload["fan_id"], 77)
        self.assertEqual(payload["count"], 500)
        self.assertTrue(payload["older_than_token"].endswith("::a::"))
        self.assertEqual(http.diff(before)["supporter_api"], 1)
        self.assertIsNotNone(cache.get_supporter_items("alice", "collection_data", "n500"))

    def test_reply_without_items_falls_back_to_page_path(self):
        with patch.object(http, "post_json", side_effect=[{"error": True}, _api_items([2])]), \
             patch.object(http, "get_html", return_value=_fan_page([1], 2)) as g:
            ids = self.rec._get_supporter_items_curl_first("alice", "collection_data")
        g.assert_called_once()
        self.assertEqual(ids, ["1", "2"])

    def test_empty_api_result_cached_short(self):
        with patch.object(http, "post_json", return_value={"items": []}):
            self.assertEqual(self.rec._get_supporter_items_curl_first("alice", "collection_data"), [])
        import sqlite3, time as _t
        conn = sqlite3.connect(os.environ["BANDCAMP_CACHE_PATH"])
        (expires,) = conn.execute("SELECT expires_ts FROM supporter_items WHERE username='alice'").fetchone()
        self.assertLess(expires - _t.time(), cache.EMPTY_SUPPORTER_TTL_S + 5)


class ExtractSupportersFanIdTests(unittest.TestCase):
    def test_fan_ids_recorded_from_the_same_parse(self):
        from bandcamp_recommender.recommendations import scraper

        blob = {"thumbs": [{"username": "a", "fan_id": 1}, {"username": "b", "fan_id": 2}]}
        page = f'<div id="collectors-data" data-blob="{htmllib.escape(json.dumps(blob))}"></div>'
        with patch.object(scraper, "fetch_page_html", return_value=page):
            self.assertEqual(scraper.extract_supporters("https://x/album/s"), ["a", "b"])
        self.assertEqual(scraper.supporter_fan_ids("https://x/album/s"), {"a": 1, "b": 2})


    def test_empty_blob_means_no_supporters(self):
        from bandcamp_recommender.recommendations import scraper

        # A track nobody bought yet: the blob is there with no thumbs. The
        # page around it links tags (bandcamp.com/discover/...) and the
        # artist's /yum page under a "supported by"-free wrapper; none of
        # them is a fan (2026-10-26: "discover" and "yum" were mined as
        # supporters, and the discover fetch started the fallback browser).
        blob = {"thumbs": []}
        page = (
            '<div class="centerWrapper">supported by nobody yet'
            f'<div id="collectors-data" data-blob="{htmllib.escape(json.dumps(blob))}"></div>'
            '<a href="https://bandcamp.com/discover/electronic?from=tralbum">electronic</a>'
            '<a href="https://artist.bandcamp.com/yum">redeem</a></div>'
        )
        with patch.object(scraper, "fetch_page_html", return_value=page):
            self.assertEqual(scraper.extract_supporters("https://x/track/s"), [])

    def test_fallback_takes_only_fan_pages(self):
        from bandcamp_recommender.recommendations import scraper

        page = (
            '<div>supported by'
            '<a href="https://bandcamp.com/realfan?from=tralbum">f</a>'
            '<a href="https://bandcamp.com/discover/rave">rave</a>'
            '<a href="https://artist.bandcamp.com/yum">redeem</a></div>'
        )
        with patch.object(scraper, "fetch_page_html", return_value=page):
            self.assertEqual(scraper.extract_supporters("https://x/track/s"), ["realfan"])

class CandidateFilterTests(unittest.TestCase):
    def test_filter_applies_before_the_pool_is_cut(self):
        from bandcamp_recommender.recommendations import supporter_recommender as sr

        rec = SupporterRecommender()
        owns = {"u1": ["A", "B", "C"], "u2": ["A", "B", "C"]}

        def fetch(username, data_key, first_page_only=False):
            for iid in owns[username]:
                rec._store_item_metadata(iid, {"item_title": iid, "band_name": iid,
                                               "item_url": f"https://x/{iid}"}, True)
            return owns[username]

        with patch.object(sr, "extract_supporters", return_value=["u1", "u2"]), \
             patch.object(sr, "extract_item_id", return_value=None), \
             patch.object(rec, "_get_supporter_items_curl_first", side_effect=fetch), \
             patch.object(rec, "_hydrate_tags_for_items", return_value=None):
            out = rec.get_recommendations(
                "https://seed/x", max_recommendations=2, min_supporters=1,
                candidate_filter=lambda meta: meta.get("item_url") != "https://x/A",
            )
        self.assertEqual(len(out), 2)
        self.assertNotIn("https://x/A", [o["item_url"] for o in out])


if __name__ == "__main__":
    unittest.main()
