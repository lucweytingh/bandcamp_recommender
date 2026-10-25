"""A fan's own lists: page + API reads, timestamps, hidden/empty lists, the
challenge fallback. Offline: fixtures follow a real fan page and real
fancollection replies (2026-10, trimmed)."""

from __future__ import annotations

import contextlib
import html as htmllib
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from bandcamp_recommender import browser, fans, http

_WISHLIST_URL = fans.API_URLS["wishlist"]
_COLLECTION_URL = fans.API_URLS["collection"]


def _fan_page(fan_id=5068627, username="tabletop13", wishlist_count=2, collection_count=3,
              wishlist_hidden=False, collection_hidden_items=0):
    wishlist_data = {"last_token": None, "item_count": wishlist_count, "batch_size": 20,
                     "sequence": [], "pending_sequence": []}
    if wishlist_hidden:
        wishlist_data.update(hidden=True, item_count=0)
    blob = {
        "fan_data": {"trackpipe_url": f"https://bandcamp.com/{username}", "username": username,
                     "name": username, "fan_id": fan_id},
        "collection_data": {"last_token": "1739209612:2110012686:t::", "item_count": collection_count,
                            "batch_size": 20, "hidden_items_count": collection_hidden_items,
                            "sequence": [],
                            "pending_sequence": []},
        "wishlist_data": wishlist_data,
        "item_cache": {"collection": {}, "wishlist": {}},
    }
    return f'<html><div id="pagedata" data-blob="{htmllib.escape(json.dumps(blob))}"></div></html>'


def _item(tralbum_id, added, token_ts, item_type="album", title=None, band="Sub Basics",
          purchased=None):
    """Real shape: the token's epoch is ``purchased`` on collection items and
    ``added`` on wishlist items (``purchased`` None)."""
    tralbum_type = "a" if item_type == "album" else "t"
    return {
        "fan_id": 5068627, "item_id": tralbum_id, "item_type": item_type, "band_id": 2044363476,
        "added": added, "updated": purchased or added, "purchased": purchased,
        "tralbum_id": tralbum_id, "tralbum_type": tralbum_type, "featured_track": 3184321110,
        "item_title": title or f"Release {tralbum_id}",
        "item_url": f"https://templeofsound.bandcamp.com/{item_type}/r{tralbum_id}",
        "item_art_url": "https://f4.bcbits.com/img/a0257506048_9.jpg",
        "band_name": band, "band_url": "https://templeofsound.bandcamp.com",
        "token": f"{token_ts}:{tralbum_id}:{tralbum_type}::",
    }


def _reply(items, more=False, last_token=None):
    return {"items": items, "more_available": more,
            "last_token": last_token or (items[-1]["token"] if items else None),
            "tracklists": {}, "redownload_urls": {}, "item_lookup": {}, "purchase_infos": {},
            "collectors": {}}


_COLLECTION = [
    _item(1168869557, "16 Sep 2026 22:52:52 GMT", 1789599172, title="Maneuver 01",
          purchased="16 Sep 2026 22:52:52 GMT"),
    # Wishlisted in 2023, bought in 2026: entered the collection in 2026.
    _item(3337769688, "08 May 2023 14:00:56 GMT", 1789067195, band="javier salazar",
          purchased="10 Sep 2026 19:06:35 GMT"),
    _item(1742731813, "04 Apr 2026 13:22:48 GMT", 1775308968, item_type="track", band="Tobias. Doltz.",
          purchased="04 Apr 2026 13:22:48 GMT"),
]
_WISHLIST = [
    _item(11, "20 Oct 2026 08:00:00 GMT", 1792483200),
    _item(12, "01 Jan 2026 00:00:00 GMT", 1767225600),
]


def _api(replies):
    """A post_json stand-in answering from ``{url: reply or [replies]}``."""
    calls = []
    queues = {url: (list(r) if isinstance(r, list) else [r]) for url, r in replies.items()}

    def post(url, payload, timeout=60, referer=None):
        calls.append((url, dict(payload)))
        reply = queues[url].pop(0) if len(queues[url]) > 1 else queues[url][0]
        if isinstance(reply, Exception):
            raise reply
        return reply

    return post, calls


class FanLibraryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = patch.dict(os.environ, {
            "BANDCAMP_CURL_BREAKER_STATE": os.path.join(self.tmp.name, "breaker.json"),
        })
        patcher.start()
        self.addCleanup(patcher.stop)
        http.clear_page_cache()
        self.addCleanup(http.clear_page_cache)

    def _read(self, page=None, replies=None, **kwargs):
        post, calls = _api(replies or {_WISHLIST_URL: _reply(_WISHLIST),
                                       _COLLECTION_URL: _reply(_COLLECTION)})
        with patch.object(http, "get_html", return_value=page or _fan_page()) as get, \
             patch.object(http, "post_json", side_effect=post):
            lib = fans.fan_library("tabletop13", **kwargs)
        return lib, get, calls

    def test_both_lists_in_full_newest_first_with_added_ts(self):
        lib, get, calls = self._read()
        self.assertEqual(lib["fan_id"], 5068627)
        self.assertEqual(lib["username"], "tabletop13")
        self.assertEqual(lib["hidden"], [])
        self.assertEqual([it["tralbum_id"] for it in lib["collection"]],
                         [1168869557, 3337769688, 1742731813])
        self.assertEqual([it["tralbum_id"] for it in lib["wishlist"]], [11, 12])
        self.assertEqual(lib["collection"][0], {
            "tralbum_id": 1168869557,
            "item_url": "https://templeofsound.bandcamp.com/album/r1168869557",
            "item_title": "Maneuver 01",
            "band_name": "Sub Basics",
            "item_type": "album",
            "added_ts": 1789599172.0,
        })
        self.assertEqual(lib["collection"][2]["item_type"], "track")
        self.assertEqual([it["added_ts"] for it in lib["collection"]],
                         [1789599172.0, 1789067195.0, 1775308968.0])
        self.assertEqual(get.call_args[0][0], "https://bandcamp.com/tabletop13/wishlist")
        self.assertEqual([url for url, _ in calls], [_WISHLIST_URL, _COLLECTION_URL])
        for _, payload in calls:
            self.assertEqual(payload["fan_id"], 5068627)
            self.assertEqual(payload["count"], fans.PAGE_SIZE)
            self.assertTrue(payload["older_than_token"].endswith("::a::"))

    def test_lists_selects_what_is_read(self):
        lib, _, calls = self._read(lists=("collection",))
        self.assertNotIn("wishlist", lib)
        self.assertEqual(len(lib["collection"]), 3)
        self.assertEqual([url for url, _ in calls], [_COLLECTION_URL])

    def test_unknown_list_name_is_a_value_error(self):
        with self.assertRaises(ValueError):
            fans.fan_library("tabletop13", lists=("purchases",))
        with self.assertRaises(ValueError):
            fans.fan_library("tabletop13", lists=())

    def test_hidden_wishlist_is_reported_and_costs_no_api_call(self):
        # Real: a private wishlist shows hidden=True on the page and the API
        # answers {"error": true}.
        lib, _, calls = self._read(
            page=_fan_page(wishlist_hidden=True),
            replies={_WISHLIST_URL: {"error": True}, _COLLECTION_URL: _reply(_COLLECTION)},
        )
        self.assertEqual(lib["wishlist"], [])
        self.assertEqual(lib["hidden"], ["wishlist"])
        self.assertEqual(len(lib["collection"]), 3)
        self.assertEqual([url for url, _ in calls], [_COLLECTION_URL])

    def test_empty_list_is_empty(self):
        lib, _, _ = self._read(
            page=_fan_page(wishlist_count=0),
            replies={_WISHLIST_URL: _reply([]), _COLLECTION_URL: _reply(_COLLECTION)},
        )
        self.assertEqual(lib["wishlist"], [])
        self.assertEqual(lib["hidden"], [])

    def test_error_reply_raises_instead_of_an_empty_list(self):
        before = http.snapshot()
        with self.assertRaises(http.Unexpected):
            self._read(replies={_WISHLIST_URL: {"error": True}, _COLLECTION_URL: _reply(_COLLECTION)})
        self.assertEqual(http.diff(before)["errors"], 1)

    def test_no_items_while_the_page_shows_some_raises(self):
        with self.assertRaises(http.Unexpected):
            self._read(replies={_WISHLIST_URL: _reply([]), _COLLECTION_URL: _reply(_COLLECTION)})

    def test_collection_of_only_hidden_items_is_empty(self):
        lib, _, _ = self._read(
            page=_fan_page(collection_count=3, collection_hidden_items=3), lists=("collection",),
            replies={_COLLECTION_URL: _reply([])},
        )
        self.assertEqual(lib["collection"], [])

    def test_unknown_fan(self):
        with patch.object(http, "get_html", side_effect=http.HTTPStatus(404, "u")), \
             patch.object(http, "post_json") as pj:
            with self.assertRaises(fans.FanNotFound):
                fans.fan_library("nobody-here")
        pj.assert_not_called()

    def test_page_without_fan_id_raises(self):
        with patch.object(http, "get_html", return_value="<html>pagedata, but no blob</html>"), \
             patch.object(http, "post_json") as pj:
            with self.assertRaises(http.Unexpected):
                fans.fan_library("tabletop13")
        pj.assert_not_called()

    def test_transient_failure_propagates_without_browser(self):
        with patch.object(http, "get_html", side_effect=http.Timeout("slow")), \
             patch.object(browser, "session") as sess:
            with self.assertRaises(http.Timeout):
                fans.fan_library("tabletop13")
        sess.assert_not_called()

    def test_follows_more_available_and_dedupes_the_seam(self):
        first = _reply(_COLLECTION[:2], more=True)
        second = _reply(_COLLECTION[1:], more=False)
        lib, _, calls = self._read(
            lists=("collection",), replies={_COLLECTION_URL: [first, second]},
        )
        self.assertEqual([it["tralbum_id"] for it in lib["collection"]],
                         [1168869557, 3337769688, 1742731813])
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1][1]["older_than_token"], _COLLECTION[1]["token"])

    def test_page_challenge_reads_everything_in_one_browser_session(self):
        driver = MagicMock()
        sessions = []

        @contextlib.contextmanager
        def fake_session():
            sessions.append(driver)
            yield driver

        def browser_post(drv, url, payload, timeout=60):
            self.assertIs(drv, driver)
            return _reply(_WISHLIST) if url == _WISHLIST_URL else _reply(_COLLECTION)

        with patch.object(http, "get_html", side_effect=http.Challenged("x")), \
             patch.object(http, "post_json") as pj, \
             patch.object(browser, "session", side_effect=fake_session), \
             patch.object(browser, "load", return_value=_fan_page()) as load, \
             patch.object(browser, "post_json", side_effect=browser_post) as bpj:
            lib = fans.fan_library("tabletop13")
        pj.assert_not_called()
        self.assertEqual(len(sessions), 1)
        load.assert_called_once_with(driver, "https://bandcamp.com/tabletop13/wishlist", "pagedata")
        self.assertEqual(bpj.call_count, 2)
        self.assertEqual(len(lib["wishlist"]), 2)
        self.assertEqual(len(lib["collection"]), 3)

    def test_api_challenge_finishes_in_the_browser_keeping_lists_already_read(self):
        driver = MagicMock()

        @contextlib.contextmanager
        def fake_session():
            yield driver

        post, calls = _api({_WISHLIST_URL: _reply(_WISHLIST),
                            _COLLECTION_URL: http.Challenged("api")})
        with patch.object(http, "get_html", return_value=_fan_page()), \
             patch.object(http, "post_json", side_effect=post), \
             patch.object(browser, "session", side_effect=fake_session), \
             patch.object(browser, "load", return_value=_fan_page()) as load, \
             patch.object(browser, "post_json", return_value=_reply(_COLLECTION)) as bpj:
            lib = fans.fan_library("tabletop13")
        load.assert_called_once()
        self.assertEqual([c[0][1] for c in bpj.call_args_list], [_COLLECTION_URL])
        self.assertEqual(len(lib["wishlist"]), 2)
        self.assertEqual(len(lib["collection"]), 3)

    def test_challenged_browser_raises(self):
        @contextlib.contextmanager
        def fake_session():
            yield MagicMock()

        with patch.object(http, "get_html", side_effect=http.Challenged("x")), \
             patch.object(browser, "session", side_effect=fake_session), \
             patch.object(browser, "load", side_effect=http.Challenged("browser")):
            with self.assertRaises(http.Challenged):
                fans.fan_library("tabletop13")


class AddedTsTests(unittest.TestCase):
    def test_token_is_the_list_entry_time(self):
        bought_later = {"added": "08 May 2023 14:00:56 GMT", "purchased": "10 Sep 2026 19:06:35 GMT",
                        "token": "1789067195:3337769688:a::"}
        self.assertEqual(fans.parse_added_ts(bought_later), 1789067195.0)
        self.assertEqual(fans.parse_added_ts({"token": "1789599172:1168869557:a:3:"}), 1789599172.0)

    def test_dates_when_the_token_does_not_parse(self):
        self.assertEqual(
            fans.parse_added_ts({"purchased": "16 Sep 2026 22:52:52 GMT", "added": "08 May 2023 14:00:56 GMT",
                                 "token": "x"}),
            1789599172.0,
        )
        self.assertEqual(fans.parse_added_ts({"added": "16 Sep 2026 22:52:52 GMT", "purchased": None}),
                         1789599172.0)
        self.assertEqual(fans.parse_added_ts({"added": "16 Sep 2026 22:52:52"}), 1789599172.0)

    def test_nothing_parseable(self):
        self.assertIsNone(fans.parse_added_ts({}))
        self.assertIsNone(fans.parse_added_ts({"added": "soon", "token": "::a::"}))
        self.assertIsNone(fans.parse_added_ts({"updated": "16 Sep 2026 22:52:52 GMT"}))


class ParsePagedataTests(unittest.TestCase):
    def test_absent_or_broken(self):
        self.assertIsNone(fans.parse_pagedata("<html></html>"))
        self.assertIsNone(fans.parse_pagedata('<div id="pagedata" data-blob="{nope"></div>'))
        self.assertIsNone(fans.parse_pagedata('<div id="pagedata" data-blob="[1]"></div>'))

    def test_decodes_the_blob(self):
        self.assertEqual(fans.parse_pagedata(_fan_page())["fan_data"]["fan_id"], 5068627)


if __name__ == "__main__":
    unittest.main()
