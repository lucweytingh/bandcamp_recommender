"""The shared page cache (bandcamp_recommender.http.get_page).

The recommender hits the seed page 3-4 times and every candidate page 2-3
times through different code paths, and consumers (a radio resolving rec
pages) read the same pages moments later. These tests pin the contract:

* a cached URL is not fetched again (any timeout, url-only key)
* failures and non-pages (bot challenge) are never cached
* entries expire (pages embed signed, expiring stream URLs)
* size 0 disables the cache; the LRU evicts the oldest entry past the bound
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from bandcamp_recommender import http


class GetPageCacheTests(unittest.TestCase):
    def setUp(self):
        os.environ["BANDCAMP_CURL_BREAKER_DISABLED"] = "1"
        saved = {k: os.environ.pop(k, None) for k in
                 ("BANDCAMP_PAGE_CACHE_SIZE", "BANDCAMP_PAGE_CACHE_TTL_S", "BANDCAMP_PAGE_CACHE_BYTES")}

        def restore():
            os.environ.pop("BANDCAMP_CURL_BREAKER_DISABLED", None)
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.addCleanup(restore)
        http.clear_page_cache()
        self.addCleanup(http.clear_page_cache)

    def _fake(self, body="<html data-tralbum>ok</html>"):
        calls = []

        def get_html(url, expect=None, timeout=20):
            calls.append((url, timeout))
            return body
        return get_html, calls

    def test_repeat_fetch_hits_cache_across_timeouts(self):
        get_html, calls = self._fake()
        with patch.object(http, "get_html", side_effect=get_html):
            a = http.get_page("https://x.test/p", timeout=5)
            b = http.get_page("https://x.test/p", timeout=20)
        self.assertEqual(a, b)
        self.assertEqual(len(calls), 1)

    def test_failed_fetch_is_not_cached(self):
        state = {"n": 0}

        def flaky(url, expect=None, timeout=20):
            state["n"] += 1
            if state["n"] == 1:
                raise http.RateLimited("429")
            return "<html>ok</html>"

        with patch.object(http, "get_html", side_effect=flaky):
            with self.assertRaises(http.RateLimited):
                http.get_page("https://x.test/q")
            self.assertEqual(http.get_page("https://x.test/q"), "<html>ok</html>")

    def test_challenge_never_cached_and_falls_back_to_browser(self):
        from bandcamp_recommender import browser

        with patch.object(http, "get_html", side_effect=http.Challenged("x")), \
             patch("bandcamp_recommender.recommendations.curl_breaker.record_outcome"), \
             patch.object(browser, "fetch_html", return_value="<html>real</html>") as fb:
            self.assertEqual(http.get_page("https://x.test/c", expect="x"), "<html>real</html>")
        fb.assert_called_once_with("https://x.test/c", "x")

    def test_challenge_without_fallback_raises(self):
        with patch.object(http, "get_html", side_effect=http.Challenged("x")), \
             patch("bandcamp_recommender.recommendations.curl_breaker.record_outcome"):
            with self.assertRaises(http.Challenged):
                http.get_page("https://x.test/c", fallback=False)

    def test_entries_expire(self):
        os.environ["BANDCAMP_PAGE_CACHE_TTL_S"] = "0.05"
        get_html, calls = self._fake()
        import time
        with patch.object(http, "get_html", side_effect=get_html):
            http.get_page("https://x.test/t")
            time.sleep(0.1)
            http.get_page("https://x.test/t")
        self.assertEqual(len(calls), 2)

    def test_size_zero_disables_cache(self):
        os.environ["BANDCAMP_PAGE_CACHE_SIZE"] = "0"
        get_html, calls = self._fake()
        with patch.object(http, "get_html", side_effect=get_html):
            http.get_page("https://x.test/z")
            http.get_page("https://x.test/z")
        self.assertEqual(len(calls), 2)

    def test_lru_eviction(self):
        os.environ["BANDCAMP_PAGE_CACHE_SIZE"] = "2"
        get_html, calls = self._fake()
        with patch.object(http, "get_html", side_effect=get_html):
            for u in ("a", "b", "c", "a"):
                http.get_page(f"https://x.test/{u}")
        # a evicted by c, so the second a is a miss.
        self.assertEqual([c[0][-1] for c in calls], ["a", "b", "c", "a"])

    def test_clear_drops_cached_entries(self):
        get_html, calls = self._fake()
        with patch.object(http, "get_html", side_effect=get_html):
            http.get_page("https://x.test/k")
            http.clear_page_cache()
            http.get_page("https://x.test/k")
        self.assertEqual(len(calls), 2)

    def test_scraper_fetch_page_html_returns_none_on_failure(self):
        from bandcamp_recommender.recommendations import scraper

        with patch.object(http, "get_html", side_effect=http.Timeout("slow")):
            self.assertIsNone(scraper.fetch_page_html("https://x.test/f"))


if __name__ == "__main__":
    unittest.main()
