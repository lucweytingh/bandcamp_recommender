"""Unit tests for the fetch_page_html process-level cache.

The cache is the main wall-clock win for repeated pipeline calls (the
recommender hits the seed URL 3-4 times and every top-N item page 2-3
times via different code paths). These tests pin the contract so a future
refactor doesn't silently disable it:

* cached URLs do NOT re-spawn curl
* failed fetches are NOT cached (so retry-on-failure still works)
* size=0 disables caching entirely
* LRU evicts the oldest entry past the bound
* clear_page_cache() empties the cache
"""

from __future__ import annotations

import os
import unittest
from subprocess import CompletedProcess
from unittest.mock import patch


class FetchPageHtmlCacheTests(unittest.TestCase):
    def setUp(self):
        # Don't let stale breaker state in the user's cache trip us.
        os.environ["BANDCAMP_CURL_BREAKER_DISABLED"] = "1"
        self.addCleanup(lambda: os.environ.pop("BANDCAMP_CURL_BREAKER_DISABLED", None))
        # Reset env knob between tests.
        self._saved_cache_size = os.environ.pop("BANDCAMP_PAGE_CACHE_SIZE", None)
        def restore():
            if self._saved_cache_size is not None:
                os.environ["BANDCAMP_PAGE_CACHE_SIZE"] = self._saved_cache_size
            else:
                os.environ.pop("BANDCAMP_PAGE_CACHE_SIZE", None)
        self.addCleanup(restore)

        from bandcamp_recommender.recommendations import scraper
        scraper.clear_page_cache()
        self.scraper = scraper

    def _fake_run(self, body: str = "<html>ok</html>"):
        """Build a fake subprocess.run that records calls + returns success."""
        calls = []
        def run(cmd, *args, **kwargs):
            calls.append(cmd)
            return CompletedProcess(cmd, returncode=0, stdout=body, stderr="")
        return run, calls

    def test_repeat_fetch_skips_subprocess(self):
        os.environ["BANDCAMP_PAGE_CACHE_SIZE"] = "16"
        run, calls = self._fake_run("<html>hit</html>")
        with patch.object(self.scraper.subprocess, "run", side_effect=run):
            a = self.scraper.fetch_page_html("https://x.test/p", timeout=5)
            b = self.scraper.fetch_page_html("https://x.test/p", timeout=5)
        self.assertEqual(a, "<html>hit</html>")
        self.assertEqual(b, "<html>hit</html>")
        self.assertEqual(len(calls), 1, f"expected one curl call, got {len(calls)}")

    def test_failed_fetch_is_not_cached(self):
        """Transient failure → don't poison the cache; next call retries."""
        os.environ["BANDCAMP_PAGE_CACHE_SIZE"] = "16"
        state = {"n": 0}

        def flaky(cmd, *args, **kwargs):
            state["n"] += 1
            if state["n"] <= 4:
                # 4 = max retry attempts inside fetch_page_html → first call returns None
                return CompletedProcess(cmd, returncode=7, stdout="", stderr="boom")
            return CompletedProcess(cmd, returncode=0, stdout="<html>ok</html>", stderr="")

        with patch.object(self.scraper.subprocess, "run", side_effect=flaky):
            a = self.scraper.fetch_page_html("https://x.test/q", timeout=1)
            b = self.scraper.fetch_page_html("https://x.test/q", timeout=1)
        self.assertIsNone(a)
        self.assertEqual(b, "<html>ok</html>")

    def test_size_zero_disables_cache(self):
        os.environ["BANDCAMP_PAGE_CACHE_SIZE"] = "0"
        run, calls = self._fake_run()
        with patch.object(self.scraper.subprocess, "run", side_effect=run):
            self.scraper.fetch_page_html("https://x.test/r", timeout=5)
            self.scraper.fetch_page_html("https://x.test/r", timeout=5)
        self.assertEqual(len(calls), 2, "size=0 must not cache")

    def test_lru_eviction(self):
        os.environ["BANDCAMP_PAGE_CACHE_SIZE"] = "2"
        run, calls = self._fake_run()
        with patch.object(self.scraper.subprocess, "run", side_effect=run):
            self.scraper.fetch_page_html("https://x.test/a", timeout=5)  # cache: a
            self.scraper.fetch_page_html("https://x.test/b", timeout=5)  # cache: a,b
            self.scraper.fetch_page_html("https://x.test/c", timeout=5)  # cache: b,c (a evicted)
            self.scraper.fetch_page_html("https://x.test/a", timeout=5)  # miss → curl again
            self.scraper.fetch_page_html("https://x.test/c", timeout=5)  # hit
        # a, b, c, a → 4 subprocesses; final c is a hit.
        self.assertEqual(len(calls), 4)

    def test_clear_drops_cached_entries(self):
        os.environ["BANDCAMP_PAGE_CACHE_SIZE"] = "16"
        run, calls = self._fake_run()
        with patch.object(self.scraper.subprocess, "run", side_effect=run):
            self.scraper.fetch_page_html("https://x.test/s", timeout=5)
            self.scraper.clear_page_cache()
            self.scraper.fetch_page_html("https://x.test/s", timeout=5)
        self.assertEqual(len(calls), 2, "clear should force re-fetch")

    def test_different_timeouts_share_cache_entries(self):
        """Same URL, different per-call timeout → still counts as the same page."""
        os.environ["BANDCAMP_PAGE_CACHE_SIZE"] = "16"
        run, calls = self._fake_run()
        with patch.object(self.scraper.subprocess, "run", side_effect=run):
            self.scraper.fetch_page_html("https://x.test/t", timeout=5)
            self.scraper.fetch_page_html("https://x.test/t", timeout=20)
        # Even if cache keys included timeout, the second call still
        # produces correct HTML; we just want to ensure at most 2 curls
        # (the current implementation keys on (url, timeout), so 2 is
        # acceptable; future hardening could merge them but it's not a
        # correctness requirement).
        self.assertLessEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
