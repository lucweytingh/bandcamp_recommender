"""Deterministic micro-benchmark for the scraper layer.

Uses the same local HTTP fixture as test_reliability.py so it never touches
Bandcamp. The shape of the benchmark mirrors what a real recommendation run
does to the scraper layer:

* one fetch of the seed URL (extract_supporters)
* one fetch of the seed URL (extract_item_id)               -- duplicate
* one fetch of the seed URL (extract_tags)                  -- duplicate
* one fetch per top-N item for tag hydration
* one fetch per top-N item for audio-URL hydration          -- duplicate

That's 2 + 3 + 2*N curl subprocesses today. With the page cache, every
unique URL only hits the network once.

We don't ``assert`` a wall-clock threshold here -- this is a benchmark, not
a regression gate. It prints two numbers so the developer / CI log shows
the speedup. Run it with ``pytest -s tests/test_page_cache_bench.py``.
"""

from __future__ import annotations

import os
import time
import unittest
from typing import List

from tests.test_reliability import local_http_server


N_ITEMS = 20  # representative top-N for get_recommendations
SIM_CANDIDATES = 30  # candidate pool for get_similar_recommendations


def _run_scraper_pipeline(base_url: str) -> None:
    """Mimic the redundant fetch pattern in supporter_recommender."""
    from bandcamp_recommender.recommendations import scraper

    # Seed URL fetched three different ways (mirrors supporter+id+tags).
    seed = f"{base_url}/instant"
    scraper.fetch_page_html(seed, timeout=5)
    scraper.fetch_page_html(seed, timeout=5)
    scraper.fetch_page_html(seed, timeout=5)

    # Per-item: one fetch for tags, one for audio_url.
    for i in range(N_ITEMS):
        item = f"{base_url}/instant?item={i}"
        scraper.fetch_page_html(item, timeout=5)
        scraper.fetch_page_html(item, timeout=5)


def _run_similar_recommendations_pipeline(base_url: str) -> None:
    """Mirror the fetch pattern of get_similar_recommendations.

    seed URL gets hit by extract_supporters + extract_item_id + extract_tags
    + get_audio_url_for_item → 4 fetches today, 1 with the cache.

    Each of the 30 candidates gets tag-hydrated, audio-url-hydrated, and
    then re-resolved inside attach_audio_features → 3 fetches per candidate
    today, 1 with the cache.

    Supporter pages are unique and don't benefit; we model only the
    de-duplicatable surface to keep the benchmark sharp.
    """
    from bandcamp_recommender.recommendations import scraper

    seed = f"{base_url}/instant"
    # Seed: 4 redundant paths.
    for _ in range(4):
        scraper.fetch_page_html(seed, timeout=5)

    # Candidates: tag + audio_url + attach_audio_features re-resolve.
    for i in range(SIM_CANDIDATES):
        item = f"{base_url}/instant?cand={i}"
        scraper.fetch_page_html(item, timeout=5)
        scraper.fetch_page_html(item, timeout=5)
        scraper.fetch_page_html(item, timeout=5)


class PageCacheBenchmark(unittest.TestCase):
    def setUp(self):
        os.environ["BANDCAMP_CURL_BREAKER_DISABLED"] = "1"
        self.addCleanup(lambda: os.environ.pop("BANDCAMP_CURL_BREAKER_DISABLED", None))

    def _run(self, base: str, runner) -> tuple[float, float]:
        from bandcamp_recommender.recommendations import scraper

        os.environ["BANDCAMP_PAGE_CACHE_SIZE"] = "0"
        scraper.clear_page_cache()
        t0 = time.monotonic()
        runner(base)
        without_cache = time.monotonic() - t0

        os.environ["BANDCAMP_PAGE_CACHE_SIZE"] = "256"
        scraper.clear_page_cache()
        t0 = time.monotonic()
        runner(base)
        with_cache = time.monotonic() - t0

        return without_cache, with_cache

    def test_bench_with_and_without_cache(self):
        with local_http_server() as base:
            no_cache, cached = self._run(base, _run_scraper_pipeline)
            no_cache_sim, cached_sim = self._run(
                base, _run_similar_recommendations_pipeline
            )

        print(
            f"\nget_recommendations          no-cache={no_cache*1000:6.0f}ms  "
            f"cached={cached*1000:6.0f}ms  speedup={no_cache/cached:5.2f}x"
        )
        print(
            f"get_similar_recommendations  no-cache={no_cache_sim*1000:6.0f}ms  "
            f"cached={cached_sim*1000:6.0f}ms  "
            f"speedup={no_cache_sim/cached_sim:5.2f}x"
        )

        # Sanity gate: the cache must never be slower than no-cache.
        self.assertLess(cached, no_cache * 1.2)
        self.assertLess(cached_sim, no_cache_sim * 1.2)


if __name__ == "__main__":
    unittest.main()
