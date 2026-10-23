"""features_for_item: featured-track preview, on-disk caching of item
metadata + audio features, and the warm path costing no request."""

from __future__ import annotations

import html as htmllib
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from bandcamp_recommender import cache, features
from bandcamp_recommender.recommendations import bpm, scraper

URL = "https://a.bandcamp.com/album/x"


def _page(featured=2):
    tralbum = {
        "featured_track_id": featured,
        "trackinfo": [
            {"id": 1, "track_id": 1, "title": "Intro", "title_link": "/track/intro",
             "file": {"mp3-128": "https://t4.bcbits.com/stream/1"}},
            {"id": 2, "track_id": 2, "title": "The Single", "title_link": "/track/single",
             "file": {"mp3-128": "https://t4.bcbits.com/stream/2"}},
        ],
    }
    return (
        f'<script data-tralbum="{htmllib.escape(json.dumps(tralbum))}"></script>'
        '<a class="tag" href="/tag/dub">dub</a><a class="tag" href="/tag/techno">techno</a>'
    )


AUDIO = {"rms_mean": 0.4, "rms_p95": 0.5, "onset_rate": 0.3, "spectral_centroid": 0.2,
         "crest_factor": 0.1, "bpm_norm": 0.4, "bpm_folded_norm": 0.6, "bpm": 120.0}


class FeatureCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = patch.dict(os.environ, {"BANDCAMP_CACHE_PATH": os.path.join(self.tmp.name, "c.sqlite")})
        p.start()
        self.addCleanup(p.stop)
        cache.reset()
        self.addCleanup(cache.reset)

    def test_preview_prefers_featured_track(self):
        with patch.object(bpm, "fetch_page_html", return_value=_page(featured=2)):
            self.assertEqual(bpm.get_preview_for_item(URL)["title"], "The Single")
            self.assertEqual(bpm.get_audio_url_for_item(URL), "https://t4.bcbits.com/stream/2")
        with patch.object(bpm, "fetch_page_html", return_value=_page(featured=99)):
            self.assertEqual(bpm.get_preview_for_item(URL)["title"], "Intro")

    def test_cold_then_warm_costs_no_request_or_decode(self):
        with patch.object(scraper, "fetch_page_html", return_value=_page()) as page, \
             patch.object(features, "extract_audio_features", return_value=dict(AUDIO)) as dec:
            vec, preview = features.features_for_item(URL)
        self.assertEqual(preview["track_id"], 2)
        self.assertEqual(preview["audio_path"], "https://t4.bcbits.com/stream/2")
        self.assertEqual(vec["bpm"], 120.0)
        self.assertEqual(page.call_count, 1)
        dec.assert_called_once()
        self.assertEqual(dec.call_args[0][0], "https://t4.bcbits.com/stream/2")

        with patch.object(scraper, "fetch_page_html") as page2, \
             patch.object(bpm, "fetch_page_html") as page3, \
             patch.object(features, "extract_audio_features") as dec2:
            vec2, preview2 = features.features_for_item(URL)
        page2.assert_not_called()
        page3.assert_not_called()
        dec2.assert_not_called()
        self.assertEqual(vec2["rms_mean"], 0.4)
        self.assertEqual(preview2["track_id"], 2)
        self.assertIsNone(preview2["audio_path"])  # signed URLs aren't cached

    def test_decode_failure_is_not_cached(self):
        with patch.object(scraper, "fetch_page_html", return_value=_page()), \
             patch.object(features, "extract_audio_features", return_value=None):
            vec, _ = features.features_for_item(URL)
        self.assertIsNone(vec["rms_mean"])
        with patch.object(bpm, "fetch_page_html", return_value=_page()), \
             patch.object(features, "extract_audio_features", return_value=dict(AUDIO)) as dec:
            vec2, _ = features.features_for_item(URL)
        dec.assert_called_once()  # retried, not pinned to "unknown"
        self.assertEqual(vec2["rms_mean"], 0.4)

    def test_unreadable_page_returns_tag_only_vector(self):
        with patch.object(scraper, "fetch_page_html", return_value=None):
            vec, preview = features.features_for_item(URL)
        self.assertIsNone(preview)
        self.assertIsNone(vec["rms_mean"])


if __name__ == "__main__":
    unittest.main()
