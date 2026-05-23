"""Unit tests for the BPM re-rank flow in SupporterRecommender."""

import os
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import patch

from bandcamp_recommender.recommendations import bpm as bpm_module
from bandcamp_recommender.recommendations import (
    supporter_recommender as sr_module,
)
from bandcamp_recommender.recommendations.bpm import (
    octave_tolerant_bpm_distance,
)
from bandcamp_recommender.recommendations.supporter_recommender import (
    SupporterRecommender,
    _bpm_rerank_score,
    _resolve_bpm_rerank_alpha,
)


# ---------------------------------------------------------------------------
# Pure math
# ---------------------------------------------------------------------------


class OctaveTolerantBpmDistanceTests(unittest.TestCase):
    def test_exact_match_is_zero(self):
        self.assertEqual(octave_tolerant_bpm_distance(128.0, 128.0), 0.0)

    def test_small_delta(self):
        self.assertAlmostEqual(octave_tolerant_bpm_distance(128.0, 130.0), 2.0)

    def test_double_time_is_near(self):
        # Seed 128, candidate 64: half-distance wins → 0
        self.assertAlmostEqual(octave_tolerant_bpm_distance(128.0, 64.0), 0.0)

    def test_half_time_is_near(self):
        # Seed 70, candidate 140: double-distance wins → 0
        self.assertAlmostEqual(octave_tolerant_bpm_distance(70.0, 140.0), 0.0)

    def test_close_to_octave_tolerated(self):
        # Seed 128, candidate 65 → |128 - 2*65|=2 wins over |128-65|=63.
        self.assertAlmostEqual(octave_tolerant_bpm_distance(128.0, 65.0), 2.0)


class BpmRerankScoreTests(unittest.TestCase):
    def test_none_distance_no_penalty(self):
        self.assertEqual(_bpm_rerank_score(10, None, 0.05), 10.0)

    def test_zero_distance_no_penalty(self):
        self.assertEqual(_bpm_rerank_score(10, 0.0, 0.05), 10.0)

    def test_nonzero_distance_subtracts_penalty(self):
        # 10 - 0.5 * 4 = 8
        self.assertEqual(_bpm_rerank_score(10, 4.0, 0.5), 8.0)


class ResolveBpmRerankAlphaTests(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.pop("BANDCAMP_BPM_RERANK_ALPHA", None)

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("BANDCAMP_BPM_RERANK_ALPHA", None)
        else:
            os.environ["BANDCAMP_BPM_RERANK_ALPHA"] = self._saved

    def test_default_when_unset(self):
        self.assertEqual(_resolve_bpm_rerank_alpha(), 0.05)

    def test_env_override(self):
        os.environ["BANDCAMP_BPM_RERANK_ALPHA"] = "0.2"
        self.assertEqual(_resolve_bpm_rerank_alpha(), 0.2)

    def test_invalid_env_falls_back_to_default(self):
        os.environ["BANDCAMP_BPM_RERANK_ALPHA"] = "not-a-float"
        self.assertEqual(_resolve_bpm_rerank_alpha(), 0.05)


# ---------------------------------------------------------------------------
# Seed BPM cache
# ---------------------------------------------------------------------------


class GetSeedBpmCacheTests(unittest.TestCase):
    def setUp(self):
        bpm_module.clear_seed_bpm_cache()

    def test_second_call_short_circuits(self):
        audio_calls: List[str] = []
        detect_calls: List[str] = []

        def fake_audio(url, track_index=0):
            audio_calls.append(url)
            return f"https://audio.example/{url}.mp3"

        def fake_detect(audio_url, method="auto", duration=60.0, use_cache=True):
            detect_calls.append(audio_url)
            return {"bpm": 128.0, "confidence": 1.0, "method": "joe_sullivan"}

        with patch.object(bpm_module, "get_audio_url_for_item", side_effect=fake_audio), \
             patch.object(bpm_module, "detect_bpm", side_effect=fake_detect):
            r1 = bpm_module.get_seed_bpm("https://x/album/seed")
            r2 = bpm_module.get_seed_bpm("https://x/album/seed")

        self.assertEqual(r1, r2)
        self.assertEqual(audio_calls, ["https://x/album/seed"])
        self.assertEqual(detect_calls, ["https://audio.example/https://x/album/seed.mp3"])

    def test_different_method_busts_cache(self):
        calls: List[tuple] = []

        def fake_audio(url, track_index=0):
            return "https://audio/x.mp3"

        def fake_detect(audio_url, method="auto", duration=60.0, use_cache=True):
            calls.append((audio_url, method))
            return {"bpm": 128.0, "confidence": 1.0, "method": method}

        with patch.object(bpm_module, "get_audio_url_for_item", side_effect=fake_audio), \
             patch.object(bpm_module, "detect_bpm", side_effect=fake_detect):
            bpm_module.get_seed_bpm("https://x/album/seed", method="auto")
            bpm_module.get_seed_bpm("https://x/album/seed", method="librosa")

        self.assertEqual(len(calls), 2)


# ---------------------------------------------------------------------------
# End-to-end get_recommendations(..., bpm_match=True)
# ---------------------------------------------------------------------------


def _candidate_metadata(item_id: str) -> Dict[str, Any]:
    return {
        "item_title": f"Track {item_id}",
        "band_name": f"Artist {item_id}",
        "item_url": f"https://example.bandcamp.com/album/{item_id}",
        "tags": [],
    }


class BpmMatchRecommendationsTests(unittest.TestCase):
    """Integration-style tests for `get_recommendations(..., bpm_match=True)`.

    We mock the network primitives (supporter scrape, item-cache hydration,
    BPM detection) so the *real* rerank path runs end-to-end against
    synthetic candidate data.
    """

    def setUp(self):
        bpm_module.clear_seed_bpm_cache()
        bpm_module.clear_bpm_cache()
        self._alpha_env = patch.dict(
            os.environ, {"BANDCAMP_BPM_RERANK_ALPHA": "1.0"}
        )
        self._alpha_env.start()
        self.addCleanup(self._alpha_env.stop)

    def _build_recommender(
        self, candidate_bpms: Dict[str, Optional[float]], counts: Dict[str, int]
    ) -> SupporterRecommender:
        """Construct a recommender pre-loaded with a synthetic candidate set.

        ``candidate_bpms`` maps item_id -> bpm (or None for items the BPM
        detector "fails" on). ``counts`` maps item_id -> supporter count.
        Returns a recommender whose ``item_cache`` is pre-populated; the
        caller is responsible for patching the supporter fetch so the
        production code sees the right counts.
        """
        recommender = SupporterRecommender()
        for item_id in candidate_bpms:
            recommender.item_cache[item_id] = _candidate_metadata(item_id)
        recommender._test_counts = counts  # consumed by the supporter-fetch stub below
        recommender._test_bpms = candidate_bpms
        return recommender

    def _run(
        self,
        recommender: SupporterRecommender,
        max_recommendations: int = 2,
        bpm_match: bool = True,
        seed_bpm: Optional[float] = 128.0,
    ):
        counts = recommender._test_counts
        bpms = recommender._test_bpms

        # Each item_id is "purchased" once per unit of supporter overlap.
        # The recommender counts purchases across supporters, so we just
        # have one supporter return the same item N times.
        def fake_curl_first(self_inner, username, data_key, first_page_only=False):
            items: List[str] = []
            for item_id, n in counts.items():
                items.extend([item_id] * n)
            return items

        def fake_hydrate(self_inner, item_ids):
            # item_cache is already populated in _build_recommender; nothing to do.
            return None

        seed_payload = (
            {"bpm": seed_bpm, "confidence": 1.0, "method": "joe_sullivan"}
            if seed_bpm is not None
            else None
        )

        def fake_attach_bpms(items, method="auto", duration=60.0, **_):
            for item in items:
                # Strip item_url's path to recover the id we used in the fixture.
                item_id = item["item_url"].rsplit("/", 1)[-1]
                bpm = bpms.get(item_id)
                if bpm is not None:
                    item["bpm"] = bpm
                    item["bpm_confidence"] = 1.0
                    item["bpm_method"] = "stub"
            return items

        with patch.object(
            sr_module, "extract_supporters", return_value=["only_supporter"]
        ), patch.object(
            sr_module, "extract_item_id", return_value=None
        ), patch.object(
            SupporterRecommender,
            "_get_supporter_items_curl_first",
            fake_curl_first,
        ), patch.object(
            SupporterRecommender,
            "_hydrate_tags_for_items",
            fake_hydrate,
        ), patch.object(
            bpm_module, "get_seed_bpm", return_value=seed_payload
        ), patch.object(
            bpm_module, "attach_bpms", side_effect=fake_attach_bpms
        ):
            return recommender.get_recommendations(
                wishlist_item_url="https://seed.bandcamp.example/album/seed",
                max_recommendations=max_recommendations,
                min_supporters=1,
                bpm_match=bpm_match,
            )

    def test_no_bpm_match_returns_top_n_by_supporters_with_distance_none(self):
        rec = self._build_recommender(
            candidate_bpms={"a": 200.0, "b": 128.0, "c": 130.0},
            counts={"a": 10, "b": 9, "c": 8},
        )
        result = self._run(rec, max_recommendations=2, bpm_match=False)
        self.assertEqual([r["item_title"] for r in result], ["Track a", "Track b"])
        for r in result:
            self.assertIn("bpm_distance", r)
            self.assertIsNone(r["bpm_distance"])

    def test_bpm_match_reranks_by_combined_score(self):
        # α=1.0 from setUp. Scores: a=10-72=-62, b=9-0=9, c=8-2=6.
        rec = self._build_recommender(
            candidate_bpms={"a": 200.0, "b": 128.0, "c": 130.0},
            counts={"a": 10, "b": 9, "c": 8},
        )
        result = self._run(rec, max_recommendations=2, bpm_match=True)
        self.assertEqual([r["item_title"] for r in result], ["Track b", "Track c"])
        self.assertAlmostEqual(result[0]["bpm_distance"], 0.0)
        self.assertAlmostEqual(result[1]["bpm_distance"], 2.0)

    def test_bpm_match_octave_tolerance(self):
        # Half-time candidate (64 BPM) vs. unrelated 100 BPM track.
        # Scores: slow=6-0=6, mid=7-28=-21. Octave-tolerant distance puts slow first.
        rec = self._build_recommender(
            candidate_bpms={"slow": 64.0, "mid": 100.0},
            counts={"slow": 6, "mid": 7},
        )
        result = self._run(rec, max_recommendations=2, bpm_match=True)
        self.assertEqual(result[0]["item_title"], "Track slow")
        self.assertAlmostEqual(result[0]["bpm_distance"], 0.0)

    def test_bpm_match_missing_bpm_keeps_supporter_rank(self):
        # "unknown" has no detectable BPM → no penalty → 20 supporters wins.
        rec = self._build_recommender(
            candidate_bpms={"known": 128.0, "unknown": None},
            counts={"known": 5, "unknown": 20},
        )
        result = self._run(rec, max_recommendations=2, bpm_match=True)
        self.assertEqual(result[0]["item_title"], "Track unknown")
        self.assertIsNone(result[0]["bpm_distance"])
        self.assertAlmostEqual(result[1]["bpm_distance"], 0.0)

    def test_bpm_match_seed_bpm_none_short_circuits(self):
        # When seed BPM resolution fails, every item should have None
        # distance and the ordering should fall back to supporters_count.
        rec = self._build_recommender(
            candidate_bpms={"a": 128.0, "b": 130.0, "c": 200.0},
            counts={"a": 5, "b": 9, "c": 7},
        )
        result = self._run(
            rec, max_recommendations=3, bpm_match=True, seed_bpm=None
        )
        self.assertEqual(
            [r["item_title"] for r in result],
            ["Track b", "Track c", "Track a"],
        )
        for r in result:
            self.assertIsNone(r["bpm_distance"])


class AttachBpmsIdempotencyTests(unittest.TestCase):
    def test_skips_items_with_bpm_already_set(self):
        items = [
            {"item_url": "https://x/already", "bpm": 128.0},
            {"item_url": "https://x/fresh"},
        ]
        audio_calls: List[str] = []

        def fake_audio(url, track_index=0):
            audio_calls.append(url)
            return None  # → no bpm set for "fresh"

        with patch.object(bpm_module, "get_audio_url_for_item", side_effect=fake_audio):
            bpm_module.attach_bpms(items)

        self.assertEqual(audio_calls, ["https://x/fresh"])
        self.assertEqual(items[0]["bpm"], 128.0)
        self.assertNotIn("bpm", items[1])


if __name__ == "__main__":
    unittest.main()
