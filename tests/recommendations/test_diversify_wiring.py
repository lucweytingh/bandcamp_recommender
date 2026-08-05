"""Wiring tests: get_similar_recommendations(..., diversify=...) applies the
in-vibe diversity re-rank only when opted in (default off = backward-compatible).

The network primitives are mocked so the real method body runs end-to-end against
synthetic candidates; the diversity re-rank itself is unit-tested in
test_diversity.py, so here we just assert it's invoked (or not) and its result
flows through.
"""
import os
import unittest
from typing import Any, Dict, List
from unittest.mock import patch

from bandcamp_recommender.recommendations import (
    supporter_recommender as sr_module,
    diversity as div_module,
)
from bandcamp_recommender.recommendations.supporter_recommender import (
    SupporterRecommender,
)


def _cands() -> List[Dict[str, Any]]:
    return [
        {"item_url": "https://x.bandcamp.com/album/a", "band_name": "A", "tags": []},
        {"item_url": "https://x.bandcamp.com/album/b", "band_name": "B", "tags": []},
        {"item_url": "https://x.bandcamp.com/album/c", "band_name": "C", "tags": []},
    ]


def _fake_features(item, **kw):
    # A constant, fully-shared feature vector so every distance is defined and
    # every candidate sits in the source's vibe band (the wiring, not the math,
    # is under test here).
    return {
        "rms_mean": 0.5, "rms_p95": 0.5, "onset_rate": 0.5,
        "spectral_centroid": 0.5, "crest_factor": 0.5,
        "tag_mood": 0.0, "tag_spikiness": 0.0,
        "bpm_folded_norm": 0.5, "bpm_norm": 0.5, "bpm": 120.0,
    }


class DiversifyWiringTests(unittest.TestCase):
    def setUp(self):
        for k in ("BANDCAMP_DIVERSIFY",):
            os.environ.pop(k, None)
        self.rec = SupporterRecommender()
        self._patchers = [
            patch.object(SupporterRecommender, "get_recommendations", return_value=_cands()),
            patch.object(SupporterRecommender, "_hydrate_audio_urls_for_items", return_value=None),
            patch.object(sr_module, "extract_tags", lambda url: []),
            patch("bandcamp_recommender.features.extract_features", side_effect=_fake_features),
            patch("bandcamp_recommender.recommendations.bpm.get_audio_url_for_item",
                  lambda url, **kw: "https://audio/x.mp3"),
        ]
        for p in self._patchers:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self._patchers])

    def test_default_does_not_diversify(self):
        with patch.object(div_module, "diversify_items") as spy:
            out = self.rec.get_similar_recommendations("https://x.bandcamp.com/album/seed",
                                                       max_recommendations=3)
        spy.assert_not_called()
        assert [c["item_url"] for c in out] == [c["item_url"] for c in _cands()]

    def test_diversify_mode_applies_rerank(self):
        # Patch the re-rank to a recognisable reversal; assert the method returns it.
        def fake_div(items, source_features, mode="mmr", **kw):
            assert mode == "mmr"
            assert isinstance(source_features, dict) and source_features  # source feats passed
            return list(reversed(items))
        with patch.object(div_module, "diversify_items", side_effect=fake_div) as spy:
            out = self.rec.get_similar_recommendations("https://x.bandcamp.com/album/seed",
                                                       max_recommendations=3, diversify="mmr")
        spy.assert_called_once()
        assert [c["item_url"] for c in out] == list(reversed([c["item_url"] for c in _cands()]))

    def test_env_default_enables_diversify(self):
        os.environ["BANDCAMP_DIVERSIFY"] = "maxmin"
        with patch.object(div_module, "diversify_items", side_effect=lambda items, sf, **kw: items) as spy:
            self.rec.get_similar_recommendations("https://x.bandcamp.com/album/seed",
                                                 max_recommendations=3)
        spy.assert_called_once()
        assert spy.call_args.kwargs.get("mode") == "maxmin"

    def test_garbage_mode_does_not_crash(self):
        # An operator typo in the env (or param) must NOT crash recommendation
        # generation — it degrades to the plain similarity ranking with a warning.
        import warnings
        os.environ["BANDCAMP_DIVERSIFY"] = "max-min"  # typo, runs the real re-rank
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = self.rec.get_similar_recommendations(
                "https://x.bandcamp.com/album/seed", max_recommendations=3)
        assert [c["item_url"] for c in out] == [c["item_url"] for c in _cands()]

    def test_garbage_param_does_not_crash(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = self.rec.get_similar_recommendations(
                "https://x.bandcamp.com/album/seed", max_recommendations=3,
                diversify="MMR ")  # case/whitespace typo still works (normalized)
        assert len(out) == 3


if __name__ == "__main__":
    unittest.main()
