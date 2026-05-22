"""Tests for bandcamp_recommender.recommendations.intensity.

librosa is imported soft-style inside intensity.py, so we patch the
functions it uses (librosa.feature.rms, librosa.feature.spectral_centroid,
librosa.onset.onset_detect) rather than supplying real audio.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bandcamp_recommender.recommendations import intensity


def _mock_librosa(rms_values, centroid_value, onsets):
    """Build a MagicMock that mimics the subset of librosa we touch."""
    mock = MagicMock()
    mock.feature.rms.return_value = np.array([rms_values], dtype=float)
    mock.feature.spectral_centroid.return_value = np.array(
        [[centroid_value]], dtype=float
    )
    mock.onset.onset_detect.return_value = np.array(onsets, dtype=int)
    return mock


def test_chill_exemplar_low_intensity():
    """Quiet ambient: low RMS, sparse onsets, dark spectrum, smooth dynamics."""
    samples = np.zeros(22050 * 10, dtype=float)
    # peak amplitude tiny so crest factor stays low too.
    samples[100] = 0.02
    sr = 22050

    mock_librosa = _mock_librosa(
        rms_values=[0.01, 0.012, 0.015, 0.013],   # very low energy
        centroid_value=600.0,                      # dark
        onsets=[1000, 50000],                      # ~0.2 onsets/sec over 10s
    )
    with patch.dict("sys.modules", {"librosa": mock_librosa}):
        score = intensity.score_intensity_from_samples(samples, sr)

    assert score is not None
    assert 0.0 <= score <= 0.35, f"chill exemplar should score low, got {score}"


def test_party_exemplar_high_intensity():
    """Loud drum-heavy track: high RMS, dense onsets, bright spectrum, punchy crest."""
    sr = 22050
    samples = np.zeros(sr * 10, dtype=float)
    # Peaks scattered through buffer push the crest factor up.
    samples[::sr // 4] = 0.9

    mock_librosa = _mock_librosa(
        rms_values=[0.28, 0.30, 0.32, 0.29],       # near max
        centroid_value=3800.0,                      # bright
        onsets=list(range(0, sr * 10, sr // 8)),    # ~8 onsets/sec
    )
    with patch.dict("sys.modules", {"librosa": mock_librosa}):
        score = intensity.score_intensity_from_samples(samples, sr)

    assert score is not None
    assert score >= 0.7, f"party exemplar should score high, got {score}"


def test_missing_audio_returns_none():
    """score_intensity should return None when the URL cannot be loaded."""
    with patch(
        "bandcamp_recommender.recommendations.intensity._load_audio_segment",
        return_value=None,
    ):
        assert intensity.score_intensity("https://example.invalid/preview.mp3") is None


def test_empty_samples_returns_none():
    """An empty samples array (decoder returned silence-only) returns None."""
    mock_librosa = _mock_librosa(rms_values=[0.0], centroid_value=0.0, onsets=[])
    with patch.dict("sys.modules", {"librosa": mock_librosa}):
        assert intensity.score_intensity_from_samples(np.array([]), 22050) is None


def test_score_clamped_to_unit_interval():
    """Even with extreme inputs the score must fall inside [0, 1]."""
    sr = 22050
    samples = np.full(sr * 10, 1.0, dtype=float)  # constant max amplitude
    mock_librosa = _mock_librosa(
        rms_values=[10.0, 10.0],                   # absurd RMS
        centroid_value=20000.0,                     # absurd centroid
        onsets=list(range(0, sr * 10, 100)),        # absurd onset density
    )
    with patch.dict("sys.modules", {"librosa": mock_librosa}):
        score = intensity.score_intensity_from_samples(samples, sr)
    assert score is not None
    assert 0.0 <= score <= 1.0
