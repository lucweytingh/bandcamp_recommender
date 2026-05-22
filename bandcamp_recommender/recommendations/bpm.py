"""BPM extraction utilities for Bandcamp tracks.

Two detection backends are available:

* ``librosa`` — wraps ``librosa.feature.tempo``. Accurate on varied music but
  pulls in the full librosa stack (numpy, scipy, audioread, …).
* ``joe_sullivan`` — pure-numpy port of the algorithm used by the Bandcamp
  BPM browser extension (Joe Sullivan). Bandpasses to the kick band
  (100–150 Hz), peak-picks in 200 ms windows, folds intervals into a
  90–180 BPM histogram, and returns the smoothed argmax. Much faster than
  librosa on kick-driven music; falls back to librosa for non-kick tracks.

Detection results are cached in-process keyed on the audio URL so a long
session (or a recommendation run that revisits items) only pays once.
"""

import asyncio
import io
import json
import os
import re
import sys
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr
from io import StringIO
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

from bandcamp_recommender.recommendations.scraper import fetch_page_html

# Suppress librosa/soundfile warnings about MP3 Xing headers
# These warnings are harmless and occur when MP3 metadata is slightly inaccurate
# The warning comes from mpg123 C library which prints directly to stderr, bypassing Python warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*Xing.*")
warnings.filterwarnings("ignore", message=".*fuzzy.*")
warnings.filterwarnings("ignore", message=".*stream size.*")
warnings.filterwarnings("ignore", message=".*Cannot seek back.*")


def detect_bpm_from_audio_url(audio_url: str, timeout: int = 300, duration: float = 60.0) -> Optional[float]:
    """Detect BPM from an audio file URL using librosa.
    
    Only downloads and analyzes the first portion of the audio file (default 60 seconds)
    for faster BPM detection. This is much faster than downloading the entire file.
    
    Args:
        audio_url: URL to the audio file (e.g., from bcbits.com)
        timeout: Request timeout in seconds (default: 300)
        duration: Duration in seconds to analyze (default: 60.0). 
                  First 30-60 seconds is usually enough for accurate BPM detection.
        
    Returns:
        BPM as float if detected, None if failed
    """
    # Suppress warnings and stderr at function level
    # The mpg123 library prints directly to stderr file descriptor, so we need to redirect it at OS level
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Redirect stderr file descriptor to devnull to suppress mpg123 C library warnings
        # This works even for C libraries that write directly to the file descriptor
        try:
            import librosa
            import numpy as np
        except ImportError:
            return None
        
        # Save original stderr file descriptor
        original_stderr_fd = sys.stderr.fileno()
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        
        try:
            # Redirect stderr at file descriptor level (works for C libraries)
            os.dup2(devnull_fd, original_stderr_fd)
            
            try:
                # Try to use HTTP range request to only download the first portion
                # Estimate bytes needed: ~128kbps MP3 = ~16KB per second, so 60s = ~960KB
                # We'll request more to be safe (2MB should cover most cases)
                import urllib.request
                
                # Create request with Range header to only download first portion
                req = urllib.request.Request(audio_url)
                # Request first 2MB (should be enough for 60 seconds at 128kbps)
                req.add_header('Range', 'bytes=0-2097151')
                
                try:
                    with urllib.request.urlopen(req, timeout=timeout) as response:
                        # Read the partial response into memory
                        audio_data = io.BytesIO(response.read())
                except Exception:
                    # Fallback: if range requests aren't supported, download full file
                    # but still only analyze first portion
                    with urlopen(audio_url, timeout=timeout) as response:
                        # Limit to first 2MB even if we download more
                        audio_data = io.BytesIO(response.read(2097152))
                
                # Load only the first 'duration' seconds with librosa
                # This is much faster than loading the entire file
                y, sr = librosa.load(audio_data, sr=None, duration=duration)
                
                # Detect tempo using librosa
                onset_env = librosa.onset.onset_strength(y=y, sr=sr)
                tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, aggregate=np.median)
                
                # tempo returns an array, get the first (median) value
                bpm = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
                
                # Round to nearest whole number (BPM is almost never fractional)
                bpm = round(bpm)
                
                return float(bpm)
                        
            except Exception:
                return None
        finally:
            # Restore original stderr file descriptor
            os.dup2(original_stderr_fd, original_stderr_fd)
            os.close(devnull_fd)


async def detect_bpm_from_audio_url_async(audio_url: str, timeout: int = 300) -> Optional[float]:
    """Async version of detect_bpm_from_audio_url.
    
    Runs BPM detection in an executor to avoid blocking the event loop.
    
    Args:
        audio_url: URL to the audio file (e.g., from bcbits.com)
        timeout: Request timeout in seconds (default: 300)
        
    Returns:
        BPM as float if detected, None if failed
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, detect_bpm_from_audio_url, audio_url, timeout)


def _find_audio_path(file_dict: Dict[str, str]) -> Optional[str]:
    """Find bcbits.com audio URL from file dictionary.
    
    Args:
        file_dict: Dictionary of file format keys to URLs
        
    Returns:
        Audio URL string if found, None otherwise
    """
    for url in file_dict.values():
        if isinstance(url, str) and re.search(r"https://\w+\.bcbits\.com", url):
            return url
    return None


def _process_trackinfo(trackinfo: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process trackinfo list and extract track data.
    
    Args:
        trackinfo: List of track dictionaries from Bandcamp data
        
    Returns:
        List of processed track dictionaries
    """
    tracks = []
    for track in trackinfo:
        # Only include playable tracks (have file and title_link)
        if track.get("title_link") and track.get("file"):
            audio_path = _find_audio_path(track.get("file", {}))
            if audio_path:
                tracks.append({
                    "url": track.get("title_link", ""),
                    "audio_path": audio_path,
                    "track_num": track.get("track_num", 0),
                    "title": track.get("title", ""),
                })
    return tracks


def extract_track_info(item_url: str) -> List[Dict[str, Any]]:
    """Extract track information including audio URLs from a Bandcamp page.
    
    Args:
        item_url: URL of the Bandcamp album or track page
        
    Returns:
        List of track info dictionaries with keys:
        - url: Track URL (title_link)
        - audio_path: Audio file URL (from bcbits.com)
        - track_num: Track number
        - title: Track title (if available)
    """
    html = fetch_page_html(item_url)
    if not html:
        return []
    
    soup = BeautifulSoup(html, features="html.parser")
    tracks = []
    
    # Method 1: Extract from data-tralbum attribute (most reliable for album/track pages)
    tralbum_elem = soup.find(attrs={"data-tralbum": True})
    if tralbum_elem:
        tralbum_json = tralbum_elem.get("data-tralbum")
        if tralbum_json:
            try:
                tralbum = json.loads(tralbum_json)
                trackinfo = tralbum.get("trackinfo", [])
                tracks = _process_trackinfo(trackinfo)
            except (json.JSONDecodeError, KeyError):
                pass
    
    # Method 2: Fallback to pagedata (for other page types)
    if not tracks:
        pagedata_elem = soup.find(id="pagedata")
        if pagedata_elem:
            data_blob = pagedata_elem.get("data-blob")
            if data_blob:
                try:
                    pagedata = json.loads(data_blob)
                    tralbum_data = pagedata.get("tralbum_data", {})
                    trackinfo = tralbum_data.get("trackinfo", [])
                    tracks = _process_trackinfo(trackinfo)
                except (json.JSONDecodeError, KeyError):
                    pass
    
    return tracks


def get_bpm_for_url(
    item_url: str,
    track_index: int = 0,
    progress_callback: Optional[callable] = None
) -> Optional[Dict[str, Any]]:
    """Get BPM information for a Bandcamp URL.
    
    This function extracts track information and detects BPM using Python (librosa).
    
    Args:
        item_url: URL of the Bandcamp album or track page
        track_index: Index of track to get BPM for (0 for first track, etc.)
        progress_callback: Optional callback function(status, elapsed_time) for progress updates
        
    Returns:
        Dictionary with keys:
        - url: Track URL
        - audio_path: Audio file URL
        - track_num: Track number
        - title: Track title
        - bpm: BPM value (if detected)
        Or None if track not found
    """
    tracks = extract_track_info(item_url)
    
    if not tracks or track_index >= len(tracks):
        return None
    
    track = tracks[track_index].copy()
    
    # Detect BPM using Python (librosa)
    if track.get("audio_path"):
        if progress_callback:
            progress_callback("Detecting BPM using Python (librosa)...", 0)
        bpm = detect_bpm_from_audio_url(track["audio_path"])
        if bpm:
            track["bpm"] = bpm
    
    return track


async def get_bpm_for_url_async(
    item_url: str,
    track_index: int = 0
) -> Optional[Dict[str, Any]]:
    """Async version of get_bpm_for_url.
    
    Args:
        item_url: URL of the Bandcamp album or track page
        track_index: Index of track to get BPM for (0 for first track, etc.)
        
    Returns:
        Dictionary with keys:
        - url: Track URL
        - audio_path: Audio file URL
        - track_num: Track number
        - title: Track title
        - bpm: BPM value (if detected)
        Or None if track not found
    """
    tracks = extract_track_info(item_url)
    
    if not tracks or track_index >= len(tracks):
        return None
    
    track = tracks[track_index].copy()
    
    # Detect BPM using Python (librosa) asynchronously
    if track.get("audio_path"):
        bpm = await detect_bpm_from_audio_url_async(track["audio_path"])
        if bpm:
            track["bpm"] = bpm
    
    return track


def get_all_track_bpms(
    item_url: str,
    progress_callback: Optional[callable] = None
) -> List[Dict[str, Any]]:
    """Get BPM information for all tracks on a Bandcamp page.
    
    Args:
        item_url: URL of the Bandcamp album or track page
        progress_callback: Optional callback function(status, elapsed_time) for progress updates
        
    Returns:
        List of track dictionaries with BPM information
    """
    tracks = extract_track_info(item_url)
    
    if not tracks:
        return []
    
    # Detect BPM for each track
    for i, track in enumerate(tracks):
        if track.get("audio_path"):
            if progress_callback:
                progress_callback(f"Track {i+1}/{len(tracks)}: Detecting BPM...", 0)
            bpm = detect_bpm_from_audio_url(track["audio_path"])
            if bpm:
                track["bpm"] = bpm
    
    return tracks


async def get_all_track_bpms_async(
    item_url: str
) -> List[Dict[str, Any]]:
    """Async version of get_all_track_bpms.
    
    Gets BPM information for all tracks on a Bandcamp page.
    
    Args:
        item_url: URL of the Bandcamp album or track page
        
    Returns:
        List of track dictionaries with BPM information
    """
    tracks = extract_track_info(item_url)
    
    if not tracks:
        return []
    
    # Create tasks for tracks that have audio paths
    track_tasks = []
    track_indices = []
    for i, track in enumerate(tracks):
        if track.get("audio_path"):
            track_tasks.append(detect_bpm_from_audio_url_async(track["audio_path"]))
            track_indices.append(i)
    
    # Detect BPM for all tracks in parallel
    bpms = await asyncio.gather(*track_tasks, return_exceptions=True)

    # Assign BPMs to tracks
    for task_idx, bpm in enumerate(bpms):
        track_idx = track_indices[task_idx]
        if not isinstance(bpm, Exception) and bpm is not None:
            tracks[track_idx]["bpm"] = bpm

    return tracks


# ---------------------------------------------------------------------------
# Joe Sullivan / Bandcamp BPM extension algorithm (kick-band peak picker).
# Ported from .context/browser-bpm-detection.md. Numpy-only after decoding.
# ---------------------------------------------------------------------------

_BPM_CACHE: Dict[str, Optional[Dict[str, Any]]] = {}
_BPM_CACHE_LOCK = threading.Lock()
_STDERR_REDIRECT_LOCK = threading.Lock()


def _download_audio_bytes(
    audio_url: str,
    max_bytes: int = 2_097_152,
    timeout: int = 300,
) -> Optional[bytes]:
    """Fetch the first `max_bytes` of an audio URL. Returns None on failure."""
    try:
        req = Request(audio_url)
        req.add_header("Range", f"bytes=0-{max_bytes - 1}")
        with urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception:
        try:
            with urlopen(audio_url, timeout=timeout) as resp:
                return resp.read(max_bytes)
        except Exception:
            return None


def _decode_audio_with_librosa(
    audio_bytes: bytes,
    duration: float,
) -> Optional[Tuple[Any, int]]:
    """Decode audio bytes to (mono_samples, sample_rate) via librosa.load.

    Suppresses mpg123/libsndfile stderr noise the same way the librosa
    backend does. Returns None if librosa is unavailable or decoding fails.
    """
    try:
        import librosa  # noqa: F401
    except ImportError:
        return None

    # Serialise the fd swap because stderr (fd 2) is process-global. Two
    # concurrent decodes could otherwise restore each other's fds or
    # leave stderr permanently redirected to devnull.
    with _STDERR_REDIRECT_LOCK:
        original_stderr_fd = sys.stderr.fileno()
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        saved_stderr_fd = os.dup(original_stderr_fd)
        try:
            os.dup2(devnull_fd, original_stderr_fd)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    import librosa
                    samples, sr = librosa.load(
                        io.BytesIO(audio_bytes),
                        sr=None,
                        duration=duration,
                        mono=True,
                    )
                    return samples, int(sr)
                except Exception:
                    return None
        finally:
            os.dup2(saved_stderr_fd, original_stderr_fd)
            os.close(saved_stderr_fd)
            os.close(devnull_fd)


def _load_audio_segment(
    audio_url: str,
    duration: float = 60.0,
    timeout: int = 300,
    max_bytes: int = 2_097_152,
) -> Optional[Tuple[Any, int]]:
    """Download + decode the first `duration` seconds of an audio URL.

    Single-shot helper for callers that want a decoded mono numpy array and
    sample rate. Returns ``None`` if the audio cannot be fetched or decoded
    (network failure, librosa missing, unsupported format).

    Intended for callers (notably ``intensity.score_intensity`` and
    ``intensity.attach_audio_features``) that want both BPM and intensity
    features off a single shared decode per track.
    """
    audio_bytes = _download_audio_bytes(audio_url, max_bytes=max_bytes, timeout=timeout)
    if not audio_bytes:
        return None
    return _decode_audio_with_librosa(audio_bytes, duration)


def detect_bpm_joe_sullivan_from_samples(
    samples: Any,
    sample_rate: int,
) -> Optional[Dict[str, Any]]:
    """Run the Joe Sullivan kick-band peak-picker on a mono numpy array.

    Returns ``{"bpm": int|None, "confidence": float, "peaks": int}`` or
    ``None`` if numpy is unavailable / samples are empty.
    """
    try:
        import numpy as np
    except ImportError:
        return None
    if samples is None or len(samples) < sample_rate:
        return None

    n = len(samples)
    # FFT-domain bandpass to 100–150 Hz — equivalent to the JS LP150+HP100
    # biquad cascade for the purposes of kick-driven peak picking, and
    # avoids depending on scipy.signal.
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    spectrum = np.fft.rfft(samples)
    spectrum[(freqs < 100) | (freqs > 150)] = 0
    filtered = np.abs(np.fft.irfft(spectrum, n=n))

    global_max = float(filtered.max())
    if global_max <= 0:
        return {"bpm": None, "confidence": 0.0, "peaks": 0}
    threshold = 0.9 * global_max
    window_size = max(1, int(sample_rate * 0.2))
    min_spacing = sample_rate * 0.15

    peaks: List[int] = []
    for start in range(0, n, window_size):
        end = min(start + window_size, n)
        chunk = filtered[start:end]
        if chunk.size == 0:
            break
        local_max = float(chunk.max())
        if local_max < threshold:
            continue
        local_idx = start + int(chunk.argmax())
        if peaks and local_idx - peaks[-1] < min_spacing:
            continue
        peaks.append(local_idx)

    if len(peaks) < 8:
        return {"bpm": None, "confidence": 0.0, "peaks": len(peaks)}

    histogram: Dict[int, int] = {}
    for i in range(len(peaks)):
        for j in range(i + 1, min(i + 11, len(peaks))):
            dt = (peaks[j] - peaks[i]) / sample_rate
            if dt <= 0:
                continue
            bpm = 60.0 / dt
            while bpm < 90:
                bpm *= 2
            while bpm > 180:
                bpm /= 2
            bucket = int(round(bpm))
            histogram[bucket] = histogram.get(bucket, 0) + 1

    if not histogram:
        return {"bpm": None, "confidence": 0.0, "peaks": len(peaks)}

    total_votes = sum(histogram.values())
    best_bucket = -1
    best_score = -1
    for bucket, count in histogram.items():
        # ±1 neighbourhood smooths quantisation jitter at fractional BPMs.
        score = (
            count
            + histogram.get(bucket - 1, 0)
            + histogram.get(bucket + 1, 0)
        )
        if score > best_score:
            best_score = score
            best_bucket = bucket
    confidence = best_score / total_votes if total_votes > 0 else 0.0
    return {
        "bpm": best_bucket,
        "confidence": confidence,
        "peaks": len(peaks),
    }


def detect_bpm_joe_sullivan(
    audio_url: str,
    duration: float = 60.0,
    timeout: int = 300,
) -> Optional[Dict[str, Any]]:
    """Joe Sullivan BPM detection from an audio URL.

    Downloads the first ~2 MB of audio, decodes via librosa.load, and runs
    the kick-band peak picker. Returns the same shape as
    :func:`detect_bpm_joe_sullivan_from_samples`, or ``None`` if the audio
    could not be fetched or decoded.
    """
    audio_bytes = _download_audio_bytes(audio_url, timeout=timeout)
    if not audio_bytes:
        return None
    decoded = _decode_audio_with_librosa(audio_bytes, duration)
    if decoded is None:
        return None
    samples, sr = decoded
    return detect_bpm_joe_sullivan_from_samples(samples, sr)


def detect_bpm(
    audio_url: str,
    method: str = "auto",
    duration: float = 60.0,
    use_cache: bool = True,
) -> Optional[Dict[str, Any]]:
    """Detect BPM for an audio URL using a chosen backend.

    Args:
        audio_url: Direct URL to the audio file (e.g. a bcbits.com preview).
        method: ``"joe_sullivan"``, ``"librosa"``, or ``"auto"``.
            ``"auto"`` tries Joe Sullivan first and falls back to librosa
            when the result is missing or low-confidence.
        duration: Seconds of audio to analyse (first N seconds).
        use_cache: Whether to consult and populate the process-level cache
            keyed on ``(method, audio_url)``.

    Returns:
        ``{"bpm": float, "confidence": float, "method": str}`` or ``None``
        if no backend produced a usable result.
    """
    if method not in ("auto", "joe_sullivan", "librosa"):
        raise ValueError(f"Unknown BPM method: {method!r}")

    cache_key = f"{method}::{audio_url}"
    if use_cache:
        with _BPM_CACHE_LOCK:
            if cache_key in _BPM_CACHE:
                return _BPM_CACHE[cache_key]

    result: Optional[Dict[str, Any]] = None

    if method in ("joe_sullivan", "auto"):
        js = detect_bpm_joe_sullivan(audio_url, duration=duration)
        if js and js.get("bpm"):
            result = {
                "bpm": float(js["bpm"]),
                "confidence": float(js.get("confidence", 0.0)),
                "method": "joe_sullivan",
            }

    if method == "librosa" or (method == "auto" and (result is None or result["confidence"] < 0.05)):
        librosa_bpm = detect_bpm_from_audio_url(audio_url, duration=duration)
        if librosa_bpm:
            # librosa's tempo tracker doesn't expose a confidence score, so we
            # report 1.0 when it returns *anything* and let callers decide
            # whether to trust it.
            result = {
                "bpm": float(librosa_bpm),
                "confidence": 1.0,
                "method": "librosa",
            }

    if use_cache:
        with _BPM_CACHE_LOCK:
            _BPM_CACHE[cache_key] = result
    return result


def clear_bpm_cache() -> None:
    """Clear the in-process BPM cache."""
    with _BPM_CACHE_LOCK:
        _BPM_CACHE.clear()


def get_audio_url_for_item(
    item_url: str,
    track_index: int = 0,
) -> Optional[str]:
    """Return the preview audio URL for a Bandcamp item.

    Defaults to the first track. Returns ``None`` if the page exposes no
    playable preview (e.g. a tracks-only release that isn't streamable).
    """
    tracks = extract_track_info(item_url)
    if not tracks or track_index >= len(tracks):
        return None
    return tracks[track_index].get("audio_path")


def attach_bpms(
    items: List[Dict[str, Any]],
    method: str = "auto",
    duration: float = 60.0,
    max_workers: int = 3,
    progress_callback: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """Attach BPM information to each item with an ``item_url``.

    Items are mutated in place and also returned for chaining. Mutations:

    * ``bpm`` — detected tempo (float).
    * ``bpm_confidence`` — detector confidence in 0..1.
    * ``bpm_method`` — which backend produced the value.

    Items without a streamable preview are left untouched (no ``bpm`` key).

    ``progress_callback`` follows the recommender convention:
    ``(status, current, total, estimated_seconds)``.
    """
    targets = [item for item in items if item.get("item_url")]
    total = len(targets)
    if total == 0:
        return items

    done = 0
    lock = threading.Lock()

    def _process(item: Dict[str, Any]) -> None:
        nonlocal done
        audio_url = get_audio_url_for_item(item["item_url"])
        if audio_url:
            result = detect_bpm(audio_url, method=method, duration=duration)
            if result and result.get("bpm"):
                item["bpm"] = result["bpm"]
                item["bpm_confidence"] = result.get("confidence", 0.0)
                item["bpm_method"] = result.get("method", method)
        with lock:
            done += 1
            current = done
        if progress_callback:
            progress_callback(
                f"Detected BPM for {current}/{total} items",
                current,
                total,
                0,
            )

    workers = max(1, min(max_workers, total))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_process, targets))

    return items
