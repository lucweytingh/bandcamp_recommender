"""BPM extraction utilities for Bandcamp tracks."""

import asyncio
import io
import json
import os
import re
import sys
import warnings
from contextlib import redirect_stderr
from io import StringIO
from typing import Any, Dict, List, Optional
from urllib.request import urlopen

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
