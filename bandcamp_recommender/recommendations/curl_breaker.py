"""File-backed circuit breaker for curl-based Bandcamp requests.

When Bandcamp blocks an IP, every curl call burns its full timeout before the
caller can fall back to Selenium. This module tracks recent curl outcomes on
disk so the trip survives process restarts (a CLI run is short — in-memory
state would forget within minutes and re-pay the timeout tax on every
invocation).

Trip policy: ``TRIP_THRESHOLD`` consecutive failures within ``TRIP_WINDOW_S``
seconds → mark the breaker tripped for ``TRIP_COOLDOWN_S`` seconds. A single
success while tripped clears the trip immediately (the IP is back).

Overrides:
- ``BANDCAMP_DISABLE_CURL=1`` — hard disable, breaker not consulted.
- ``BANDCAMP_CURL_BREAKER_DISABLED=1`` — skip breaker logic entirely (tests).
- ``BANDCAMP_CURL_TRIP_THRESHOLD``, ``BANDCAMP_CURL_TRIP_WINDOW_S``,
  ``BANDCAMP_CURL_TRIP_COOLDOWN_S`` — tune defaults.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

_DEFAULT_THRESHOLD = 3
_DEFAULT_WINDOW_S = 60.0
_DEFAULT_COOLDOWN_S = 24 * 60 * 60  # 24 hours

_lock = threading.Lock()


def _state_path() -> Path:
    override = os.environ.get("BANDCAMP_CURL_BREAKER_STATE")
    if override:
        return Path(override)
    cache_home = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return Path(cache_home) / "bandcamp_recommender" / "curl_breaker.json"


def _read_state() -> dict:
    path = _state_path()
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return {"recent_failures": [], "tripped_until": 0.0}


def _write_state(state: dict) -> None:
    path = _state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    try:
        fd, tmp = tempfile.mkstemp(prefix=".curl_breaker.", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state, f)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except OSError:
        pass


def _config() -> tuple[int, float, float]:
    def _env_float(name: str, default: float) -> float:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    threshold = int(_env_float("BANDCAMP_CURL_TRIP_THRESHOLD", _DEFAULT_THRESHOLD))
    window_s = _env_float("BANDCAMP_CURL_TRIP_WINDOW_S", _DEFAULT_WINDOW_S)
    cooldown_s = _env_float("BANDCAMP_CURL_TRIP_COOLDOWN_S", _DEFAULT_COOLDOWN_S)
    return threshold, window_s, cooldown_s


def _breaker_off() -> bool:
    return os.environ.get("BANDCAMP_CURL_BREAKER_DISABLED") == "1"


def should_skip_curl(now: Optional[float] = None) -> bool:
    """Return True if curl should be skipped (manual override or tripped breaker)."""
    if os.environ.get("BANDCAMP_DISABLE_CURL") == "1":
        return True
    if _breaker_off():
        return False
    now = time.time() if now is None else now
    with _lock:
        state = _read_state()
        return float(state.get("tripped_until", 0.0)) > now


def record_outcome(success: bool, now: Optional[float] = None) -> None:
    """Record a curl outcome. Trips the breaker after enough recent failures."""
    if _breaker_off():
        return
    threshold, window_s, cooldown_s = _config()
    now = time.time() if now is None else now
    with _lock:
        state = _read_state()
        recent = [t for t in state.get("recent_failures", []) if now - t <= window_s]
        if success:
            recent = []
            state["tripped_until"] = 0.0
        else:
            recent.append(now)
            if len(recent) >= threshold:
                state["tripped_until"] = now + cooldown_s
                recent = []
        state["recent_failures"] = recent
        _write_state(state)


def reset() -> None:
    """Clear breaker state. Useful for tests and manual recovery."""
    with _lock:
        try:
            _state_path().unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


if __name__ == "__main__":  # pragma: no cover
    import sys

    if len(sys.argv) >= 2 and sys.argv[1] == "reset":
        reset()
        print("curl breaker state cleared")
    elif len(sys.argv) >= 2 and sys.argv[1] == "status":
        state = _read_state()
        now = time.time()
        tripped_until = float(state.get("tripped_until", 0.0))
        if tripped_until > now:
            mins = (tripped_until - now) / 60.0
            print(f"TRIPPED — curl skipped for {mins:.1f} more minutes")
        else:
            print(f"ok — recent failures: {len(state.get('recent_failures', []))}")
    else:
        print("usage: python -m bandcamp_recommender.recommendations.curl_breaker [reset|status]")
        sys.exit(1)
