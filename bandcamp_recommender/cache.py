"""On-disk cache for work that is expensive to redo and slow to change.

* Supporter collections: each similarity call reads every supporter's
  collection (one page + one API call each, ~80 supporters for a popular
  release). Seeds recur and supporters overlap across a scene, so without a
  cache the same collections are re-downloaded every call, which is also
  what pushes a caller into Bandcamp's rate limiter.
* Audio features: one preview download + decode per track (decodes are
  serialized process-wide), and the same candidates recur.

sqlite (WAL) at ``$XDG_CACHE_HOME/bandcamp_recommender/cache.sqlite``;
``BANDCAMP_CACHE_PATH`` overrides, ``BANDCAMP_CACHE_DISABLED=1`` turns it
off. If the location isn't writable the cache disables itself (with one
warning) instead of failing the caller. Values are zlib-compressed JSON
blobs read and written atomically per key.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import zlib
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

SUPPORTER_TTL_S = 7 * 24 * 3600
EMPTY_SUPPORTER_TTL_S = 24 * 3600
FEATURE_TTL_S = 90 * 24 * 3600
ITEM_META_TTL_S = 30 * 24 * 3600
NO_PREVIEW_TTL_S = 24 * 3600

# Bump when extract_features' audio output changes meaning, so stale vectors
# are never compared against fresh ones.
FEATURE_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS supporter_items (
    username TEXT NOT NULL,
    data_key TEXT NOT NULL,
    depth TEXT NOT NULL,
    expires_ts REAL NOT NULL,
    blob BLOB NOT NULL,
    PRIMARY KEY (username, data_key, depth)
);
CREATE TABLE IF NOT EXISTS item_meta (
    item_key TEXT PRIMARY KEY,
    expires_ts REAL NOT NULL,
    blob BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS audio_features (
    item_key TEXT NOT NULL,
    version INTEGER NOT NULL,
    expires_ts REAL NOT NULL,
    blob BLOB NOT NULL,
    PRIMARY KEY (item_key, version)
);
"""

_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None
_disabled = False


def _path() -> Path:
    override = os.environ.get("BANDCAMP_CACHE_PATH")
    if override:
        return Path(override)
    cache_home = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return Path(cache_home) / "bandcamp_recommender" / "cache.sqlite"


def _connect() -> Optional[sqlite3.Connection]:
    """Open (once) and prune expired rows. Caller holds ``_lock``."""
    global _conn, _disabled
    if _conn is not None or _disabled:
        return _conn
    if os.environ.get("BANDCAMP_CACHE_DISABLED") == "1":
        _disabled = True
        return None
    path = _path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), check_same_thread=False, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        conn.executescript(_SCHEMA)
        now = time.time()
        conn.execute("DELETE FROM supporter_items WHERE expires_ts < ?", (now,))
        conn.execute("DELETE FROM audio_features WHERE expires_ts < ?", (now,))
        conn.execute("DELETE FROM item_meta WHERE expires_ts < ?", (now,))
        conn.commit()
    except (OSError, sqlite3.Error) as e:
        logger.warning("bandcamp_recommender cache disabled (%s: %s)", path, e)
        _disabled = True
        return None
    _conn = conn
    return _conn


def _encode(value: Any) -> bytes:
    return zlib.compress(json.dumps(value, separators=(",", ":")).encode("utf-8"))


def _decode(blob: bytes) -> Any:
    return json.loads(zlib.decompress(blob).decode("utf-8"))


def _get(sql: str, params: tuple) -> Optional[Any]:
    with _lock:
        conn = _connect()
        if conn is None:
            return None
        try:
            row = conn.execute(sql, params).fetchone()
        except sqlite3.Error as e:
            logger.warning("cache read failed: %s", e)
            return None
    if row is None:
        return None
    try:
        return _decode(row[0])
    except (zlib.error, ValueError):
        return None


def _put(sql: str, params: tuple) -> None:
    with _lock:
        conn = _connect()
        if conn is None:
            return
        try:
            conn.execute(sql, params)
            conn.commit()
        except sqlite3.Error as e:
            logger.warning("cache write failed: %s", e)


# ---- supporter collections ---------------------------------------------------

def get_supporter_items(username: str, data_key: str, depth: str) -> Optional[list]:
    """``[[item_id, title, band, url], ...]`` or None on miss/expiry."""
    return _get(
        "SELECT blob FROM supporter_items WHERE username=? AND data_key=? AND depth=? AND expires_ts>=?",
        (username, data_key, depth, time.time()),
    )


def put_supporter_items(username: str, data_key: str, depth: str, rows: list,
                        ttl_s: float = SUPPORTER_TTL_S) -> None:
    _put(
        "INSERT OR REPLACE INTO supporter_items VALUES (?, ?, ?, ?, ?)",
        (username, data_key, depth, time.time() + ttl_s, _encode(rows)),
    )


# ---- item metadata ----------------------------------------------------------

def get_item_meta(item_key: str) -> Optional[dict]:
    """``{"track_id", "title", "tags"}`` for an item page, or None."""
    return _get(
        "SELECT blob FROM item_meta WHERE item_key=? AND expires_ts>=?",
        (item_key, time.time()),
    )


def put_item_meta(item_key: str, value: dict, ttl_s: float = ITEM_META_TTL_S) -> None:
    _put(
        "INSERT OR REPLACE INTO item_meta VALUES (?, ?, ?)",
        (item_key, time.time() + ttl_s, _encode(value)),
    )


# ---- audio features ---------------------------------------------------------

def get_audio_features(item_key: str) -> Optional[dict]:
    """The audio half of a feature vector (see features.extract_audio_features)."""
    return _get(
        "SELECT blob FROM audio_features WHERE item_key=? AND version=? AND expires_ts>=?",
        (item_key, FEATURE_VERSION, time.time()),
    )


def put_audio_features(item_key: str, value: dict, ttl_s: float = FEATURE_TTL_S) -> None:
    _put(
        "INSERT OR REPLACE INTO audio_features VALUES (?, ?, ?, ?)",
        (item_key, FEATURE_VERSION, time.time() + ttl_s, _encode(value)),
    )


def reset() -> None:
    """Close the connection and re-read env on next use (tests)."""
    global _conn, _disabled
    with _lock:
        if _conn is not None:
            _conn.close()
        _conn = None
        _disabled = False
