"""Stable per-item metadata: the featured preview track and the tags.

Both come from the item page, change rarely, and every similarity call needs
them for ~30 candidates. Bandcamp allows ~1 request/second, so re-reading
those pages on every call costs ~30 s even when nothing changed. They are
cached on disk (``bandcamp_recommender.cache``); the page's signed stream
URL is NOT (it expires), so ``audio_path`` is only present when the page was
read just now.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup

from bandcamp_recommender import cache


def normalize_item_url(url: str) -> str:
    return (url or "").split("?", 1)[0].split("#", 1)[0].rstrip("/").lower()


def preview_from_tralbum(tralbum: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The preview track (featured if playable, else first playable) from an
    already-decoded ``data-tralbum`` blob, for callers that parsed the page."""
    from bandcamp_recommender.recommendations.bpm import _process_trackinfo, select_preview_track

    return select_preview_track(
        _process_trackinfo((tralbum or {}).get("trackinfo") or []),
        (tralbum or {}).get("featured_track_id"),
    )


def parse_item_page(html: str) -> Dict[str, Any]:
    """``{"tags": [...], "preview": track dict or None}`` from an item page."""
    from bandcamp_recommender.recommendations.bpm import _parse_tracks, select_preview_track

    soup = BeautifulSoup(html, features="html.parser")
    tags = [
        a.get_text(strip=True)
        for a in soup.find_all("a", class_=re.compile("tag"))
        if a.get_text(strip=True)
    ]
    return {"tags": tags, "preview": select_preview_track(*_parse_tracks(html))}


def item_meta(item_url: str) -> Optional[Dict[str, Any]]:
    """``{"track_id", "title", "tags"}`` (+ ``audio_path`` when the page was
    fetched now), from the disk cache or one page read. None when the page
    can't be read."""
    from bandcamp_recommender.recommendations.scraper import fetch_page_html

    key = normalize_item_url(item_url)
    cached = cache.get_item_meta(key)
    if cached is not None:
        return dict(cached)
    html = fetch_page_html(item_url)
    if not html:
        return None
    parsed = parse_item_page(html)
    preview = parsed["preview"] or {}
    meta = {
        "track_id": preview.get("track_id"),
        "title": preview.get("title"),
        "tags": parsed["tags"],
    }
    cache.put_item_meta(
        key, meta,
        ttl_s=cache.ITEM_META_TTL_S if meta["track_id"] is not None else cache.NO_PREVIEW_TTL_S,
    )
    return {**meta, "audio_path": preview.get("audio_path")}
