"""Bandcamp item (album/track) pages: the one parse, and cached metadata.

:func:`parse_item_html` is the single reading of an item page (tracks,
preview choice, tags, location, artist, art, description, release date);
everything in the library that needs a piece of an item page goes through
it, and :func:`item_info` exposes it to consumers on top of the shared page
policy (:func:`bandcamp_recommender.http.get_page`).

:func:`item_meta` is the stable subset every similarity call needs for ~30
candidates: the featured preview track and the tags. Bandcamp allows ~1
request/second, so re-reading those pages on every call costs ~30 s even
when nothing changed. They are cached on disk
(``bandcamp_recommender.cache``); the page's signed stream URL is NOT (it
expires), so ``audio_path`` is only present when the page was read just now.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from bandcamp_recommender import cache, http

logger = logging.getLogger(__name__)

# Every Bandcamp item page carries this attribute; a 200 without it is not
# the page (bot challenge, error shell, parked domain).
ITEM_PAGE_MARKER = "data-tralbum"

# Cover art lives on Bandcamp's image CDN as a{art_id}_{size}.jpg (the URL
# pattern is undocumented, hence the og:image fallback for the id).
_ART_URL = "https://f4.bcbits.com/img/a{art_id}_{size}.jpg"
_OG_ART_RE = re.compile(r"/img/a?(\d+)_\d+\.\w+")
_MONTHS = {m: i for i, m in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1)}
# Bandcamp's own date format: "19 Feb 2022 00:00:00 GMT".
_BC_DATE_RE = re.compile(r"(\d{1,2}) ([A-Za-z]{3})[a-z]* (\d{4})")


def normalize_item_url(url: str) -> str:
    return (url or "").split("?", 1)[0].split("#", 1)[0].rstrip("/").lower()


def _str(value: Any) -> Optional[str]:
    return (value.strip() or None) if isinstance(value, str) else None


def _dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def art_url(art_id: Any, size: int = 10) -> Optional[str]:
    """Cover-art URL for an ``art_id`` (from :func:`item_info`) at a
    Bandcamp image size: 10 = 1200 px (lock screens), 2 = 700 px (grids).
    None without an art id."""
    if not art_id:
        return None
    return _ART_URL.format(art_id=art_id, size=size)


def select_preview_track(tracks: List[Dict[str, Any]], featured_track_id: Any) -> Optional[Dict[str, Any]]:
    """The release's featured track if it is playable, else the first
    playable one. This is the track a listener hears when they open the page,
    so it is the one worth describing (and, downstream, airing). ``tracks``
    are the playable tracks."""
    if not tracks:
        return None
    if featured_track_id is not None:
        for track in tracks:
            if str(track.get("track_id")) == str(featured_track_id):
                return track
    return tracks[0]


def _find_audio_path(files: Dict[str, Any]) -> Optional[str]:
    """The first bcbits.com stream URL among a track's file formats."""
    for url in files.values():
        if isinstance(url, str) and re.search(r"https://\w+\.bcbits\.com", url):
            return url
    return None


def _tracks(trackinfo: Any, base_url: Optional[str]) -> List[Dict[str, Any]]:
    """Every track of the release, in page order, playable or not.

    ``playable`` = a bcbits stream and a track page (``title_link``);
    unreleased preorder tracks have neither (and ``file`` is then ``[]``,
    not a dict)."""
    out = []
    for t in trackinfo if isinstance(trackinfo, list) else []:
        if not isinstance(t, dict):
            continue
        raw_files = t.get("file")
        files = {k: v for k, v in raw_files.items() if v} if isinstance(raw_files, dict) else {}
        link = t.get("title_link") or None
        audio_path = _find_audio_path(files)
        out.append({
            "track_id": t.get("track_id") or t.get("id"),
            "title": t.get("title") or None,
            # Per-track artist: set on compilations, None otherwise.
            "artist": t.get("artist") or None,
            "track_num": t.get("track_num"),
            # Unreleased tracks report 0.0: unknown, not zero seconds.
            "duration": t.get("duration") or None,
            "url": urljoin(base_url, link) if link and base_url else link,
            "files": files,
            "audio_path": audio_path,
            "playable": bool(link and audio_path),
        })
    return out


def preview_from_tralbum(tralbum: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The preview track (featured if playable, else first playable) from an
    already-decoded ``data-tralbum`` blob, for callers that parsed the page."""
    tralbum = tralbum if isinstance(tralbum, dict) else {}
    tracks = _tracks(tralbum.get("trackinfo"), _str(tralbum.get("url")))
    return select_preview_track([t for t in tracks if t["playable"]], tralbum.get("featured_track_id"))


def _json_attr(tag, attr: str) -> Dict[str, Any]:
    if tag is None or not tag.get(attr):
        return {}
    try:
        return _dict(json.loads(tag.get(attr)))
    except (TypeError, ValueError):
        return {}


def _ld_json(soup) -> Dict[str, Any]:
    """The page's schema.org blob (MusicAlbum / MusicRecording)."""
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(tag.string or tag.get_text() or "")
        except (TypeError, ValueError):
            continue
        if isinstance(data, dict):
            return data
    return {}


def _page_tags(soup) -> List[str]:
    """Tag link texts in page order (case-insensitively unique), from the
    release's tags block when there is one."""
    scope = soup.find(class_="tralbum-tags") or soup
    out, seen = [], set()
    for a in scope.find_all("a", class_="tag"):
        text = a.get_text(strip=True)
        if text and text.casefold() not in seen:
            seen.add(text.casefold())
            out.append(text)
    return out


def _location_tag(tags: List[str], band_location: Optional[str]) -> Optional[str]:
    """Bandcamp appends the artist's location ("Berlin" for "Berlin,
    Germany") as the last tag; its link looks like any genre tag's. It is
    split off only when it matches the page's band location, so a page
    without a known location keeps all of its tags."""
    if not tags or not band_location:
        return None
    names = {band_location.casefold()} | {p.strip().casefold() for p in band_location.split(",")}
    return tags[-1] if tags[-1].casefold() in names else None


def _release_date(*raw: Any) -> Optional[str]:
    """ISO date (YYYY-MM-DD) from the first parseable Bandcamp date."""
    for value in raw:
        m = _BC_DATE_RE.search(value) if isinstance(value, str) else None
        month = _MONTHS.get(m.group(2).lower()) if m else None
        if month:
            return f"{int(m.group(3)):04d}-{month:02d}-{int(m.group(1)):02d}"
    return None


def _album_url(tralbum: Dict[str, Any], ld: Dict[str, Any], base_url: Optional[str]) -> Optional[str]:
    """Absolute URL of the album a track page belongs to; None for albums
    and standalone singles."""
    raw = _str(tralbum.get("album_url"))
    if raw:
        return urljoin(base_url, raw) if base_url else raw
    releases = _dict(ld.get("inAlbum")).get("albumRelease")
    first = _dict(releases[0]) if isinstance(releases, list) and releases else {}
    candidate = _str(first.get("@id"))
    # A standalone single lists itself as its own "album" release.
    if candidate and "/track/" not in candidate:
        return candidate
    return None


def parse_item_html(html: str, item_url: Optional[str] = None) -> Dict[str, Any]:
    """Everything the library knows how to read from an album or track page,
    as a plain dict. Pure (no I/O) and total: a missing or malformed piece is
    None / empty, never an exception, so a challenge page or an error shell
    parses to a dict with no page data (only what ``item_url`` implies).

    Keys:
        ``item_url``          ``item_url`` as given, else the page's own URL
        ``item_type``         ``"album"`` | ``"track"`` | None
        ``item_id``           Bandcamp's tralbum id
        ``title``             release (album page) or track (track page) title
        ``artist``            the release's artist as the page shows it ("by
                              ..."); on a label's page this is the artist,
                              not the label
        ``page_name``         the Bandcamp page it is on (``data-band``): the
                              artist's own page, or the label's
        ``album_url``         a track page's album (absolute), else None
        ``tracks``            every track in order: ``track_id``, ``title``,
                              ``artist`` (compilations), ``track_num``,
                              ``duration`` (s), ``url`` (absolute when the
                              item URL is known), ``files`` ({format: url}),
                              ``audio_path`` (bcbits stream), ``playable``
        ``featured_track_id`` the artist's featured track (album pages)
        ``preview``           the track that plays: featured if playable,
                              else the first playable (one of ``tracks``)
        ``tags``              genre tags, page order, location excluded
        ``location``          the location tag ("Berlin"), or None
        ``description``       the page's own description (ld+json, else the
                              "about" text); :func:`item_info` gives a track
                              page its album's
        ``art_id``            cover art id, see :func:`art_url`
        ``release_date``      ISO date, or None

    ``feature_tags(info)`` gives the tag list the tag features expect
    (location included)."""
    soup = BeautifulSoup(html or "", features="html.parser")
    script = soup.find(attrs={"data-tralbum": True})
    tralbum = _json_attr(script, "data-tralbum")
    band = _json_attr(script, "data-band")
    if not tralbum:
        # Older page shapes carried the release in the pagedata blob only.
        pagedata = _json_attr(soup.find(id="pagedata"), "data-blob")
        tralbum = _dict(pagedata.get("tralbum_data"))
    current = _dict(tralbum.get("current"))
    ld = _ld_json(soup)
    publisher = _dict(ld.get("publisher"))
    base_url = _str(item_url) or _str(tralbum.get("url"))

    tracks = _tracks(tralbum.get("trackinfo"), base_url)
    featured = tralbum.get("featured_track_id")

    band_location = _str(_dict(publisher.get("foundingLocation")).get("name"))
    if band_location is None:
        loc = soup.select_one("#band-name-location .location")
        band_location = _str(loc.get_text()) if loc else None
    tags = _page_tags(soup)
    location = _location_tag(tags, band_location)
    if location is not None:
        tags = tags[:-1]

    item_type = _str(tralbum.get("item_type")) or _str(current.get("type"))
    if item_type not in ("album", "track"):
        path = base_url or ""
        item_type = "track" if "/track/" in path else "album" if "/album/" in path else None

    art_id = current.get("art_id") or tralbum.get("art_id")
    if not art_id:
        og = soup.find("meta", attrs={"property": "og:image"})
        m = _OG_ART_RE.search(og.get("content") or "") if og else None
        art_id = int(m.group(1)) if m else None

    return {
        "item_url": base_url,
        "item_type": item_type,
        "item_id": tralbum.get("id") or current.get("id"),
        "title": _str(current.get("title")) or _str(ld.get("name")),
        "artist": (_str(tralbum.get("artist")) or _str(current.get("artist"))
                   or _str(_dict(ld.get("byArtist")).get("name")) or _str(band.get("name"))),
        "page_name": _str(band.get("name")) or _str(publisher.get("name")),
        "album_url": _album_url(tralbum, ld, base_url) if item_type == "track" else None,
        "tracks": tracks,
        "featured_track_id": featured,
        "preview": select_preview_track([t for t in tracks if t["playable"]], featured),
        "tags": tags,
        "location": location,
        "description": _str(ld.get("description")) or _str(current.get("about")),
        "art_id": art_id or None,
        "release_date": _release_date(
            current.get("release_date"), tralbum.get("album_release_date"), ld.get("datePublished"),
        ),
    }


def feature_tags(info: Dict[str, Any]) -> List[str]:
    """Every tag the page shows, location last: the input the tag features
    (``recommendations.mood_tags``) expect. The location is kept there on
    purpose: it pairs with genre tags into place genres ("detroit" +
    "techno", "chicago" + "house", "japan" + "jazz")."""
    tags = list(info.get("tags") or [])
    return tags + [info["location"]] if info.get("location") else tags


def item_info(item_url: str, *, follow_album_description: bool = True) -> Dict[str, Any]:
    """:func:`parse_item_html` of the live page, read through the shared page
    policy (plain HTTP, payload validation, short-lived cache, the shared
    browser on a bot challenge).

    A track page describes its album, so with ``follow_album_description``
    a track page that belongs to an album takes the album page's description
    (one more page read); if that page can't be read the track's own
    description (usually None) stays.

    Raises :class:`bandcamp_recommender.http.FetchError` (typed: Challenged,
    RateLimited, Timeout, HTTPStatus, Unexpected) when the item page itself
    can't be read; never returns a dict parsed from a junk page."""
    info = parse_item_html(http.get_page(item_url, expect=ITEM_PAGE_MARKER), item_url)
    if follow_album_description and info["item_type"] == "track" and info["album_url"]:
        try:
            album_html = http.get_page(info["album_url"], expect=ITEM_PAGE_MARKER)
        except http.FetchError as e:
            logger.warning("album page for %s unavailable: %s", item_url, e)
        else:
            album = parse_item_html(album_html, info["album_url"])
            info["description"] = album["description"] or info["description"]
    return info


def parse_item_page(html: str) -> Dict[str, Any]:
    """``{"tags": feature_tags, "preview": track dict or None}`` from an
    item page (the subset :func:`item_meta` caches)."""
    info = parse_item_html(html)
    return {"tags": feature_tags(info), "preview": info["preview"]}


def item_meta(item_url: str) -> Optional[Dict[str, Any]]:
    """``{"track_id", "title", "tags"}`` (+ ``audio_path`` when the page was
    fetched now), from the disk cache or one page read. ``tags`` are
    :func:`feature_tags` (location included). None when the page can't be
    read."""
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
