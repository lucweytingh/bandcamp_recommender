"""A fan's own lists (wishlist + collection), read in full.

A fan page (``bandcamp.com/<username>/wishlist``) embeds a ``pagedata``
blob with the fan's id, each list's item count and whether the fan keeps it
private. The items come from the fancollection API, newest first, in pages
of up to 10000 (``older_than_token`` = where the previous page ended).

Plain HTTP first. If Bandcamp challenges the page or the API, the shared
fallback browser loads the fan page (same-origin cookies) and calls the API
from inside it. Failures are typed (:mod:`bandcamp_recommender.http`
errors) and a list is never silently empty: a private list is named in
``hidden`` (measured: a private wishlist shows ``hidden: true`` on the page
and the API answers ``{"error": true}``), any other error reply raises.

The supporter reads in ``SupporterRecommender`` use the same page parse and
API call (:func:`parse_pagedata`, :func:`read_list_page`).
"""

from __future__ import annotations

import email.utils
import json
import logging
import time
from typing import Any, Dict, List, Optional, Sequence

from bs4 import BeautifulSoup

from bandcamp_recommender import browser, http

logger = logging.getLogger(__name__)

FAN_PAGE_MARKER = "pagedata"
LISTS = ("wishlist", "collection")
API_URLS = {
    "wishlist": "https://bandcamp.com/api/fancollection/1/wishlist_items",
    "collection": "https://bandcamp.com/api/fancollection/1/collection_items",
}
# The API's maximum page size.
PAGE_SIZE = 10000
# Full collections take Bandcamp ~3 s per 1000 items to serve (measured:
# 2789 items = 8.7 s, 7 MB); the 10000-item maximum is ~30 s, more under load.
LIST_TIMEOUT_S = 120
# 20 x 10000 items: a bound on a misbehaving token, not a real fan.
_MAX_PAGES = 20


class FanNotFound(http.FetchError):
    """Bandcamp has no fan page for this username (404/410)."""


def fan_page_url(username: str) -> str:
    return f"https://bandcamp.com/{username}/wishlist"


def parse_pagedata(html: str) -> Optional[Dict[str, Any]]:
    """The decoded ``pagedata`` blob of a Bandcamp page, None if absent or
    not a JSON object."""
    elem = BeautifulSoup(html, features="html.parser").find(id="pagedata")
    if elem is None:
        return None
    try:
        blob = json.loads(elem.get("data-blob") or "{}")
    except ValueError:
        return None
    return blob if isinstance(blob, dict) else None


def newest_first_token() -> str:
    """An ``older_than_token`` in the future: the API answers with a list's
    newest items, the same head the fan page shows."""
    return f"{int(time.time()) + 86400}::a::"


def read_list_page(name: str, fan_id: Any, older_than_token: str, count: int, *,
                   referer: str, driver=None, timeout: float = LIST_TIMEOUT_S) -> Any:
    """One fancollection API page of the fan's ``name`` list, over plain
    HTTP, or from inside ``driver``'s current bandcamp.com page. Returns the
    decoded reply (``{"items", "more_available", "last_token", ...}`` or an
    error object); raises :class:`http.FetchError` subclasses."""
    payload = {"fan_id": fan_id, "older_than_token": older_than_token, "count": count}
    if driver is None:
        return http.post_json(API_URLS[name], payload, timeout=timeout, referer=referer)
    return browser.post_json(driver, API_URLS[name], payload, timeout=timeout)


def _parse_date(value: Any) -> Optional[float]:
    """Epoch seconds from an API date like ``"16 Sep 2026 22:52:52 GMT"``."""
    if not isinstance(value, str):
        return None
    try:
        parsed = email.utils.parsedate_tz(value.strip())
        if parsed is None:
            return None
        if parsed[9] is None:
            parsed = parsed[:9] + (0,)  # Bandcamp dates are GMT
        return float(email.utils.mktime_tz(parsed))
    except (IndexError, OverflowError, TypeError, ValueError):
        return None


def parse_added_ts(item: Dict[str, Any]) -> Optional[float]:
    """Epoch seconds when a fancollection API item entered this list: the
    epoch prefix of its ``token`` (``"1789599172:1168869557:a::"``), the
    feed's own sort key. Measured (2026-10, 136 items): it equals
    ``purchased`` on every collection item and ``added`` on every wishlist
    item. A collection item's ``added`` is not when it was bought (it can be
    the wishlist date, years older, or a later rewrite), and ``updated``
    moves on edits. Falls back to ``purchased``, then ``added``, when the
    token doesn't parse; None when nothing does."""
    token = item.get("token")
    if isinstance(token, str):
        head = token.split(":", 1)[0]
        if head.isdigit() and int(head) > 0:
            return float(head)
    for key in ("purchased", "added"):
        ts = _parse_date(item.get(key))
        if ts is not None:
            return ts
    return None


def _item(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "tralbum_id": raw.get("tralbum_id"),
        "item_url": raw.get("item_url") or "",
        "item_title": raw.get("item_title") or "",
        "band_name": raw.get("band_name") or "",
        "item_type": raw.get("item_type") or "",
        "added_ts": parse_added_ts(raw),
    }


def _unexpected(message: str) -> http.Unexpected:
    http.count("errors")
    return http.Unexpected(message)


def _parse_fan(html: str, username: str, url: str) -> Dict[str, Any]:
    """fan_id, canonical username and per-list state from a fan page."""
    pagedata = parse_pagedata(html) or {}
    fan_data = pagedata.get("fan_data") or {}
    fan_id = fan_data.get("fan_id")
    if isinstance(fan_id, bool) or not isinstance(fan_id, int) or fan_id <= 0:
        raise _unexpected(f"no fan_id in the pagedata of {url}")
    lists = {}
    for name in LISTS:
        section = pagedata.get(f"{name}_data")
        section = section if isinstance(section, dict) else {}
        count = section.get("item_count")
        count = count if isinstance(count, int) else None
        if count is not None and isinstance(section.get("hidden_items_count"), int):
            # Only used to tell an empty list from a failed read: count
            # conservatively, items the fan hid are not served.
            count -= section["hidden_items_count"]
        lists[name] = {"hidden": section.get("hidden") is True, "item_count": count}
    return {"fan_id": fan_id, "username": fan_data.get("username") or username, "lists": lists}


def _read_list(fan: Dict[str, Any], name: str, referer: str, driver) -> List[Dict[str, Any]]:
    """Every item of one list, newest first. Raises on an error reply, or on
    an empty reply for a list the fan page says has items."""
    token = newest_first_token()
    items: List[Dict[str, Any]] = []
    seen = set()
    for _ in range(_MAX_PAGES):
        data = read_list_page(name, fan["fan_id"], token, PAGE_SIZE, referer=referer, driver=driver)
        batch = data.get("items") if isinstance(data, dict) else None
        if not isinstance(batch, list):
            raise _unexpected(f"no items in the {name} reply for {fan['username']}: {str(data)[:80]}")
        for raw in batch:
            if not isinstance(raw, dict) or not raw.get("tralbum_id"):
                continue
            key = (raw.get("tralbum_type") or raw.get("item_type"), raw["tralbum_id"])
            if key in seen:
                continue
            seen.add(key)
            items.append(_item(raw))
        next_token = data.get("last_token")
        if not data.get("more_available") or not batch or not next_token or next_token == token:
            break
        token = next_token
    else:
        logger.warning("%s's %s still has more after %d pages; returning %d items",
                       fan["username"], name, _MAX_PAGES, len(items))
    advertised = fan["lists"][name]["item_count"]
    if not items and advertised:
        raise _unexpected(f"{fan['username']}'s {name}: the page shows {advertised} items, the API none")
    return items


def _read_lists(fan, names, out, referer, driver) -> None:
    for name in names:
        if name not in out:
            out[name] = [] if fan["lists"][name]["hidden"] else _read_list(fan, name, referer, driver)


def _result(fan, names, out) -> Dict[str, Any]:
    result: Dict[str, Any] = {"fan_id": fan["fan_id"], "username": fan["username"]}
    for name in names:
        result[name] = out[name]
    result["hidden"] = [name for name in names if fan["lists"][name]["hidden"]]
    return result


def fan_library(username: str, *, lists: Sequence[str] = LISTS) -> Dict[str, Any]:
    """A fan's own lists in full, in the API's newest-first order (not
    cached).

    Returns ``{"fan_id", "username", "hidden", "<list>": [item, ...]}`` with
    one key per requested list. Each item is ``{"tralbum_id", "item_url",
    "item_title", "band_name", "item_type", "added_ts"}`` (``added_ts``:
    epoch seconds or None, see :func:`parse_added_ts`; near-monotonic, the
    feed itself has the odd item out of place). ``hidden`` names the
    requested lists the fan keeps private; those are ``[]`` and cost no API
    call. ``username`` is the page's spelling.

    Costs one page read plus one API call per visible list (more past 10000
    items). All or nothing: raises :class:`FanNotFound` for an unknown
    username, :class:`http.Unexpected` for a page or reply without the
    expected data, other :class:`http.FetchError` subclasses when Bandcamp
    can't be read (``http.Challenged`` only if the fallback browser is
    challenged too)."""
    names = tuple(dict.fromkeys(lists))
    if not names or any(name not in API_URLS for name in names):
        raise ValueError(f"lists must be drawn from {LISTS}, got {lists!r}")
    url = fan_page_url(username)
    out: Dict[str, List[Dict[str, Any]]] = {}
    try:
        html: Optional[str] = http.get_page(url, expect=FAN_PAGE_MARKER, cache=False, fallback=False)
    except http.Challenged:
        html = None
    except http.HTTPStatus as e:
        if e.status in (404, 410):
            raise FanNotFound(username) from e
        raise
    if html is not None:
        fan = _parse_fan(html, username, url)
        try:
            _read_lists(fan, names, out, url, None)
            return _result(fan, names, out)
        except http.Challenged:
            from bandcamp_recommender.recommendations import curl_breaker

            if not curl_breaker.should_skip_curl():
                curl_breaker.record_outcome(success=False)
    logger.warning("bot challenge reading %s's lists; using the fallback browser", username)
    with browser.session() as driver:
        fan = _parse_fan(browser.load(driver, url, FAN_PAGE_MARKER), username, url)
        _read_lists(fan, names, out, url, driver)
    return _result(fan, names, out)
