"""Shared HTTP client for every Bandcamp page and API request.

Why this exists
---------------
The previous transport shelled out to ``curl`` with a spoofed desktop-Chrome
User-Agent. Bandcamp's CDN (Fastly) answers a client that *claims* to be a
browser but speaks HTTP/2 with curl's fingerprint with a ~3 KB JavaScript
challenge page, **with HTTP 200**. That page looked like a successful fetch:
it was cached, every parser silently found nothing, and callers fell back to
a full Chrome session per request (minutes per call, GBs of RAM on a
server), while audio-feature hydration silently returned nothing. Verified
on a datacenter box: the same URL with the same spoofed UA passes over
HTTP/1.1, and an honest UA passes over both.

The systemic lesson is that success was judged by the transport, not the
payload. So this client:

* speaks HTTP/1.1 (``requests``) with an honest User-Agent;
* validates the payload: ``get_html(url, expect=...)`` only returns a page
  that contains the marker the caller needs; anything else is a typed
  failure (:class:`Challenged` when it is the bot-challenge page);
* bounds every request by a wall-clock deadline (``requests``' own timeout
  is per socket operation, so a slow trickle could hold a slot for minutes);
* paces requests to Bandcamp (its origin rate-limits bursts with
  ``429 Retry-After: 3``), backing off multiplicatively on 429 and easing
  back on success; retries live only here;
* counts outcomes (:func:`snapshot`) so callers can log what a unit of work
  cost and spot a silent degradation.

Audio previews (``*.bcbits.com`` CDN) are not paced.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests
import urllib3

logger = logging.getLogger(__name__)

_DEFAULT_USER_AGENT = "bandcamp-recommender (+https://github.com/lucweytingh/bandcamp_recommender)"
# Measured from a datacenter IP (2026-10-24): 100 sequential fan-page GETs
# at 1.0 s spacing -> 0 x 429; at 0.6 s the first 429 lands at request ~96,
# i.e. a ~60/min bucket with a ~40-request burst.
_DEFAULT_MIN_INTERVAL_S = 1.0
_MAX_INTERVAL_S = 3.0
_DEFAULT_MAX_CONCURRENCY = 3
_MAX_RETRY_AFTER_S = 10.0
_MAX_429_ATTEMPTS = 3
_TRANSPORT_BACKOFF_S = (0.4, 1.2)
# A full fan collection is ~2.5 KB/item (measured); the API's 10000-item
# maximum is ~25 MB. Leave headroom.
_MAX_BODY_BYTES = 64 * 1024 * 1024
_CHUNK = 64 * 1024

# Fastly's JS challenge serves its assets from /_fs-ch-<token>/. Only used to
# LABEL a failed validation; validation itself is the caller's `expect`.
_CHALLENGE_MARKER = "/_fs-ch-"


class FetchError(Exception):
    """A Bandcamp request did not produce usable content."""


class Challenged(FetchError):
    """Bandcamp served its bot-challenge page instead of the content."""


class RateLimited(FetchError):
    """Bandcamp kept answering 429 after the retry budget."""


class Timeout(FetchError):
    """The request exceeded its wall-clock deadline."""


class HTTPStatus(FetchError):
    def __init__(self, status: int, url: str):
        super().__init__(f"HTTP {status}: {url}")
        self.status = status


class Unexpected(FetchError):
    """A 200 whose body lacks the content the caller asked for."""


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def is_challenge_page(body: str) -> bool:
    return _CHALLENGE_MARKER in body[:8192]


# ---- counters ---------------------------------------------------------------

_COUNTER_NAMES = (
    "requests", "challenged", "rate_limited", "timeouts", "errors", "selenium_fallbacks",
    # How supporter collections were read: one API call (fan_id known) vs
    # fan page + API. A rising supporter_page share doubles cold-seed cost.
    "supporter_api", "supporter_page", "supporter_cached",
)
_counters: Dict[str, int] = {k: 0 for k in _COUNTER_NAMES}
_counters_lock = threading.Lock()


def count(name: str, n: int = 1) -> None:
    with _counters_lock:
        _counters[name] = _counters.get(name, 0) + n


def snapshot() -> Dict[str, int]:
    """Monotonic process-wide counters; diff two snapshots for a unit of work."""
    with _counters_lock:
        return dict(_counters)


def diff(before: Dict[str, int], after: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    after = after if after is not None else snapshot()
    return {k: after.get(k, 0) - before.get(k, 0) for k in after}


# ---- pacing -----------------------------------------------------------------

class Pacer:
    """Process-wide politeness gate for Bandcamp.

    * request *starts* spaced at least ``interval`` apart: a reservation on
      ``next_allowed`` is taken under the lock and slept outside it, BEFORE
      taking a concurrency slot, so a long backoff never idles a slot;
    * at most ``max_concurrency`` requests in flight;
    * AIMD: a 429 widens the interval x1.5 (capped) and pauses every thread
      until its Retry-After; each success shrinks it additively back toward
      the floor (multiplicative easing saw-tooths straight back into the
      limiter; a tiny step strands the pacer at the cap for minutes).
    """

    ADDITIVE_DECREASE_S = 0.1
    BACKOFF_FACTOR = 1.5

    def __init__(self, min_interval: float, max_concurrency: int):
        self.min_interval = max(0.0, min_interval)
        self.interval = self.min_interval
        self._slots = threading.BoundedSemaphore(max(1, max_concurrency))
        self._lock = threading.Lock()
        self._next_allowed = 0.0
        self._paused_until = 0.0

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                start = max(now, self._next_allowed, self._paused_until)
                self._next_allowed = start + self.interval
            if start > now:
                time.sleep(start - now)
            self._slots.acquire()
            with self._lock:
                paused = self._paused_until > time.monotonic()
            if not paused:
                return
            # A 429 landed while we waited: give the slot back and re-reserve.
            self._slots.release()

    def release(self) -> None:
        self._slots.release()

    def on_success(self) -> None:
        with self._lock:
            self.interval = max(self.min_interval, self.interval - self.ADDITIVE_DECREASE_S)

    def on_rate_limited(self, retry_after: float) -> None:
        with self._lock:
            self.interval = min(
                max(_MAX_INTERVAL_S, self.min_interval),
                max(self.interval * self.BACKOFF_FACTOR, self.min_interval, 0.1),
            )
            self._paused_until = max(self._paused_until, time.monotonic() + retry_after)


_session: Optional[requests.Session] = None
_pacer: Optional[Pacer] = None
_init_lock = threading.Lock()


def _state():
    global _session, _pacer
    with _init_lock:
        if _session is None:
            concurrency = int(_env_float("BANDCAMP_MAX_CONCURRENCY", _DEFAULT_MAX_CONCURRENCY))
            session = requests.Session()
            # Redirects are followed inside ONE paced slot. Some custom artist
            # domains ping-pong forever with their *.bandcamp.com twin
            # (303 -> 301 -> ...); the default of 30 turned each into a
            # 30-request burst against a ~1 req/s limit.
            session.max_redirects = 5
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=16, pool_maxsize=max(8, concurrency * 2)
            )
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            session.headers.update({
                "User-Agent": os.environ.get("BANDCAMP_USER_AGENT", _DEFAULT_USER_AGENT),
                "Accept-Language": "en-US,en;q=0.8",
            })
            _pacer = Pacer(
                _env_float("BANDCAMP_MIN_REQUEST_INTERVAL", _DEFAULT_MIN_INTERVAL_S),
                concurrency,
            )
            _session = session
        return _session, _pacer


def pacer() -> Pacer:
    """The shared pacer, for non-HTTP Bandcamp loads (Selenium) to take a slot."""
    return _state()[1]


def reset() -> None:
    """Drop the shared session + pacer (tests, or after an env change)."""
    global _session, _pacer
    with _init_lock:
        if _session is not None:
            _session.close()
        _session = None
        _pacer = None


def _is_paced(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return not (host == "bcbits.com" or host.endswith(".bcbits.com"))


def _retry_after_seconds(resp: requests.Response) -> float:
    try:
        value = float(resp.headers.get("Retry-After", "3"))
    except ValueError:
        value = 3.0
    return min(max(value, 0.5), _MAX_RETRY_AFTER_S)


class Response:
    """A fully read response: status, headers, bytes, and decoded text."""

    def __init__(self, status: int, headers, content: bytes, encoding: Optional[str]):
        self.status = status
        self.headers = headers
        self.content = content
        self._encoding = encoding

    @property
    def text(self) -> str:
        # requests falls back to ISO-8859-1 for text/* without a charset,
        # which turns UTF-8 titles into mojibake (read aloud by TTS
        # downstream). Bandcamp serves UTF-8; trust a declared charset only.
        return self.content.decode(self._encoding or "utf-8", errors="replace")

    def json(self) -> Any:
        return json.loads(self.content)


def _read_body(resp: requests.Response, deadline: float, url: str) -> bytes:
    """Read the whole body, bounded by ``deadline``. ``read1`` returns what
    has arrived instead of blocking for a full chunk, so a slow trickle can't
    outlive the deadline (``iter_content`` would wait for 64 KB)."""
    raw = resp.raw
    buf = bytearray()
    while True:
        if time.monotonic() > deadline:
            raise Timeout(f"wall-clock deadline exceeded reading {url}")
        # Reading `raw` bypasses requests' exception translation: map
        # urllib3/socket errors onto our typed errors so a mid-body stall,
        # reset or bad gzip is a counted FetchError, never an escape.
        try:
            chunk = raw.read1(_CHUNK, decode_content=True)
        except urllib3.exceptions.ReadTimeoutError as e:
            raise Timeout(f"read timeout mid-body: {url}") from e
        except (urllib3.exceptions.HTTPError, OSError) as e:
            raise FetchError(f"transport error mid-body: {url}: {e}") from e
        if not chunk:
            return bytes(buf)
        buf.extend(chunk)
        if len(buf) > _MAX_BODY_BYTES:
            raise FetchError(f"body over {_MAX_BODY_BYTES} bytes: {url}")


def _declared_charset(resp: requests.Response) -> Optional[str]:
    ctype = resp.headers.get("Content-Type", "")
    for part in ctype.split(";")[1:]:
        key, _, value = part.strip().partition("=")
        if key.lower() == "charset" and value:
            return value.strip("\"'")
    return None


def _attempt(session, gate: Optional[Pacer], method: str, url: str,
             timeout: float, kwargs: Dict[str, Any]):
    """One request holding one pacer slot for exactly its own duration.
    Returns ``(status, Response | retry_after_seconds)``."""
    if gate is not None:
        gate.acquire()
    try:
        count("requests")
        deadline = time.monotonic() + timeout
        # (connect, read-gap) bound each socket op; the deadline bounds the
        # whole body read.
        with session.request(method, url, timeout=(min(10.0, timeout), timeout),
                             stream=True, **kwargs) as resp:
            if resp.status_code == 429:
                return 429, _retry_after_seconds(resp)
            content = _read_body(resp, deadline, url)
            return resp.status_code, Response(
                resp.status_code, resp.headers, content, _declared_charset(resp)
            )
    finally:
        if gate is not None:
            gate.release()


def request(method: str, url: str, *, timeout: float, **kwargs: Any) -> Response:
    """Paced request with 429 + transient-transport retries, each attempt
    bounded by a wall-clock ``timeout``. Returns the final non-429 response.
    Raises :class:`RateLimited`, :class:`Timeout` or :class:`FetchError`.
    Timeouts are not retried: ``timeout`` is the caller's budget."""
    session, shared_gate = _state()
    gate = shared_gate if _is_paced(url) else None
    transport_failures = 0
    rate_limited = 0
    while True:
        try:
            status, result = _attempt(session, gate, method, url, timeout, kwargs)
        except Timeout:
            count("timeouts")
            raise
        except FetchError:
            count("errors")
            raise
        except requests.Timeout as e:
            count("timeouts")
            raise Timeout(f"timeout after {timeout}s: {url}") from e
        except requests.TooManyRedirects as e:
            count("errors")  # deterministic: a loop doesn't heal on retry
            raise FetchError(f"redirect loop: {url}") from e
        except requests.RequestException as e:
            if transport_failures >= len(_TRANSPORT_BACKOFF_S):
                count("errors")
                raise FetchError(f"transport error: {url}: {e}") from e
            time.sleep(_TRANSPORT_BACKOFF_S[transport_failures])
            transport_failures += 1
            continue
        if status != 429:
            if gate is not None:
                gate.on_success()
            return result
        count("rate_limited")
        rate_limited += 1
        logger.warning("429 from %s, backing off %.1fs (attempt %d/%d)",
                       urlparse(url).hostname, result, rate_limited, _MAX_429_ATTEMPTS)
        if rate_limited >= _MAX_429_ATTEMPTS:
            raise RateLimited(f"still 429 after {_MAX_429_ATTEMPTS} attempts: {url}")
        if gate is not None:
            # Pushes next_allowed for every thread; our next acquire waits it out.
            gate.on_rate_limited(result)
        else:
            time.sleep(result)


def get_html(url: str, expect: Optional[str] = None, timeout: float = 20) -> str:
    """GET a Bandcamp page and validate it contains ``expect`` (e.g.
    ``"data-tralbum"`` for an item page, ``"pagedata"`` for a fan page).

    Raises :class:`Challenged` for the bot-challenge page, :class:`Unexpected`
    for any other 200 without the marker, :class:`HTTPStatus` for non-200."""
    resp = request("GET", url, timeout=timeout)
    if resp.status != 200:
        count("errors")
        raise HTTPStatus(resp.status, url)
    body = resp.text
    if is_challenge_page(body):
        count("challenged")
        raise Challenged(url)
    if expect is not None and expect not in body:
        count("errors")
        raise Unexpected(f"no {expect!r} in {url}")
    return body


def post_json(url: str, payload: Dict[str, Any], timeout: float = 60,
              referer: Optional[str] = None) -> Any:
    """POST a JSON payload to a Bandcamp API endpoint and decode the reply."""
    headers = {"Referer": referer} if referer else None
    resp = request("POST", url, timeout=timeout, json=payload, headers=headers)
    if resp.status != 200:
        count("errors")
        raise HTTPStatus(resp.status, url)
    try:
        return resp.json()
    except ValueError as e:
        if is_challenge_page(resp.text):
            count("challenged")
            raise Challenged(url) from e
        count("errors")
        raise Unexpected(f"non-JSON reply from {url}") from e


# ---- page cache + fallback policy -------------------------------------------
#
# One policy for every Bandcamp page read, used by the library's scrapers and
# by consumers (e.g. a radio resolving rec pages), so a page fetched by the
# recommender is not fetched again moments later, and a challenged fetch
# degrades the same way everywhere.

_DEFAULT_PAGE_CACHE_ENTRIES = 256
_DEFAULT_PAGE_CACHE_BYTES = 32 * 1024 * 1024
# Pages embed signed, expiring bcbits stream URLs: keep entries short-lived.
_DEFAULT_PAGE_CACHE_TTL_S = 600.0

_page_cache: "OrderedDict[str, tuple]" = OrderedDict()
_page_cache_bytes = 0
_page_cache_lock = threading.Lock()


def _page_cache_limits():
    return (
        max(0, int(_env_float("BANDCAMP_PAGE_CACHE_SIZE", _DEFAULT_PAGE_CACHE_ENTRIES))),
        max(0, int(_env_float("BANDCAMP_PAGE_CACHE_BYTES", _DEFAULT_PAGE_CACHE_BYTES))),
        _env_float("BANDCAMP_PAGE_CACHE_TTL_S", _DEFAULT_PAGE_CACHE_TTL_S),
    )


def _page_cache_get(url: str) -> Optional[str]:
    with _page_cache_lock:
        entry = _page_cache.get(url)
        if entry is None:
            return None
        expires, body = entry
        if time.monotonic() > expires:
            _page_cache_drop(url)
            return None
        _page_cache.move_to_end(url)
        return body


def _page_cache_drop(url: str) -> None:
    global _page_cache_bytes
    entry = _page_cache.pop(url, None)
    if entry is not None:
        _page_cache_bytes -= len(entry[1])


def _page_cache_put(url: str, body: str) -> None:
    global _page_cache_bytes
    max_entries, max_bytes, ttl = _page_cache_limits()
    if max_entries <= 0 or max_bytes <= 0 or ttl <= 0 or len(body) > max_bytes:
        return
    with _page_cache_lock:
        _page_cache_drop(url)
        _page_cache[url] = (time.monotonic() + ttl, body)
        _page_cache_bytes += len(body)
        while _page_cache and (len(_page_cache) > max_entries or _page_cache_bytes > max_bytes):
            oldest = next(iter(_page_cache))
            _page_cache_drop(oldest)


_FAILURE_TTL_S = 60.0
_failures: Dict[str, tuple] = {}


def _remember_failure(url: str, error: FetchError) -> None:
    with _page_cache_lock:
        _failures[url] = (time.monotonic() + _FAILURE_TTL_S, error)
        if len(_failures) > 512:
            now = time.monotonic()
            for k in [k for k, (exp, _) in _failures.items() if exp < now]:
                del _failures[k]


def _recent_failure(url: str) -> Optional[FetchError]:
    with _page_cache_lock:
        entry = _failures.get(url)
        if entry is None:
            return None
        if time.monotonic() > entry[0]:
            del _failures[url]
            return None
        return entry[1]


def clear_page_cache() -> None:
    global _page_cache_bytes
    with _page_cache_lock:
        _page_cache.clear()
        _failures.clear()
        _page_cache_bytes = 0


def get_page(url: str, expect: Optional[str] = None, timeout: float = 20,
             cache: bool = True, fallback: bool = True) -> str:
    """The standard way to read a Bandcamp page.

    Plain HTTP first; on :class:`Challenged`, the shared fallback browser
    (one Chrome, serialized). While the challenge breaker is tripped, plain
    HTTP is skipped. Real pages are cached for a few minutes (``cache``).
    Raises :class:`FetchError` subclasses; never returns a junk page."""
    from bandcamp_recommender.recommendations import curl_breaker

    if cache:
        cached = _page_cache_get(url)
        if cached is not None and (expect is None or expect in cached):
            return cached
    failed = _recent_failure(url)
    if failed is not None:
        # Asked again moments after it failed (e.g. tags then audio for the
        # same candidate): don't spend another paced request or browser load.
        raise failed

    body: Optional[str] = None
    if not curl_breaker.should_skip_curl():
        try:
            body = get_html(url, expect=expect, timeout=timeout)
            curl_breaker.record_outcome(success=True)
        except Challenged as e:
            curl_breaker.record_outcome(success=False)
            if not fallback:
                _remember_failure(url, e)
                raise
            logger.warning("bot challenge on %s; using the fallback browser", url)
        except FetchError as e:
            _remember_failure(url, e)
            raise
    elif not fallback:
        raise Challenged(f"challenge breaker tripped: {url}")

    try:
        if body is None:
            from bandcamp_recommender import browser

            body = browser.fetch_html(url, expect)
    except FetchError as e:
        _remember_failure(url, e)
        raise
    if cache:
        _page_cache_put(url, body)
    return body
