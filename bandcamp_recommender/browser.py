"""The one Selenium browser used as a fallback when Bandcamp challenges
plain HTTP (see :mod:`bandcamp_recommender.http`).

Before this module, every fallback path made its own Chrome: a fresh
``DriverManager`` per page in the scraper, a driver pool per recommender
call, one per caller elsewhere. On a small server that meant several
1+ GB Chromes alive at once whenever the fallback fired. Now there is one
process-wide driver:

* created lazily on first use, serialized by a lock (acquire with a
  timeout, so a wedged page can't stall every thread forever);
* recreated if it died;
* quit by :func:`close` / :func:`close_if_idle` (long-running callers should
  call one of them between units of work; :mod:`atexit` covers the rest).

Lock order: the driver lock is always taken BEFORE a pacer slot, never while
holding one.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import os
import threading
import time
from typing import Any, Dict, Iterator, Optional

from bandcamp_recommender import http

logger = logging.getLogger(__name__)

_DEFAULT_ACQUIRE_TIMEOUT_S = 180.0
_EXPECT_POLL_S = 0.25
_EXPECT_WAIT_S = 6.0

_lock = threading.Lock()
_driver = None
_last_used = 0.0


class BrowserUnavailable(http.FetchError):
    """The fallback browser could not be started or acquired in time."""


def _acquire_timeout() -> float:
    try:
        return float(os.environ.get("BANDCAMP_BROWSER_ACQUIRE_TIMEOUT", _DEFAULT_ACQUIRE_TIMEOUT_S))
    except ValueError:
        return _DEFAULT_ACQUIRE_TIMEOUT_S


def _quit(driver) -> None:
    try:
        driver.quit()
    except Exception:
        logger.warning("fallback browser quit failed", exc_info=os.environ.get("BANDCAMP_DEBUG") == "1")


@contextlib.contextmanager
def session() -> Iterator[Any]:
    """Exclusive use of the shared driver for a sequence of steps that must
    stay on one page (e.g. load a fan page, then call the API from it)."""
    global _driver, _last_used
    if not _lock.acquire(timeout=_acquire_timeout()):
        raise BrowserUnavailable("fallback browser busy")
    try:
        from bandcamp_recommender.recommendations.driver_manager import DriverManager

        if _driver is not None and not DriverManager.is_driver_alive(_driver):
            _quit(_driver)
            _driver = None
        if _driver is None:
            try:
                _driver = DriverManager().create_driver()
            except Exception as e:
                raise BrowserUnavailable(f"could not start the fallback browser: {e}") from e
        try:
            yield _driver
        finally:
            _last_used = time.monotonic()
            if not DriverManager.is_driver_alive(_driver):
                _quit(_driver)
                _driver = None
    finally:
        _lock.release()


def load(driver, url: str, expect: Optional[str]) -> str:
    """Navigate (holding a pacer slot: same origin as plain HTTP) and return
    the page source once ``expect`` appears. Raises :class:`http.Challenged`
    if the browser itself is challenged, :class:`http.Unexpected` if the
    marker never shows."""
    http.count("selenium_fallbacks")
    gate = http.pacer()
    gate.acquire()
    try:
        driver.get(url)
    except Exception as e:
        raise http.FetchError(f"browser load failed for {url}: {e}") from e
    finally:
        gate.release()
    deadline = time.monotonic() + _EXPECT_WAIT_S
    while True:
        source = driver.page_source
        if expect is None or expect in source:
            if not http.is_challenge_page(source):
                return source
        if time.monotonic() > deadline:
            if http.is_challenge_page(source):
                http.count("challenged")
                raise http.Challenged(url)
            raise http.Unexpected(f"no {expect!r} in browser page {url}")
        time.sleep(_EXPECT_POLL_S)


_FETCH_JS = """
var callback = arguments[arguments.length - 1];
fetch(arguments[0], {method: 'POST', headers: {'Content-Type': 'application/json'},
                     body: arguments[1], credentials: 'include'})
  .then(function(r) { return r.text().then(function(t) { callback([r.status, t]); }); })
  .catch(function(e) { callback([0, String(e)]); });
"""


def post_json(driver, url: str, payload: Dict[str, Any], timeout: float = 60) -> Any:
    """POST from inside the current page (same-origin cookies), for when the
    API also refuses plain HTTP. The driver must be on a bandcamp.com page."""
    http.count("selenium_fallbacks")
    gate = http.pacer()
    gate.acquire()
    try:
        driver.set_script_timeout(timeout)
        status, body = driver.execute_async_script(_FETCH_JS, url, json.dumps(payload))
    except Exception as e:
        raise http.FetchError(f"in-browser fetch failed for {url}: {e}") from e
    finally:
        gate.release()
    if status != 200:
        raise http.HTTPStatus(int(status or 0), url)
    try:
        return json.loads(body)
    except ValueError as e:
        raise http.Unexpected(f"non-JSON in-browser reply from {url}") from e


def fetch_html(url: str, expect: Optional[str]) -> str:
    with session() as driver:
        return load(driver, url, expect)


def close() -> None:
    """Quit the shared driver now (waits for an in-flight user)."""
    global _driver
    with _lock:
        if _driver is not None:
            _quit(_driver)
            _driver = None


def close_if_idle(max_idle_s: float) -> bool:
    """Quit the driver if nobody used it for ``max_idle_s``. Never blocks."""
    global _driver
    if not _lock.acquire(blocking=False):
        return False
    try:
        if _driver is not None and time.monotonic() - _last_used >= max_idle_s:
            _quit(_driver)
            _driver = None
            return True
        return False
    finally:
        _lock.release()


def is_running() -> bool:
    return _driver is not None


atexit.register(close)
