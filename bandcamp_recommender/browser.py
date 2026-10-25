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
* quit once nobody used it for ``BANDCAMP_BROWSER_IDLE_S`` (default 120 s,
  0 = never), by :func:`close` / :func:`close_if_idle`, or at exit;
* cleaned up after: each driver runs on its own library-owned profile, and
  quitting it kills whatever still runs on that profile and deletes it.
  :func:`reap_orphans` does the same for drivers of library processes that
  died. See :mod:`bandcamp_recommender.chrome_procs`.

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

from bandcamp_recommender import chrome_procs, http

logger = logging.getLogger(__name__)

_DEFAULT_ACQUIRE_TIMEOUT_S = 180.0
_DEFAULT_IDLE_S = 120.0
_EXPECT_POLL_S = 0.25
_EXPECT_WAIT_S = 6.0

_lock = threading.Lock()
_driver = None
_last_used = 0.0
_idle_wake = threading.Event()
_idle_thread: Optional[threading.Thread] = None

# Kill Chrome left behind by library processes that died; spares live ones.
# Runs by itself once per process (first Bandcamp request or driver start).
reap_orphans = chrome_procs.reap_orphans


class BrowserUnavailable(http.FetchError):
    """The fallback browser could not be started or acquired in time."""


def _acquire_timeout() -> float:
    try:
        return float(os.environ.get("BANDCAMP_BROWSER_ACQUIRE_TIMEOUT", _DEFAULT_ACQUIRE_TIMEOUT_S))
    except ValueError:
        return _DEFAULT_ACQUIRE_TIMEOUT_S


def _idle_timeout() -> float:
    try:
        return float(os.environ.get("BANDCAMP_BROWSER_IDLE_S", _DEFAULT_IDLE_S))
    except ValueError:
        return _DEFAULT_IDLE_S


def _drop() -> None:
    """Quit the shared driver and forget it. Caller holds ``_lock``."""
    global _driver
    chrome_procs.quit_driver(_driver)
    _driver = None


def _start_idle_watch() -> None:
    """Caller holds ``_lock``. One watcher thread per process, started with
    the first driver; it sleeps until a session ends when there's nothing
    to watch."""
    global _idle_thread
    if _idle_thread is None:
        _idle_thread = threading.Thread(target=_idle_watch, name="bandcamp-browser-idle", daemon=True)
        _idle_thread.start()


def _idle_watch() -> None:
    """Quit the shared driver once nobody has used it for the idle timeout."""
    timeout: Optional[float] = None
    while True:
        _idle_wake.wait(timeout)
        # Cleared before reading the state, so a session ending from here on
        # sets it again: no lost wake-up.
        _idle_wake.clear()
        idle_s = _idle_timeout()
        if idle_s <= 0 or _driver is None:
            timeout = None
            continue
        left = _last_used + idle_s - time.monotonic()
        if left > 0:
            timeout = left
            continue
        timeout = None  # quit now, or in use: the end of that session wakes us
        if not _lock.acquire(blocking=False):
            continue
        try:
            if _driver is not None and time.monotonic() - _last_used >= idle_s:
                _drop()
                logger.info("closed the fallback browser after %.0f s idle", idle_s)
        finally:
            _lock.release()


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
            _drop()
        if _driver is None:
            try:
                _driver = DriverManager().create_driver()
            except Exception as e:
                raise BrowserUnavailable(f"could not start the fallback browser: {e}") from e
            _start_idle_watch()
        try:
            yield _driver
        finally:
            if not DriverManager.is_driver_alive(_driver):
                _drop()
            _last_used = time.monotonic()
            _idle_wake.set()
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
        try:
            source = driver.page_source
        except Exception as e:  # e.g. Chrome OOM-killed under its scope cap
            raise http.FetchError(f"browser died reading {url}: {e}") from e
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
    with _lock:
        if _driver is not None:
            _drop()


def close_if_idle() -> bool:
    """Quit the driver unless someone is using it right now. Never blocks;
    True if a driver was quit."""
    if not _lock.acquire(blocking=False):
        return False
    try:
        if _driver is None:
            return False
        _drop()
        return True
    finally:
        _lock.release()


atexit.register(close)
