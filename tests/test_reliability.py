"""Reliability tests for the Selenium / scraping surface.

Simulates the failure modes that broke the downstream radio worker: hung page
loads, slow-trickle responses, blocked IPs, leaked drivers. Every test has a
wall-clock budget; if any individual surface call exceeds it, that's a
reliability regression.

The local HTTP server fixture runs on a random port. We never hit Bandcamp.
"""

from __future__ import annotations

import contextlib
import http.server
import os
import socket
import socketserver
import sys
import threading
import time
import unittest
from typing import Optional, Tuple
from unittest.mock import MagicMock, patch

from selenium.common.exceptions import TimeoutException, WebDriverException


# ---------- local HTTP server fixture ----------


class _ReliabilityHandler(http.server.BaseHTTPRequestHandler):
    """Routes:

    /instant     → 200, instant body
    /hang        → accept connection, never write a byte
    /slow        → drip 1 byte / second forever
    /forbidden   → 403
    /reset       → accept then close the socket
    """

    # Silence the per-request stderr log line.
    def log_message(self, format, *args):  # noqa: A002
        return

    def do_GET(self):  # noqa: N802 (BaseHTTPRequestHandler API)
        path = self.path.split("?", 1)[0]
        if path == "/instant":
            body = b"<html><body>ok</body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/forbidden":
            self.send_response(403)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if path == "/reset":
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self.connection.close()
            except OSError:
                pass
            return
        if path == "/hang":
            # Block until the client times out / disconnects. Polling on the
            # socket lets us notice the disconnect promptly so the server can
            # shut down cleanly.
            self.connection.settimeout(0.2)
            while True:
                try:
                    chunk = self.connection.recv(1)
                    if not chunk:
                        return
                except socket.timeout:
                    if getattr(self.server, "_shutdown_flag", False):
                        return
                    continue
                except OSError:
                    return
        if path == "/slow":
            self.send_response(200)
            self.send_header("Content-Type", "audio/mpeg")
            # No Content-Length — keep the client reading forever.
            self.end_headers()
            try:
                while True:
                    self.wfile.write(b"\x00")
                    self.wfile.flush()
                    time.sleep(1.0)
                    if getattr(self.server, "_shutdown_flag", False):
                        return
            except OSError:
                return
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):  # noqa: N802
        # Some callers might POST. Mirror GET semantics.
        return self.do_GET()


class _ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


@contextlib.contextmanager
def local_http_server():
    """Yield a base URL like 'http://127.0.0.1:54321' for the test."""
    server = _ThreadingHTTPServer(("127.0.0.1", 0), _ReliabilityHandler)
    server._shutdown_flag = False
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base_url = f"http://{host}:{port}"
    try:
        yield base_url
    finally:
        server._shutdown_flag = True
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@contextlib.contextmanager
def wall_clock_budget(seconds: float, label: str = "<unlabeled>"):
    """Assert the wrapped block completes inside ``seconds``."""
    start = time.monotonic()
    yield
    elapsed = time.monotonic() - start
    if elapsed > seconds:
        raise AssertionError(
            f"wall-clock budget exceeded for {label}: "
            f"{elapsed:.2f}s > {seconds:.2f}s"
        )


def _make_fake_chrome() -> MagicMock:
    """Return a MagicMock shaped like selenium.webdriver.Chrome."""
    fake = MagicMock(name="FakeChrome")
    fake.current_url = "about:blank"
    fake.page_source = "<html></html>"
    fake.get_cookies.return_value = []
    return fake


# ---------- Phase 1: timeouts on every driver ----------


class DriverManagerTimeoutTests(unittest.TestCase):
    """Every driver this manager creates must have page-load + script timeouts set."""

    def setUp(self):
        # Make sure stale env doesn't leak across tests.
        for var in ("BANDCAMP_PAGE_LOAD_TIMEOUT", "BANDCAMP_SCRIPT_TIMEOUT"):
            os.environ.pop(var, None)

    def test_init_driver_sets_both_timeouts(self):
        from bandcamp_recommender.recommendations import driver_manager

        fake = _make_fake_chrome()
        with patch.object(driver_manager.webdriver, "Chrome", return_value=fake) as ctor:
            dm = driver_manager.DriverManager()
            dm.init_driver()

        ctor.assert_called_once()
        fake.set_page_load_timeout.assert_called_once_with(30)
        fake.set_script_timeout.assert_called_once_with(30)

    def test_create_driver_sets_both_timeouts(self):
        from bandcamp_recommender.recommendations import driver_manager

        fake = _make_fake_chrome()
        with patch.object(driver_manager.webdriver, "Chrome", return_value=fake):
            dm = driver_manager.DriverManager()
            driver = dm.create_driver()

        self.assertIs(driver, fake)
        fake.set_page_load_timeout.assert_called_once_with(30)
        fake.set_script_timeout.assert_called_once_with(30)

    def test_pool_drivers_all_get_timeouts(self):
        from bandcamp_recommender.recommendations import driver_manager

        fakes = [_make_fake_chrome() for _ in range(3)]
        with patch.object(
            driver_manager.webdriver, "Chrome", side_effect=fakes
        ):
            dm = driver_manager.DriverManager()
            pool = dm.get_driver_pool(pool_size=3)

        self.assertEqual(pool.qsize(), 3)
        for fake in fakes:
            fake.set_page_load_timeout.assert_called_once_with(30)
            fake.set_script_timeout.assert_called_once_with(30)

    def test_env_overrides_timeout_values(self):
        from bandcamp_recommender.recommendations import driver_manager

        with patch.dict(
            os.environ,
            {
                "BANDCAMP_PAGE_LOAD_TIMEOUT": "12",
                "BANDCAMP_SCRIPT_TIMEOUT": "7",
            },
        ):
            fake = _make_fake_chrome()
            with patch.object(driver_manager.webdriver, "Chrome", return_value=fake):
                dm = driver_manager.DriverManager()
                dm.init_driver()

        fake.set_page_load_timeout.assert_called_once_with(12)
        fake.set_script_timeout.assert_called_once_with(7)


# ---------- Phase 1 + 2: the fallback browser is one shared, bounded driver ----------


class SharedBrowserTests(unittest.TestCase):
    def setUp(self):
        from bandcamp_recommender import browser, http as bchttp

        browser.close()
        self.addCleanup(browser.close)
        patcher = patch.dict(os.environ, {"BANDCAMP_MIN_REQUEST_INTERVAL": "0"})
        patcher.start()
        bchttp.reset()
        self.addCleanup(bchttp.reset)
        self.addCleanup(patcher.stop)

    def test_failed_loads_reuse_one_driver_and_close_quits_it(self):
        from bandcamp_recommender import browser
        from bandcamp_recommender.recommendations import driver_manager, scraper

        fake = _make_fake_chrome()
        fake.get.side_effect = TimeoutException("simulated page-load timeout")
        with patch.object(driver_manager.webdriver, "Chrome", return_value=fake) as chrome:
            with wall_clock_budget(2.0, "scraper fallback on hang"):
                self.assertIsNone(scraper._fetch_page_with_selenium("http://localhost/a"))
                self.assertIsNone(scraper._fetch_page_with_selenium("http://localhost/b"))
            self.assertEqual(chrome.call_count, 1, "fallback must not start a Chrome per page")
            browser.close()
        fake.quit.assert_called_once()
        self.assertIsNone(browser._driver)

    def test_dead_driver_is_quit_and_replaced(self):
        from bandcamp_recommender import browser
        from bandcamp_recommender.recommendations import driver_manager

        dead = _make_fake_chrome()
        type(dead).current_url = property(lambda self: (_ for _ in ()).throw(WebDriverException("gone")))
        alive = _make_fake_chrome()
        alive.page_source = "<html>marker</html>"
        with patch.object(driver_manager.webdriver, "Chrome", side_effect=[dead, alive]):
            with browser.session():
                pass  # first session hands out `dead`; liveness check on exit quits it
            self.assertEqual(browser.fetch_html("http://localhost/x", "marker"), "<html>marker</html>")
        dead.quit.assert_called()

    def test_busy_browser_times_out_instead_of_blocking_forever(self):
        from bandcamp_recommender import browser, http as bchttp
        from bandcamp_recommender.recommendations import driver_manager

        with patch.dict(os.environ, {"BANDCAMP_BROWSER_ACQUIRE_TIMEOUT": "0.2"}), \
             patch.object(driver_manager.webdriver, "Chrome", return_value=_make_fake_chrome()):
            with browser.session():
                errors = []

                def other():
                    try:
                        with browser.session():
                            pass
                    except bchttp.FetchError as e:
                        errors.append(e)

                t = threading.Thread(target=other)
                t.start()
                t.join(timeout=3)
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], browser.BrowserUnavailable)


# ---------- HTTP path timeouts, against the local server ----------


class HttpPathReliabilityTests(unittest.TestCase):
    def setUp(self):
        os.environ["BANDCAMP_CURL_BREAKER_DISABLED"] = "1"
        self.addCleanup(lambda: os.environ.pop("BANDCAMP_CURL_BREAKER_DISABLED", None))
        from bandcamp_recommender import http as bchttp

        bchttp.reset()
        bchttp.clear_page_cache()
        self.addCleanup(bchttp.reset)

    def test_fetch_page_html_against_hang_returns_under_budget(self):
        from bandcamp_recommender.recommendations import scraper

        with local_http_server() as base:
            with wall_clock_budget(5.0, "fetch_page_html against /hang"):
                result = scraper.fetch_page_html(f"{base}/hang", timeout=2, expect=None)
        self.assertIsNone(result)

    def test_fetch_page_html_against_instant_returns_body(self):
        from bandcamp_recommender.recommendations import scraper

        with local_http_server() as base:
            with wall_clock_budget(2.0, "fetch_page_html against /instant"):
                result = scraper.fetch_page_html(f"{base}/instant", timeout=2, expect=None)
        self.assertIn("ok", result)

    def test_page_without_expected_marker_is_none(self):
        from bandcamp_recommender.recommendations import scraper

        with local_http_server() as base:
            self.assertIsNone(scraper.fetch_page_html(f"{base}/instant", timeout=2))

    def test_forbidden_is_none_and_not_retried_forever(self):
        from bandcamp_recommender.recommendations import scraper

        with local_http_server() as base:
            with wall_clock_budget(2.0, "403"):
                self.assertIsNone(scraper.fetch_page_html(f"{base}/forbidden", timeout=2, expect=None))

    def test_transport_error_retries_then_gives_up(self):
        import requests
        from bandcamp_recommender import http as bchttp

        session, _ = bchttp._state()
        with patch.object(session, "request", side_effect=requests.ConnectionError("reset")) as req:
            with wall_clock_budget(4.0, "transport retries"):
                with self.assertRaises(bchttp.FetchError):
                    bchttp.get_html("http://127.0.0.1:9/x", timeout=1)
        self.assertEqual(req.call_count, 3)  # 1 + 2 backoffs


# ---------- Phase 3: audio download ----------


class AudioDownloadTimeoutTests(unittest.TestCase):
    def setUp(self):
        os.environ.pop("BANDCAMP_AUDIO_TIMEOUT", None)

    def test_download_against_hang_under_budget(self):
        from bandcamp_recommender.recommendations import bpm

        with local_http_server() as base:
            with wall_clock_budget(4.0, "_download_audio_bytes against /hang"):
                result = bpm._download_audio_bytes(
                    f"{base}/hang", max_bytes=1024, timeout=2
                )

        self.assertIsNone(result)

    def test_download_against_slow_trickle_under_budget(self):
        from bandcamp_recommender.recommendations import bpm

        with local_http_server() as base:
            with wall_clock_budget(4.0, "_download_audio_bytes against /slow"):
                result = bpm._download_audio_bytes(
                    f"{base}/slow", max_bytes=4096, timeout=2
                )

        # Either we got nothing (timeout fired during connect/header) or we
        # got at most a couple of bytes (slow trickle was killed mid-read).
        if result is not None:
            self.assertLessEqual(len(result), 16)

    def test_default_audio_timeout_is_thirty_seconds(self):
        from bandcamp_recommender.recommendations import bpm

        # The default value lives on the function signature.
        defaults = bpm._download_audio_bytes.__defaults__
        # signature: (max_bytes=..., timeout=...)
        self.assertIn(30, defaults, msg=f"expected 30 in defaults, got {defaults!r}")


# ---------- Phase 2: orphan reaper ----------


class OrphanReaperTests(unittest.TestCase):
    @unittest.skipIf(sys.platform.startswith("win"), "POSIX only")
    def test_reap_orphans_is_safe_no_op_when_none_present(self):
        from bandcamp_recommender.recommendations import driver_manager

        dm = driver_manager.DriverManager()
        # Should not raise and should return an int.
        count = dm.reap_orphans()
        self.assertIsInstance(count, int)
        self.assertGreaterEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
