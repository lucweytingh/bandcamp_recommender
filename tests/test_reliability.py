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


# ---------- Phase 1 + 2: scraper Selenium fallback never leaks a driver ----------


class ScraperSeleniumFallbackTests(unittest.TestCase):
    def test_fallback_quits_driver_on_timeout(self):
        from bandcamp_recommender.recommendations import driver_manager, scraper

        fake = _make_fake_chrome()
        fake.get.side_effect = TimeoutException("simulated page-load timeout")

        with patch.object(driver_manager.webdriver, "Chrome", return_value=fake):
            with wall_clock_budget(2.0, "scraper fallback on hang"):
                result = scraper._fetch_page_with_selenium("http://localhost/anything")

        self.assertIsNone(result)
        fake.quit.assert_called()

    def test_fallback_quits_driver_on_unexpected_exception(self):
        from bandcamp_recommender.recommendations import driver_manager, scraper

        fake = _make_fake_chrome()
        fake.get.side_effect = WebDriverException("simulated webdriver crash")

        with patch.object(driver_manager.webdriver, "Chrome", return_value=fake):
            result = scraper._fetch_page_with_selenium("http://localhost/anything")

        self.assertIsNone(result)
        fake.quit.assert_called()


# ---------- curl-side timeouts (already in place, verify against the local server) ----------


class CurlPathReliabilityTests(unittest.TestCase):
    def setUp(self):
        # Don't let stale breaker state in the user's cache trip us.
        os.environ["BANDCAMP_CURL_BREAKER_DISABLED"] = "1"
        self.addCleanup(lambda: os.environ.pop("BANDCAMP_CURL_BREAKER_DISABLED", None))

    def test_fetch_page_html_against_hang_returns_under_budget(self):
        from bandcamp_recommender.recommendations import scraper

        with local_http_server() as base:
            # Hard timeouts are not retried — single 2 s call, returns None
            # well under the budget.
            with wall_clock_budget(4.0, "fetch_page_html against /hang"):
                result = scraper.fetch_page_html(f"{base}/hang", timeout=2)

        self.assertIsNone(result)

    def test_fetch_page_html_against_instant_returns_body(self):
        from bandcamp_recommender.recommendations import scraper

        with local_http_server() as base:
            with wall_clock_budget(2.0, "fetch_page_html against /instant"):
                result = scraper.fetch_page_html(f"{base}/instant", timeout=2)

        self.assertIsNotNone(result)
        self.assertIn("ok", result)

    def test_fetch_page_html_retries_on_transient_then_succeeds(self):
        """Transient curl failure (non-zero exit) retries with backoff."""
        from bandcamp_recommender.recommendations import scraper

        call_count = {"n": 0}
        real_run = scraper.subprocess.run

        def flaky_run(cmd, *args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # First call: return a fake CompletedProcess with non-zero exit.
                from subprocess import CompletedProcess
                return CompletedProcess(cmd, returncode=7, stdout="", stderr="simulated network blip")
            return real_run(cmd, *args, **kwargs)

        with local_http_server() as base:
            with patch.object(scraper.subprocess, "run", side_effect=flaky_run):
                with wall_clock_budget(6.0, "fetch_page_html with one transient failure"):
                    result = scraper.fetch_page_html(f"{base}/instant", timeout=2)

        self.assertIsNotNone(result)
        self.assertIn("ok", result)
        self.assertEqual(call_count["n"], 2, "expected exactly one retry")

    def test_fetch_page_html_gives_up_after_repeated_transient(self):
        """All-failing transient: returns None after 4 attempts (1 + 3 retries)."""
        from bandcamp_recommender.recommendations import scraper
        from subprocess import CompletedProcess

        call_count = {"n": 0}

        def always_fail(cmd, *args, **kwargs):
            call_count["n"] += 1
            return CompletedProcess(cmd, returncode=7, stdout="", stderr="boom")

        with patch.object(scraper.subprocess, "run", side_effect=always_fail):
            with wall_clock_budget(8.0, "fetch_page_html giving up"):
                result = scraper.fetch_page_html("http://invalid.test/x", timeout=2)

        self.assertIsNone(result)
        # 1 initial + 3 retries = 4 attempts total.
        self.assertEqual(call_count["n"], 4)


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


# ---------- Phase 5: retry on Selenium fallback ----------


class SupporterRetryTests(unittest.TestCase):
    """Hangs on the first Selenium attempt should retry once with a fresh driver."""

    def test_retry_once_on_first_driver_timeout(self):
        from bandcamp_recommender.recommendations import supporter_recommender

        # Two drivers: first hangs on .get, second succeeds.
        driver_a = _make_fake_chrome()
        driver_a.get.side_effect = TimeoutException("first try hangs")

        driver_b = _make_fake_chrome()
        # Minimal pagedata blob with one tralbum_id so the parse path succeeds.
        driver_b.page_source = (
            '<html><body>'
            '<div id="pagedata" data-blob=\'{"fan_data":{"fan_id":42},'
            '"collection_data":{"sequence":["k1"],"pending_sequence":[],'
            '"last_token":"","item_count":1},'
            '"item_cache":{"collection":{"k1":{"tralbum_id":12345,'
            '"item_title":"T","band_name":"B","item_url":"https://x/album/y"}}}}\'></div>'
            '</body></html>'
        )

        # Bypass the curl-first path so we hit Selenium fallback directly.
        with patch.object(
            supporter_recommender.SupporterRecommender,
            "_get_supporter_items_via_curl",
            return_value=None,
        ):
            # Mock the DriverManager pool: hand out driver_a first, then driver_b.
            rec = supporter_recommender.SupporterRecommender()
            from queue import Queue

            pool: Queue = Queue()
            pool.put(driver_a)
            pool.put(driver_b)
            with patch.object(
                rec._driver_manager,
                "get_driver_pool",
                return_value=pool,
            ):
                items = rec._get_supporter_items_curl_first(
                    "someuser", "collection_data"
                )

        self.assertEqual(items, ["12345"])
        # Both drivers were used (first failed, retry on second succeeded).
        driver_a.get.assert_called()
        driver_b.get.assert_called()


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
