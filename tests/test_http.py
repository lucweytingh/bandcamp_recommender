"""Tests for the shared Bandcamp HTTP client (bandcamp_recommender.http).

Runs against a local server; never touches Bandcamp."""

from __future__ import annotations

import contextlib
import http.server
import socketserver
import threading
import time
import unittest

from bandcamp_recommender import http as bchttp

CHALLENGE = (
    '<!DOCTYPE html><html><head><link href="/_fs-ch-1T1wmsGaOgGaSxcX/assets/styles.css" '
    'rel="stylesheet" /></head><body></body></html>'
)


class _Handler(http.server.BaseHTTPRequestHandler):
    hits: dict = {}
    lock = threading.Lock()

    def log_message(self, *args):
        return

    def _send(self, status, body=b"", ctype="text/html", headers=()):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        for k, v in headers:
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        with self.lock:
            self.hits[path] = self.hits.get(path, 0) + 1
            n = self.hits[path]
        if path == "/item":
            self._send(200, b'<html><script data-tralbum="{}"></script></html>')
        elif path == "/challenge":
            self._send(200, CHALLENGE.encode())
        elif path == "/utf8-no-charset":
            self._send(200, "<p data-tralbum>Sigur Rós – Ágætis</p>".encode("utf-8"))
        elif path == "/429-then-ok":
            if n == 1:
                self._send(429, headers=[("Retry-After", "1")])
            else:
                self._send(200, b"data-tralbum ok")
        elif path == "/always-429":
            self._send(429, headers=[("Retry-After", "1")])
        elif path == "/404":
            self._send(404)
        elif path == "/trickle":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            try:
                for _ in range(100):
                    self.wfile.write(b"x" * 10)
                    self.wfile.flush()
                    time.sleep(0.3)
            except OSError:
                return
        else:
            self._send(404)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        if self.path == "/api":
            self._send(200, b'{"items": [1, 2]}', ctype="application/json")
        else:
            self._send(200, CHALLENGE.encode())


class _Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


@contextlib.contextmanager
def server():
    _Handler.hits = {}
    srv = _Server(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()


class HttpClientTests(unittest.TestCase):
    def setUp(self):
        bchttp.reset()
        self.addCleanup(bchttp.reset)

    def test_get_html_returns_validated_page(self):
        with server() as base:
            self.assertIn("data-tralbum", bchttp.get_html(f"{base}/item", expect="data-tralbum"))

    def test_challenge_page_is_typed_not_returned(self):
        before = bchttp.snapshot()
        with server() as base:
            with self.assertRaises(bchttp.Challenged):
                bchttp.get_html(f"{base}/challenge", expect="data-tralbum")
        self.assertEqual(bchttp.diff(before)["challenged"], 1)

    def test_200_without_expected_marker_is_unexpected(self):
        with server() as base:
            with self.assertRaises(bchttp.Unexpected):
                bchttp.get_html(f"{base}/item", expect="pagedata")

    def test_http_error_is_typed(self):
        with server() as base:
            with self.assertRaises(bchttp.HTTPStatus) as ctx:
                bchttp.get_html(f"{base}/404")
        self.assertEqual(ctx.exception.status, 404)

    def test_utf8_decoded_without_declared_charset(self):
        with server() as base:
            body = bchttp.get_html(f"{base}/utf8-no-charset")
        self.assertIn("Sigur Rós – Ágætis", body)

    def test_429_honours_retry_after_then_succeeds(self):
        before = bchttp.snapshot()
        with server() as base:
            t0 = time.monotonic()
            body = bchttp.get_html(f"{base}/429-then-ok", expect="data-tralbum")
            elapsed = time.monotonic() - t0
        self.assertIn("ok", body)
        self.assertGreaterEqual(elapsed, 0.9)
        self.assertEqual(bchttp.diff(before)["rate_limited"], 1)

    def test_persistent_429_raises_rate_limited(self):
        with server() as base:
            with self.assertRaises(bchttp.RateLimited):
                bchttp.get_html(f"{base}/always-429")

    def test_wall_clock_deadline_bounds_a_trickle(self):
        with server() as base:
            t0 = time.monotonic()
            with self.assertRaises(bchttp.Timeout):
                bchttp.get_html(f"{base}/trickle", timeout=1.0)
            self.assertLess(time.monotonic() - t0, 3.0)

    def test_mid_body_transport_error_is_typed(self):
        import urllib3
        from unittest.mock import patch as _patch

        with server() as base:
            with _patch.object(urllib3.response.HTTPResponse, "read1",
                               side_effect=urllib3.exceptions.ProtocolError("reset")):
                with self.assertRaises(bchttp.FetchError):
                    bchttp.get_html(f"{base}/item")
            with _patch.object(urllib3.response.HTTPResponse, "read1",
                               side_effect=urllib3.exceptions.ReadTimeoutError(None, "u", "slow")):
                with self.assertRaises(bchttp.Timeout):
                    bchttp.get_html(f"{base}/item")

    def test_post_json_decodes_and_detects_challenge(self):
        with server() as base:
            self.assertEqual(bchttp.post_json(f"{base}/api", {"a": 1}), {"items": [1, 2]})
            with self.assertRaises(bchttp.Challenged):
                bchttp.post_json(f"{base}/other", {"a": 1})


class PacerTests(unittest.TestCase):
    def test_starts_are_spaced_by_interval(self):
        gate = bchttp.Pacer(min_interval=0.1, max_concurrency=5)
        starts = []
        lock = threading.Lock()

        def worker():
            gate.acquire()
            with lock:
                starts.append(time.monotonic())
            gate.release()

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        starts.sort()
        gaps = [b - a for a, b in zip(starts, starts[1:])]
        self.assertTrue(all(g >= 0.09 for g in gaps), gaps)

    def test_concurrency_cap(self):
        gate = bchttp.Pacer(min_interval=0.0, max_concurrency=2)
        inflight = [0]
        peak = [0]
        lock = threading.Lock()

        def worker():
            gate.acquire()
            with lock:
                inflight[0] += 1
                peak[0] = max(peak[0], inflight[0])
            time.sleep(0.05)
            with lock:
                inflight[0] -= 1
            gate.release()

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(peak[0], 2)

    def test_aimd_backs_off_on_429_and_recovers(self):
        gate = bchttp.Pacer(min_interval=0.4, max_concurrency=1)
        gate.on_rate_limited(0.0)
        self.assertAlmostEqual(gate.interval, 0.6)
        gate.on_success()
        self.assertAlmostEqual(gate.interval, 0.5)
        for _ in range(50):
            gate.on_success()
        self.assertAlmostEqual(gate.interval, 0.4)

    def test_retry_after_delays_every_thread(self):
        gate = bchttp.Pacer(min_interval=0.0, max_concurrency=4)
        gate.on_rate_limited(0.5)
        t0 = time.monotonic()
        gate.acquire()
        gate.release()
        self.assertGreaterEqual(time.monotonic() - t0, 0.45)


if __name__ == "__main__":
    unittest.main()
