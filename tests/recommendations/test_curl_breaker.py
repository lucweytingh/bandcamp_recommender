"""Unit tests for the curl circuit breaker."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bandcamp_recommender.recommendations import curl_breaker


class CurlBreakerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._env = patch.dict(
            os.environ,
            {
                "BANDCAMP_CURL_BREAKER_STATE": str(Path(self._tmp.name) / "state.json"),
                "BANDCAMP_CURL_TRIP_THRESHOLD": "3",
                "BANDCAMP_CURL_TRIP_WINDOW_S": "60",
                "BANDCAMP_CURL_TRIP_COOLDOWN_S": "3600",
            },
            clear=False,
        )
        self._env.start()
        self.addCleanup(self._env.stop)
        # Make sure stray override flags from the shell don't leak in.
        for var in ("BANDCAMP_DISABLE_CURL", "BANDCAMP_CURL_BREAKER_DISABLED"):
            if var in os.environ:
                os.environ.pop(var)
        curl_breaker.reset()

    def test_initial_state_does_not_skip(self):
        self.assertFalse(curl_breaker.should_skip_curl())

    def test_three_consecutive_failures_trip_breaker(self):
        for _ in range(3):
            curl_breaker.record_outcome(success=False, now=1000.0)
        self.assertTrue(curl_breaker.should_skip_curl(now=1001.0))

    def test_failures_outside_window_do_not_trip(self):
        curl_breaker.record_outcome(success=False, now=0.0)
        curl_breaker.record_outcome(success=False, now=30.0)
        # Third failure lands outside the 60s window from the first one.
        curl_breaker.record_outcome(success=False, now=200.0)
        self.assertFalse(curl_breaker.should_skip_curl(now=201.0))

    def test_success_clears_failure_streak(self):
        curl_breaker.record_outcome(success=False, now=0.0)
        curl_breaker.record_outcome(success=False, now=1.0)
        curl_breaker.record_outcome(success=True, now=2.0)
        curl_breaker.record_outcome(success=False, now=3.0)
        self.assertFalse(curl_breaker.should_skip_curl(now=4.0))

    def test_trip_expires_after_cooldown(self):
        for _ in range(3):
            curl_breaker.record_outcome(success=False, now=1000.0)
        self.assertTrue(curl_breaker.should_skip_curl(now=2000.0))
        # 3600s cooldown — past it, breaker should release.
        self.assertFalse(curl_breaker.should_skip_curl(now=1000.0 + 3601))

    def test_success_while_tripped_clears_trip(self):
        for _ in range(3):
            curl_breaker.record_outcome(success=False, now=1000.0)
        self.assertTrue(curl_breaker.should_skip_curl(now=1100.0))
        curl_breaker.record_outcome(success=True, now=1200.0)
        self.assertFalse(curl_breaker.should_skip_curl(now=1201.0))

    def test_state_persists_across_calls(self):
        for _ in range(3):
            curl_breaker.record_outcome(success=False, now=1000.0)
        # New read picks up the persisted trip.
        self.assertTrue(curl_breaker.should_skip_curl(now=1001.0))

    def test_disable_env_var_hard_disables_curl(self):
        with patch.dict(os.environ, {"BANDCAMP_DISABLE_CURL": "1"}):
            self.assertTrue(curl_breaker.should_skip_curl())

    def test_breaker_disabled_env_var_skips_logic(self):
        with patch.dict(os.environ, {"BANDCAMP_CURL_BREAKER_DISABLED": "1"}):
            for _ in range(10):
                curl_breaker.record_outcome(success=False, now=1000.0)
            self.assertFalse(curl_breaker.should_skip_curl(now=1001.0))

    def test_reset_clears_trip(self):
        for _ in range(3):
            curl_breaker.record_outcome(success=False, now=1000.0)
        self.assertTrue(curl_breaker.should_skip_curl(now=1001.0))
        curl_breaker.reset()
        self.assertFalse(curl_breaker.should_skip_curl(now=1002.0))


if __name__ == "__main__":
    unittest.main()
