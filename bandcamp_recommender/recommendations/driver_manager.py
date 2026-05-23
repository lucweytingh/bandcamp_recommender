"""Selenium WebDriver management for Bandcamp scraping."""

import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from queue import Queue
from threading import Lock
from typing import Optional

from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager


logger = logging.getLogger(__name__)


_DEFAULT_PAGE_LOAD_TIMEOUT = 30
_DEFAULT_SCRIPT_TIMEOUT = 30


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _page_load_timeout() -> int:
    return _env_int("BANDCAMP_PAGE_LOAD_TIMEOUT", _DEFAULT_PAGE_LOAD_TIMEOUT)


def _script_timeout() -> int:
    return _env_int("BANDCAMP_SCRIPT_TIMEOUT", _DEFAULT_SCRIPT_TIMEOUT)


class DriverManager:
    """Manages Selenium WebDriver instances and pooling for parallel processing."""

    def __init__(self):
        """Initialize the driver manager."""
        self.driver: Optional[webdriver.Chrome] = None
        self._driver_pool: Optional[Queue] = None
        self._driver_pool_lock = Lock()
        self._chrome_service: Optional[Service] = None

    def _get_chromedriver_service(self) -> Service:
        """Get a ChromeDriver Service, preferring env var or system binary over webdriver_manager."""
        # 1. Check CHROMEDRIVER env var
        chromedriver_env = os.environ.get("CHROMEDRIVER", "")
        if chromedriver_env and os.path.exists(chromedriver_env):
            return Service(chromedriver_env)

        # 2. Check system chromedriver on PATH
        system_chromedriver = shutil.which("chromedriver")
        if system_chromedriver:
            return Service(system_chromedriver)

        # 3. Fall back to webdriver_manager auto-download
        return Service(ChromeDriverManager().install())

    def get_driver_options(self) -> Options:
        """Get optimized driver options (reusable).

        Returns:
            Configured Chrome Options object
        """
        options = Options()
        # Always run headless to avoid popup windows
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        # Skip images. --blink-settings is the canonical flag; --disable-images
        # was deprecated upstream and silently ignored in newer Chromium.
        options.add_argument("--blink-settings=imagesEnabled=false")
        # Opt-in: disable JS for collection-page fetches. pagedata is in the
        # initial HTML so this is safe for read-only page loads, but it
        # breaks api.py:_fetch_via_driver (which uses execute_async_script).
        # Default off to preserve the curl-403 fallback path.
        if os.environ.get("BANDCAMP_DISABLE_JS") == "1":
            options.add_argument("--disable-javascript")
        options.page_load_strategy = "eager"  # Don't wait for all resources to load
        options.add_experimental_option("excludeSwitches", ["enable-logging", "enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)

        # Set Chrome binary only if explicitly configured via env var.
        # Otherwise let chromedriver find Chrome itself (works for both
        # snap and apt installs; auto-detection via shutil.which is fragile
        # because wrapper scripts like /usr/bin/google-chrome may exist but
        # not be functional Chrome binaries).
        chrome_binary = os.environ.get("CHROME_BINARY", "")
        if chrome_binary and os.path.exists(chrome_binary):
            options.binary_location = chrome_binary

        return options

    def _configure_driver(self, driver: webdriver.Chrome) -> None:
        """Apply wall-clock timeouts to every driver this manager produces.

        Without these, ``driver.get(...)`` and ``execute_async_script`` will
        block forever on a stalled Bandcamp request. The radio worker
        outage that triggered this hardening was exactly that hang.
        """
        try:
            driver.set_page_load_timeout(_page_load_timeout())
            driver.set_script_timeout(_script_timeout())
        except WebDriverException:
            logger.warning(
                "failed to configure driver timeouts",
                exc_info=os.environ.get("BANDCAMP_DEBUG") == "1",
            )

    def init_driver(self):
        """Initialize the Selenium webdriver with appropriate options.

        Only initialized when needed (for collection pages that require cookies).
        """
        options = self.get_driver_options()
        service = self._get_chromedriver_service()
        self.driver = webdriver.Chrome(service=service, options=options)
        self._configure_driver(self.driver)

    def ensure_driver(self):
        """Ensure driver is initialized (lazy initialization)."""
        if self.driver is None:
            self.init_driver()

    def get_driver_pool(self, pool_size: int = 10, progress_callback=None) -> Queue:
        """Get or create a driver pool for parallel processing.

        Args:
            pool_size: Number of drivers to create in the pool
            progress_callback: Optional callback for progress updates

        Returns:
            Queue of driver instances
        """
        with self._driver_pool_lock:
            if self._driver_pool is None:
                self._driver_pool = Queue(maxsize=pool_size)

                # Pre-create ChromeDriver service (expensive operation, do once)
                if self._chrome_service is None:
                    self._chrome_service = self._get_chromedriver_service()

                # Pre-create drivers (this can take a while, but we do it once)
                options = self.get_driver_options()
                for i in range(pool_size):
                    try:
                        if i > 0:
                            time.sleep(0.1)
                        driver = webdriver.Chrome(
                            service=self._chrome_service,
                            options=options
                        )
                        self._configure_driver(driver)
                        self._driver_pool.put(driver)
                        if progress_callback:
                            progress_callback(f"Initialized driver {i+1}/{pool_size}...")
                    except Exception as e:
                        # If driver creation fails, continue with fewer drivers
                        print(f"Warning: Failed to create driver {i+1}/{pool_size}: {e}")
                        break

        return self._driver_pool

    def create_driver(self) -> webdriver.Chrome:
        """Create a new driver instance (for parallel processing).

        Note: Prefer using driver pool for better performance.

        Returns:
            New Chrome WebDriver instance
        """
        options = self.get_driver_options()
        if self._chrome_service is None:
            self._chrome_service = self._get_chromedriver_service()
        driver = webdriver.Chrome(
            service=self._chrome_service,
            options=options
        )
        self._configure_driver(driver)
        return driver

    def close(self):
        """Close the webdriver and cleanup driver pool."""
        if self.driver:
            try:
                self.driver.quit()
            except WebDriverException:
                logger.warning(
                    "driver.quit() failed in close()",
                    exc_info=os.environ.get("BANDCAMP_DEBUG") == "1",
                )
            self.driver = None

        # Clean up driver pool
        if self._driver_pool:
            while not self._driver_pool.empty():
                try:
                    driver = self._driver_pool.get_nowait()
                    driver.quit()
                except Exception:
                    pass
            self._driver_pool = None

    @staticmethod
    def is_driver_alive(driver) -> bool:
        """Cheap liveness probe — reading ``current_url`` round-trips to chromedriver.

        Used by the supporter-recommender fallback to decide whether a driver
        returned from a worker is safe to put back in the pool or must be
        quit instead.
        """
        try:
            _ = driver.current_url
            return True
        except WebDriverException:
            return False
        except Exception:
            return False

    @staticmethod
    def reap_orphans() -> int:
        """SIGKILL local chromedriver / chrome / chromium processes whose
        parent is init (PID 1) — i.e. orphans from a previous crashed run.

        Returns the number of processes reaped. Safe no-op on Windows and
        when ``ps`` is unavailable. Opt-in: callers run this on cold start.
        """
        if sys.platform.startswith("win"):
            return 0
        ps = shutil.which("ps")
        if not ps:
            return 0
        try:
            result = subprocess.run(
                [ps, "-axo", "pid=,ppid=,comm="],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (subprocess.TimeoutExpired, OSError):
            logger.warning("reap_orphans: ps invocation failed")
            return 0
        if result.returncode != 0:
            return 0

        targets = ("chromedriver", "chrome", "chromium", "Google Chrome")
        reaped = 0
        for line in result.stdout.splitlines():
            parts = line.strip().split(None, 2)
            if len(parts) < 3:
                continue
            try:
                pid = int(parts[0])
                ppid = int(parts[1])
            except ValueError:
                continue
            comm = parts[2]
            if ppid != 1:
                continue
            base = os.path.basename(comm).lower()
            if not any(t.lower() in base for t in targets):
                continue
            try:
                os.kill(pid, signal.SIGKILL)
                reaped += 1
            except ProcessLookupError:
                continue
            except PermissionError:
                logger.warning("reap_orphans: no permission to kill pid %d", pid)
                continue
        if reaped:
            logger.warning("reap_orphans: killed %d orphan chromium process(es)", reaped)
        return reaped
