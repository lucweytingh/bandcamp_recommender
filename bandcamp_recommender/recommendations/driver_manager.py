"""Selenium WebDriver management for Bandcamp scraping."""

import logging
import os
import shutil
import time
from queue import Queue
from threading import Lock
from typing import Optional

from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

from bandcamp_recommender import chrome_procs


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
        self._chromedriver: Optional[str] = None

    def _chromedriver_path(self) -> str:
        """Resolve chromedriver once per manager, preferring env var or system binary over webdriver_manager."""
        if self._chromedriver is None:
            # 1. CHROMEDRIVER env var, 2. chromedriver on PATH, 3. webdriver_manager auto-download
            chromedriver_env = os.environ.get("CHROMEDRIVER", "")
            if chromedriver_env and os.path.exists(chromedriver_env):
                self._chromedriver = chromedriver_env
            else:
                self._chromedriver = shutil.which("chromedriver") or ChromeDriverManager().install()
        return self._chromedriver

    def _launch(self) -> webdriver.Chrome:
        """Start one Chrome on its own library-owned profile, with its own
        chromedriver Service (a Service shared across drivers only stops the
        last chromedriver it started: the rest leaked). See chrome_procs."""
        chromedriver = self._chromedriver_path()
        options = self.get_driver_options()
        profile = chrome_procs.new_profile()
        try:
            options.add_argument(f"--user-data-dir={profile}")
            service = Service(chromedriver, service_args=chrome_procs.chromedriver_args(profile))
            driver = webdriver.Chrome(service=service, options=options)
        except Exception:
            chrome_procs.release_profile(profile)
            raise
        setattr(driver, chrome_procs.PROFILE_ATTR, profile)
        try:
            self._configure_driver(driver)
        except Exception:
            chrome_procs.quit_driver(driver)
            raise
        return driver

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
        self.driver = self._launch()

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

                # Pre-create drivers (this can take a while, but we do it once)
                for i in range(pool_size):
                    try:
                        if i > 0:
                            time.sleep(0.1)
                        driver = self._launch()
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
        return self._launch()

    def close(self):
        """Close the webdriver and cleanup driver pool."""
        if self.driver:
            chrome_procs.quit_driver(self.driver)
            self.driver = None

        # Clean up driver pool
        if self._driver_pool:
            while not self._driver_pool.empty():
                try:
                    driver = self._driver_pool.get_nowait()
                except Exception:
                    break
                chrome_procs.quit_driver(driver)
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
        """SIGKILL chromedriver / Chrome left behind by library processes that
        died (a previous crashed run), and delete their profiles. Spares the
        browser of every live library process. Opt-in: callers run this on
        cold start.

        This is :func:`bandcamp_recommender.chrome_procs.reap_orphans` with
        ``legacy=True``: unlike the default, it keeps reaching what versions
        without profile markers (before 0.8) leave behind: an unmarked
        chromedriver whose spawner died, or a Chrome on a chromedriver temp
        profile whose chromedriver died (reparented to init or a
        subreaper), with their descendants. Caveat: that also kills a
        chromedriver run as a service straight under init, systemd or
        launchd (e.g. a Selenium Grid node). It no longer kills any other
        Chrome with PPID 1 (on macOS that included the user's own Chrome
        launched from the Dock).

        Returns the number of processes reaped. Safe no-op on Windows.
        """
        return chrome_procs.reap_orphans(legacy=True)
