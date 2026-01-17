"""Selenium WebDriver management for Bandcamp scraping."""

import warnings
from queue import Queue
from threading import Lock
from typing import Optional

from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from seleniumwire import webdriver as wire_webdriver
from webdriver_manager.chrome import ChromeDriverManager

# Suppress selenium-wire RuntimeWarnings (harmless coroutine warnings)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="seleniumwire")


class DriverManager:
    """Manages Selenium WebDriver instances and pooling for parallel processing."""

    def __init__(self):
        """Initialize the driver manager."""
        self.driver: Optional[wire_webdriver.Chrome] = None
        self._driver_pool: Optional[Queue] = None
        self._driver_pool_lock = Lock()
        self._chrome_service: Optional[Service] = None

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
        options.add_argument("--disable-images")  # Don't load images - faster page loads
        options.page_load_strategy = "eager"  # Don't wait for all resources to load
        options.add_experimental_option("excludeSwitches", ["enable-logging", "enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)

        # Auto-detect Chrome binary
        chrome_paths = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
            "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
            "/Applications/Arc.app/Contents/MacOS/Arc",
        ]
        for path in chrome_paths:
            try:
                options.binary_location = path
                break
            except Exception:
                continue

        return options

    def init_driver(self):
        """Initialize the Selenium webdriver with appropriate options.
        
        Only initialized when needed (for collection pages that require cookies).
        """
        options = self.get_driver_options()
        service = Service(ChromeDriverManager().install())
        self.driver = wire_webdriver.Chrome(service=service, options=options)

    def ensure_driver(self):
        """Ensure driver is initialized (lazy initialization)."""
        if self.driver is None:
            self.init_driver()

    def get_driver_pool(self, pool_size: int = 15, progress_callback=None) -> Queue:
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
                    self._chrome_service = Service(ChromeDriverManager().install())

                # Pre-create drivers (this can take a while, but we do it once)
                options = self.get_driver_options()
                for i in range(pool_size):
                    try:
                        driver = wire_webdriver.Chrome(service=self._chrome_service, options=options)
                        self._driver_pool.put(driver)
                        if progress_callback:
                            progress_callback(f"Initialized driver {i+1}/{pool_size}...")
                    except Exception as e:
                        # If driver creation fails, continue with fewer drivers
                        print(f"Warning: Failed to create driver {i+1}/{pool_size}: {e}")
                        break

        return self._driver_pool

    def create_driver(self) -> wire_webdriver.Chrome:
        """Create a new driver instance (for parallel processing).
        
        Note: Prefer using driver pool for better performance.
        
        Returns:
            New Chrome WebDriver instance
        """
        options = self.get_driver_options()
        if self._chrome_service is None:
            self._chrome_service = Service(ChromeDriverManager().install())
        return wire_webdriver.Chrome(service=self._chrome_service, options=options)

    def close(self):
        """Close the webdriver and cleanup driver pool."""
        if self.driver:
            self.driver.quit()
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


