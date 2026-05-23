"""Bandcamp API interaction utilities."""

import json
import logging
import os
import subprocess
from typing import Dict, List, Optional

from bs4 import BeautifulSoup
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from . import curl_breaker


logger = logging.getLogger(__name__)


def _debug_exc_info() -> bool:
    return os.environ.get("BANDCAMP_DEBUG") == "1"


def get_fan_id_from_page(driver: WebDriver, username: str) -> Optional[int]:
    """Get fan_id from a supporter's page.
    
    Args:
        driver: Selenium WebDriver instance
        username: Supporter username
        
    Returns:
        fan_id as integer, or None if not found
    """
    try:
        # Navigate to supporter's wishlist page to get fan_id and cookies
        # (wishlist/profile pages have fan_data, /music page doesn't)
        wishlist_url = f"https://bandcamp.com/{username}/wishlist"
        try:
            driver.get(wishlist_url)
        except TimeoutException:
            logger.warning("get_fan_id_from_page: wishlist page-load timeout for %s", username)
            return None

        # Wait for pagedata element. Pagedata is server-rendered so it
        # resolves quickly when the page loads at all; long timeouts mostly
        # wait out failures.
        try:
            WebDriverWait(driver, 1.5).until(
                EC.presence_of_element_located((By.ID, "pagedata"))
            )
        except Exception:
            # If wishlist page doesn't work, try profile page
            profile_url = f"https://bandcamp.com/{username}"
            try:
                driver.get(profile_url)
            except TimeoutException:
                logger.warning("get_fan_id_from_page: profile page-load timeout for %s", username)
                return None
            try:
                WebDriverWait(driver, 1.5).until(
                    EC.presence_of_element_located((By.ID, "pagedata"))
                )
            except Exception:
                return None

        soup = BeautifulSoup(driver.page_source, features="html.parser")
        pagedata_elem = soup.find(id="pagedata")
        if not pagedata_elem:
            return None

        pagedata = json.loads(pagedata_elem.get("data-blob", "{}"))

        # Get fan_id for API call
        fan_data = pagedata.get("fan_data", {})
        fan_id = fan_data.get("fan_id")

        return fan_id
    except WebDriverException:
        logger.warning("get_fan_id_from_page: webdriver error for %s", username, exc_info=_debug_exc_info())
        return None
    except Exception:
        logger.warning("get_fan_id_from_page: unexpected error for %s", username, exc_info=_debug_exc_info())
        return None


def get_cookies_from_driver(driver: WebDriver) -> Dict[str, str]:
    """Extract cookies from Selenium driver.
    
    Args:
        driver: Selenium WebDriver instance
        
    Returns:
        Dictionary of cookie name -> cookie value
    """
    cookies = {}
    for cookie in driver.get_cookies():
        cookies[cookie["name"]] = cookie["value"]
    return cookies


def fetch_collection_items_api(
    fan_id: int,
    last_token: str,
    cookies: Dict[str, str],
    referer_url: str,
    timeout: int = 30,
    driver: Optional[WebDriver] = None
) -> List[Dict]:
    """Fetch collection items from Bandcamp API.

    Uses the Selenium browser session (via driver.execute_async_script) when a
    driver is provided, which avoids 403s from Bandcamp's bot protection on
    headless servers. Falls back to curl if no driver is given.

    Args:
        fan_id: Bandcamp fan ID
        last_token: Token from last page
        cookies: Authentication cookies
        referer_url: Referer URL for the request
        timeout: Request timeout in seconds
        driver: Optional Selenium WebDriver to execute the fetch inside the browser

    Returns:
        List of item dictionaries from API response
    """
    api_url = "https://bandcamp.com/api/fancollection/1/collection_items"
    payload = {
        "fan_id": fan_id,
        "older_than_token": last_token,
        "count": 10000,
    }

    if driver:
        return _fetch_via_driver(driver, api_url, payload, timeout)

    return _fetch_via_curl(api_url, payload, cookies, referer_url, timeout)


def _fetch_via_driver(
    driver: WebDriver,
    api_url: str,
    payload: Dict,
    timeout: int
) -> List[Dict]:
    """Execute a fetch() inside the browser session to avoid bot protection."""
    try:
        script = """
            var callback = arguments[arguments.length - 1];
            fetch(arguments[0], {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: arguments[1],
                credentials: 'include'
            })
            .then(function(r) { return r.json(); })
            .then(function(data) { callback(JSON.stringify(data)); })
            .catch(function(e) { callback(JSON.stringify({error: e.toString()})); });
        """
        old_timeout = driver.timeouts.script
        driver.set_script_timeout(timeout)
        try:
            result_json = driver.execute_async_script(
                script, api_url, json.dumps(payload)
            )
        finally:
            driver.set_script_timeout(old_timeout)

        if result_json:
            data = json.loads(result_json)
            return data.get("items", [])
    except TimeoutException:
        logger.warning("_fetch_via_driver: script timeout after %ds", timeout)
    except WebDriverException:
        logger.warning("_fetch_via_driver: webdriver error", exc_info=_debug_exc_info())
    except Exception:
        logger.warning("_fetch_via_driver: unexpected error", exc_info=_debug_exc_info())
    return []


def _fetch_via_curl(
    api_url: str,
    payload: Dict,
    cookies: Dict[str, str],
    referer_url: str,
    timeout: int
) -> List[Dict]:
    """Fetch via curl subprocess (original approach, may 403 on headless servers)."""
    if curl_breaker.should_skip_curl():
        return []

    cookie_string = "; ".join([f"{k}={v}" for k, v in cookies.items()])
    curl_cmd = [
        "curl",
        "-s",
        "-X",
        "POST",
        "-H",
        "Content-Type: application/json",
        "-H",
        f"Cookie: {cookie_string}",
        "-H",
        "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "-H",
        f"Referer: {referer_url}",
        "-d",
        json.dumps(payload),
        api_url,
    ]

    try:
        result = subprocess.run(
            curl_cmd, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        curl_breaker.record_outcome(success=False)
        return []

    if result.returncode == 0:
        try:
            data = json.loads(result.stdout)
            curl_breaker.record_outcome(success=True)
            return data.get("items", [])
        except (json.JSONDecodeError, KeyError):
            pass

    curl_breaker.record_outcome(success=False)
    return []


