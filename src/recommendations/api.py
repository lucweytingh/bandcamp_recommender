"""Bandcamp API interaction utilities."""

import json
import subprocess
from typing import Dict, List, Optional

from bs4 import BeautifulSoup
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


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
        driver.get(wishlist_url)
        
        # Wait for pagedata element
        try:
            WebDriverWait(driver, 3).until(
                EC.presence_of_element_located((By.ID, "pagedata"))
            )
        except Exception:
            # If wishlist page doesn't work, try profile page
            profile_url = f"https://bandcamp.com/{username}"
            driver.get(profile_url)
            try:
                WebDriverWait(driver, 3).until(
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
    except Exception:
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
    timeout: int = 30
) -> List[Dict]:
    """Fetch collection items from Bandcamp API.
    
    Args:
        fan_id: Bandcamp fan ID
        last_token: Token from last page
        cookies: Authentication cookies
        referer_url: Referer URL for the request
        timeout: Request timeout in seconds
        
    Returns:
        List of item dictionaries from API response
    """
    api_url = "https://bandcamp.com/api/fancollection/1/collection_items"
    payload = {
        "fan_id": fan_id,
        "older_than_token": last_token,
        "count": 10000,
    }
    
    cookie_string = "; ".join([f"{k}={v}" for k, v in cookies.items()])
    curl_cmd = [
        "curl",
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
    
    result = subprocess.run(
        curl_cmd, capture_output=True, text=True, timeout=timeout
    )
    
    if result.returncode == 0:
        try:
            data = json.loads(result.stdout)
            return data.get("items", [])
        except (json.JSONDecodeError, KeyError):
            pass
    
    return []

