"""Web scraping utilities for Bandcamp pages."""

import json
import logging
import os
import re
import shutil
import subprocess
import time
from typing import List, Optional

from bs4 import BeautifulSoup

from . import curl_breaker


logger = logging.getLogger(__name__)


def _debug_exc_info() -> bool:
    return os.environ.get("BANDCAMP_DEBUG") == "1"


# Default curl timeout, raised 15→20 to accommodate slow Bandcamp custom
# domains (e.g. craigieknowes.com routinely takes 15–20s).
_DEFAULT_CURL_TIMEOUT = 20

# Backoff schedule for the curl retry loop. Three attempts total — the
# downstream radio worker confirmed 3 is the sweet spot: enough to ride
# out a per-IP throttle, few enough to not extend the slice budget too far.
_RETRY_BACKOFF_SECONDS = (0.4, 1.2, 3.0)


def fetch_page_html(url: str, timeout: int = _DEFAULT_CURL_TIMEOUT) -> Optional[str]:
    """Fetch HTML content from a URL using curl, with bounded retry.

    Strategy:

    * If the curl-breaker has tripped (IP blocked), skip straight to Selenium.
    * Otherwise try curl up to 3 times with exponential backoff
      (0.4 s → 1.2 s → 3.0 s) on subprocess timeout or non-zero exit.
      Bandcamp custom domains (e.g. craigieknowes.com) can spike to 15–20 s
      on a single request; a single transient timeout shouldn't fail the
      whole pipeline.
    * Parse failures are out of scope here — this layer only retries on
      transport-level failure.

    Args:
        url: URL to fetch
        timeout: Per-attempt request timeout in seconds (default 20 s,
            up from 15 s to match real-world custom-domain latency).

    Returns:
        HTML content as string, or None if all attempts fail.
    """
    if curl_breaker.should_skip_curl():
        return _fetch_page_with_selenium(url)

    curl_cmd = [
        "curl",
        "-s",  # Silent mode
        "-L",  # Follow redirects
        "--compressed",  # Automatically decompress gzip/deflate
        "-H", "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "-H", "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "-H", "Accept-Language: en-US,en;q=0.5",
        "-H", "Connection: keep-alive",
        url,
    ]

    # Retry strategy: on transient HTTP-level failure (non-zero exit, but the
    # subprocess didn't time out → the request completed, just badly), retry
    # up to 3 times with backoff. On a hard timeout (subprocess.TimeoutExpired)
    # we give up immediately — the per-call timeout (20 s default) is already
    # generous enough for Bandcamp custom domains, and stacking 4 × 20 s on a
    # dead URL would blow the per-supporter slice budget for everyone fanned
    # out alongside it.
    attempts = len(_RETRY_BACKOFF_SECONDS) + 1  # initial try + N backoffs
    for attempt in range(attempts):
        try:
            result = subprocess.run(
                curl_cmd, capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            curl_breaker.record_outcome(success=False)
            logger.warning(
                "curl timeout fetching %s (timeout=%ds, attempt %d/%d, not retrying timeouts)",
                url,
                timeout,
                attempt + 1,
                attempts,
            )
            return None
        except Exception:
            logger.warning(
                "curl error fetching %s (attempt %d/%d)",
                url,
                attempt + 1,
                attempts,
                exc_info=_debug_exc_info(),
            )
            return None  # Non-timeout errors aren't worth retrying.

        if result.returncode == 0:
            curl_breaker.record_outcome(success=True)
            return result.stdout

        curl_breaker.record_outcome(success=False)
        logger.warning(
            "curl returned %d for %s (attempt %d/%d): %s",
            result.returncode,
            url,
            attempt + 1,
            attempts,
            result.stderr.strip()[:200],
        )
        if attempt < attempts - 1:
            time.sleep(_RETRY_BACKOFF_SECONDS[attempt])

    logger.error("curl gave up after %d attempts: %s", attempts, url)
    return None


def extract_supporters(item_url: str) -> List[str]:
    """Extract supporter usernames from an item page.
    
    Uses curl instead of Selenium for better performance and no popups.
    
    Args:
        item_url: URL of the Bandcamp item
        
    Returns:
        List of supporter usernames
    """
    html = fetch_page_html(item_url)
    if not html:
        return []

    soup = BeautifulSoup(html, features="html.parser")
    supporters = []
    
    # Extract from collectors-data JSON blob (most reliable)
    collectors_data = soup.find("div", id="collectors-data")
    if collectors_data:
        data_blob = collectors_data.get("data-blob")
        if data_blob:
            try:
                collectors_json = json.loads(data_blob)
                # Extract usernames from thumbs array
                thumbs = collectors_json.get("thumbs", [])
                for thumb in thumbs:
                    username = thumb.get("username")
                    if username:
                        supporters.append(username)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("error parsing collectors-data: %s", e)
    
    # Fallback - look for links with class "fan pic" or near supporter thumbnails
    if not supporters:
        # Try fan pic links first
        fan_links = soup.find_all("a", class_=re.compile("fan.*pic|pic.*fan"))
        for link in fan_links:
            href = link.get("href", "")
            # Extract username from href like https://bandcamp.com/username?from=...
            match = re.search(r"bandcamp\.com/([^/?]+)", href)
            if match:
                username = match.group(1)
                if username and username != "compliments":  # Exclude special accounts
                    supporters.append(username)
        
        # If still no supporters, look for "supported by" section (track pages)
        if not supporters:
            # Find the section containing "supported by" text
            for elem in soup.find_all(["div", "section", "span", "p"]):
                text = elem.get_text()
                if "supported by" in text.lower():
                    # Find all links within this section
                    links = elem.find_all("a", href=re.compile(r"bandcamp\.com/[^/?]+"))
                    for link in links:
                        href = link.get("href", "")
                        match = re.search(r"bandcamp\.com/([^/?]+)", href)
                        if match:
                            username = match.group(1)
                            # Exclude common non-supporter links
                            excluded = ["artists", "music", "merch", "community", "partner", 
                                       "sign", "log", "help", "settings", "compliments", 
                                       "album", "track", "EmbeddedPlayer"]
                            if username and username not in excluded:
                                supporters.append(username)
                    break  # Found the section, no need to continue
        
        # Final fallback - look for links near thumbnail images (works for track pages)
        if not supporters:
            # Find thumbnail images and get their parent links
            thumbnails = soup.find_all("img", alt=re.compile(".*thumbnail"))
            for thumb in thumbnails:
                # Check parent link
                parent = thumb.parent
                if parent and parent.name == "a":
                    href = parent.get("href", "")
                    match = re.search(r"bandcamp\.com/([^/?]+)", href)
                    if match:
                        username = match.group(1)
                        # Exclude common non-supporter links
                        excluded = ["artists", "music", "merch", "community", "partner", 
                                   "sign", "log", "help", "settings", "compliments",
                                   "album", "track", "EmbeddedPlayer", "discover"]
                        if username and username not in excluded:
                            supporters.append(username)
                # Also check if thumbnail is in a link itself
                elif thumb.parent and thumb.parent.parent:
                    grandparent = thumb.parent.parent
                    if grandparent.name == "a":
                        href = grandparent.get("href", "")
                        match = re.search(r"bandcamp\.com/([^/?]+)", href)
                        if match:
                            username = match.group(1)
                            excluded = ["artists", "music", "merch", "community", "partner", 
                                       "sign", "log", "help", "settings", "compliments",
                                       "album", "track", "EmbeddedPlayer", "discover"]
                            if username and username not in excluded:
                                supporters.append(username)

    # Selenium fallback if curl returned no supporters (e.g. datacenter IP blocked by Bandcamp)
    if not supporters:
        selenium_html = _fetch_page_with_selenium(item_url)
        if selenium_html:
            supporters = _parse_supporters_from_html(selenium_html)

    # Remove duplicates while preserving order
    seen = set()
    unique_supporters = []
    for supporter in supporters:
        if supporter not in seen:
            seen.add(supporter)
            unique_supporters.append(supporter)

    return unique_supporters


def _parse_supporters_from_html(html: str) -> List[str]:
    """Parse supporter usernames from raw HTML."""
    soup = BeautifulSoup(html, features="html.parser")
    supporters = []

    collectors_data = soup.find("div", id="collectors-data")
    if collectors_data:
        data_blob = collectors_data.get("data-blob")
        if data_blob:
            try:
                collectors_json = json.loads(data_blob)
                for thumb in collectors_json.get("thumbs", []):
                    username = thumb.get("username")
                    if username:
                        supporters.append(username)
            except (json.JSONDecodeError, KeyError):
                pass

    if not supporters:
        fan_links = soup.find_all("a", class_=re.compile("fan.*pic|pic.*fan"))
        for link in fan_links:
            href = link.get("href", "")
            match = re.search(r"bandcamp\.com/([^/?]+)", href)
            if match:
                username = match.group(1)
                if username and username != "compliments":
                    supporters.append(username)

    return supporters


def _fetch_page_with_selenium(url: str) -> Optional[str]:
    """Fetch page HTML using Selenium. Fallback for when curl is blocked (e.g. datacenter IPs).

    Always cleans up the driver — even when ``driver.get(...)`` raises
    ``TimeoutException`` because the page-load timeout fired. Pre-hardening
    this routine could leak a chromedriver on every hung Bandcamp request.
    """
    from .driver_manager import DriverManager
    from selenium.common.exceptions import TimeoutException, WebDriverException

    dm = DriverManager()
    try:
        dm.init_driver()
    except WebDriverException:
        logger.warning("Selenium fallback: driver init failed", exc_info=_debug_exc_info())
        return None

    try:
        dm.driver.get(url)
        time.sleep(3)
        return dm.driver.page_source
    except TimeoutException:
        logger.warning("Selenium fallback: page-load timeout on %s", url)
        return None
    except WebDriverException as e:
        logger.warning("Selenium fallback: webdriver error on %s: %s", url, e)
        return None
    except Exception:
        logger.warning("Selenium fallback failed", exc_info=_debug_exc_info())
        return None
    finally:
        try:
            dm.close()
        except Exception:
            logger.warning("Selenium fallback: driver close failed", exc_info=_debug_exc_info())


def extract_item_id(item_url: str) -> Optional[str]:
    """Extract tralbum_id from an item URL or page.
    
    Uses curl instead of Selenium.
    
    Args:
        item_url: URL of the Bandcamp item
        
    Returns:
        tralbum_id as string, or None if not found
    """
    html = fetch_page_html(item_url, timeout=_DEFAULT_CURL_TIMEOUT)
    if not html:
        return None

    try:
        soup = BeautifulSoup(html, features="html.parser")
        pagedata_elem = soup.find(id="pagedata")
        if pagedata_elem:
            pagedata = json.loads(pagedata_elem.get("data-blob", "{}"))
            # Try multiple possible locations for tralbum_id
            tralbum_id = None
            
            # Try tralbum_data first
            tralbum_data = pagedata.get("tralbum_data")
            if isinstance(tralbum_data, dict):
                tralbum_id = tralbum_data.get("tralbum_id")
            
            # Try fan_tralbum_data
            if not tralbum_id:
                fan_tralbum_data = pagedata.get("fan_tralbum_data")
                if isinstance(fan_tralbum_data, dict):
                    tralbum_id = fan_tralbum_data.get("tralbum_id")
            
            # Try album_id as fallback
            if not tralbum_id:
                tralbum_id = pagedata.get("album_id")
            
            if tralbum_id:
                return str(tralbum_id)
    except Exception:
        logger.warning("error extracting item id from %s", item_url, exc_info=_debug_exc_info())

    return None


def extract_tags(item_url: str) -> List[str]:
    """Extract tags from a Bandcamp item page.
    
    Tags are extracted from DOM elements with class 'tag'.
    
    Args:
        item_url: URL of the Bandcamp item
        
    Returns:
        List of tag strings, or empty list if not found
    """
    html = fetch_page_html(item_url, timeout=_DEFAULT_CURL_TIMEOUT)
    if not html:
        return []
    
    try:
        soup = BeautifulSoup(html, features="html.parser")
        
        # Extract tags from DOM elements with class 'tag'
        tag_links = soup.find_all("a", class_=re.compile("tag"))
        tags = [tag.get_text(strip=True) for tag in tag_links if tag.get_text(strip=True)]
        
        return tags
    except Exception:
        logger.warning("error extracting tags from %s", item_url, exc_info=_debug_exc_info())
        return []


