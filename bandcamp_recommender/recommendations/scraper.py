"""Web scraping utilities for Bandcamp pages."""

import json
import logging
import os
import re
from typing import Dict, List, Optional

from bs4 import BeautifulSoup

from .. import http


logger = logging.getLogger(__name__)


def _debug_exc_info() -> bool:
    return os.environ.get("BANDCAMP_DEBUG") == "1"


# Default curl timeout, raised 15→20 to accommodate slow Bandcamp custom
# domains (e.g. craigieknowes.com routinely takes 15–20s).
_DEFAULT_CURL_TIMEOUT = 20


# Every Bandcamp item page carries this attribute; a 200 without it is not
# the page (bot challenge, error shell, parked domain).
ITEM_PAGE_MARKER = "data-tralbum"


def clear_page_cache() -> None:
    """Drop every cached page (tests + long-running consumers)."""
    http.clear_page_cache()


def fetch_page_html(url: str, timeout: int = _DEFAULT_CURL_TIMEOUT,
                    expect: Optional[str] = ITEM_PAGE_MARKER) -> Optional[str]:
    """Fetch a Bandcamp page through the shared policy
    (:func:`bandcamp_recommender.http.get_page`: plain HTTP, payload
    validation, short-lived cache, browser fallback on a bot challenge).

    Returns None on any failure; callers treat None as "no data". The
    default ``expect`` suits item (album/track) pages; pass another marker,
    or None, for other page types."""
    try:
        return http.get_page(url, expect=expect, timeout=timeout)
    except http.FetchError as e:
        logger.warning("page fetch failed: %s", e)
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

    # No Selenium retry here: a challenged fetch already went through the
    # browser inside fetch_page_html, so an empty list is a real "no
    # supporters" (or a failed fetch, logged there).

    # Remove duplicates while preserving order
    seen = set()
    unique_supporters = []
    for supporter in supporters:
        if supporter not in seen:
            seen.add(supporter)
            unique_supporters.append(supporter)

    return unique_supporters


def supporter_fan_ids(item_url: str) -> Dict[str, int]:
    """``{username: fan_id}`` for the supporters shown on an item page, read
    from the page cache only (``extract_supporters`` just fetched it). Lets a
    supporter's collection be read with ONE API call instead of fan page +
    API call. Empty when the page isn't cached or has no ids."""
    html = http.peek_page(item_url)
    if not html:
        return {}
    soup = BeautifulSoup(html, features="html.parser")
    collectors_data = soup.find("div", id="collectors-data")
    if not collectors_data:
        return {}
    try:
        blob = json.loads(collectors_data.get("data-blob") or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}
    out: Dict[str, int] = {}
    for thumb in blob.get("thumbs", []) or []:
        username, fan_id = thumb.get("username"), thumb.get("fan_id")
        if username and fan_id:
            out[username] = fan_id
    return out


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


def _fetch_page_with_selenium(url: str, expect: Optional[str] = None) -> Optional[str]:
    """Fetch page HTML with the shared fallback browser (kept for callers of
    the old helper; :func:`fetch_page_html` already falls back on its own)."""
    from bandcamp_recommender import browser

    try:
        return browser.fetch_html(url, expect)
    except http.FetchError as e:
        logger.warning("Selenium fallback failed for %s: %s", url, e)
        return None


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


