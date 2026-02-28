"""Web scraping utilities for Bandcamp pages."""

import json
import os
import re
import shutil
import subprocess
import time
from typing import List, Optional

from bs4 import BeautifulSoup


def fetch_page_html(url: str, timeout: int = 15) -> Optional[str]:
    """Fetch HTML content from a URL using curl.
    
    Args:
        url: URL to fetch
        timeout: Request timeout in seconds
        
    Returns:
        HTML content as string, or None if failed
    """
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
    
    try:
        result = subprocess.run(
            curl_cmd, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            print(f"Error fetching page with curl: {result.stderr}")
            return None
        return result.stdout
    except Exception as e:
        print(f"Error running curl: {e}")
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
                print(f"Error parsing collectors-data: {e}")
    
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
    """Fetch page HTML using Selenium. Fallback for when curl is blocked (e.g. datacenter IPs)."""
    try:
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from selenium import webdriver

        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-dev-shm-usage")

        for path in ["/usr/bin/chromium-browser", "/usr/bin/chromium", "/usr/bin/google-chrome",
                     "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]:
            if os.path.isfile(path):
                options.binary_location = path
                break

        chromedriver_path = shutil.which("chromedriver")
        if chromedriver_path:
            service = Service(chromedriver_path)
        else:
            from webdriver_manager.chrome import ChromeDriverManager
            service = Service(ChromeDriverManager().install())

        driver = webdriver.Chrome(service=service, options=options)
        driver.get(url)
        time.sleep(3)
        html = driver.page_source
        driver.quit()
        return html
    except Exception as e:
        print(f"Selenium fallback failed: {e}")
        return None


def extract_item_id(item_url: str) -> Optional[str]:
    """Extract tralbum_id from an item URL or page.
    
    Uses curl instead of Selenium.
    
    Args:
        item_url: URL of the Bandcamp item
        
    Returns:
        tralbum_id as string, or None if not found
    """
    html = fetch_page_html(item_url, timeout=10)
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
    except Exception as e:
        print(f"Error extracting item ID: {e}")
        pass

    return None


def extract_tags(item_url: str) -> List[str]:
    """Extract tags from a Bandcamp item page.
    
    Tags are extracted from DOM elements with class 'tag'.
    
    Args:
        item_url: URL of the Bandcamp item
        
    Returns:
        List of tag strings, or empty list if not found
    """
    html = fetch_page_html(item_url, timeout=10)
    if not html:
        return []
    
    try:
        soup = BeautifulSoup(html, features="html.parser")
        
        # Extract tags from DOM elements with class 'tag'
        tag_links = soup.find_all("a", class_=re.compile("tag"))
        tags = [tag.get_text(strip=True) for tag in tag_links if tag.get_text(strip=True)]
        
        return tags
    except Exception as e:
        # Log error for debugging but don't fail silently
        print(f"Error extracting tags from {item_url}: {e}")
        return []


