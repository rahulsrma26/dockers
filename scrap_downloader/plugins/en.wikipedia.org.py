import os
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

INFOBOX_SELECTOR = "#mw-content-text > div.mw-content-ltr.mw-parser-output > table.infobox"


DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; scrap-downloader/1.0)",
    "Referer": "https://en.wikipedia.org/",
}


def extract(url: str) -> list[tuple]:
    response = httpx.get(url, headers=DOWNLOAD_HEADERS, follow_redirects=True, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")
    infobox = soup.select_one(INFOBOX_SELECTOR)
    if not infobox:
        return []

    results = []
    for img in infobox.select("img"):
        src = img.get("src", "")
        if not src:
            continue
        if src.startswith("//"):
            src = "https:" + src
        filename = os.path.basename(urlparse(src).path)
        results.append((src, "image", filename, DOWNLOAD_HEADERS))

    return results[:1]  # first image only for now
