"""
Example plugin for example.com.
Copy this file, rename it to <domain>.py (e.g. reddit.com.py),
and implement extract() to return a list of tuples or DownloadItems.

Tuple format:
  (url, "video")                         # yt-dlp, auto-named
  (url, "video", "My Title")             # yt-dlp, custom filename stem
  (url, "video", "My Title", headers)    # yt-dlp, custom name + headers
  (url, "image", "filename.jpg")         # httpx download
  (url, "image", "filename.jpg", headers) # httpx with headers
"""


def extract(url: str) -> list[tuple]:
    return []
