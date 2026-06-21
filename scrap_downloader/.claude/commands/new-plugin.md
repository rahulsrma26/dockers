Create a new domain plugin for scrap_downloader.

## Usage

/new-plugin <domain>

Example: `/new-plugin example.com`

## What to do

1. Create `plugins/<domain>.py` based on `plugins/example.com.py`.
2. Implement `extract(url)` to return a list of `(download_url, filename_or_None, headers_or_None)` tuples:
   - Images: `(direct_url, "filename.ext", headers_dict_or_None)`
   - Videos: `(video_url, None, None)` — yt-dlp handles naming/download
3. The file must be named exactly `<domain>.py` (e.g. `example.com.py`). `www.` prefix is stripped automatically by the worker.
4. Use `httpx`, `beautifulsoup4`, `requests`, or `cloudscraper` for scraping — all are already in `pyproject.toml`.
5. Test by running the app (`/run`) and submitting a URL from that domain in the Auto tab.

## Plugin contract

```python
def extract(url: str) -> list[tuple]:
    # (url, type, filename, headers) — filename and headers optional
    # type must be "video" or "image"
    (url, "video")                          # yt-dlp, auto-named
    (url, "video", "My Title")             # yt-dlp, custom filename stem
    (url, "video", "My Title", headers)    # yt-dlp with headers
    (url, "image", "photo.jpg")            # httpx
    (url, "image", "photo.jpg", headers)   # httpx with headers
```
