# CLAUDE.md

See README.md for project docs, stack, and architecture.

## Rules

- Always use `uv` — never `pip` directly (`uv add`, `uv sync`, `uv run`)
- Format and lint with `uv run ruff format . && uv run ruff check --fix .` (runs automatically via PostToolUse hook after every `.py` edit)

## Plugin contract

When adding a new domain plugin, create `plugins/<domain>.py` (e.g. `reddit.com.py`):

```python
def extract(url: str) -> list[tuple]:
    # (url, type, filename, headers)  — filename and headers are optional
    # type must be "video" or "image" (enforced by worker)
```

Examples:
```python
(url, "video")                           # yt-dlp, auto-named
(url, "video", "My Title")              # yt-dlp, saved as "My Title.<ext>"
(url, "video", "My Title", headers)     # yt-dlp with request headers
(url, "image", "photo.jpg")             # httpx
(url, "image", "photo.jpg", headers)    # httpx with headers
```

Available scraping libs: `httpx`, `beautifulsoup4`, `requests`, `cloudscraper` (all in `pyproject.toml`).
