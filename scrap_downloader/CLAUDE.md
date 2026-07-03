# CLAUDE.md

See README.md for project docs, stack, and architecture.

## Rules

- Always use `uv` — never `pip` directly (`uv add`, `uv sync`, `uv run`)
- Format and lint with `uv run ruff format . && uv run ruff check --fix .` (runs automatically via PostToolUse hook after every `.py` edit)
- **Never drop imports when editing a file.** Before every Edit that touches the top of a file or replaces a large block, read the current imports first. After the edit, verify the import list is intact. If ruff removes an import as "unused", it means the usage site was also dropped — restore both.
- **Before bumping the version, run `uv run pytest -q` and confirm all tests pass.** Only then bump the patch version in `pyproject.toml` (e.g. `0.1.0` → `0.1.1`). The UI reads the version at startup via `importlib.metadata`.

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
