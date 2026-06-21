# scrap_downloader

Personal self-hosted archival server for images and videos. Paste a URL + tag → queues and downloads to an organized archive. Supports 1000+ sites via yt-dlp, gallery-dl, and a per-domain plugin system.

## Stack

- **UI:** NiceGUI (pure Python, `main.py`)
- **DB:** SQLite via SQLAlchemy (`db.py`) — single user, no concurrency needed
- **Worker:** `threading.Thread` in-process background worker, polls DB every 2s (`worker.py`)
- **HTTP:** `httpx` for images, `yt-dlp` for videos, `gallery-dl` for image galleries
- **Auth:** Optional password via `APP_PASSWORD` env var, NiceGUI cookie storage

## File layout

```
main.py         # NiceGUI app, UI, routes, startup hooks
db.py           # SQLAlchemy Task model, TaskStatus/TaskTool enums
worker.py       # background thread: plugin → yt-dlp fallback → gallery-dl tab
downloader.py   # download_image (httpx), download_video (yt-dlp)
plugins/        # domain plugins, auto-seeded to PLUGINS_DIR on startup
Dockerfile      # python:3.13-slim + ffmpeg + uv
docker-compose.yml
```

## Plugins

Each `plugins/<domain>.py` exposes a single function:

```python
def extract(url: str) -> list[tuple]:
    # (url, type, filename, headers) — filename and headers are optional
    # type must be "video" or "image"
```

Copy `plugins/example.com.py` as the starting template. Name the file `<domain>.py` (e.g. `reddit.com.py`). `www.` prefix is stripped automatically.

**Worker fallback chain (Auto tab):**
1. Match domain to `plugins/<domain>.py` → use plugin
2. No plugin → yt-dlp fallback (handles 1000+ video sites)

## Env vars

| Var | Default | Purpose |
|-----|---------|---------|
| `APP_PASSWORD` | _(empty = no auth)_ | Optional login password |
| `STORAGE_SECRET` | `change-me-in-prod` | NiceGUI cookie signing key |
| `SESSION_TIMEOUT_MINUTES` | `60` | Inactivity logout timeout |
| `PORT` | `8080` | Listen port |
| `DATA_DIR` | `./data` | Host path for db (`/data` in container) |
| `DOWNLOAD_DIR` | `./downloads` | Host path for archive (`/downloads` in container) |
| `PLUGINS_DIR` | `./plugins` | Host path for plugins (`/plugins` in container) |

## Dev

```bash
uv sync                 # install/sync deps
uv run python main.py   # run locally (set DEV=1 for auto-reload)
uv run pytest           # run tests
```

## Docker

### docker run

```bash
mkdir -p data downloads plugins
docker run -d \
  -p 8080:8080 \
  -v $(pwd)/data:/data \
  -v $(pwd)/downloads:/downloads \
  -v $(pwd)/plugins:/plugins \
  -e APP_PASSWORD=secret \
  -e STORAGE_SECRET=change-me \
  --name scrap-downloader \
  --restart unless-stopped \
  welcometors/scrap_downloader:latest
```

To archive to a network share, replace `$(pwd)/downloads` with the mount path:

```bash
  -v /mnt/nas/media:/downloads \
```

### docker compose

Copy `.env.example` to `.env` and edit, then:

```bash
docker compose up -d                          # start
docker compose down                           # stop
docker compose pull && docker compose up -d   # update to latest image
```

Set `DOWNLOAD_DIR` in `.env` to point to a NAS or any other path. `DATA_DIR` and `PLUGINS_DIR` can be overridden independently.

Single container with ffmpeg bundled. Bundled plugins are seeded to the plugins dir on first start.
