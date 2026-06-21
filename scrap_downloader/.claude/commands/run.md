Run the scrap_downloader app locally for development.

```bash
uv sync
uv run python main.py
```

The app starts on http://localhost:8080. Set `DEV=1` in the environment for auto-reload. Set `APP_PASSWORD` if you want the login prompt.

After starting, open the browser and verify:
- The queue table loads (even if empty)
- The Auto and gallery-dl tabs are both present
- Submitting a URL queues a task and it appears in the table
