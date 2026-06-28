FROM python:3.13-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN mkdir -p /data /downloads /plugins

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

ENV DOWNLOAD_DIR=/downloads \
    PLUGINS_DIR=/plugins \
    DB_PATH=/data/scrap_downloader.db \
    HOST=0.0.0.0 \
    PORT=8080

EXPOSE 8080

CMD ["uv", "run", "scrap-downloader"]
