import logging
import os
from dataclasses import dataclass
from typing import Callable, Literal

import httpx
import yt_dlp

logger = logging.getLogger(__name__)


@dataclass
class DownloadItem:
    url: str
    type: Literal["video", "image"]
    filename: str | None = None  # images: output name; videos: yt-dlp stem (None = auto)
    headers: dict | None = None


def _item_from_tuple(t: tuple) -> DownloadItem:
    if len(t) < 2:
        raise ValueError(f"Plugin tuple must have at least (url, type), got: {t!r}")
    url, type_ = t[0], t[1]
    if type_ not in ("video", "image"):
        raise ValueError(f"Invalid type {type_!r}, must be 'video' or 'image'")
    filename = t[2] if len(t) > 2 else None
    headers = t[3] if len(t) > 3 else None
    return DownloadItem(url=url, type=type_, filename=filename, headers=headers)


def to_download_item(item) -> DownloadItem:
    if isinstance(item, DownloadItem):
        return item
    if isinstance(item, tuple):
        return _item_from_tuple(item)
    raise TypeError(f"Plugin must return DownloadItem or tuple, got {type(item)!r}")


def download_image(
    url: str,
    save_dir: str,
    filename: str | None = None,
    headers: dict | None = None,
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    if not filename:
        from urllib.parse import urlparse

        filename = os.path.basename(urlparse(url).path) or "image"
    save_path = os.path.join(save_dir, filename)
    with httpx.stream("GET", url, headers=headers or {}, follow_redirects=True, timeout=60) as r:
        r.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=65536):
                f.write(chunk)
    logger.info(f"Saved image: {save_path}")
    return save_path


def download_video(
    url: str,
    save_dir: str,
    on_progress: Callable[[float], None] | None = None,
    filename: str | None = None,
    audio_only: bool = False,
) -> str:
    os.makedirs(save_dir, exist_ok=True)

    def _progress_hook(d: dict):
        if on_progress and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes", 0)
            if total:
                on_progress(downloaded / total * 100)
        elif on_progress and d.get("status") == "finished":
            on_progress(100.0)

    outtmpl = os.path.join(save_dir, f"{filename}.%(ext)s" if filename else "%(title)s.%(ext)s")
    ydl_opts = {
        "outtmpl": outtmpl,
        "progress_hooks": [_progress_hook],
        "quiet": True,
        "no_warnings": True,
    }
    if audio_only:
        ydl_opts["format"] = "bestaudio/best"

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        out = ydl.prepare_filename(info)

    logger.info(f"Saved video: {out}")
    return out
