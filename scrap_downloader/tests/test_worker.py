import textwrap

import scrap_downloader.worker as worker_mod
from scrap_downloader.worker import _load_plugin, _parse_media_type


def test_load_plugin_found(tmp_path, monkeypatch):
    plugin_file = tmp_path / "example.com.py"
    plugin_file.write_text(
        textwrap.dedent("""
            def extract(url):
                return [(url, "video")]
        """)
    )
    monkeypatch.setattr(worker_mod, "PLUGINS_DIR", str(tmp_path))

    plugin = _load_plugin("example.com")
    assert plugin is not None
    result = plugin.extract("https://example.com/v.mp4")
    assert result == [("https://example.com/v.mp4", "video")]


def test_load_plugin_strips_www(tmp_path, monkeypatch):
    plugin_file = tmp_path / "example.com.py"
    plugin_file.write_text("def extract(url): return []")

    monkeypatch.setattr(worker_mod, "PLUGINS_DIR", str(tmp_path))

    plugin = _load_plugin("www.example.com")
    assert plugin is not None


def test_load_plugin_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_mod, "PLUGINS_DIR", str(tmp_path))

    plugin = _load_plugin("notfound.com")
    assert plugin is None


# ── _parse_media_type ─────────────────────────────────────────────────────────


def test_parse_media_type_none():
    assert _parse_media_type(None) == ""


def test_parse_media_type_empty():
    assert _parse_media_type("") == ""


def test_parse_media_type_images():
    assert _parse_media_type("--media-type images") == "images"


def test_parse_media_type_videos():
    assert _parse_media_type("--media-type videos") == "videos"


def test_parse_media_type_missing_value():
    assert _parse_media_type("--media-type") == ""


def test_parse_media_type_unrelated_args():
    assert _parse_media_type("--audio-only") == ""


# ── _run_auto media type filtering ────────────────────────────────────────────


def _make_task(extra_args=None, url="https://example.com/page"):
    import types

    return types.SimpleNamespace(id=1, url=url, tag="test", extra_args=extra_args)


def test_run_auto_images_only_filters_videos(tmp_path, monkeypatch):
    """Images-only filter: video items from plugin are dropped."""
    from scrap_downloader.downloader import DownloadItem

    plugin_items = [
        DownloadItem(url="http://x.com/a.jpg", type="image", filename="a.jpg", headers=None),
        DownloadItem(url="http://x.com/b.mp4", type="video", filename="b.mp4", headers=None),
    ]

    downloaded = []

    def fake_plugin_extract(_url):
        return plugin_items

    import types

    fake_plugin = types.SimpleNamespace(extract=fake_plugin_extract)
    monkeypatch.setattr(worker_mod, "_load_plugin", lambda domain: fake_plugin)
    monkeypatch.setattr(worker_mod, "DOWNLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(
        worker_mod, "download_image", lambda url, d, fn, headers=None: downloaded.append(url)
    )
    monkeypatch.setattr(worker_mod, "download_video", lambda *a, **kw: downloaded.append(a[0]))

    task = _make_task(extra_args="--media-type images")
    worker_mod._run_auto(task)

    assert downloaded == ["http://x.com/a.jpg"]


def test_run_auto_videos_only_filters_images(tmp_path, monkeypatch):
    """Videos-only filter: image items from plugin are dropped."""
    from scrap_downloader.downloader import DownloadItem

    plugin_items = [
        DownloadItem(url="http://x.com/a.jpg", type="image", filename="a.jpg", headers=None),
        DownloadItem(url="http://x.com/b.mp4", type="video", filename="b.mp4", headers=None),
    ]

    downloaded = []

    import types

    fake_plugin = types.SimpleNamespace(extract=lambda url: plugin_items)
    monkeypatch.setattr(worker_mod, "_load_plugin", lambda domain: fake_plugin)
    monkeypatch.setattr(worker_mod, "DOWNLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(
        worker_mod, "download_image", lambda url, d, fn, headers=None: downloaded.append(url)
    )
    monkeypatch.setattr(worker_mod, "download_video", lambda *a, **kw: downloaded.append(a[0]))

    task = _make_task(extra_args="--media-type videos")
    worker_mod._run_auto(task)

    assert downloaded == ["http://x.com/b.mp4"]


def test_run_auto_no_filter_downloads_all(tmp_path, monkeypatch):
    """No filter: all items downloaded regardless of type."""
    from scrap_downloader.downloader import DownloadItem

    plugin_items = [
        DownloadItem(url="http://x.com/a.jpg", type="image", filename="a.jpg", headers=None),
        DownloadItem(url="http://x.com/b.mp4", type="video", filename="b.mp4", headers=None),
    ]

    downloaded = []

    import types

    fake_plugin = types.SimpleNamespace(extract=lambda url: plugin_items)
    monkeypatch.setattr(worker_mod, "_load_plugin", lambda domain: fake_plugin)
    monkeypatch.setattr(worker_mod, "DOWNLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(
        worker_mod, "download_image", lambda url, d, fn, headers=None: downloaded.append(url)
    )
    monkeypatch.setattr(worker_mod, "download_video", lambda *a, **kw: downloaded.append(a[0]))

    task = _make_task(extra_args=None)
    worker_mod._run_auto(task)

    assert set(downloaded) == {"http://x.com/a.jpg", "http://x.com/b.mp4"}


def test_run_auto_images_only_no_plugin_skips_ytdlp(tmp_path, monkeypatch):
    """Images-only with no plugin: yt-dlp fallback is skipped (yt-dlp only gives videos)."""
    downloaded = []
    monkeypatch.setattr(worker_mod, "_load_plugin", lambda domain: None)
    monkeypatch.setattr(worker_mod, "DOWNLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(worker_mod, "download_video", lambda *a, **kw: downloaded.append(a[0]))

    task = _make_task(extra_args="--media-type images")
    worker_mod._run_auto(task)

    assert downloaded == []


def test_run_auto_videos_only_no_plugin_uses_ytdlp(tmp_path, monkeypatch):
    """Videos-only with no plugin: yt-dlp fallback still runs."""
    downloaded = []
    monkeypatch.setattr(worker_mod, "_load_plugin", lambda domain: None)
    monkeypatch.setattr(worker_mod, "DOWNLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(worker_mod, "download_video", lambda *a, **kw: downloaded.append(a[0]))

    task = _make_task(extra_args="--media-type videos")
    worker_mod._run_auto(task)

    assert downloaded == [task.url]
