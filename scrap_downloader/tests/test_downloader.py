import pytest

from scrap_downloader.downloader import DownloadItem, to_download_item


def test_video_minimal():
    item = to_download_item(("https://example.com/v.mp4", "video"))
    assert item.type == "video"
    assert item.url == "https://example.com/v.mp4"
    assert item.filename is None
    assert item.headers is None


def test_video_with_filename():
    item = to_download_item(("https://example.com/v.mp4", "video", "My Title"))
    assert item.filename == "My Title"
    assert item.headers is None


def test_video_with_headers():
    h = {"Referer": "https://example.com"}
    item = to_download_item(("https://example.com/v.mp4", "video", "My Title", h))
    assert item.filename == "My Title"
    assert item.headers == h


def test_image_minimal():
    item = to_download_item(("https://example.com/photo.jpg", "image", "photo.jpg"))
    assert item.type == "image"
    assert item.filename == "photo.jpg"
    assert item.headers is None


def test_image_with_headers():
    h = {"Referer": "https://example.com"}
    item = to_download_item(("https://example.com/photo.jpg", "image", "photo.jpg", h))
    assert item.headers == h


def test_passthrough_download_item():
    original = DownloadItem(url="https://example.com/v.mp4", type="video")
    assert to_download_item(original) is original


def test_invalid_type():
    with pytest.raises(ValueError, match="Invalid type"):
        to_download_item(("https://example.com/v.mp4", "audio"))


def test_missing_type():
    with pytest.raises(ValueError, match="at least"):
        to_download_item(("https://example.com/v.mp4",))


def test_wrong_input_type():
    with pytest.raises(TypeError):
        to_download_item("not-a-tuple")
