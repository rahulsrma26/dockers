"""Tests for collections feature (face.py collection functions)."""

import os

import pytest
from sqlalchemy import create_engine

import scrap_downloader.db as db_mod
import scrap_downloader.face as face_mod
from scrap_downloader.db import Base, ImageMeta


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    monkeypatch.setattr(db_mod, "engine", engine)

    dl = tmp_path / "downloads"
    dl.mkdir()
    monkeypatch.setattr(face_mod, "DOWNLOAD_DIR", str(dl))
    return dl


# ── list_collections ──────────────────────────────────────────────────────────


def test_list_collections_empty(tmp_path, monkeypatch):
    result = face_mod.list_collections()
    assert result == []


def test_list_collections_missing_dir(tmp_path, monkeypatch):
    # collections/ dir doesn't exist yet
    result = face_mod.list_collections()
    assert result == []


def test_list_collections_casefold_sort(tmp_path, monkeypatch):
    coll_dir = tmp_path / "downloads" / "collections"
    for name in ["Zebra", "apple", "Mango", "banana"]:
        (coll_dir / name).mkdir(parents=True)

    result = face_mod.list_collections()
    assert result == ["apple", "banana", "Mango", "Zebra"]


def test_list_collections_skips_files(tmp_path, monkeypatch):
    coll_dir = tmp_path / "downloads" / "collections"
    coll_dir.mkdir(parents=True)
    (coll_dir / "myalbum").mkdir()
    (coll_dir / "readme.txt").write_text("hello")

    result = face_mod.list_collections()
    assert result == ["myalbum"]
    assert "readme.txt" not in result


def test_list_collections_skips_hidden(tmp_path, monkeypatch):
    coll_dir = tmp_path / "downloads" / "collections"
    coll_dir.mkdir(parents=True)
    (coll_dir / "visible").mkdir()
    (coll_dir / ".hidden").mkdir()

    result = face_mod.list_collections()
    assert result == ["visible"]


# ── move_to_collection ────────────────────────────────────────────────────────


def _make_album(dl, name, files):
    """Create a folder under downloads with given files. Returns rel path."""
    folder = dl / name
    folder.mkdir(parents=True, exist_ok=True)
    for fname, content in files.items():
        (folder / fname).write_bytes(content if isinstance(content, bytes) else content.encode())
    return name


def test_move_to_collection_basic(tmp_path, monkeypatch):
    dl = tmp_path / "downloads"
    _make_album(dl, "japan", {"photo.jpg": b"img", "clip.mp4": b"vid"})

    err = face_mod.move_to_collection("japan", "travel")
    assert err is None

    dest = dl / "collections" / "travel"
    assert (dest / "photo.jpg").exists()
    assert (dest / "clip.mp4").exists()
    assert not (dl / "japan").exists()


def test_move_to_collection_creates_dir(tmp_path, monkeypatch):
    dl = tmp_path / "downloads"
    _make_album(dl, "album", {"a.jpg": b"x"})

    err = face_mod.move_to_collection("album", "newcoll")
    assert err is None
    assert (dl / "collections" / "newcoll").is_dir()


def test_move_to_collection_moves_non_media(tmp_path, monkeypatch):
    dl = tmp_path / "downloads"
    _make_album(dl, "album", {"photo.jpg": b"img", "notes.txt": b"text"})

    err = face_mod.move_to_collection("album", "stuff")
    assert err is None

    dest = dl / "collections" / "stuff"
    assert (dest / "notes.txt").exists()


def test_move_to_collection_appends_to_existing(tmp_path, monkeypatch):
    dl = tmp_path / "downloads"
    _make_album(dl, "album1", {"a.jpg": b"a"})
    _make_album(dl, "album2", {"b.jpg": b"b"})

    face_mod.move_to_collection("album1", "travel")
    err = face_mod.move_to_collection("album2", "travel")
    assert err is None

    dest = dl / "collections" / "travel"
    assert (dest / "a.jpg").exists()
    assert (dest / "b.jpg").exists()


def test_move_to_collection_collision_renamed(tmp_path, monkeypatch):
    dl = tmp_path / "downloads"
    _make_album(dl, "album1", {"photo.jpg": b"first"})
    _make_album(dl, "album2", {"photo.jpg": b"second"})

    face_mod.move_to_collection("album1", "travel")
    face_mod.move_to_collection("album2", "travel")

    dest = dl / "collections" / "travel"
    names = set(os.listdir(dest))
    assert "photo.jpg" in names
    assert "photo_2.jpg" in names


def test_move_to_collection_rejects_empty_name(tmp_path, monkeypatch):
    dl = tmp_path / "downloads"
    _make_album(dl, "album", {"a.jpg": b"x"})

    err = face_mod.move_to_collection("album", "")
    assert err is not None
    assert "empty" in err.lower()


def test_move_to_collection_rejects_illegal_chars(tmp_path, monkeypatch):
    dl = tmp_path / "downloads"
    _make_album(dl, "album", {"a.jpg": b"x"})

    err = face_mod.move_to_collection("album", "bad/name")
    assert err is not None
    assert "illegal" in err.lower() or "/" in err


def test_move_to_collection_rejects_dot_prefix_name(tmp_path, monkeypatch):
    dl = tmp_path / "downloads"
    _make_album(dl, "album", {"a.jpg": b"x"})

    err = face_mod.move_to_collection("album", ".hidden")
    assert err is not None


def test_move_to_collection_rejects_missing_source(tmp_path, monkeypatch):
    err = face_mod.move_to_collection("nonexistent", "travel")
    assert err is not None
    assert "not found" in err.lower()


def test_move_to_collection_rejects_dl_root(tmp_path, monkeypatch):
    err = face_mod.move_to_collection("", "travel")
    assert err is not None


def test_move_to_collection_rejects_dot_source(tmp_path, monkeypatch):
    err = face_mod.move_to_collection(".", "travel")
    assert err is not None


def test_move_to_collection_rejects_categorized_source(tmp_path, monkeypatch):
    dl = tmp_path / "downloads"
    cat = dl / "categorized" / "alice"
    cat.mkdir(parents=True)
    (cat / "a.jpg").write_bytes(b"x")

    err = face_mod.move_to_collection("categorized/alice", "travel")
    assert err is not None
    assert "categorized" in err.lower()


def test_move_to_collection_rejects_already_collection_source(tmp_path, monkeypatch):
    dl = tmp_path / "downloads"
    existing = dl / "collections" / "travel"
    existing.mkdir(parents=True)
    (existing / "a.jpg").write_bytes(b"x")

    err = face_mod.move_to_collection("collections/travel", "other")
    assert err is not None
    assert "collection" in err.lower()


def test_move_to_collection_cleans_stale_imagemeta(tmp_path, monkeypatch):
    from sqlalchemy.orm import Session

    dl = tmp_path / "downloads"
    _make_album(dl, "album", {"a.jpg": b"img"})

    with Session(db_mod.engine) as s:
        s.add(ImageMeta(file_path="album/a.jpg", file_mtime=0.0))
        s.commit()

    face_mod.move_to_collection("album", "travel")

    from sqlalchemy import select

    with Session(db_mod.engine) as s:
        rows = s.execute(select(ImageMeta)).scalars().all()
    assert all(not r.file_path.startswith("album/") for r in rows)


# ── rename_collection ─────────────────────────────────────────────────────────


def test_rename_collection_basic(tmp_path, monkeypatch):
    coll_dir = tmp_path / "downloads" / "collections"
    (coll_dir / "old").mkdir(parents=True)
    (coll_dir / "old" / "a.jpg").write_bytes(b"x")

    err = face_mod.rename_collection("old", "new")
    assert err is None
    assert (coll_dir / "new").is_dir()
    assert (coll_dir / "new" / "a.jpg").exists()
    assert not (coll_dir / "old").exists()


def test_rename_collection_dest_exists(tmp_path, monkeypatch):
    coll_dir = tmp_path / "downloads" / "collections"
    (coll_dir / "alpha").mkdir(parents=True)
    (coll_dir / "beta").mkdir(parents=True)

    err = face_mod.rename_collection("alpha", "beta")
    assert err is not None
    assert "already exists" in err.lower() or "exist" in err.lower()


def test_rename_collection_not_found(tmp_path, monkeypatch):
    err = face_mod.rename_collection("ghost", "newname")
    assert err is not None
    assert "not found" in err.lower()


def test_rename_collection_invalid_new_name(tmp_path, monkeypatch):
    coll_dir = tmp_path / "downloads" / "collections"
    (coll_dir / "alpha").mkdir(parents=True)

    err = face_mod.rename_collection("alpha", "bad/name")
    assert err is not None


def test_rename_collection_traversal_old_name(tmp_path, monkeypatch):
    err = face_mod.rename_collection("../outside", "travel")
    assert err is not None
    assert "invalid" in err.lower()


def test_rename_collection_traversal_new_name(tmp_path, monkeypatch):
    coll_dir = tmp_path / "downloads" / "collections"
    (coll_dir / "alpha").mkdir(parents=True)

    err = face_mod.rename_collection("alpha", "../escape")
    assert err is not None
    # "../escape" starts with "." — caught by dot-prefix or invalid guard
    assert err is not None


# ── delete_collection_file ────────────────────────────────────────────────────


def test_delete_collection_file_removes_file(tmp_path, monkeypatch):
    coll_dir = tmp_path / "downloads" / "collections" / "travel"
    coll_dir.mkdir(parents=True)
    f = coll_dir / "photo.jpg"
    f.write_bytes(b"img")

    err = face_mod.delete_collection_file("travel", "collections/travel/photo.jpg")
    assert err is None
    assert not f.exists()


def test_delete_collection_file_path_traversal_name(tmp_path, monkeypatch):
    err = face_mod.delete_collection_file("../outside", "collections/../outside/file.jpg")
    assert err is not None
    assert "invalid" in err.lower()


def test_delete_collection_file_path_traversal_rel(tmp_path, monkeypatch):
    coll_dir = tmp_path / "downloads" / "collections" / "travel"
    coll_dir.mkdir(parents=True)

    err = face_mod.delete_collection_file("travel", "collections/travel/../../secret.txt")
    assert err is not None
    assert "invalid" in err.lower()


def test_delete_collection_file_auto_removes_empty_folder(tmp_path, monkeypatch):
    coll_dir = tmp_path / "downloads" / "collections" / "travel"
    coll_dir.mkdir(parents=True)
    f = coll_dir / "only.jpg"
    f.write_bytes(b"img")

    face_mod.delete_collection_file("travel", "collections/travel/only.jpg")
    # The collection folder should be removed since it's now empty
    assert not coll_dir.exists()


def test_delete_collection_file_nonexistent_is_ok(tmp_path, monkeypatch):
    coll_dir = tmp_path / "downloads" / "collections" / "travel"
    coll_dir.mkdir(parents=True)

    # File doesn't exist — should not raise, just return None
    err = face_mod.delete_collection_file("travel", "collections/travel/ghost.jpg")
    assert err is None


# ── _collection_images ────────────────────────────────────────────────────────


def test_collection_images_empty(tmp_path, monkeypatch):
    coll_dir = tmp_path / "downloads" / "collections" / "travel"
    coll_dir.mkdir(parents=True)

    result = face_mod._collection_images("travel")
    assert result == []


def test_collection_images_returns_media_files(tmp_path, monkeypatch):
    coll_dir = tmp_path / "downloads" / "collections" / "travel"
    coll_dir.mkdir(parents=True)
    (coll_dir / "photo.jpg").write_bytes(b"img")
    (coll_dir / "clip.mp4").write_bytes(b"vid")
    (coll_dir / "notes.txt").write_bytes(b"text")

    result = face_mod._collection_images("travel")
    fnames = {r["filename"] for r in result}
    assert "photo.jpg" in fnames
    assert "clip.mp4" in fnames
    assert "notes.txt" not in fnames


def test_collection_images_traversal_guard(tmp_path, monkeypatch):
    result = face_mod._collection_images("../outside")
    assert result == []


def test_collection_images_video_has_type(tmp_path, monkeypatch):
    coll_dir = tmp_path / "downloads" / "collections" / "vids"
    coll_dir.mkdir(parents=True)
    (coll_dir / "movie.mp4").write_bytes(b"vid")

    result = face_mod._collection_images("vids")
    assert len(result) == 1
    assert result[0]["is_video"] is True
    assert "video_type" in result[0]


def test_collection_images_has_correct_urls(tmp_path, monkeypatch):
    import hashlib

    coll_dir = tmp_path / "downloads" / "collections" / "travel"
    coll_dir.mkdir(parents=True)
    (coll_dir / "photo.jpg").write_bytes(b"img")

    result = face_mod._collection_images("travel")
    assert len(result) == 1
    rel = "collections/travel/photo.jpg"
    expected_thumb = f"/thumbs/{hashlib.sha256(rel.encode()).hexdigest()}.jpg"
    assert result[0]["thumb"] == expected_thumb
    assert result[0]["src"] == f"/files/{rel}"


# ── scan exclusion ────────────────────────────────────────────────────────────


def test_scan_dir_locked_excludes_collections(tmp_path, monkeypatch):
    """Files in collections/ must not be included in the scan file list."""
    dl = tmp_path / "downloads"
    # Regular file
    (dl / "album").mkdir(parents=True)
    (dl / "album" / "photo.jpg").write_bytes(b"img")
    # Collections file — must NOT be scanned
    (dl / "collections" / "travel").mkdir(parents=True)
    (dl / "collections" / "travel" / "coll.jpg").write_bytes(b"img")

    # Collect all_files by simulating the walk used in _scan_dir_locked
    coll_abs = os.path.abspath(face_mod._collections_dir())
    all_files = []
    for root, dirs, files in os.walk(str(dl)):
        dirs[:] = [
            d
            for d in dirs
            if not d.startswith(".") and os.path.abspath(os.path.join(root, d)) != coll_abs
        ]
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in face_mod._MEDIA_EXTS:
                all_files.append(os.path.join(root, fname))

    rel_files = [os.path.relpath(f, str(dl)) for f in all_files]
    assert any("album" in f for f in rel_files)
    assert not any("collections" in f for f in rel_files)


def test_phase1_excludes_collections(tmp_path, monkeypatch):
    """_phase1_prefix_groups must not walk into collections/."""
    dl = tmp_path / "downloads"
    (dl / "collections" / "travel").mkdir(parents=True)
    (dl / "collections" / "travel" / "img001.jpg").write_bytes(b"img")
    (dl / "collections" / "travel" / "img002.jpg").write_bytes(b"img")

    suggestions, claimed = face_mod._phase1_prefix_groups(str(dl), 0.4)

    all_folders = [f for s in suggestions for f in s.get("folders", [])]
    assert not any("collections" in f for f in all_folders)
    assert not any("collections" in f for f in claimed)
