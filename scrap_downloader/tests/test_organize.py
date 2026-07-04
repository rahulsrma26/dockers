"""Tests for organize.py helper functions."""

import json

import pytest
from sqlalchemy import create_engine

import scrap_downloader.db as db_mod
from scrap_downloader.db import Base, ImageMeta, MergeLog


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch, tmp_path):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    monkeypatch.setattr(db_mod, "engine", engine)

    dl = tmp_path / "downloads"
    dl.mkdir()
    import scrap_downloader.organize as org_mod

    monkeypatch.setattr(org_mod, "DOWNLOAD_DIR", str(dl))
    return dl


def _dl(tmp_path):
    return tmp_path / "downloads"


# ── _human_size ───────────────────────────────────────────────────────────────


def test_human_size_none():
    from scrap_downloader.organize import _human_size

    assert _human_size(None) == "—"


def test_human_size_zero():
    from scrap_downloader.organize import _human_size

    assert _human_size(0) == "0 B"


def test_human_size_bytes():
    from scrap_downloader.organize import _human_size

    assert _human_size(512) == "512 B"


def test_human_size_kilobytes_exact():
    from scrap_downloader.organize import _human_size

    assert _human_size(1024) == "1.0 KB"


def test_human_size_kilobytes_fractional():
    from scrap_downloader.organize import _human_size

    assert _human_size(1536) == "1.5 KB"


def test_human_size_megabytes():
    from scrap_downloader.organize import _human_size

    assert _human_size(1024 * 1024) == "1.0 MB"


def test_human_size_gigabytes():
    from scrap_downloader.organize import _human_size

    assert _human_size(2 * 1024 * 1024 * 1024) == "2.0 GB"


# ── _list_people casefold sort ────────────────────────────────────────────────


def test_list_people_casefold_sort(tmp_path, monkeypatch):
    from scrap_downloader.organize import _list_people

    cat = tmp_path / "downloads" / "categorized"
    # Create folders with mixed case — casefold sort: alice < bob < Charlie < zack
    for name in ["bob", "Charlie", "alice", "Zack"]:
        (cat / name).mkdir(parents=True)

    people = _list_people()
    assert people == ["alice", "bob", "Charlie", "Zack"]


def test_list_people_empty_when_no_categorized(tmp_path):
    from scrap_downloader.organize import _list_people

    people = _list_people()
    assert people == []


def test_list_people_skips_hidden_dirs(tmp_path, monkeypatch):
    from scrap_downloader.organize import _list_people

    cat = tmp_path / "downloads" / "categorized"
    cat.mkdir(parents=True)
    (cat / "alice").mkdir()
    (cat / ".hidden").mkdir()

    people = _list_people()
    assert people == ["alice"]
    assert ".hidden" not in people


# ── _delete_gallery_file path traversal ──────────────────────────────────────


def test_delete_gallery_file_rejects_dotdot_person(tmp_path, monkeypatch):
    from scrap_downloader.organize import _delete_gallery_file

    cat = tmp_path / "downloads" / "categorized"
    cat.mkdir(parents=True)

    result = _delete_gallery_file("../outside", "categorized/../outside/file.jpg")
    assert result is not None
    assert "invalid" in result.lower()


def test_delete_gallery_file_rejects_escaped_rel(tmp_path, monkeypatch):
    from scrap_downloader.organize import _delete_gallery_file

    cat = tmp_path / "downloads" / "categorized" / "alice"
    cat.mkdir(parents=True)

    # rel_path that would escape alice/ directory
    result = _delete_gallery_file("alice", "categorized/alice/../../secret.jpg")
    assert result is not None
    assert "invalid" in result.lower()


def test_delete_gallery_file_removes_file(tmp_path, monkeypatch):
    from sqlalchemy.orm import Session

    from scrap_downloader.organize import _delete_gallery_file

    cat = tmp_path / "downloads" / "categorized" / "alice"
    cat.mkdir(parents=True)
    f = cat / "photo.jpg"
    f.write_bytes(b"img")

    rel = "categorized/alice/photo.jpg"
    with Session(db_mod.engine) as s:
        s.add(ImageMeta(file_path=rel, file_mtime=0.0))
        s.commit()

    result = _delete_gallery_file("alice", rel)
    assert result is None
    assert not f.exists()


# ── _last_undoable_merge ──────────────────────────────────────────────────────


def test_last_undoable_merge_empty_db():
    from scrap_downloader.organize import _last_undoable_merge

    assert _last_undoable_merge() is None


def test_last_undoable_merge_no_file_map():
    from datetime import datetime, timezone

    from sqlalchemy.orm import Session

    from scrap_downloader.organize import _last_undoable_merge

    with Session(db_mod.engine) as s:
        s.add(
            MergeLog(
                sources=json.dumps(["album"]),
                dest="categorized/alice",
                status="done",
                completed_at=datetime.now(timezone.utc).replace(tzinfo=None),
                file_map=None,
            )
        )
        s.commit()

    assert _last_undoable_merge() is None


def test_last_undoable_merge_recent_returns_dict():
    from datetime import datetime, timezone

    from sqlalchemy.orm import Session

    from scrap_downloader.organize import _last_undoable_merge

    file_map = json.dumps({"album/photo.jpg": "categorized/alice/photo.jpg"})
    with Session(db_mod.engine) as s:
        s.add(
            MergeLog(
                sources=json.dumps(["album"]),
                dest="categorized/alice",
                status="done",
                completed_at=datetime.now(timezone.utc).replace(tzinfo=None),
                file_map=file_map,
            )
        )
        s.commit()

    result = _last_undoable_merge()
    assert result is not None
    assert result["dest"] == "categorized/alice"
    assert "album" in result["sources"]


def test_last_undoable_merge_old_returns_none():
    from datetime import datetime, timedelta, timezone

    from sqlalchemy.orm import Session

    from scrap_downloader.organize import _last_undoable_merge

    old_time = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=31)
    file_map = json.dumps({"album/photo.jpg": "categorized/alice/photo.jpg"})
    with Session(db_mod.engine) as s:
        s.add(
            MergeLog(
                sources=json.dumps(["album"]),
                dest="categorized/alice",
                status="done",
                completed_at=old_time,
                file_map=file_map,
            )
        )
        s.commit()

    assert _last_undoable_merge() is None


# ── Settings defaults ─────────────────────────────────────────────────────────


def test_suggestions_page_size_default():
    from scrap_downloader.db import get_setting

    assert get_setting("suggestions_page_size") == "10"


def test_auto_logout_minutes_default():
    from scrap_downloader.db import get_setting

    assert get_setting("auto_logout_minutes") == "15"


# ── init_db migrations ────────────────────────────────────────────────────────


def test_init_db_adds_file_size_column(tmp_path, monkeypatch):
    """init_db must add file_size to image_meta if missing (migration path)."""
    from sqlalchemy import create_engine, inspect, text

    # Create a legacy DB without file_size
    engine = create_engine(f"sqlite:///{tmp_path}/legacy.db")
    with engine.connect() as conn:
        conn.execute(
            text(
                "CREATE TABLE image_meta ("
                "  id INTEGER PRIMARY KEY,"
                "  file_path TEXT UNIQUE,"
                "  phash TEXT,"
                "  sha256 TEXT,"
                "  face_count INTEGER,"
                "  max_face_ratio REAL,"
                "  file_mtime REAL,"
                "  scanned_at DATETIME"
                ")"
            )
        )
        conn.commit()

    monkeypatch.setattr(db_mod, "engine", engine)
    db_mod.init_db()

    inspector = inspect(engine)
    cols = [c["name"] for c in inspector.get_columns("image_meta")]
    assert "file_size" in cols


def test_init_db_adds_file_map_column(tmp_path, monkeypatch):
    """init_db must add file_map to merge_log if missing (migration path)."""
    from sqlalchemy import create_engine, inspect, text

    engine = create_engine(f"sqlite:///{tmp_path}/legacy.db")
    with engine.connect() as conn:
        conn.execute(
            text(
                "CREATE TABLE merge_log ("
                "  id INTEGER PRIMARY KEY,"
                "  sources TEXT,"
                "  dest TEXT,"
                "  status TEXT,"
                "  created_at DATETIME,"
                "  completed_at DATETIME"
                ")"
            )
        )
        conn.commit()

    monkeypatch.setattr(db_mod, "engine", engine)
    db_mod.init_db()

    inspector = inspect(engine)
    cols = [c["name"] for c in inspector.get_columns("merge_log")]
    assert "file_map" in cols
