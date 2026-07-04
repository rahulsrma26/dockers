"""Tests for face.py — pure logic, DB helpers, and filesystem operations.

InsightFace / PIL / cv2 are NOT imported here; all image-scanning functions
are tested via mocks or by exercising the parts that have no heavy deps.
"""

import hashlib
import json
import os

import numpy as np
import pytest
from sqlalchemy import create_engine

import scrap_downloader.db as db_mod
import scrap_downloader.face as face_mod
from scrap_downloader.db import (
    Base,
    DismissedMatch,
    FaceEmbedding,
    ImageMeta,
    MergeLog,
    Notification,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch, tmp_path):
    """Each test gets a fresh in-memory SQLite and a temporary download dir."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    monkeypatch.setattr(db_mod, "engine", engine)

    dl = tmp_path / "downloads"
    dl.mkdir()
    monkeypatch.setattr(face_mod, "DOWNLOAD_DIR", str(dl))
    return dl


def _dl() -> str:
    return face_mod.DOWNLOAD_DIR


def _make_file(rel_path: str, content: bytes = b"data") -> str:
    abs_path = os.path.join(_dl(), rel_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "wb") as f:
        f.write(content)
    return abs_path


def _insert_image_meta(
    rel_path: str,
    sha256: str | None = None,
    phash: str | None = None,
    max_face_ratio: float | None = None,
):
    from sqlalchemy.orm import Session

    with Session(db_mod.engine) as s:
        s.add(
            ImageMeta(
                file_path=rel_path,
                sha256=sha256,
                phash=phash,
                max_face_ratio=max_face_ratio,
                file_mtime=0.0,
            )
        )
        s.commit()


def _insert_face_embedding(rel_path: str, face_index: int = 0, embedding: list | None = None):
    from sqlalchemy.orm import Session

    emb = np.array(embedding or [1.0] * 512, dtype=np.float32)
    with Session(db_mod.engine) as s:
        s.add(
            FaceEmbedding(
                file_path=rel_path,
                face_index=face_index,
                embedding=emb.tobytes(),
                model_name="buffalo_s",
            )
        )
        s.commit()


# ── Pure functions ────────────────────────────────────────────────────────────


def test_normalize_pair_ordering():
    assert face_mod._normalize_pair("b", "a") == ("a", "b")
    assert face_mod._normalize_pair("a", "b") == ("a", "b")
    assert face_mod._normalize_pair("z", "z") == ("z", "z")


def test_cosine_sim_identical():
    v = np.array([1.0, 0.0, 0.0])
    assert face_mod._cosine_sim(v, v) == pytest.approx(1.0)


def test_cosine_sim_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert face_mod._cosine_sim(a, b) == pytest.approx(0.0)


def test_cosine_sim_zero_vector():
    z = np.array([0.0, 0.0])
    v = np.array([1.0, 0.0])
    assert face_mod._cosine_sim(z, v) == 0.0
    assert face_mod._cosine_sim(v, z) == 0.0


def test_sha256_file(tmp_path):
    content = b"hello world"
    f = tmp_path / "test.bin"
    f.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    assert face_mod._sha256_file(str(f)) == expected


def test_sha256_file_limit(tmp_path):
    content = b"abcdefghij"  # 10 bytes
    f = tmp_path / "test.bin"
    f.write_bytes(content)
    # limit to first 5 bytes
    expected = hashlib.sha256(content[:5]).hexdigest()
    assert face_mod._sha256_file(str(f), limit_bytes=5) == expected


def test_embedding_roundtrip():
    arr = np.random.rand(512).astype(np.float32)
    b = face_mod._embedding_to_bytes(arr.tolist())
    recovered = face_mod._bytes_to_embedding(b)
    np.testing.assert_array_almost_equal(arr, recovered)


def test_thumb_key_stable():
    k1 = face_mod._thumb_key("a/b.jpg")
    k2 = face_mod._thumb_key("a/b.jpg")
    assert k1 == k2
    assert face_mod._thumb_key("a/b.jpg") != face_mod._thumb_key("a/c.jpg")


# ── _remove_empty_dirs ────────────────────────────────────────────────────────


def test_remove_empty_dirs_removes_nested_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(face_mod, "DOWNLOAD_DIR", str(tmp_path))
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)
    face_mod._remove_empty_dirs(str(tmp_path / "a"))
    assert not (tmp_path / "a").exists()


def test_remove_empty_dirs_keeps_dir_with_files(tmp_path, monkeypatch):
    monkeypatch.setattr(face_mod, "DOWNLOAD_DIR", str(tmp_path))
    folder = tmp_path / "album"
    folder.mkdir()
    (folder / "photo.jpg").write_bytes(b"img")
    face_mod._remove_empty_dirs(str(folder))
    assert folder.exists()


def test_remove_empty_dirs_does_not_remove_download_root(tmp_path, monkeypatch):
    monkeypatch.setattr(face_mod, "DOWNLOAD_DIR", str(tmp_path))
    face_mod._remove_empty_dirs(str(tmp_path))
    assert tmp_path.exists()


def test_remove_empty_dirs_partial(tmp_path, monkeypatch):
    """Only fully-empty branches are removed; branches with files survive."""
    monkeypatch.setattr(face_mod, "DOWNLOAD_DIR", str(tmp_path))
    empty = tmp_path / "album" / "empty"
    empty.mkdir(parents=True)
    with_file = tmp_path / "album" / "kept"
    with_file.mkdir()
    (with_file / "x.jpg").write_bytes(b"x")
    face_mod._remove_empty_dirs(str(tmp_path / "album"))
    assert not empty.exists()
    assert with_file.exists()


# ── DB helpers ────────────────────────────────────────────────────────────────


def test_get_setting_returns_default():
    assert db_mod.get_setting("face_model") == "buffalo_s"


def test_set_and_get_setting():
    db_mod.set_setting("face_model", "buffalo_l")
    assert db_mod.get_setting("face_model") == "buffalo_l"


def test_set_setting_overwrite():
    db_mod.set_setting("face_threshold", "0.5")
    db_mod.set_setting("face_threshold", "0.7")
    assert db_mod.get_setting("face_threshold") == "0.7"


def test_get_scan_meta_default():
    assert db_mod.get_scan_meta("nonexistent", "fallback") == "fallback"
    assert db_mod.get_scan_meta("nonexistent") is None


def test_set_and_get_scan_meta():
    db_mod.set_scan_meta("status", "running")
    assert db_mod.get_scan_meta("status") == "running"
    db_mod.set_scan_meta("status", None)
    assert db_mod.get_scan_meta("status") is None


# ── next_unknown_name ─────────────────────────────────────────────────────────


def test_next_unknown_name_sequential():
    assert face_mod.next_unknown_name() == "unknown-1"
    assert face_mod.next_unknown_name() == "unknown-2"
    assert face_mod.next_unknown_name() == "unknown-3"


# ── dismiss / undismiss ───────────────────────────────────────────────────────


def test_dismiss_suggestion_stores_pairs():
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    face_mod.dismiss_suggestion(["a", "b", "c"])
    with Session(db_mod.engine) as s:
        rows = s.execute(select(DismissedMatch)).scalars().all()
    pairs = {(r.folder_a, r.folder_b) for r in rows}
    assert ("a", "b") in pairs
    assert ("a", "c") in pairs
    assert ("b", "c") in pairs


def test_dismiss_idempotent():
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    face_mod.dismiss_suggestion(["x", "y"])
    face_mod.dismiss_suggestion(["x", "y"])
    with Session(db_mod.engine) as s:
        count = len(s.execute(select(DismissedMatch)).scalars().all())
    assert count == 1


def test_undismiss_removes_pairs():
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    face_mod.dismiss_suggestion(["a", "b", "c"])
    face_mod.undismiss_suggestion(["a", "b", "c"])
    with Session(db_mod.engine) as s:
        rows = s.execute(select(DismissedMatch)).scalars().all()
    assert rows == []


def test_dismiss_normalizes_order():
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    face_mod.dismiss_suggestion(["z", "a"])
    with Session(db_mod.engine) as s:
        rows = s.execute(select(DismissedMatch)).scalars().all()
    assert rows[0].folder_a == "a"
    assert rows[0].folder_b == "z"


# ── rename_person — validation ────────────────────────────────────────────────


def test_rename_person_empty_name():
    err = face_mod.rename_person("alice", "")
    assert err is not None
    assert "empty" in err.lower()


def test_rename_person_whitespace_only():
    err = face_mod.rename_person("alice", "   ")
    assert err is not None


def test_rename_person_illegal_chars():
    err = face_mod.rename_person("alice", "al/ice")
    assert err is not None
    assert "illegal" in err.lower()


def test_rename_person_src_not_exists():
    err = face_mod.rename_person("nobody", "newname")
    assert err is not None
    assert "does not exist" in err.lower()


def test_rename_person_dest_exists():
    cat = os.path.join(_dl(), "categorized")
    os.makedirs(os.path.join(cat, "alice"), exist_ok=True)
    os.makedirs(os.path.join(cat, "bob"), exist_ok=True)
    err = face_mod.rename_person("alice", "bob")
    assert err is not None
    assert "already exists" in err.lower()


def test_rename_person_success():
    cat = os.path.join(_dl(), "categorized")
    os.makedirs(os.path.join(cat, "alice"), exist_ok=True)
    (cat_path := os.path.join(cat, "alice", "photo.jpg")) and open(cat_path, "wb").write(b"img")

    _insert_image_meta("categorized/alice/photo.jpg", sha256="abc")
    _insert_face_embedding("categorized/alice/photo.jpg")

    err = face_mod.rename_person("alice", "carol")
    assert err is None
    assert os.path.isdir(os.path.join(cat, "carol"))
    assert not os.path.isdir(os.path.join(cat, "alice"))

    from sqlalchemy import select
    from sqlalchemy.orm import Session

    with Session(db_mod.engine) as s:
        im = s.execute(select(ImageMeta)).scalars().first()
        fe = s.execute(select(FaceEmbedding)).scalars().first()
    assert im.file_path == "categorized/carol/photo.jpg"
    assert fe.file_path == "categorized/carol/photo.jpg"


def test_rename_person_updates_notifications():
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    cat = os.path.join(_dl(), "categorized")
    os.makedirs(os.path.join(cat, "alice"), exist_ok=True)
    with Session(db_mod.engine) as s:
        s.add(Notification(person_name="alice", new_folder="downloads/album"))
        s.commit()

    face_mod.rename_person("alice", "carol")

    with Session(db_mod.engine) as s:
        n = s.execute(select(Notification)).scalars().first()
    assert n.person_name == "carol"


# ── analyse_merge ─────────────────────────────────────────────────────────────


def test_analyse_merge_exact_duplicate():
    """Source file with matching sha256 in dest → exact_dupes."""
    cat = os.path.join(_dl(), "categorized", "alice")
    os.makedirs(cat, exist_ok=True)

    src_file = _make_file("album/photo.jpg", b"same content")
    dest_file = os.path.join(cat, "photo_old.jpg")
    with open(dest_file, "wb") as f:
        f.write(b"same content")

    sha = hashlib.sha256(b"same content").hexdigest()
    _insert_image_meta("album/photo.jpg", sha256=sha)
    _insert_image_meta("categorized/alice/photo_old.jpg", sha256=sha)

    report = face_mod.analyse_merge(["album"], "alice")
    assert src_file in report["exact_dupes"]
    assert report["clean"] == []


def test_analyse_merge_clean_file():
    """New file with no conflicts → clean."""
    cat = os.path.join(_dl(), "categorized", "alice")
    os.makedirs(cat, exist_ok=True)

    src_file = _make_file("album/new.jpg", b"unique content")
    sha = hashlib.sha256(b"unique content").hexdigest()
    _insert_image_meta("album/new.jpg", sha256=sha)

    report = face_mod.analyse_merge(["album"], "alice")
    assert src_file in report["clean"]
    assert report["exact_dupes"] == []


def test_analyse_merge_filename_collision():
    """Same filename, different content → collision."""
    cat = os.path.join(_dl(), "categorized", "alice")
    os.makedirs(cat, exist_ok=True)

    src_file = _make_file("album/photo.jpg", b"content A")
    dest_file = os.path.join(cat, "photo.jpg")
    with open(dest_file, "wb") as f:
        f.write(b"content B")

    sha_a = hashlib.sha256(b"content A").hexdigest()
    sha_b = hashlib.sha256(b"content B").hexdigest()
    _insert_image_meta("album/photo.jpg", sha256=sha_a)
    _insert_image_meta("categorized/alice/photo.jpg", sha256=sha_b)

    report = face_mod.analyse_merge(["album"], "alice")
    assert src_file in report["collisions"]
    assert report["exact_dupes"] == []


def test_analyse_merge_dest_not_exists():
    """dest_exists is False when folder doesn't exist yet."""
    _make_file("album/photo.jpg", b"data")
    _insert_image_meta("album/photo.jpg", sha256="abc")
    report = face_mod.analyse_merge(["album"], "newperson")
    assert report["dest_exists"] is False


def test_analyse_merge_dest_exists():
    cat = os.path.join(_dl(), "categorized", "alice")
    os.makedirs(cat, exist_ok=True)
    report = face_mod.analyse_merge(["album"], "alice")
    assert report["dest_exists"] is True


# ── merge_into ────────────────────────────────────────────────────────────────


def test_merge_into_moves_file():
    src_file = _make_file("album/photo.jpg", b"data")
    _insert_image_meta("album/photo.jpg", sha256="abc123")

    face_mod.merge_into(["album"], "alice")

    dest_file = os.path.join(_dl(), "categorized", "alice", "photo.jpg")
    assert os.path.exists(dest_file)
    assert not os.path.exists(src_file)


def test_merge_into_updates_image_meta_path():
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    _make_file("album/photo.jpg", b"data")
    _insert_image_meta("album/photo.jpg", sha256="abc123")

    face_mod.merge_into(["album"], "alice")

    with Session(db_mod.engine) as s:
        rows = s.execute(select(ImageMeta)).scalars().all()
    paths = [r.file_path for r in rows]
    assert "categorized/alice/photo.jpg" in paths
    assert "album/photo.jpg" not in paths


def test_merge_into_updates_face_embedding_path():
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    _make_file("album/photo.jpg", b"data")
    _insert_image_meta("album/photo.jpg", sha256="abc")
    _insert_face_embedding("album/photo.jpg")

    face_mod.merge_into(["album"], "alice")

    with Session(db_mod.engine) as s:
        fe = s.execute(select(FaceEmbedding)).scalars().first()
    assert fe.file_path == "categorized/alice/photo.jpg"


def test_merge_into_clears_last_scan():
    db_mod.set_scan_meta("last_scan", "2025-01-01T00:00:00")
    _make_file("album/photo.jpg", b"data")
    _insert_image_meta("album/photo.jpg", sha256="abc")

    face_mod.merge_into(["album"], "alice")

    assert db_mod.get_scan_meta("last_scan") is None


def test_merge_into_deletes_exact_duplicate():
    """Source file whose sha256 matches dest → deleted from source; dest untouched."""
    sha = hashlib.sha256(b"same").hexdigest()
    src_file = _make_file("album/photo.jpg", b"same")
    cat = os.path.join(_dl(), "categorized", "alice")
    os.makedirs(cat, exist_ok=True)
    dest_file = os.path.join(cat, "photo_old.jpg")
    with open(dest_file, "wb") as f:
        f.write(b"same")

    _insert_image_meta("album/photo.jpg", sha256=sha)
    _insert_image_meta("categorized/alice/photo_old.jpg", sha256=sha)

    face_mod.merge_into(["album"], "alice")

    # Exact duplicate in source is deleted (dest already has identical copy)
    assert not os.path.exists(src_file)
    assert os.path.exists(dest_file)


def test_merge_into_handles_filename_collision():
    """Two files with the same name but different content → second gets _2 suffix."""
    cat = os.path.join(_dl(), "categorized", "alice")
    os.makedirs(cat, exist_ok=True)
    dest_file = os.path.join(cat, "photo.jpg")
    with open(dest_file, "wb") as f:
        f.write(b"original")

    _make_file("album/photo.jpg", b"new content")
    sha_orig = hashlib.sha256(b"original").hexdigest()
    sha_new = hashlib.sha256(b"new content").hexdigest()
    _insert_image_meta("album/photo.jpg", sha256=sha_new)
    _insert_image_meta("categorized/alice/photo.jpg", sha256=sha_orig)

    face_mod.merge_into(["album"], "alice")

    assert os.path.exists(os.path.join(cat, "photo_2.jpg"))
    assert os.path.exists(dest_file)


def test_merge_into_removes_empty_source_dir():
    _make_file("album/photo.jpg", b"data")
    _insert_image_meta("album/photo.jpg", sha256="abc")

    face_mod.merge_into(["album"], "alice")

    assert not os.path.isdir(os.path.join(_dl(), "album"))


def test_merge_into_marks_merge_log_done():
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    _make_file("album/photo.jpg", b"data")
    _insert_image_meta("album/photo.jpg", sha256="abc")

    face_mod.merge_into(["album"], "alice")

    with Session(db_mod.engine) as s:
        log = s.execute(select(MergeLog)).scalars().first()
    assert log.status == "done"
    assert log.completed_at is not None


def test_merge_into_deletes_dismissed_matches():
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    face_mod.dismiss_suggestion(["album", "other"])
    _make_file("album/photo.jpg", b"data")
    _insert_image_meta("album/photo.jpg", sha256="abc")

    face_mod.merge_into(["album"], "alice")

    with Session(db_mod.engine) as s:
        rows = s.execute(select(DismissedMatch)).scalars().all()
    assert rows == []


def test_merge_progress_callback():
    _make_file("album/a.jpg", b"a")
    _make_file("album/b.jpg", b"b")
    _insert_image_meta("album/a.jpg", sha256="aaa")
    _insert_image_meta("album/b.jpg", sha256="bbb")

    calls = []
    face_mod.merge_into(["album"], "alice", on_progress=lambda c, t: calls.append((c, t)))

    assert len(calls) == 2
    assert calls[-1] == (2, 2)


# ── _pick_sample_thumbs ───────────────────────────────────────────────────────


def test_pick_sample_thumbs_empty():
    assert face_mod._pick_sample_thumbs([]) == []


def test_pick_sample_thumbs_files_not_in_db():
    _make_file("folder/photo.jpg")
    result = face_mod._pick_sample_thumbs(["folder/photo.jpg"])
    assert len(result) == 1
    assert result[0].startswith("/thumbs/")


def test_pick_sample_thumbs_sorted_by_ratio():
    from sqlalchemy.orm import Session

    _make_file("folder/low.jpg")
    _make_file("folder/high.jpg")
    with Session(db_mod.engine) as s:
        s.add(ImageMeta(file_path="folder/low.jpg", max_face_ratio=0.1, file_mtime=0.0))
        s.add(ImageMeta(file_path="folder/high.jpg", max_face_ratio=0.9, file_mtime=0.0))
        s.commit()

    result = face_mod._pick_sample_thumbs(["folder/low.jpg", "folder/high.jpg"])
    assert result[0] == f"/thumbs/{face_mod._thumb_key('folder/high.jpg')}.jpg"


def test_pick_sample_thumbs_respects_n():
    files = []
    for i in range(10):
        rel = f"folder/img{i}.jpg"
        _make_file(rel)
        files.append(rel)
    result = face_mod._pick_sample_thumbs(files, n=3)
    assert len(result) == 3


# ── _find_phash_match ─────────────────────────────────────────────────────────


def test_find_phash_match_exact():
    phash_str = "0" * 16  # all-zero 64-bit hash
    dest_map = {phash_str: "/path/photo.jpg"}
    assert face_mod._find_phash_match(phash_str, dest_map, threshold=8) == "/path/photo.jpg"


def test_find_phash_match_above_threshold():
    src_phash = "0" * 16  # all zeros
    dst_phash = "f" * 16  # all ones — Hamming distance 64
    dest_map = {dst_phash: "/path/photo.jpg"}
    assert face_mod._find_phash_match(src_phash, dest_map, threshold=8) is None


def test_find_phash_match_empty_map():
    assert face_mod._find_phash_match("0" * 16, {}, threshold=8) is None


# ── _cleanup_stale ────────────────────────────────────────────────────────────


def test_cleanup_stale_removes_missing_files():
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    _insert_image_meta("ghost/photo.jpg", sha256="abc")
    _insert_face_embedding("ghost/photo.jpg")

    face_mod._cleanup_stale(_dl())

    with Session(db_mod.engine) as s:
        im_rows = s.execute(select(ImageMeta)).scalars().all()
        fe_rows = s.execute(select(FaceEmbedding)).scalars().all()
    assert im_rows == []
    assert fe_rows == []


def test_cleanup_stale_keeps_existing_files():
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    _make_file("album/photo.jpg", b"data")
    _insert_image_meta("album/photo.jpg", sha256="abc")

    face_mod._cleanup_stale(_dl())

    with Session(db_mod.engine) as s:
        rows = s.execute(select(ImageMeta)).scalars().all()
    assert len(rows) == 1


def test_cleanup_stale_removes_dismissed_for_missing_folder():
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    face_mod.dismiss_suggestion(["missing_a", "missing_b"])

    face_mod._cleanup_stale(_dl())

    with Session(db_mod.engine) as s:
        rows = s.execute(select(DismissedMatch)).scalars().all()
    assert rows == []


def test_cleanup_stale_keeps_dismissed_for_existing_folders():
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    os.makedirs(os.path.join(_dl(), "folder_a"), exist_ok=True)
    os.makedirs(os.path.join(_dl(), "folder_b"), exist_ok=True)
    face_mod.dismiss_suggestion(["folder_a", "folder_b"])

    face_mod._cleanup_stale(_dl())

    with Session(db_mod.engine) as s:
        rows = s.execute(select(DismissedMatch)).scalars().all()
    assert len(rows) == 1


# ── resume_incomplete_merges ──────────────────────────────────────────────────


def test_resume_incomplete_merges():
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    _make_file("album/photo.jpg", b"data")
    _insert_image_meta("album/photo.jpg", sha256="abc123")

    with Session(db_mod.engine) as s:
        s.add(MergeLog(sources=json.dumps(["album"]), dest="alice", status="pending"))
        s.commit()
        s.execute(select(MergeLog)).scalars().first()  # verify row exists

    face_mod.resume_incomplete_merges()

    assert os.path.exists(os.path.join(_dl(), "categorized", "alice", "photo.jpg"))

    with Session(db_mod.engine) as s:
        rows = s.execute(select(MergeLog)).scalars().all()
    pending = [r for r in rows if r.status == "pending"]
    assert pending == []


def test_resume_incomplete_merges_noop_when_none_pending():
    face_mod.resume_incomplete_merges()  # should not raise


# ── merge_into phash larger wins ─────────────────────────────────────────────


def test_merge_into_phash_larger_wins():
    """Source larger than dest phash-match → source replaces dest."""
    shared_phash = "0" * 16

    cat = os.path.join(_dl(), "categorized", "alice")
    os.makedirs(cat, exist_ok=True)

    dest_file = os.path.join(cat, "small.jpg")
    with open(dest_file, "wb") as f:
        f.write(b"tiny")  # 4 bytes

    src_file = _make_file("album/large.jpg", b"much_larger_content")  # 19 bytes

    sha_large = hashlib.sha256(b"much_larger_content").hexdigest()
    sha_small = hashlib.sha256(b"tiny").hexdigest()
    _insert_image_meta("album/large.jpg", sha256=sha_large, phash=shared_phash)
    _insert_image_meta("categorized/alice/small.jpg", sha256=sha_small, phash=shared_phash)

    face_mod.merge_into(["album"], "alice")

    assert not os.path.exists(src_file)  # src was moved out
    # dest path reused by the larger file
    assert os.path.exists(dest_file)
    with open(dest_file, "rb") as fh:
        assert fh.read() == b"much_larger_content"


def test_merge_into_phash_smaller_src_deleted():
    """Source smaller than dest phash-match → source deleted (dest has better copy)."""
    shared_phash = "1" * 16

    cat = os.path.join(_dl(), "categorized", "alice")
    os.makedirs(cat, exist_ok=True)

    dest_file = os.path.join(cat, "big.jpg")
    with open(dest_file, "wb") as f:
        f.write(b"larger_file_content")  # 19 bytes

    src_file = _make_file("album/small.jpg", b"tiny")  # 4 bytes

    sha_big = hashlib.sha256(b"larger_file_content").hexdigest()
    sha_tiny = hashlib.sha256(b"tiny").hexdigest()
    _insert_image_meta("album/small.jpg", sha256=sha_tiny, phash=shared_phash)
    _insert_image_meta("categorized/alice/big.jpg", sha256=sha_big, phash=shared_phash)

    face_mod.merge_into(["album"], "alice")

    assert not os.path.exists(src_file)  # smaller src deleted (dest has better copy)
    assert os.path.exists(dest_file)  # dest untouched


# ── rename_person dismissed_matches edge cases ────────────────────────────────


def test_rename_person_updates_dismissed_matches():
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    cat = os.path.join(_dl(), "categorized")
    os.makedirs(os.path.join(cat, "alice"), exist_ok=True)

    with Session(db_mod.engine) as s:
        s.add(DismissedMatch(folder_a="categorized/alice", folder_b="other/folder"))
        s.commit()

    face_mod.rename_person("alice", "carol")

    with Session(db_mod.engine) as s:
        rows = s.execute(select(DismissedMatch)).scalars().all()
    assert len(rows) == 1
    assert {(r.folder_a, r.folder_b) for r in rows} == {("categorized/carol", "other/folder")}


def test_rename_person_dismissed_match_no_duplicate():
    """No IntegrityError when renamed pair already exists as a DismissedMatch."""
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    cat = os.path.join(_dl(), "categorized")
    os.makedirs(os.path.join(cat, "alice"), exist_ok=True)
    # Do NOT mkdir carol — so rename can proceed

    with Session(db_mod.engine) as s:
        s.add(DismissedMatch(folder_a="categorized/alice", folder_b="other"))
        s.add(DismissedMatch(folder_a="categorized/carol", folder_b="other"))
        s.commit()

    err = face_mod.rename_person("alice", "carol")
    assert err is None

    with Session(db_mod.engine) as s:
        rows = s.execute(select(DismissedMatch)).scalars().all()
    assert len(rows) == 1
    assert rows[0].folder_a == "categorized/carol"
    assert rows[0].folder_b == "other"


# ── dismiss_suggestion edge cases ─────────────────────────────────────────────


def test_dismiss_suggestion_single_folder_no_pairs():
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    face_mod.dismiss_suggestion(["only_folder"])

    with Session(db_mod.engine) as s:
        rows = s.execute(select(DismissedMatch)).scalars().all()
    assert rows == []


def test_undismiss_nonexistent_pairs_is_noop():
    face_mod.undismiss_suggestion(["x", "y"])  # should not raise


# ── get_setting edge cases ────────────────────────────────────────────────────


def test_get_setting_unknown_key_returns_none():
    assert db_mod.get_setting("totally_unknown_key") is None


def test_get_setting_unknown_key_with_explicit_default():
    assert db_mod.get_setting("totally_unknown_key", "fallback") == "fallback"


# ── _remove_empty_dirs path boundary ─────────────────────────────────────────


def test_remove_empty_dirs_does_not_touch_sibling_dir(tmp_path, monkeypatch):
    """Path boundary fix: /downloads2/ must not be treated as inside /downloads/."""
    parent = tmp_path / "parent"
    parent.mkdir()
    dl = parent / "downloads"
    dl.mkdir()
    sibling = parent / "downloads2"
    sibling.mkdir()
    folder = sibling / "album"
    folder.mkdir()

    monkeypatch.setattr(face_mod, "DOWNLOAD_DIR", str(dl))

    face_mod._remove_empty_dirs(str(folder))

    assert folder.exists()


# ── scan_dir concurrency ──────────────────────────────────────────────────────


def test_scan_dir_raises_if_already_locked():
    face_mod._scan_lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="in progress"):
            face_mod.scan_dir(_dl())
    finally:
        face_mod._scan_lock.release()


# ── _sha256_file edge cases ───────────────────────────────────────────────────


def test_sha256_file_empty(tmp_path):
    f = tmp_path / "empty.bin"
    f.write_bytes(b"")
    assert face_mod._sha256_file(str(f)) == hashlib.sha256(b"").hexdigest()


# ── next_unknown_name with pre-existing counter ───────────────────────────────


def test_next_unknown_name_respects_existing_counter():
    db_mod.set_setting("unknown_counter", "10")
    assert face_mod.next_unknown_name() == "unknown-11"
    assert face_mod.next_unknown_name() == "unknown-12"


# ── _cosine_sim anti-parallel ─────────────────────────────────────────────────


def test_cosine_sim_opposite():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert face_mod._cosine_sim(a, b) == pytest.approx(-1.0)


# ── merge_into edge cases ─────────────────────────────────────────────────────


def test_merge_into_empty_source():
    """Source folder with no files — merge still succeeds and log is marked done."""
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    os.makedirs(os.path.join(_dl(), "empty_album"), exist_ok=True)

    face_mod.merge_into(["empty_album"], "alice")

    with Session(db_mod.engine) as s:
        log = s.execute(select(MergeLog)).scalars().first()
    assert log.status == "done"


def test_merge_into_multiple_sources():
    _make_file("album1/a.jpg", b"aaa")
    _make_file("album2/b.jpg", b"bbb")
    _insert_image_meta("album1/a.jpg", sha256="sha_a")
    _insert_image_meta("album2/b.jpg", sha256="sha_b")

    face_mod.merge_into(["album1", "album2"], "alice")

    cat = os.path.join(_dl(), "categorized", "alice")
    names = os.listdir(cat)
    assert "a.jpg" in names
    assert "b.jpg" in names
    assert not os.path.isdir(os.path.join(_dl(), "album1"))
    assert not os.path.isdir(os.path.join(_dl(), "album2"))


def test_merge_into_collision_counter_increments_past_2():
    """When photo.jpg and photo_2.jpg both exist, incoming src gets photo_3.jpg."""
    cat = os.path.join(_dl(), "categorized", "alice")
    os.makedirs(cat, exist_ok=True)
    for name, content in [("photo.jpg", b"orig"), ("photo_2.jpg", b"v2")]:
        with open(os.path.join(cat, name), "wb") as f:
            f.write(content)

    src_file = _make_file("album/photo.jpg", b"new")
    _insert_image_meta("album/photo.jpg", sha256=hashlib.sha256(b"new").hexdigest())
    _insert_image_meta("categorized/alice/photo.jpg", sha256=hashlib.sha256(b"orig").hexdigest())
    _insert_image_meta("categorized/alice/photo_2.jpg", sha256=hashlib.sha256(b"v2").hexdigest())

    face_mod.merge_into(["album"], "alice")

    assert os.path.exists(os.path.join(cat, "photo_3.jpg"))
    assert not os.path.exists(src_file)


# ── analyse_merge phash_dupe ──────────────────────────────────────────────────


def test_analyse_merge_phash_dupe():
    """Source with same phash as dest → appears in phash_dupes, keep=dest (larger)."""
    cat = os.path.join(_dl(), "categorized", "alice")
    os.makedirs(cat, exist_ok=True)

    shared_phash = "0" * 16
    dest_file = os.path.join(cat, "large.jpg")
    with open(dest_file, "wb") as f:
        f.write(b"large_content_here")

    _make_file("album/small.jpg", b"tiny")
    _insert_image_meta(
        "album/small.jpg",
        sha256=hashlib.sha256(b"tiny").hexdigest(),
        phash=shared_phash,
    )
    _insert_image_meta(
        "categorized/alice/large.jpg",
        sha256=hashlib.sha256(b"large_content_here").hexdigest(),
        phash=shared_phash,
    )

    report = face_mod.analyse_merge(["album"], "alice")
    assert len(report["phash_dupes"]) == 1
    dupe = report["phash_dupes"][0]
    assert dupe["keep"] == dest_file


# ── rename_person — FolderCentroid and MergeLog updates ──────────────────────


def test_rename_person_updates_folder_centroid():
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from scrap_downloader.db import FolderCentroid

    cat = os.path.join(_dl(), "categorized")
    os.makedirs(os.path.join(cat, "alice"), exist_ok=True)
    emb = np.ones(512, dtype=np.float32)
    with Session(db_mod.engine) as s:
        s.add(
            FolderCentroid(
                folder="categorized/alice",
                centroid=emb.tobytes(),
                model_name="buffalo_s",
            )
        )
        s.commit()

    face_mod.rename_person("alice", "carol")

    with Session(db_mod.engine) as s:
        rows = s.execute(select(FolderCentroid)).scalars().all()
    assert len(rows) == 1
    assert rows[0].folder == "categorized/carol"


def test_rename_person_updates_pending_merge_log():
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    cat = os.path.join(_dl(), "categorized")
    os.makedirs(os.path.join(cat, "alice"), exist_ok=True)
    with Session(db_mod.engine) as s:
        s.add(MergeLog(sources=json.dumps(["src"]), dest="alice", status="pending"))
        s.commit()

    face_mod.rename_person("alice", "carol")

    with Session(db_mod.engine) as s:
        log = s.execute(select(MergeLog)).scalars().first()
    assert log.dest == "carol"


# ── rename_person — dot/dotdot validation ────────────────────────────────────


def test_rename_person_rejects_dot():
    cat = os.path.join(_dl(), "categorized")
    os.makedirs(os.path.join(cat, "alice"), exist_ok=True)
    err = face_mod.rename_person("alice", ".")
    assert err is not None
    assert "dot" in err.lower()


def test_rename_person_rejects_dotdot():
    cat = os.path.join(_dl(), "categorized")
    os.makedirs(os.path.join(cat, "alice"), exist_ok=True)
    err = face_mod.rename_person("alice", "..")
    assert err is not None
    assert "dot" in err.lower()


def test_rename_person_rejects_hidden_name():
    cat = os.path.join(_dl(), "categorized")
    os.makedirs(os.path.join(cat, "alice"), exist_ok=True)
    err = face_mod.rename_person("alice", ".hidden")
    assert err is not None
    assert "dot" in err.lower()


# ── _like_prefix — LIKE wildcard escaping ─────────────────────────────────────


def test_like_prefix_escapes_underscore():
    pattern = face_mod._like_prefix("john_doe")
    assert "\\_" in pattern
    assert pattern.endswith("/%")


def test_like_prefix_escapes_percent():
    pattern = face_mod._like_prefix("100%real")
    assert "\\%" in pattern


def test_like_prefix_escapes_backslash():
    pattern = face_mod._like_prefix("back\\slash")
    assert "\\\\" in pattern


def test_rename_person_with_underscore_only_renames_exact_folder():
    """LIKE wildcard bug: john_doe should NOT match johnXdoe rows."""
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    cat = os.path.join(_dl(), "categorized")
    os.makedirs(os.path.join(cat, "john_doe"), exist_ok=True)
    # Insert meta for the target folder AND a similarly-named decoy
    _insert_image_meta("categorized/john_doe/photo.jpg", sha256="a")
    _insert_image_meta("categorized/johnXdoe/photo.jpg", sha256="b")

    face_mod.rename_person("john_doe", "carol")

    with Session(db_mod.engine) as s:
        rows = s.execute(select(ImageMeta)).scalars().all()
    paths = {r.file_path for r in rows}
    # Only john_doe row should be renamed; johnXdoe stays unchanged
    assert "categorized/carol/photo.jpg" in paths
    assert "categorized/johnXdoe/photo.jpg" in paths
    assert "categorized/john_doe/photo.jpg" not in paths


# ── scan_new_download path-boundary tests ─────────────────────────────────────


def test_scan_new_download_skips_categorized_folder(monkeypatch):
    """A folder exactly equal to categorized/ is skipped."""
    called = []
    monkeypatch.setattr(face_mod, "_scan_file", lambda *a, **k: called.append(a))
    db_mod.set_scan_meta("status", "idle")

    face_mod.scan_new_download("categorized")

    assert called == []


def test_scan_new_download_skips_subfolder_of_categorized(monkeypatch):
    """A folder inside categorized/ (already categorized person) is skipped."""
    called = []
    monkeypatch.setattr(face_mod, "_scan_file", lambda *a, **k: called.append(a))
    db_mod.set_scan_meta("status", "idle")

    face_mod.scan_new_download("categorized/alice")

    assert called == []


def test_scan_new_download_does_not_skip_adjacent_folder(monkeypatch):
    """A folder named categorized_old must NOT be treated as categorized/."""
    scanned = []

    def fake_scan_file(fpath, model):
        scanned.append(fpath)
        return 0

    monkeypatch.setattr(face_mod, "_scan_file", fake_scan_file)
    db_mod.set_scan_meta("status", "idle")
    db_mod.set_setting("mini_scan_enabled", "false")

    _make_file("categorized_old/photo.jpg", b"data")

    face_mod.scan_new_download("categorized_old")

    assert any("photo.jpg" in p for p in scanned)


def test_scan_new_download_skips_when_scan_running(monkeypatch):
    """Mini-scan is a no-op when a full scan is in progress."""
    called = []
    monkeypatch.setattr(face_mod, "_scan_file", lambda *a, **k: called.append(a))
    db_mod.set_scan_meta("status", "running")

    _make_file("album/photo.jpg", b"data")
    face_mod.scan_new_download("album")

    assert called == []


# ── merge_into — source file not in DB ───────────────────────────────────────


def test_merge_into_source_not_in_db_still_moves():
    """Files with no ImageMeta row (never scanned) are still moved."""
    src_file = _make_file("album/photo.jpg", b"data")
    # Intentionally do NOT insert ImageMeta

    face_mod.merge_into(["album"], "alice")

    dest_file = os.path.join(_dl(), "categorized", "alice", "photo.jpg")
    assert os.path.exists(dest_file)
    assert not os.path.exists(src_file)


# ── merge_into — hidden files skipped ────────────────────────────────────────


def test_merge_into_skips_hidden_files():
    """Hidden files (dot-prefixed) in source are not moved."""
    _make_file("album/.DS_Store", b"meta")
    _make_file("album/photo.jpg", b"data")
    _insert_image_meta("album/photo.jpg", sha256="abc")

    face_mod.merge_into(["album"], "alice")

    cat = os.path.join(_dl(), "categorized", "alice")
    names = os.listdir(cat)
    assert "photo.jpg" in names
    assert ".DS_Store" not in names


# ── _all_image_files ──────────────────────────────────────────────────────────


def test_all_image_files_excludes_videos():
    folder = os.path.join(_dl(), "mixed")
    os.makedirs(folder, exist_ok=True)
    for name in ["a.jpg", "b.mp4", "c.png", "d.webm"]:
        open(os.path.join(folder, name), "wb").write(b"x")

    result = face_mod._all_image_files(folder)
    names = {os.path.basename(p) for p in result}
    assert "a.jpg" in names
    assert "c.png" in names
    assert "b.mp4" not in names
    assert "d.webm" not in names


# ── analyse_downloads — sort key correctness ──────────────────────────────────


def test_analyse_downloads_confidence_zero_sorts_before_none(monkeypatch):
    """confidence=0.0 must rank above confidence=None (or -1 bug re-appears)."""
    series = [
        {
            "type": "series",
            "confidence": 0.0,
            "folders": [],
            "files": [],
            "dest_name": "p",
            "sample_images": [],
            "prefix": "p",
        }
    ]
    face = [
        {
            "type": "face",
            "confidence": None,
            "folders": [],
            "dest_name": "",
            "sample_images": [],
            "cat_folders": [],
        }
    ]
    monkeypatch.setattr(face_mod, "_phase1_prefix_groups", lambda *a, **k: (series, set()))
    monkeypatch.setattr(face_mod, "_phase2_face_clustering", lambda *a, **k: (face, []))

    result = face_mod.analyse_downloads(_dl())
    suggs = result["suggestions"]
    assert len(suggs) == 2
    assert suggs[0]["type"] == "series"  # 0.0 before None
    assert suggs[1]["type"] == "face"


# ── _scan_dir_locked — status reset on exception ──────────────────────────────


def test_scan_dir_resets_status_to_idle_on_exception(monkeypatch):
    """scan status must return to 'idle' even when _cleanup_stale raises."""

    def bad_cleanup(*a, **k):
        raise RuntimeError("simulated disk error")

    monkeypatch.setattr(face_mod, "_cleanup_stale", bad_cleanup)

    with pytest.raises(RuntimeError, match="simulated disk error"):
        face_mod.scan_dir(_dl())

    assert db_mod.get_scan_meta("status") == "idle"
    assert db_mod.get_scan_meta("last_error") is not None


def test_scan_dir_records_last_error_message_on_exception(monkeypatch):
    """last_error scan_meta key contains the exception message after failure."""

    def bad_recompute(*a, **k):
        raise ValueError("centroid failure")

    monkeypatch.setattr(face_mod, "_recompute_centroids", bad_recompute)

    with pytest.raises(ValueError):
        face_mod.scan_dir(_dl())

    assert "centroid failure" in (db_mod.get_scan_meta("last_error") or "")


# ── FAISS + union-find clustering ─────────────────────────────────────────────


def test_phase2_clustering_groups_similar_embeddings():
    """Two groups of similar embeddings → two clusters; dissimilar points → noise."""

    # Person A: 3 embeddings near [1, 0, 0, ...]
    base_a = np.zeros(512, dtype=np.float32)
    base_a[0] = 1.0
    # Person B: 3 embeddings near [0, 1, 0, ...]
    base_b = np.zeros(512, dtype=np.float32)
    base_b[1] = 1.0

    rng = np.random.default_rng(42)

    # scale=0.001: noise norm ≈ sqrt(512)*0.001 ≈ 0.02, so intra-group
    # cosine sim stays >0.999 — safely above the face_threshold=0.7 used below.
    def _noisy(base, n=3, scale=0.001):
        vecs = base + rng.normal(0, scale, (n, 512)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    embs_a = _noisy(base_a)
    embs_b = _noisy(base_b)

    # Seed DB with embeddings
    for i, emb in enumerate(embs_a):
        _make_file(f"folder_a/img{i}.jpg")
        _insert_image_meta(f"folder_a/img{i}.jpg")
        _insert_face_embedding(f"folder_a/img{i}.jpg", face_index=0, embedding=emb.tolist())
    for i, emb in enumerate(embs_b):
        _make_file(f"folder_b/img{i}.jpg")
        _insert_image_meta(f"folder_b/img{i}.jpg")
        _insert_face_embedding(f"folder_b/img{i}.jpg", face_index=0, embedding=emb.tolist())

    suggestions, ambiguous = face_mod._phase2_face_clustering(
        _dl(), set(), face_threshold=0.7, min_images=1
    )

    # Two distinct clusters → two suggestions (no categorized/ folders → type "face")
    assert len(suggestions) == 2
    assert ambiguous == []
    cluster_folder_sets = [frozenset(s["folders"]) for s in suggestions]
    assert frozenset(["folder_a"]) in cluster_folder_sets
    assert frozenset(["folder_b"]) in cluster_folder_sets


def test_phase2_clustering_no_embeddings_returns_empty():
    suggestions, ambiguous = face_mod._phase2_face_clustering(
        _dl(), set(), face_threshold=0.4, min_images=1
    )
    assert suggestions == []
    assert ambiguous == []


# ── score_against_people ──────────────────────────────────────────────────────


def test_score_against_people_empty_folders():
    assert face_mod.score_against_people([]) == {}


def test_score_against_people_no_embeddings_in_db():
    # Folder exists on disk but has no FaceEmbedding rows → empty result
    os.makedirs(os.path.join(_dl(), "album"), exist_ok=True)
    result = face_mod.score_against_people(["album"])
    assert result == {}


def test_score_against_people_returns_sorted_scores():
    """Source centroid identical to person A → person A scores near 1.0, person B near 0."""
    from sqlalchemy.orm import Session

    from scrap_downloader.db import FolderCentroid

    base_a = np.zeros(512, dtype=np.float32)
    base_a[0] = 1.0
    base_b = np.zeros(512, dtype=np.float32)
    base_b[1] = 1.0

    # Source folder has embeddings near base_a
    _make_file("album/img0.jpg")
    _insert_image_meta("album/img0.jpg")
    _insert_face_embedding("album/img0.jpg", embedding=base_a.tolist())

    # Two categorized people with orthogonal centroids
    with Session(db_mod.engine) as s:
        s.add(
            FolderCentroid(
                folder="categorized/person_a", centroid=base_a.tobytes(), model_name="buffalo_s"
            )
        )
        s.add(
            FolderCentroid(
                folder="categorized/person_b", centroid=base_b.tobytes(), model_name="buffalo_s"
            )
        )
        s.commit()

    scores = face_mod.score_against_people(["album"])

    assert set(scores.keys()) == {"person_a", "person_b"}
    assert scores["person_a"] > 0.99, f"Expected near 1.0, got {scores['person_a']}"
    assert abs(scores["person_b"]) < 0.05, f"Expected near 0.0, got {scores['person_b']}"
    # Result must be sorted descending
    items = list(scores.items())
    assert items[0][0] == "person_a"
    assert items[1][0] == "person_b"


# ── file_map population ───────────────────────────────────────────────────────


def test_merge_into_populates_file_map():
    """merge_into saves file_map JSON in MergeLog with old→new mapping."""
    import json

    from sqlalchemy import select
    from sqlalchemy.orm import Session

    _make_file("album/photo.jpg", b"data")
    _insert_image_meta("album/photo.jpg", sha256="aaa")

    face_mod.merge_into(["album"], "alice")

    with Session(db_mod.engine) as s:
        log = s.execute(select(MergeLog)).scalars().first()

    assert log.file_map is not None
    fm = json.loads(log.file_map)
    assert isinstance(fm, dict)
    assert len(fm) == 1
    key = list(fm.keys())[0]
    assert key == "album/photo.jpg"
    assert fm[key] == "categorized/alice/photo.jpg"


def test_merge_into_file_map_duplicate_is_null():
    """Exact duplicate deleted during merge is stored as null in file_map."""
    import json

    from sqlalchemy import select
    from sqlalchemy.orm import Session

    _make_file("album/photo.jpg", b"same")
    _make_file("categorized/alice/photo.jpg", b"same")
    _insert_image_meta("album/photo.jpg", sha256="same_hash")
    _insert_image_meta("categorized/alice/photo.jpg", sha256="same_hash")

    face_mod.merge_into(["album"], "alice")

    with Session(db_mod.engine) as s:
        log = s.execute(select(MergeLog)).scalars().first()

    assert log.file_map is not None
    fm = json.loads(log.file_map)
    assert "album/photo.jpg" in fm
    assert fm["album/photo.jpg"] is None


# ── undo_merge ────────────────────────────────────────────────────────────────


def test_undo_merge_not_found():
    """undo_merge returns error when log_id doesn't exist."""
    result = face_mod.undo_merge(9999)
    assert result is not None
    assert "not found" in result.lower() or "9999" in result


def test_undo_merge_scan_running(monkeypatch):
    """undo_merge refuses when a scan is in progress."""
    from scrap_downloader.db import set_scan_meta

    set_scan_meta("status", "running")
    result = face_mod.undo_merge(1)
    assert result is not None
    assert "scan" in result.lower() or "running" in result.lower()


def test_undo_merge_preflight_new_file_missing():
    """undo_merge fails pre-flight if file was moved/deleted after merge."""
    from sqlalchemy.orm import Session

    _make_file("album/photo.jpg", b"data")
    _insert_image_meta("album/photo.jpg", sha256="abc")
    face_mod.merge_into(["album"], "alice")

    # Remove the newly merged file to simulate missing file
    merged_path = os.path.join(_dl(), "categorized/alice/photo.jpg")
    os.remove(merged_path)

    with Session(db_mod.engine) as s:
        log = (
            s.execute(__import__("sqlalchemy", fromlist=["select"]).select(MergeLog))
            .scalars()
            .first()
        )

    result = face_mod.undo_merge(log.id)
    assert result is not None


def test_undo_merge_preflight_old_path_occupied():
    """undo_merge fails pre-flight if new file already at old_rel path."""
    from sqlalchemy.orm import Session

    _make_file("album/photo.jpg", b"data")
    _insert_image_meta("album/photo.jpg", sha256="abc")
    face_mod.merge_into(["album"], "alice")

    # Recreate a file at the original path (simulating a new download there)
    _make_file("album/photo.jpg", b"new_data")

    with Session(db_mod.engine) as s:
        log = (
            s.execute(__import__("sqlalchemy", fromlist=["select"]).select(MergeLog))
            .scalars()
            .first()
        )

    result = face_mod.undo_merge(log.id)
    assert result is not None
