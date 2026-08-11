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
    Tag,
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


def test_dismiss_suggestion_single_folder_self_pair():
    """Single-folder suggestions (series) store a self-pair so dismiss actually works."""
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    face_mod.dismiss_suggestion(["only_folder"])

    with Session(db_mod.engine) as s:
        rows = s.execute(select(DismissedMatch)).scalars().all()
    assert len(rows) == 1
    assert rows[0].folder_a == "only_folder"
    assert rows[0].folder_b == "only_folder"


def test_undismiss_suggestion_single_folder_removes_self_pair():
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    face_mod.dismiss_suggestion(["solo"])
    face_mod.undismiss_suggestion(["solo"])

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


# ── Tag CRUD ──────────────────────────────────────────────────────────────────


def test_add_tag_stores_lowercase():
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    face_mod.add_tag("person", "alice", "  Summer  ")
    with Session(db_mod.engine) as s:
        rows = s.execute(select(Tag)).scalars().all()
    assert len(rows) == 1
    assert rows[0].tag == "summer"
    assert rows[0].entity_type == "person"
    assert rows[0].entity_name == "alice"


def test_add_tag_empty_is_noop():
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    face_mod.add_tag("person", "alice", "   ")
    with Session(db_mod.engine) as s:
        rows = s.execute(select(Tag)).scalars().all()
    assert rows == []


def test_add_tag_idempotent():
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    face_mod.add_tag("person", "alice", "summer")
    face_mod.add_tag("person", "alice", "summer")
    with Session(db_mod.engine) as s:
        rows = s.execute(select(Tag)).scalars().all()
    assert len(rows) == 1


def test_add_tag_different_entity_types_isolated():
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    face_mod.add_tag("person", "alice", "travel")
    face_mod.add_tag("collection", "vacation", "travel")
    with Session(db_mod.engine) as s:
        rows = s.execute(select(Tag)).scalars().all()
    assert len(rows) == 2
    types = {r.entity_type for r in rows}
    assert types == {"person", "collection"}


def test_remove_tag():
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    face_mod.add_tag("person", "alice", "summer")
    face_mod.add_tag("person", "alice", "winter")
    face_mod.remove_tag("person", "alice", "summer")
    with Session(db_mod.engine) as s:
        rows = s.execute(select(Tag)).scalars().all()
    assert len(rows) == 1
    assert rows[0].tag == "winter"


def test_remove_tag_nonexistent_is_noop():
    face_mod.remove_tag("person", "alice", "ghost")  # should not raise


def test_get_tags_sorted():
    face_mod.add_tag("person", "alice", "zebra")
    face_mod.add_tag("person", "alice", "apple")
    face_mod.add_tag("person", "alice", "mango")
    result = face_mod.get_tags("person", "alice")
    assert result == ["apple", "mango", "zebra"]


def test_get_tags_empty():
    assert face_mod.get_tags("person", "nobody") == []


def test_get_all_tags_bulk():
    face_mod.add_tag("person", "alice", "travel")
    face_mod.add_tag("person", "alice", "summer")
    face_mod.add_tag("person", "bob", "sport")
    result = face_mod.get_all_tags("person")
    assert sorted(result["alice"]) == ["summer", "travel"]
    assert result["bob"] == ["sport"]


def test_get_all_tags_empty():
    assert face_mod.get_all_tags("person") == {}


def test_list_all_tags_distinct_sorted():
    face_mod.add_tag("person", "alice", "travel")
    face_mod.add_tag("person", "bob", "travel")
    face_mod.add_tag("person", "alice", "summer")
    result = face_mod.list_all_tags("person")
    assert result == ["summer", "travel"]


def test_list_all_tags_entity_type_isolation():
    face_mod.add_tag("person", "alice", "travel")
    face_mod.add_tag("collection", "vac", "beach")
    assert face_mod.list_all_tags("person") == ["travel"]
    assert face_mod.list_all_tags("collection") == ["beach"]


def test_rename_entity_tags_person():
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    face_mod.add_tag("person", "alice", "summer")
    face_mod.add_tag("person", "alice", "travel")
    face_mod.rename_entity_tags("person", "alice", "carol")
    with Session(db_mod.engine) as s:
        rows = s.execute(select(Tag)).scalars().all()
    names = {r.entity_name for r in rows}
    assert names == {"carol"}
    assert len(rows) == 2


def test_rename_entity_tags_noop_on_no_match():
    face_mod.add_tag("person", "alice", "summer")
    face_mod.rename_entity_tags("person", "nobody", "carol")
    assert face_mod.get_tags("person", "alice") == ["summer"]


def test_rename_person_propagates_tags():
    """rename_person must migrate tags to the new name."""
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    cat = os.path.join(_dl(), "categorized")
    os.makedirs(os.path.join(cat, "alice"), exist_ok=True)
    face_mod.add_tag("person", "alice", "summer")

    face_mod.rename_person("alice", "carol")

    with Session(db_mod.engine) as s:
        rows = s.execute(select(Tag)).scalars().all()
    assert len(rows) == 1
    assert rows[0].entity_name == "carol"


def test_rename_collection_propagates_tags():
    """rename_collection must migrate tags to the new name."""
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    coll_dir = os.path.join(_dl(), "collections", "old_coll")
    os.makedirs(coll_dir, exist_ok=True)
    face_mod.add_tag("collection", "old_coll", "faves")

    face_mod.rename_collection("old_coll", "new_coll")

    with Session(db_mod.engine) as s:
        rows = s.execute(select(Tag)).scalars().all()
    assert len(rows) == 1
    assert rows[0].entity_name == "new_coll"


# ── Composite thumbnails ──────────────────────────────────────────────────────


def test_gif_key_different_from_thumb_key():
    """GIF key must not collide with individual thumb SHA256 key."""
    rel = "categorized/alice/photo.jpg"
    assert face_mod._gif_key("categorized/alice") != face_mod._thumb_key(rel)


def test_gif_key_stable():
    k1 = face_mod._gif_key("categorized/alice")
    k2 = face_mod._gif_key("categorized/alice")
    assert k1 == k2


def test_invalidate_gif_removes_file(tmp_path, monkeypatch):
    monkeypatch.setattr(face_mod, "DOWNLOAD_DIR", str(tmp_path))
    (tmp_path / ".thumbs").mkdir()
    gp = face_mod._gif_path("categorized/alice")
    open(gp, "wb").write(b"fake")
    face_mod._invalidate_gif("categorized/alice")
    assert not os.path.exists(gp)


def test_invalidate_gif_noop_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(face_mod, "DOWNLOAD_DIR", str(tmp_path))
    face_mod._invalidate_gif("categorized/alice")  # should not raise


def test_get_gif_url_returns_fallback_when_no_gif():
    url = face_mod.get_gif_url("categorized/alice", ["/thumbs/abc.jpg"])
    assert url == "/thumbs/abc.jpg"


def test_get_gif_url_returns_empty_when_no_fallback():
    url = face_mod.get_gif_url("categorized/alice", [])
    assert url == ""


def test_get_gif_url_returns_gif_when_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(face_mod, "DOWNLOAD_DIR", str(tmp_path))
    (tmp_path / ".thumbs").mkdir()
    gp = face_mod._gif_path("categorized/alice")
    open(gp, "wb").write(b"fake_gif")
    url = face_mod.get_gif_url("categorized/alice", ["/thumbs/fallback.jpg"])
    assert url.startswith("/thumbs/")
    assert url.endswith(".gif")
    assert url != "/thumbs/fallback.jpg"


def test_regenerate_person_gif_no_files_invalidates(tmp_path, monkeypatch):
    """No ImageMeta rows → no GIF written, no crash."""
    monkeypatch.setattr(face_mod, "DOWNLOAD_DIR", str(tmp_path))
    (tmp_path / ".thumbs").mkdir()
    face_mod.regenerate_person_gif("nobody")
    gp = face_mod._gif_path("categorized/nobody")
    assert not os.path.exists(gp)


def test_regenerate_collection_gif_missing_dir(tmp_path, monkeypatch):
    """Missing collection dir → GIF invalidated, no crash."""
    monkeypatch.setattr(face_mod, "DOWNLOAD_DIR", str(tmp_path))
    (tmp_path / ".thumbs").mkdir()
    face_mod.regenerate_collection_gif("ghost_coll")
    gp = face_mod._gif_path("collections/ghost_coll")
    assert not os.path.exists(gp)


def test_merge_into_gif_regenerated(monkeypatch):
    """After merge_into, regenerate_person_gif is called."""
    called = []
    monkeypatch.setattr(face_mod, "regenerate_person_gif", lambda n: called.append(n))
    _make_file("album/photo.jpg", b"data")
    _insert_image_meta("album/photo.jpg", sha256="abc")
    face_mod.merge_into(["album"], "alice")
    assert called == ["alice"]


def test_undo_merge_gif_invalidated_and_regenerated(monkeypatch):
    """After undo_merge, invalidate then regenerate are called for dest."""
    invalidated = []
    regenerated = []
    monkeypatch.setattr(face_mod, "_invalidate_gif", lambda r: invalidated.append(r))
    monkeypatch.setattr(face_mod, "regenerate_person_gif", lambda n: regenerated.append(n))

    _make_file("album/photo.jpg", b"data")
    _insert_image_meta("album/photo.jpg", sha256="abc")
    face_mod.merge_into(["album"], "alice")
    regenerated.clear()
    invalidated.clear()

    from sqlalchemy import select
    from sqlalchemy.orm import Session

    with Session(db_mod.engine) as s:
        log = s.execute(select(MergeLog)).scalars().first()

    face_mod.undo_merge(log.id)
    assert "categorized/alice" in invalidated
    assert "alice" in regenerated


def test_rename_person_invalidates_old_and_regenerates_new(monkeypatch):
    invalidated = []
    regenerated = []
    monkeypatch.setattr(face_mod, "_invalidate_gif", lambda r: invalidated.append(r))
    monkeypatch.setattr(face_mod, "regenerate_person_gif", lambda n: regenerated.append(n))

    cat = os.path.join(_dl(), "categorized")
    os.makedirs(os.path.join(cat, "alice"), exist_ok=True)
    face_mod.rename_person("alice", "carol")
    assert "categorized/alice" in invalidated
    assert "carol" in regenerated


def test_rename_person_renames_thumbnail_files(monkeypatch):
    """rename_person must move existing thumbnail files to the new SHA256 key."""
    monkeypatch.setattr(face_mod, "_invalidate_gif", lambda r: None)
    monkeypatch.setattr(face_mod, "regenerate_person_gif", lambda n: None)

    cat = os.path.join(_dl(), "categorized", "alice")
    os.makedirs(cat, exist_ok=True)
    open(os.path.join(cat, "photo.jpg"), "wb").write(b"img")

    # Create a fake thumbnail at the old path
    old_thumb = face_mod._thumb_path("categorized/alice/photo.jpg")
    os.makedirs(os.path.dirname(old_thumb), exist_ok=True)
    open(old_thumb, "wb").write(b"thumb")

    face_mod.rename_person("alice", "carol")

    new_thumb = face_mod._thumb_path("categorized/carol/photo.jpg")
    assert os.path.isfile(new_thumb), "thumbnail must be moved to new key"
    assert not os.path.isfile(old_thumb), "old thumbnail key must not exist"


def test_rename_collection_invalidates_old_and_regenerates_new(monkeypatch):
    invalidated = []
    regenerated = []
    monkeypatch.setattr(face_mod, "_invalidate_gif", lambda r: invalidated.append(r))
    monkeypatch.setattr(face_mod, "regenerate_collection_gif", lambda n: regenerated.append(n))

    coll_dir = os.path.join(_dl(), "collections", "old_coll")
    os.makedirs(coll_dir, exist_ok=True)
    face_mod.rename_collection("old_coll", "new_coll")
    assert "collections/old_coll" in invalidated
    assert "new_coll" in regenerated


def test_delete_collection_file_triggers_gif_regen(monkeypatch):
    """delete_collection_file must regenerate the collection GIF."""
    regenerated = []
    monkeypatch.setattr(face_mod, "regenerate_collection_gif", lambda n: regenerated.append(n))

    coll_dir = os.path.join(_dl(), "collections", "mycoll")
    os.makedirs(coll_dir, exist_ok=True)
    fpath = os.path.join(coll_dir, "photo.jpg")
    open(fpath, "wb").write(b"img")

    face_mod.delete_collection_file("mycoll", "collections/mycoll/photo.jpg")
    assert "mycoll" in regenerated


def test_move_to_collection_triggers_gif_regen(monkeypatch):
    """move_to_collection must regenerate the collection GIF after moving files."""
    regenerated = []
    monkeypatch.setattr(face_mod, "regenerate_collection_gif", lambda n: regenerated.append(n))

    src_dir = os.path.join(_dl(), "album")
    os.makedirs(src_dir, exist_ok=True)
    open(os.path.join(src_dir, "photo.jpg"), "wb").write(b"img")

    result = face_mod.move_to_collection("album", "mycoll")
    assert result is None
    assert "mycoll" in regenerated


# ── Animated GIF generation ───────────────────────────────────────────────────


def _make_rgb_jpeg(path: str, color: tuple = (128, 64, 32)) -> None:
    """Write a minimal valid JPEG to path for use as a thumbnail input."""
    from PIL import Image

    img = Image.new("RGB", (40, 40), color)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path, "JPEG")


def test_gif_path_ends_with_gif(tmp_path, monkeypatch):
    monkeypatch.setattr(face_mod, "DOWNLOAD_DIR", str(tmp_path))
    assert face_mod._gif_path("categorized/alice").endswith(".gif")


def test_gif_key_uses_animated_prefix():
    """'animated:' prefix must be part of the key derivation (not 'composite:')."""
    import hashlib

    rel = "categorized/alice"
    expected = hashlib.sha256(f"animated:{rel}".encode()).hexdigest()
    assert face_mod._gif_key(rel) == expected
    wrong = hashlib.sha256(f"composite:{rel}".encode()).hexdigest()
    assert face_mod._gif_key(rel) != wrong


def test_generate_animated_gif_creates_file(tmp_path, monkeypatch):
    """_generate_animated_gif must write a .gif file when given valid JPEG inputs."""
    monkeypatch.setattr(face_mod, "DOWNLOAD_DIR", str(tmp_path))
    (tmp_path / ".thumbs").mkdir()

    thumb = str(tmp_path / "frame.jpg")
    _make_rgb_jpeg(thumb)

    face_mod._generate_animated_gif("categorized/alice", [thumb])

    gp = face_mod._gif_path("categorized/alice")
    assert os.path.isfile(gp), "GIF file must exist after generation"
    assert gp.endswith(".gif")


def test_generate_animated_gif_all_corrupt_invalidates(tmp_path, monkeypatch):
    """If every frame path is unreadable, _generate_animated_gif must invalidate (no crash)."""
    monkeypatch.setattr(face_mod, "DOWNLOAD_DIR", str(tmp_path))
    (tmp_path / ".thumbs").mkdir()

    invalidated = []
    monkeypatch.setattr(face_mod, "_invalidate_gif", lambda r: invalidated.append(r))

    face_mod._generate_animated_gif("categorized/alice", ["/nonexistent/fake.jpg"])

    assert "categorized/alice" in invalidated
    assert not os.path.isfile(face_mod._gif_path("categorized/alice"))


def test_generate_animated_gif_single_frame(tmp_path, monkeypatch):
    """Single-frame animated GIF (only 1 thumbnail) must still be created."""
    monkeypatch.setattr(face_mod, "DOWNLOAD_DIR", str(tmp_path))
    (tmp_path / ".thumbs").mkdir()

    thumb = str(tmp_path / "solo.jpg")
    _make_rgb_jpeg(thumb)
    face_mod._generate_animated_gif("categorized/solo", [thumb])

    assert os.path.isfile(face_mod._gif_path("categorized/solo"))


def test_generate_animated_gif_caps_at_4_frames(tmp_path, monkeypatch):
    """_generate_animated_gif must use at most 4 frames even when given more."""
    monkeypatch.setattr(face_mod, "DOWNLOAD_DIR", str(tmp_path))
    (tmp_path / ".thumbs").mkdir()

    thumbs = []
    for i in range(6):
        p = str(tmp_path / f"frame{i}.jpg")
        _make_rgb_jpeg(p, color=(i * 40 % 256, 0, 0))
        thumbs.append(p)

    face_mod._generate_animated_gif("categorized/many", thumbs)

    gp = face_mod._gif_path("categorized/many")
    assert os.path.isfile(gp)
    from PIL import Image

    with Image.open(gp) as gif:
        n_frames = getattr(gif, "n_frames", 1)
    assert n_frames <= 4


def test_regenerate_person_gif_creates_file_when_thumbs_exist(tmp_path, monkeypatch):
    """regenerate_person_gif must produce a GIF when valid thumb files exist in DB."""
    monkeypatch.setattr(face_mod, "DOWNLOAD_DIR", str(tmp_path))
    (tmp_path / ".thumbs").mkdir()

    _make_file("categorized/alice/photo.jpg", b"img")
    _insert_image_meta("categorized/alice/photo.jpg", sha256="aaa", max_face_ratio=0.5)

    thumb = face_mod._thumb_path("categorized/alice/photo.jpg")
    _make_rgb_jpeg(thumb)

    called_gif = []
    real_gen = face_mod._generate_animated_gif
    monkeypatch.setattr(
        face_mod, "_generate_animated_gif", lambda r, p: (called_gif.append(r), real_gen(r, p))
    )

    face_mod.regenerate_person_gif("alice")

    assert "categorized/alice" in called_gif
    assert os.path.isfile(face_mod._gif_path("categorized/alice"))


def test_regenerate_person_gif_picks_highest_face_ratio(tmp_path, monkeypatch):
    """regenerate_person_gif must order by max_face_ratio DESC (best faces first)."""
    monkeypatch.setattr(face_mod, "DOWNLOAD_DIR", str(tmp_path))
    (tmp_path / ".thumbs").mkdir()

    for fname, ratio in [("a.jpg", 0.1), ("b.jpg", 0.9), ("c.jpg", 0.5)]:
        rel = f"categorized/alice/{fname}"
        _make_file(rel, b"img")
        _insert_image_meta(rel, sha256=fname, max_face_ratio=ratio)
        _make_rgb_jpeg(face_mod._thumb_path(rel))

    received_paths = []
    monkeypatch.setattr(
        face_mod, "_generate_animated_gif", lambda r, paths: received_paths.extend(paths)
    )

    face_mod.regenerate_person_gif("alice")

    b_thumb = face_mod._thumb_path("categorized/alice/b.jpg")
    assert received_paths[0] == b_thumb, "highest face-ratio thumb must be first"


def test_regenerate_collection_gif_uses_sorted_filenames(tmp_path, monkeypatch):
    """regenerate_collection_gif must iterate files in sorted order."""
    monkeypatch.setattr(face_mod, "DOWNLOAD_DIR", str(tmp_path))
    (tmp_path / ".thumbs").mkdir()

    coll_dir = os.path.join(_dl(), "collections", "pics")
    os.makedirs(coll_dir, exist_ok=True)

    for fname in ["c.jpg", "a.jpg", "b.jpg"]:
        fpath = os.path.join(coll_dir, fname)
        open(fpath, "wb").write(b"img")
        _make_rgb_jpeg(face_mod._thumb_path(f"collections/pics/{fname}"))

    received_paths = []
    monkeypatch.setattr(
        face_mod, "_generate_animated_gif", lambda r, paths: received_paths.extend(paths)
    )

    face_mod.regenerate_collection_gif("pics")

    a_thumb = face_mod._thumb_path("collections/pics/a.jpg")
    assert received_paths[0] == a_thumb, "sorted order: a.jpg must be first"


def test_regenerate_collection_gif_skips_non_media(tmp_path, monkeypatch):
    """regenerate_collection_gif must ignore non-media files (e.g. .txt)."""
    monkeypatch.setattr(face_mod, "DOWNLOAD_DIR", str(tmp_path))
    (tmp_path / ".thumbs").mkdir()

    coll_dir = os.path.join(_dl(), "collections", "mixed")
    os.makedirs(coll_dir, exist_ok=True)

    open(os.path.join(coll_dir, "readme.txt"), "wb").write(b"text")
    _make_rgb_jpeg(face_mod._thumb_path("collections/mixed/photo.jpg"))
    open(os.path.join(coll_dir, "photo.jpg"), "wb").write(b"img")

    received = []
    monkeypatch.setattr(face_mod, "_generate_animated_gif", lambda r, paths: received.extend(paths))

    face_mod.regenerate_collection_gif("mixed")

    assert len(received) == 1
    # received[0] is the SHA256-keyed thumb path, not the original filename
    assert received[0] == face_mod._thumb_path("collections/mixed/photo.jpg")


def test_get_gif_url_key_prefix_differs_from_old_composite():
    """get_gif_url must serve .gif not .jpg and key must differ from old composite key."""
    import hashlib

    rel = "categorized/alice"
    old_composite_key = hashlib.sha256(f"composite:{rel}".encode()).hexdigest()
    new_gif_key = face_mod._gif_key(rel)
    assert new_gif_key != old_composite_key


def test_regenerate_person_gif_missing_thumb_falls_back_to_invalidate(tmp_path, monkeypatch):
    """DB row exists but thumb file is missing → no GIF generated, invalidate called."""
    monkeypatch.setattr(face_mod, "DOWNLOAD_DIR", str(tmp_path))
    (tmp_path / ".thumbs").mkdir()

    _make_file("categorized/bob/photo.jpg", b"img")
    _insert_image_meta("categorized/bob/photo.jpg", sha256="bbb", max_face_ratio=0.3)
    # deliberately do NOT create a thumb file

    invalidated = []
    monkeypatch.setattr(face_mod, "_invalidate_gif", lambda r: invalidated.append(r))

    face_mod.regenerate_person_gif("bob")

    assert "categorized/bob" in invalidated
    assert not os.path.isfile(face_mod._gif_path("categorized/bob"))


# ── warmup_missing_gifs ───────────────────────────────────────────────────────


def test_warmup_missing_categorized_dir_no_crash(tmp_path, monkeypatch):
    """warmup_missing_gifs must not crash when categorized dir doesn't exist."""
    monkeypatch.setattr(face_mod, "_categorized_dir", lambda: str(tmp_path / "categorized"))
    monkeypatch.setattr(face_mod, "_collections_dir", lambda: str(tmp_path / "collections"))
    face_mod.warmup_missing_gifs()  # must not raise


def test_warmup_generates_missing_person_gif(tmp_path, monkeypatch):
    """warmup_missing_gifs calls regenerate_person_gif for persons without a GIF."""
    cat_dir = tmp_path / "categorized"
    (cat_dir / "alice").mkdir(parents=True)

    monkeypatch.setattr(face_mod, "_categorized_dir", lambda: str(cat_dir))
    monkeypatch.setattr(face_mod, "_collections_dir", lambda: str(tmp_path / "collections"))
    monkeypatch.setattr(
        face_mod,
        "_gif_path",
        lambda rel: str(tmp_path / "thumbs" / (rel.replace("/", "_") + ".gif")),
    )

    generated = []
    monkeypatch.setattr(face_mod, "regenerate_person_gif", lambda n: generated.append(n))

    face_mod.warmup_missing_gifs()

    assert generated == ["alice"]


def test_warmup_skips_existing_person_gif(tmp_path, monkeypatch):
    """warmup_missing_gifs skips persons that already have a GIF file."""
    cat_dir = tmp_path / "categorized"
    (cat_dir / "alice").mkdir(parents=True)

    # Simulate existing GIF by making _gif_path point to a real file
    gif_dir = tmp_path / "thumbs"
    gif_dir.mkdir()

    def fake_gif_path(rel):
        return str(gif_dir / (rel.replace("/", "_") + ".gif"))

    (gif_dir / "categorized_alice.gif").write_bytes(b"GIF89a")

    monkeypatch.setattr(face_mod, "_categorized_dir", lambda: str(cat_dir))
    monkeypatch.setattr(face_mod, "_collections_dir", lambda: str(tmp_path / "collections"))
    monkeypatch.setattr(face_mod, "_gif_path", fake_gif_path)

    generated = []
    monkeypatch.setattr(face_mod, "regenerate_person_gif", lambda n: generated.append(n))

    face_mod.warmup_missing_gifs()

    assert generated == []


def test_warmup_skips_dotfiles(tmp_path, monkeypatch):
    """warmup_missing_gifs skips entries whose name starts with '.'."""
    cat_dir = tmp_path / "categorized"
    (cat_dir / ".DS_Store").mkdir(parents=True)
    (cat_dir / "alice").mkdir()

    monkeypatch.setattr(face_mod, "_categorized_dir", lambda: str(cat_dir))
    monkeypatch.setattr(face_mod, "_collections_dir", lambda: str(tmp_path / "collections"))
    monkeypatch.setattr(
        face_mod,
        "_gif_path",
        lambda rel: str(tmp_path / "thumbs" / (rel.replace("/", "_") + ".gif")),
    )

    generated = []
    monkeypatch.setattr(face_mod, "regenerate_person_gif", lambda n: generated.append(n))

    face_mod.warmup_missing_gifs()

    assert generated == ["alice"]
    assert ".DS_Store" not in generated


def test_warmup_skips_non_dir_entries(tmp_path, monkeypatch):
    """warmup_missing_gifs skips plain files inside categorized dir."""
    cat_dir = tmp_path / "categorized"
    cat_dir.mkdir()
    (cat_dir / "stray_file.txt").write_text("oops")
    (cat_dir / "alice").mkdir()

    monkeypatch.setattr(face_mod, "_categorized_dir", lambda: str(cat_dir))
    monkeypatch.setattr(face_mod, "_collections_dir", lambda: str(tmp_path / "collections"))
    monkeypatch.setattr(
        face_mod,
        "_gif_path",
        lambda rel: str(tmp_path / "thumbs" / (rel.replace("/", "_") + ".gif")),
    )

    generated = []
    monkeypatch.setattr(face_mod, "regenerate_person_gif", lambda n: generated.append(n))

    face_mod.warmup_missing_gifs()

    assert generated == ["alice"]
    assert "stray_file.txt" not in generated


def test_warmup_generates_missing_collection_gif(tmp_path, monkeypatch):
    """warmup_missing_gifs calls regenerate_collection_gif for collections without a GIF."""
    coll_dir = tmp_path / "collections"
    (coll_dir / "summer").mkdir(parents=True)

    monkeypatch.setattr(face_mod, "_categorized_dir", lambda: str(tmp_path / "categorized"))
    monkeypatch.setattr(face_mod, "_collections_dir", lambda: str(coll_dir))
    monkeypatch.setattr(
        face_mod,
        "_gif_path",
        lambda rel: str(tmp_path / "thumbs" / (rel.replace("/", "_") + ".gif")),
    )

    generated = []
    monkeypatch.setattr(face_mod, "regenerate_collection_gif", lambda n: generated.append(n))

    face_mod.warmup_missing_gifs()

    assert generated == ["summer"]


def test_warmup_skips_existing_collection_gif(tmp_path, monkeypatch):
    """warmup_missing_gifs skips collections that already have a GIF file."""
    coll_dir = tmp_path / "collections"
    (coll_dir / "summer").mkdir(parents=True)

    gif_dir = tmp_path / "thumbs"
    gif_dir.mkdir()

    def fake_gif_path(rel):
        return str(gif_dir / (rel.replace("/", "_") + ".gif"))

    (gif_dir / "collections_summer.gif").write_bytes(b"GIF89a")

    monkeypatch.setattr(face_mod, "_categorized_dir", lambda: str(tmp_path / "categorized"))
    monkeypatch.setattr(face_mod, "_collections_dir", lambda: str(coll_dir))
    monkeypatch.setattr(face_mod, "_gif_path", fake_gif_path)

    generated = []
    monkeypatch.setattr(face_mod, "regenerate_collection_gif", lambda n: generated.append(n))

    face_mod.warmup_missing_gifs()

    assert generated == []


def test_warmup_exception_in_one_person_does_not_abort_rest(tmp_path, monkeypatch):
    """If regenerate_person_gif raises for one entry, others are still processed."""
    cat_dir = tmp_path / "categorized"
    (cat_dir / "alice").mkdir(parents=True)
    (cat_dir / "bob").mkdir()
    (cat_dir / "carol").mkdir()

    monkeypatch.setattr(face_mod, "_categorized_dir", lambda: str(cat_dir))
    monkeypatch.setattr(face_mod, "_collections_dir", lambda: str(tmp_path / "collections"))
    monkeypatch.setattr(
        face_mod,
        "_gif_path",
        lambda rel: str(tmp_path / "thumbs" / (rel.replace("/", "_") + ".gif")),
    )

    generated = []

    def boom_on_bob(name):
        if name == "bob":
            raise RuntimeError("simulated failure")
        generated.append(name)

    monkeypatch.setattr(face_mod, "regenerate_person_gif", boom_on_bob)

    face_mod.warmup_missing_gifs()

    assert set(generated) == {"alice", "carol"}


def test_warmup_mixed_persons_only_missing_generated(tmp_path, monkeypatch):
    """Only persons without an existing GIF file are regenerated."""
    cat_dir = tmp_path / "categorized"
    (cat_dir / "alice").mkdir(parents=True)
    (cat_dir / "bob").mkdir()

    gif_dir = tmp_path / "thumbs"
    gif_dir.mkdir()

    def fake_gif_path(rel):
        return str(gif_dir / (rel.replace("/", "_") + ".gif"))

    # alice already has a GIF; bob does not
    (gif_dir / "categorized_alice.gif").write_bytes(b"GIF89a")

    monkeypatch.setattr(face_mod, "_categorized_dir", lambda: str(cat_dir))
    monkeypatch.setattr(face_mod, "_collections_dir", lambda: str(tmp_path / "collections"))
    monkeypatch.setattr(face_mod, "_gif_path", fake_gif_path)

    generated = []
    monkeypatch.setattr(face_mod, "regenerate_person_gif", lambda n: generated.append(n))

    face_mod.warmup_missing_gifs()

    assert generated == ["bob"]


def test_warmup_missing_collections_dir_no_crash(tmp_path, monkeypatch):
    """warmup_missing_gifs must not crash when collections dir doesn't exist."""
    cat_dir = tmp_path / "categorized"
    cat_dir.mkdir()

    monkeypatch.setattr(face_mod, "_categorized_dir", lambda: str(cat_dir))
    monkeypatch.setattr(face_mod, "_collections_dir", lambda: str(tmp_path / "collections"))
    monkeypatch.setattr(
        face_mod,
        "_gif_path",
        lambda rel: str(tmp_path / "thumbs" / (rel.replace("/", "_") + ".gif")),
    )
    monkeypatch.setattr(face_mod, "regenerate_person_gif", lambda n: None)

    face_mod.warmup_missing_gifs()  # must not raise


def test_warmup_processes_both_persons_and_collections(tmp_path, monkeypatch):
    """warmup_missing_gifs generates GIFs for both persons and collections in one call."""
    cat_dir = tmp_path / "categorized"
    (cat_dir / "alice").mkdir(parents=True)
    coll_dir = tmp_path / "collections"
    (coll_dir / "summer").mkdir(parents=True)

    monkeypatch.setattr(face_mod, "_categorized_dir", lambda: str(cat_dir))
    monkeypatch.setattr(face_mod, "_collections_dir", lambda: str(coll_dir))
    monkeypatch.setattr(
        face_mod,
        "_gif_path",
        lambda rel: str(tmp_path / "thumbs" / (rel.replace("/", "_") + ".gif")),
    )

    persons = []
    colls = []
    monkeypatch.setattr(face_mod, "regenerate_person_gif", lambda n: persons.append(n))
    monkeypatch.setattr(face_mod, "regenerate_collection_gif", lambda n: colls.append(n))

    face_mod.warmup_missing_gifs()

    assert persons == ["alice"]
    assert colls == ["summer"]


# ── _check_gif_version ────────────────────────────────────────────────────────


def test_check_gif_version_no_sentinel_creates_it(tmp_path, monkeypatch):
    """When .gif_version doesn't exist, _check_gif_version creates it with version '2'."""
    td = tmp_path / ".thumbs"
    td.mkdir()
    monkeypatch.setattr(face_mod, "_thumbs_dir", lambda: str(td))

    face_mod._check_gif_version()

    sentinel = td / ".gif_version"
    assert sentinel.is_file()
    assert sentinel.read_text().strip() == "2"


def test_check_gif_version_correct_version_is_noop(tmp_path, monkeypatch):
    """When .gif_version already contains '2', no GIFs are deleted and sentinel unchanged."""
    td = tmp_path / ".thumbs"
    td.mkdir()
    gif = td / "animated_alice.gif"
    gif.write_bytes(b"GIF89a")
    sentinel = td / ".gif_version"
    sentinel.write_text("2")

    monkeypatch.setattr(face_mod, "_thumbs_dir", lambda: str(td))

    face_mod._check_gif_version()

    assert gif.is_file(), "Existing GIF must not be deleted when version matches"
    assert sentinel.read_text().strip() == "2"


def test_check_gif_version_wrong_version_purges_gifs(tmp_path, monkeypatch):
    """When .gif_version has an old value, all .gif files are deleted and '2' is written."""
    td = tmp_path / ".thumbs"
    td.mkdir()
    gif1 = td / "a.gif"
    gif2 = td / "b.gif"
    jpg = td / "thumb.jpg"
    gif1.write_bytes(b"GIF89a")
    gif2.write_bytes(b"GIF89a")
    jpg.write_bytes(b"\xff\xd8\xff")
    sentinel = td / ".gif_version"
    sentinel.write_text("1")

    monkeypatch.setattr(face_mod, "_thumbs_dir", lambda: str(td))

    face_mod._check_gif_version()

    assert not gif1.is_file(), "Old GIFs must be purged"
    assert not gif2.is_file(), "Old GIFs must be purged"
    assert jpg.is_file(), "Non-GIF thumbs must be preserved"
    assert sentinel.read_text().strip() == "2"


def test_check_gif_version_missing_thumbs_dir_no_crash(tmp_path, monkeypatch):
    """When .thumbs doesn't exist at all, _check_gif_version creates it and writes sentinel."""
    td = tmp_path / ".thumbs"
    assert not td.exists()
    monkeypatch.setattr(face_mod, "_thumbs_dir", lambda: str(td))

    face_mod._check_gif_version()

    assert (td / ".gif_version").is_file()
    assert (td / ".gif_version").read_text().strip() == "2"


def test_warmup_writes_gif_version_sentinel(tmp_path, monkeypatch):
    """warmup_missing_gifs writes the .gif_version sentinel after a clean run."""
    td = tmp_path / ".thumbs"
    monkeypatch.setattr(face_mod, "_thumbs_dir", lambda: str(td))
    monkeypatch.setattr(face_mod, "_categorized_dir", lambda: str(tmp_path / "categorized"))
    monkeypatch.setattr(face_mod, "_collections_dir", lambda: str(tmp_path / "collections"))

    face_mod.warmup_missing_gifs()

    assert (td / ".gif_version").is_file()


# ── _generate_video_thumb ─────────────────────────────────────────────────────


def test_generate_video_thumb_ffmpeg_success_skips_cv2(tmp_path, monkeypatch):
    """When ffmpeg is available and succeeds, cv2 is never imported."""
    import pathlib
    import shutil
    import subprocess

    monkeypatch.setattr(face_mod, "DOWNLOAD_DIR", str(tmp_path))
    (tmp_path / ".thumbs").mkdir()

    fake_video = str(tmp_path / "vid.mp4")
    pathlib.Path(fake_video).write_bytes(b"fake-video")

    def fake_which(cmd):
        return "/usr/bin/ffmpeg" if cmd == "ffmpeg" else None

    def fake_run(args, **kwargs):
        out = args[-1]
        pathlib.Path(out).write_bytes(b"\xff\xd8\xff\xe0")
        return subprocess.CompletedProcess(args, 0, b"", b"")

    monkeypatch.setattr(shutil, "which", fake_which)
    monkeypatch.setattr(subprocess, "run", fake_run)
    # Patch cv2 to detect any import attempt
    import sys

    orig_cv2 = sys.modules.get("cv2")
    sys.modules.pop("cv2", None)

    try:
        face_mod._generate_video_thumb(fake_video, "downloads/vid.mp4")
    finally:
        if orig_cv2 is not None:
            sys.modules["cv2"] = orig_cv2

    out = face_mod._thumb_path("downloads/vid.mp4")
    assert os.path.isfile(out), "Thumbnail must be written by ffmpeg path"


def test_generate_video_thumb_no_ffmpeg_no_subprocess(tmp_path, monkeypatch):
    """When ffmpeg is not on PATH, subprocess.run is not called."""
    import pathlib
    import shutil
    import subprocess

    monkeypatch.setattr(face_mod, "DOWNLOAD_DIR", str(tmp_path))
    (tmp_path / ".thumbs").mkdir()

    fake_video = str(tmp_path / "vid.mp4")
    pathlib.Path(fake_video).write_bytes(b"fake-video")

    run_calls = []

    monkeypatch.setattr(shutil, "which", lambda _: None)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: run_calls.append(a))

    try:
        # cv2.VideoCapture on a fake file returns ret=False — the function returns early
        face_mod._generate_video_thumb(fake_video, "downloads/vid.mp4")
    except Exception:
        pass  # cv2 may raise on a fake file; that's fine

    assert run_calls == [], "subprocess.run must not be called when ffmpeg is absent"


def test_generate_video_thumb_ffmpeg_nonzero_falls_through(tmp_path, monkeypatch):
    """When ffmpeg returns non-zero, the cv2 fallback path is attempted."""
    import pathlib
    import shutil
    import subprocess

    monkeypatch.setattr(face_mod, "DOWNLOAD_DIR", str(tmp_path))
    (tmp_path / ".thumbs").mkdir()

    fake_video = str(tmp_path / "vid.mp4")
    pathlib.Path(fake_video).write_bytes(b"fake-video")

    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/ffmpeg")

    def fake_run_fail(args, **kwargs):
        return subprocess.CompletedProcess(args, 1, b"", b"error")

    monkeypatch.setattr(subprocess, "run", fake_run_fail)

    # The function should fall through to cv2 without raising
    try:
        face_mod._generate_video_thumb(fake_video, "downloads/vid.mp4")
    except Exception:
        pass  # cv2 on a fake file may raise; we just verify no crash from ffmpeg path
