"""Face scanning, clustering, and merge logic for the Organize feature."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections import defaultdict
from typing import Callable

import numpy as np
from sqlalchemy import delete, select, update

from .db import (
    DismissedMatch,
    FaceEmbedding,
    FolderCentroid,
    ImageMeta,
    MergeLog,
    Notification,
    Setting,
    Tag,
    get_scan_meta,
    get_session,
    get_setting,
    set_scan_meta,
    utcnow,
)

logger = logging.getLogger(__name__)

DOWNLOAD_DIR = os.environ.get("DOWNLOAD_DIR", "downloads")

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
_MEDIA_EXTS = _IMAGE_EXTS | {
    ".mp4",
    ".webm",
    ".mkv",
    ".avi",
    ".mov",
    ".flv",
    ".m4v",
    ".wmv",
    ".mp3",
    ".m4a",
    ".opus",
    ".flac",
}
_VIDEO_MIME: dict[str, str] = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mkv": "video/mp4",
    ".avi": "video/mp4",
    ".mov": "video/mp4",
    ".flv": "video/mp4",
    ".m4v": "video/mp4",
    ".wmv": "video/mp4",
}

_scan_lock = threading.Lock()

# InsightFace app is expensive to initialise — cache per model name
_face_app_cache: dict[str, object] = {}
_face_app_lock = threading.Lock()


def _get_face_app(model_name: str):
    with _face_app_lock:
        if model_name not in _face_app_cache:
            from insightface.app import FaceAnalysis

            models_dir = os.path.join(DOWNLOAD_DIR, ".models")
            os.makedirs(models_dir, exist_ok=True)
            fa = FaceAnalysis(name=model_name, root=models_dir, providers=["CPUExecutionProvider"])
            fa.prepare(ctx_id=0, det_size=(640, 640))
            _face_app_cache[model_name] = fa
        return _face_app_cache[model_name]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _thumbs_dir() -> str:
    return os.path.join(DOWNLOAD_DIR, ".thumbs")


def _thumb_key(rel_path: str) -> str:
    return hashlib.sha256(rel_path.encode()).hexdigest()


def _thumb_path(rel_path: str) -> str:
    return os.path.join(_thumbs_dir(), _thumb_key(rel_path) + ".jpg")


def _sha256_file(path: str, limit_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        remaining = limit_bytes
        while True:
            chunk_size = 65536
            if remaining is not None:
                chunk_size = min(chunk_size, remaining)
                if chunk_size == 0:
                    break
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return h.hexdigest()


def _embedding_to_bytes(arr) -> bytes:
    a = np.array(arr, dtype=np.float32)
    return a.tobytes()


def _bytes_to_embedding(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _normalize_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _all_image_files(folder: str) -> list[str]:
    result = []
    for root, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in _IMAGE_EXTS:
                result.append(os.path.join(root, f))
    return result


def _categorized_dir() -> str:
    return os.path.join(DOWNLOAD_DIR, "categorized")


def _collections_dir() -> str:
    return os.path.join(DOWNLOAD_DIR, "collections")


def _rel(path: str) -> str:
    return os.path.relpath(path, DOWNLOAD_DIR)


def _like_prefix(rel: str) -> str:
    """Return a LIKE pattern that matches rel as a path prefix (rel/%).

    Escapes SQL LIKE wildcards %, _, \\ so folder names like 'john_doe'
    only match themselves, not 'john doe' or 'johnXdoe'.
    Use with .like(..., escape='\\\\').
    """
    escaped = rel.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return escaped + "/%"


# ── Thumbnail generation ──────────────────────────────────────────────────────


def _generate_thumb(image_path: str, rel_path: str) -> None:
    from PIL import Image

    out_path = _thumb_path(rel_path)
    os.makedirs(_thumbs_dir(), exist_ok=True)
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((256, 256))
        img.save(out_path, "JPEG", quality=80)


# ── Tag CRUD ──────────────────────────────────────────────────────────────────


def add_tag(entity_type: str, entity_name: str, tag: str) -> None:
    tag = tag.strip().lower()
    if not tag:
        return
    with get_session() as s:
        existing = s.execute(
            select(Tag).where(
                Tag.entity_type == entity_type,
                Tag.entity_name == entity_name,
                Tag.tag == tag,
            )
        ).scalar_one_or_none()
        if not existing:
            s.add(Tag(entity_type=entity_type, entity_name=entity_name, tag=tag))
        s.commit()


def remove_tag(entity_type: str, entity_name: str, tag: str) -> None:
    with get_session() as s:
        s.execute(
            delete(Tag).where(
                Tag.entity_type == entity_type,
                Tag.entity_name == entity_name,
                Tag.tag == tag,
            )
        )
        s.commit()


def get_tags(entity_type: str, entity_name: str) -> list[str]:
    with get_session() as s:
        rows = (
            s.execute(
                select(Tag.tag)
                .where(Tag.entity_type == entity_type, Tag.entity_name == entity_name)
                .order_by(Tag.tag)
            )
            .scalars()
            .all()
        )
    return list(rows)


def get_all_tags(entity_type: str) -> dict[str, list[str]]:
    """Bulk-load all tags for an entity type — one query instead of N per card."""
    with get_session() as s:
        rows = s.execute(select(Tag).where(Tag.entity_type == entity_type)).scalars().all()
    result: dict[str, list[str]] = {}
    for row in rows:
        result.setdefault(row.entity_name, []).append(row.tag)
    for v in result.values():
        v.sort()
    return result


def list_all_tags(entity_type: str) -> list[str]:
    with get_session() as s:
        rows = (
            s.execute(
                select(Tag.tag).where(Tag.entity_type == entity_type).distinct().order_by(Tag.tag)
            )
            .scalars()
            .all()
        )
    return list(rows)


def rename_entity_tags(entity_type: str, old_name: str, new_name: str) -> None:
    """Migrate tag rows when a person or collection is renamed."""
    with get_session() as s:
        s.execute(
            update(Tag)
            .where(Tag.entity_type == entity_type, Tag.entity_name == old_name)
            .values(entity_name=new_name)
        )
        s.commit()


# ── Animated GIF thumbnails ───────────────────────────────────────────────────


def _gif_key(entity_rel: str) -> str:
    return hashlib.sha256(f"animated:{entity_rel}".encode()).hexdigest()


def _gif_path(entity_rel: str) -> str:
    return os.path.join(_thumbs_dir(), _gif_key(entity_rel) + ".gif")


def _invalidate_gif(entity_rel: str) -> None:
    p = _gif_path(entity_rel)
    if os.path.exists(p):
        try:
            os.remove(p)
        except OSError:
            pass


def _generate_animated_gif(entity_rel: str, thumb_paths: list[str]) -> None:
    """Stitch up to 4 thumbs into an animated GIF (1500 ms/frame, infinite loop). Executor-only."""
    from PIL import Image, ImageOps

    size = 320
    frames: list[Image.Image] = []
    for path in thumb_paths[:4]:
        try:
            with Image.open(path) as img:
                frames.append(ImageOps.fit(img.convert("RGB"), (size, size), method=Image.LANCZOS))
        except Exception:
            pass
    if not frames:
        _invalidate_gif(entity_rel)
        return
    out = _gif_path(entity_rel)
    os.makedirs(_thumbs_dir(), exist_ok=True)
    frames[0].save(
        out,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=1500,
    )


def regenerate_person_gif(person_name: str) -> None:
    """Pick best 4 thumb paths for a person and regenerate animated GIF. Executor-only."""
    entity_rel = f"categorized/{person_name}"
    pattern = _like_prefix(entity_rel)
    with get_session() as s:
        rows = (
            s.execute(
                select(ImageMeta)
                .where(ImageMeta.file_path.like(pattern, escape="\\"))
                .order_by(ImageMeta.max_face_ratio.desc().nulls_last())
                .limit(4)
            )
            .scalars()
            .all()
        )
    paths = [_thumb_path(r.file_path) for r in rows if os.path.isfile(_thumb_path(r.file_path))]
    if paths:
        _generate_animated_gif(entity_rel, paths)
    else:
        _invalidate_gif(entity_rel)


def regenerate_collection_gif(coll_name: str) -> None:
    """Pick first 4 media thumb paths in a collection and regenerate animated GIF. Executor-only."""
    entity_rel = f"collections/{coll_name}"
    coll_abs = os.path.join(_collections_dir(), coll_name)
    if not os.path.isdir(coll_abs):
        _invalidate_gif(entity_rel)
        return
    paths = []
    for fname in sorted(os.listdir(coll_abs)):
        if len(paths) >= 4:
            break
        ext = os.path.splitext(fname)[1].lower()
        if ext not in _MEDIA_EXTS:
            continue
        rel = f"{entity_rel}/{fname}"
        tp = _thumb_path(rel)
        if os.path.isfile(tp):
            paths.append(tp)
    if paths:
        _generate_animated_gif(entity_rel, paths)
    else:
        _invalidate_gif(entity_rel)


def get_gif_url(entity_rel: str, fallback_rels: list[str]) -> str:
    """Return animated GIF URL if it exists, else first fallback URL, else empty string.

    Pure filesystem stat — safe to call from the event loop.
    """
    gp = _gif_path(entity_rel)
    if os.path.isfile(gp):
        return f"/thumbs/{_gif_key(entity_rel)}.gif"
    if fallback_rels:
        return fallback_rels[0]
    return ""


_GIF_SPEED_VERSION = "2"  # bump this when the animation duration constant changes


def _check_gif_version() -> None:
    """Delete all animated GIFs if the speed version sentinel doesn't match. Executor-only."""
    td = _thumbs_dir()
    sentinel = os.path.join(td, ".gif_version")
    if os.path.isfile(sentinel):
        try:
            with open(sentinel) as f:
                if f.read().strip() == _GIF_SPEED_VERSION:
                    return
        except OSError:
            pass  # unreadable sentinel → fall through to purge + recreate
    if os.path.isdir(td):
        for fn in os.listdir(td):
            if fn.endswith(".gif"):
                try:
                    os.remove(os.path.join(td, fn))
                except Exception:
                    pass
    os.makedirs(td, exist_ok=True)
    try:
        with open(sentinel, "w") as f:
            f.write(_GIF_SPEED_VERSION)
    except OSError:
        pass  # non-fatal; will retry on next startup


def warmup_missing_gifs() -> None:
    """Generate animated GIFs for all persons/collections that don't have one yet. Executor-only."""
    _check_gif_version()
    cat_dir = _categorized_dir()
    if os.path.isdir(cat_dir):
        for name in os.listdir(cat_dir):
            if name.startswith("."):
                continue
            if not os.path.isdir(os.path.join(cat_dir, name)):
                continue
            if not os.path.isfile(_gif_path(f"categorized/{name}")):
                try:
                    regenerate_person_gif(name)
                except Exception:
                    pass

    coll_dir = _collections_dir()
    if os.path.isdir(coll_dir):
        for name in os.listdir(coll_dir):
            if name.startswith("."):
                continue
            if not os.path.isdir(os.path.join(coll_dir, name)):
                continue
            if not os.path.isfile(_gif_path(f"collections/{name}")):
                try:
                    regenerate_collection_gif(name)
                except Exception:
                    pass


# ── scan_dir ──────────────────────────────────────────────────────────────────


def scan_dir(path: str, on_progress: Callable[[int, int], None] | None = None) -> dict:
    """Scan all media files under path, extract embeddings and hashes."""
    if not _scan_lock.acquire(blocking=False):
        raise RuntimeError("Scan already in progress")

    try:
        return _scan_dir_locked(path, on_progress)
    finally:
        _scan_lock.release()


def _scan_dir_locked(path: str, on_progress) -> dict:
    face_model = get_setting("face_model", "buffalo_s")

    # Detect model change → wipe all embeddings, meta, centroids, thumbs
    stored_model = get_scan_meta("model")
    if stored_model and stored_model != face_model:
        logger.info(f"Model changed from {stored_model} to {face_model} — wiping embeddings")
        with get_session() as s:
            s.execute(delete(FaceEmbedding))
            s.execute(delete(ImageMeta))
            s.execute(delete(FolderCentroid))
            s.commit()
        thumbs = _thumbs_dir()
        if os.path.isdir(thumbs):
            import shutil

            shutil.rmtree(thumbs)
        with _face_app_lock:
            _face_app_cache.clear()

    os.makedirs(_thumbs_dir(), exist_ok=True)

    set_scan_meta("status", "running")
    set_scan_meta("last_error", None)

    # Collect all files (excluding collections/ subtree)
    coll_abs = os.path.abspath(_collections_dir())
    all_files = []
    for root, dirs, files in os.walk(path):
        dirs[:] = [
            d
            for d in dirs
            if not d.startswith(".") and os.path.abspath(os.path.join(root, d)) != coll_abs
        ]
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in _MEDIA_EXTS:
                all_files.append(os.path.join(root, fname))

    total = len(all_files)
    set_scan_meta("total", str(total))
    set_scan_meta("progress", "0")

    scanned = skipped = faces_found = 0

    try:
        for idx, file_path in enumerate(all_files):
            if get_scan_meta("status") == "cancelling":
                return {"scanned": scanned, "skipped": skipped, "faces_found": faces_found}

            if idx % 10 == 0:
                set_scan_meta("progress", str(idx))
            if on_progress:
                on_progress(idx, total)

            try:
                faces_found += _scan_file(file_path, face_model)
                scanned += 1
            except Exception as e:
                logger.warning(f"Skipped {file_path}: {e}")
                skipped += 1

        # Cleanup stale rows
        _cleanup_stale(path)

        # Recompute centroids for all categorized/ folders
        _recompute_centroids(face_model)

        set_scan_meta("model", face_model)
        set_scan_meta("last_scan", utcnow().isoformat())
        set_scan_meta("progress", str(total))

        return {"scanned": scanned, "skipped": skipped, "faces_found": faces_found}
    except Exception as e:
        set_scan_meta("last_error", str(e))
        raise
    finally:
        set_scan_meta("status", "idle")


def _scan_file(file_path: str, face_model: str) -> int:
    """Returns the number of faces found (0 for videos/no-face images)."""
    mtime = os.path.getmtime(file_path)
    rel = _rel(file_path)

    with get_session() as s:
        existing = s.execute(
            select(ImageMeta).where(ImageMeta.file_path == rel)
        ).scalar_one_or_none()
        if existing and existing.file_mtime == mtime:
            # Backfill file_size for files scanned before this column was added.
            if existing.file_size is None:
                try:
                    with get_session() as s2:
                        row = s2.execute(
                            select(ImageMeta).where(ImageMeta.file_path == rel)
                        ).scalar_one_or_none()
                        if row:
                            row.file_size = os.path.getsize(file_path)
                            s2.commit()
                except Exception:
                    pass
            # Backfill: if this is a video with no thumbnail yet, generate it now.
            # Handles collections scanned before video thumb support was added.
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in _IMAGE_EXTS and not os.path.exists(_thumb_path(rel)):
                try:
                    _generate_video_thumb(file_path, rel)
                except Exception:
                    pass
            return existing.face_count or 0

    ext = os.path.splitext(file_path)[1].lower()
    if ext in _IMAGE_EXTS:
        return _scan_image_file(file_path, rel, mtime, face_model)
    else:
        _scan_video_file(file_path, rel, mtime)
        return 0


def _scan_image_file(file_path: str, rel: str, mtime: float, face_model: str) -> int:
    import cv2
    import imagehash
    from PIL import Image

    file_size = os.path.getsize(file_path)
    phash_val = None
    sha256_val = _sha256_file(file_path)
    face_count = 0
    max_face_ratio = None
    embeddings_data = []

    try:
        with Image.open(file_path) as img:
            w, h = img.size
            img_area = w * h
            phash_val = str(imagehash.phash(img))
    except Exception as e:
        raise RuntimeError(f"PIL failed: {e}") from e

    try:
        fa = _get_face_app(face_model)
        img_cv = cv2.imread(file_path)
        if img_cv is not None:
            for i, face in enumerate(fa.get(img_cv)):
                emb = face.embedding
                if emb is None or face.det_score < 0.5:
                    continue
                bbox = face.bbox  # [x1, y1, x2, y2]
                face_w = bbox[2] - bbox[0]
                face_h = bbox[3] - bbox[1]
                ratio = (face_w * face_h) / img_area if img_area > 0 else 0
                if ratio < 0.005:
                    continue
                embeddings_data.append((i, emb.tolist(), ratio))
                face_count += 1
                if max_face_ratio is None or ratio > max_face_ratio:
                    max_face_ratio = ratio
    except Exception as e:
        logger.debug(f"InsightFace failed for {file_path}: {e}")

    # Generate thumbnail
    try:
        _generate_thumb(file_path, rel)
    except Exception as e:
        logger.debug(f"Thumb failed for {file_path}: {e}")

    with get_session() as s:
        # Upsert image_meta
        existing = s.execute(
            select(ImageMeta).where(ImageMeta.file_path == rel)
        ).scalar_one_or_none()
        if existing:
            existing.phash = phash_val
            existing.sha256 = sha256_val
            existing.face_count = face_count
            existing.max_face_ratio = max_face_ratio
            existing.file_size = file_size
            existing.file_mtime = mtime
            existing.scanned_at = utcnow()
        else:
            s.add(
                ImageMeta(
                    file_path=rel,
                    phash=phash_val,
                    sha256=sha256_val,
                    face_count=face_count,
                    max_face_ratio=max_face_ratio,
                    file_size=file_size,
                    file_mtime=mtime,
                )
            )

        # Remove old embeddings for this file, re-insert
        s.execute(delete(FaceEmbedding).where(FaceEmbedding.file_path == rel))
        for face_idx, emb, _ in embeddings_data:
            s.add(
                FaceEmbedding(
                    file_path=rel,
                    face_index=face_idx,
                    embedding=_embedding_to_bytes(emb),
                    model_name=face_model,
                )
            )
        s.commit()

    return face_count


def _generate_video_thumb(file_path: str, rel_path: str) -> None:
    import shutil
    import subprocess

    out_path = _thumb_path(rel_path)
    os.makedirs(_thumbs_dir(), exist_ok=True)

    if shutil.which("ffmpeg"):
        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    "00:00:01",
                    "-i",
                    file_path,
                    "-vframes",
                    "1",
                    "-vf",
                    "scale=256:256:force_original_aspect_ratio=decrease",
                    "-loglevel",
                    "error",
                    out_path,
                ],
                timeout=30,
                capture_output=True,
            )
            if (
                result.returncode == 0
                and os.path.isfile(out_path)
                and os.path.getsize(out_path) > 0
            ):
                return
        except Exception:
            pass

    import cv2

    cv2.setLogLevel(0)
    cap = cv2.VideoCapture(file_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        if not ret:
            return
        frame_small = cv2.resize(frame, (256, 256))
        cv2.imwrite(out_path, frame_small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    finally:
        cap.release()


def _scan_video_file(file_path: str, rel: str, mtime: float) -> None:
    file_size = os.path.getsize(file_path)
    sha256_val = _sha256_file(file_path, limit_bytes=10 * 1024 * 1024)

    with get_session() as s:
        existing = s.execute(
            select(ImageMeta).where(ImageMeta.file_path == rel)
        ).scalar_one_or_none()
        if existing:
            existing.sha256 = sha256_val
            existing.file_size = file_size
            existing.file_mtime = mtime
            existing.scanned_at = utcnow()
        else:
            s.add(
                ImageMeta(
                    file_path=rel,
                    sha256=sha256_val,
                    file_size=file_size,
                    file_mtime=mtime,
                )
            )
        s.commit()

    try:
        _generate_video_thumb(file_path, rel)
    except Exception as e:
        logger.debug(f"Video thumb failed for {file_path}: {e}")


def _cleanup_stale(base_path: str) -> None:
    with get_session() as s:
        all_meta = s.execute(select(ImageMeta)).scalars().all()
        for row in all_meta:
            abs_path = os.path.join(DOWNLOAD_DIR, row.file_path)
            if not os.path.exists(abs_path):
                thumb = _thumb_path(row.file_path)
                if os.path.exists(thumb):
                    os.remove(thumb)
                s.execute(delete(FaceEmbedding).where(FaceEmbedding.file_path == row.file_path))
                s.delete(row)

        # Remove dismissed pairs where either folder no longer exists
        dismissed = s.execute(select(DismissedMatch)).scalars().all()
        for row in dismissed:
            a_exists = os.path.isdir(os.path.join(DOWNLOAD_DIR, row.folder_a))
            b_exists = os.path.isdir(os.path.join(DOWNLOAD_DIR, row.folder_b))
            if not a_exists or not b_exists:
                s.delete(row)

        s.commit()


def _recompute_centroids(face_model: str) -> None:
    cat_dir = _categorized_dir()
    if not os.path.isdir(cat_dir):
        return

    # One bulk fetch of all categorized embeddings, grouped in Python
    with get_session() as s:
        all_rows = (
            s.execute(select(FaceEmbedding).where(FaceEmbedding.file_path.like("categorized/%")))
            .scalars()
            .all()
        )
        folder_embs: dict[str, list[np.ndarray]] = defaultdict(list)
        for row in all_rows:
            folder_embs[os.path.dirname(row.file_path)].append(_bytes_to_embedding(row.embedding))

        for person in os.listdir(cat_dir):
            if not os.path.isdir(os.path.join(cat_dir, person)):
                continue
            folder_rel = os.path.join("categorized", person)
            embs = folder_embs.get(folder_rel)
            if not embs:
                continue
            centroid = np.mean(embs, axis=0).astype(np.float32)

            existing = s.execute(
                select(FolderCentroid).where(FolderCentroid.folder == folder_rel)
            ).scalar_one_or_none()
            if existing:
                existing.centroid = centroid.tobytes()
                existing.model_name = face_model
                existing.updated_at = utcnow()
            else:
                s.add(
                    FolderCentroid(
                        folder=folder_rel,
                        centroid=centroid.tobytes(),
                        model_name=face_model,
                    )
                )
        s.commit()


def score_against_people(folders: list[str]) -> dict[str, float]:
    """Return {person_name: cosine_sim} for each categorized person, sorted descending."""
    from sqlalchemy import or_

    if not folders:
        return {}
    patterns = [or_(*[FaceEmbedding.file_path.like(_like_prefix(f), escape="\\") for f in folders])]
    with get_session() as s:
        rows = s.execute(select(FaceEmbedding).where(*patterns)).scalars().all()
        if not rows:
            return {}
        embs = [_bytes_to_embedding(r.embedding) for r in rows]
        centroid = np.mean(embs, axis=0).astype(np.float32)

        centroids = (
            s.execute(select(FolderCentroid).where(FolderCentroid.folder.like("categorized/%")))
            .scalars()
            .all()
        )

    scores: dict[str, float] = {}
    for c in centroids:
        person = os.path.basename(c.folder)
        scores[person] = _cosine_sim(centroid, _bytes_to_embedding(c.centroid))

    return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))


# ── analyse_downloads ─────────────────────────────────────────────────────────


def analyse_downloads(path: str) -> dict:
    """Return suggestions and ambiguous items for the Suggestions tab."""
    face_threshold = float(get_setting("face_threshold", "0.4"))
    min_images = int(get_setting("min_images_per_folder", "3"))

    suggestions = []
    claimed_files: set[str] = set()

    # Phase 1 — prefix grouping
    series_suggestions, series_claimed = _phase1_prefix_groups(path, face_threshold)
    suggestions.extend(series_suggestions)
    claimed_files.update(series_claimed)

    # Phase 2 — face clustering on remaining files
    face_suggestions, ambiguous = _phase2_face_clustering(
        path, claimed_files, face_threshold, min_images
    )
    suggestions.extend(face_suggestions)

    # Sort by confidence descending (None sorts last; 0.0 must rank above None)
    suggestions.sort(
        key=lambda s: s.get("confidence") if s.get("confidence") is not None else -1,
        reverse=True,
    )

    # Pre-compute similarity scores here (in the executor thread) so the UI render
    # loop doesn't have to run DB queries on the event loop for every card.
    for sugg in suggestions:
        sugg["scores"] = score_against_people(sugg.get("folders", []))

    # Persist to DB so the UI can restore them after a reconnect without re-scanning.
    set_scan_meta("last_suggestions", json.dumps(suggestions))
    set_scan_meta("last_ambiguous", json.dumps(ambiguous))

    return {"suggestions": suggestions, "ambiguous": ambiguous}


def _phase1_prefix_groups(path: str, face_threshold: float) -> tuple[list[dict], set[str]]:
    """Group files by common filename prefix within the same directory."""
    # Walk non-categorized, non-collections tree
    cat_dir = _categorized_dir()
    coll_dir = _collections_dir()
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)

    for root, dirs, files in os.walk(path):
        dirs[:] = [
            d
            for d in dirs
            if not d.startswith(".")
            and os.path.abspath(os.path.join(root, d)) != os.path.abspath(cat_dir)
            and os.path.abspath(os.path.join(root, d)) != os.path.abspath(coll_dir)
        ]
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            ext = os.path.splitext(fname)[1].lower()
            if ext not in _MEDIA_EXTS:
                continue
            stem = os.path.splitext(fname)[0]
            # Strip trailing digits/spaces to find prefix
            prefix = stem.rstrip("0123456789").rstrip(" _-")
            if prefix:
                groups[(root, prefix)].append(fpath)

    suggestions = []
    claimed: set[str] = set()

    for (parent_dir, prefix), files in groups.items():
        if len(files) < 2:
            continue

        rel_files = [_rel(f) for f in files]
        image_files = [f for f in files if os.path.splitext(f)[1].lower() in _IMAGE_EXTS]

        # Compute confidence from face similarity within group
        confidence = None
        embeddings_by_file: dict[str, list[np.ndarray]] = defaultdict(list)
        if image_files:
            rels = [_rel(img_f) for img_f in image_files]
            with get_session() as s:
                fe_rows = (
                    s.execute(select(FaceEmbedding).where(FaceEmbedding.file_path.in_(rels)))
                    .scalars()
                    .all()
                )
            for row in fe_rows:
                embeddings_by_file[row.file_path].append(_bytes_to_embedding(row.embedding))

        if len(embeddings_by_file) >= 2:
            sims = []
            emb_lists = list(embeddings_by_file.values())
            for i in range(len(emb_lists)):
                for j in range(i + 1, len(emb_lists)):
                    for e1 in emb_lists[i]:
                        for e2 in emb_lists[j]:
                            sims.append(_cosine_sim(e1, e2))
            confidence = float(np.mean(sims)) if sims else None

        sample_thumbs = _pick_sample_thumbs(image_files, n=24)

        folder_rel = os.path.relpath(parent_dir, DOWNLOAD_DIR)
        suggestions.append(
            {
                "type": "series",
                "folders": [folder_rel],
                "files": rel_files,
                "dest_name": prefix.strip(),
                "sample_images": sample_thumbs,
                "confidence": confidence,
                "prefix": prefix,
            }
        )
        claimed.update(rel_files)

    return suggestions, claimed


def _phase2_face_clustering(
    path: str,
    claimed_files: set[str],
    face_threshold: float,
    min_images: int,
) -> tuple[list[dict], list[dict]]:
    """FAISS + union-find clustering on unclaimed files; classifies against categorized/."""
    cat_dir = _categorized_dir()

    # Single pass over all embeddings — split into source and categorized buckets
    folder_embeddings: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
    cat_folder_embeddings: dict[str, list[np.ndarray]] = defaultdict(list)
    path_abs = os.path.abspath(path)

    with get_session() as s:
        all_emb_rows = s.execute(select(FaceEmbedding)).scalars().all()
        dismissed_rows = s.execute(select(DismissedMatch)).scalars().all()
        dismissed_pairs: set[tuple[str, str]] = {(r.folder_a, r.folder_b) for r in dismissed_rows}

    for row in all_emb_rows:
        if row.file_path.startswith("collections/"):
            continue  # collections are excluded from face analysis
        if row.file_path.startswith("categorized/"):
            folder = os.path.dirname(row.file_path)
            cat_folder_embeddings[folder].append(_bytes_to_embedding(row.embedding))
        else:
            if row.file_path in claimed_files:
                continue
            abs_path_file = os.path.abspath(os.path.join(DOWNLOAD_DIR, row.file_path))
            if not abs_path_file.startswith(path_abs + os.sep):
                continue
            folder = os.path.dirname(row.file_path)
            folder_embeddings[folder].append((row.file_path, _bytes_to_embedding(row.embedding)))

    cat_dir_abs = os.path.abspath(cat_dir)

    # Batch min_images filter — one OR query instead of N per-folder queries
    non_cat_folders = [
        folder
        for folder in folder_embeddings
        if not os.path.abspath(os.path.join(DOWNLOAD_DIR, folder)).startswith(cat_dir_abs + os.sep)
        and os.path.abspath(os.path.join(DOWNLOAD_DIR, folder)) != cat_dir_abs
    ]
    source_folders: dict[str, list] = {}
    if non_cat_folders:
        from sqlalchemy import or_ as _or_

        conditions = [
            ImageMeta.file_path.like(_like_prefix(f), escape="\\") for f in non_cat_folders
        ]
        folder_img_count: dict[str, int] = defaultdict(int)
        with get_session() as s:
            meta_rows = s.execute(select(ImageMeta).where(_or_(*conditions))).scalars().all()
        for m in meta_rows:
            d = os.path.dirname(m.file_path)
            if os.path.splitext(m.file_path)[1].lower() in _IMAGE_EXTS:
                folder_img_count[d] += 1
        source_folders = {
            folder: folder_embeddings[folder]
            for folder in non_cat_folders
            if folder_img_count.get(folder, 0) >= min_images
        }

    if not source_folders and not cat_folder_embeddings:
        return [], []

    # Build embedding matrix
    all_items: list[tuple[str, str, np.ndarray]] = []  # (folder, file_path, embedding)
    for folder, items in source_folders.items():
        for fp, emb in items:
            all_items.append((folder, fp, emb))
    for folder, embs in cat_folder_embeddings.items():
        for emb in embs:
            all_items.append((folder, "", emb))

    if not all_items:
        return [], []

    import faiss

    n = len(all_items)
    X = np.stack([item[2] for item in all_items]).astype(np.float32)
    faiss.normalize_L2(X)  # cosine sim → inner product on unit vectors

    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)

    # Top-k neighbours per point (k capped at n; sorted descending by similarity)
    k = min(100, n)
    D, nn_idx = index.search(X, k)

    # Union-Find: link all pairs with cosine_sim >= face_threshold
    parent = list(range(n))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return x

    for i in range(n):
        for j_pos in range(k):
            if float(D[i, j_pos]) < face_threshold:
                break  # results are sorted descending — no better neighbours follow
            j = int(nn_idx[i, j_pos])
            if j == i:
                continue
            ri, rj = _find(i), _find(j)
            if ri != rj:
                parent[ri] = rj

    # Assign cluster labels; singletons → -1 (mirrors DBSCAN noise)
    root_count: dict[int, int] = {}
    for i in range(n):
        root_count[_find(i)] = root_count.get(_find(i), 0) + 1

    root_to_label: dict[int, int] = {}
    next_label = 0
    for root, count in root_count.items():
        if count >= 2:
            root_to_label[root] = next_label
            next_label += 1

    labels = [root_to_label.get(_find(i), -1) for i in range(n)]

    # Group by cluster
    clusters: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for i, label in enumerate(labels):
        if label == -1:
            continue  # noise point
        folder, fp, _ = all_items[i]
        clusters[label].append((folder, fp))

    suggestions = []
    ambiguous = []

    for cluster_id, members in clusters.items():
        source_folders_in_cluster: set[str] = set()
        cat_folders_in_cluster: set[str] = set()

        for folder, _ in members:
            if folder.startswith("categorized/"):
                cat_folders_in_cluster.add(folder)
            else:
                source_folders_in_cluster.add(folder)

        if not source_folders_in_cluster:
            continue

        # Remove dismissed pairs from source_folders_in_cluster
        filtered_sources = set()
        for src in source_folders_in_cluster:
            dismissed = False
            for other in source_folders_in_cluster | cat_folders_in_cluster:
                if other == src:
                    continue
                pair = _normalize_pair(src, other)
                if pair in dismissed_pairs:
                    dismissed = True
                    break
            if not dismissed:
                filtered_sources.add(src)

        if not filtered_sources:
            continue

        # Get sample files for each source folder
        sample_image_files = []
        for folder in filtered_sources:
            img_files = sorted(
                [
                    os.path.join(DOWNLOAD_DIR, fp)
                    for f, fp in members
                    if f == folder and os.path.splitext(fp)[1].lower() in _IMAGE_EXTS
                ]
            )
            sample_image_files.extend(img_files[:3])

        sample_thumbs = _pick_sample_thumbs(sample_image_files, n=6)

        if len(cat_folders_in_cluster) == 0:
            # Rule 4: new unknown person
            suggestions.append(
                {
                    "type": "face",
                    "folders": sorted(filtered_sources),
                    "dest_name": "",
                    "sample_images": sample_thumbs,
                    "confidence": None,
                    "cat_folders": [],
                }
            )
        elif len(cat_folders_in_cluster) == 1:
            # Rule 2: single match → add_to
            cat_folder = next(iter(cat_folders_in_cluster))
            person_name = os.path.basename(cat_folder)
            suggestions.append(
                {
                    "type": "add_to",
                    "folders": sorted(filtered_sources),
                    "dest_name": person_name,
                    "sample_images": sample_thumbs,
                    "confidence": None,
                    "cat_folders": [cat_folder],
                }
            )
        else:
            # Rule 3: multiple matches → ambiguous
            ambiguous.append(
                {
                    "folders": sorted(filtered_sources),
                    "matched_people": sorted(cat_folders_in_cluster),
                    "sample_images": sample_thumbs,
                }
            )

    return suggestions, ambiguous


def _pick_sample_thumbs(image_files: list[str], n: int = 6) -> list[str]:
    """Return up to n thumbnail URLs, preferring high max_face_ratio images."""
    if not image_files:
        return []

    rel_files = []
    for f in image_files:
        if os.path.isabs(f):
            rel_files.append(_rel(f))
        else:
            rel_files.append(f)

    with get_session() as s:
        meta_rows = (
            s.execute(select(ImageMeta).where(ImageMeta.file_path.in_(rel_files))).scalars().all()
        )
    ratio_map = {r.file_path: (r.max_face_ratio or 0) for r in meta_rows}
    ranked = [(ratio_map.get(rel, 0), rel) for rel in rel_files]

    ranked.sort(reverse=True)
    selected = [r for _, r in ranked[:n]]
    return [f"/thumbs/{_thumb_key(r)}.jpg" for r in selected]


# ── analyse_merge ─────────────────────────────────────────────────────────────


def analyse_merge(sources: list[str], dest_name: str) -> dict:
    """Return conflict report dict — UI owns all dialogs."""
    phash_threshold = int(get_setting("phash_threshold", "8"))
    cat_dir = _categorized_dir()
    dest_path = os.path.join(cat_dir, dest_name)
    dest_exists = os.path.isdir(dest_path)

    dest_sample_thumb = None
    if dest_exists:
        dest_files = _all_image_files(dest_path)
        if dest_files:
            thumbs = _pick_sample_thumbs(dest_files[:1], n=1)
            dest_sample_thumb = thumbs[0] if thumbs else None

    # Collect all source files
    source_files: list[str] = []
    for src in sources:
        src_abs = os.path.join(DOWNLOAD_DIR, src) if not os.path.isabs(src) else src
        for root, _, files in os.walk(src_abs):
            for f in files:
                if not f.startswith("."):
                    source_files.append(os.path.join(root, f))

    # Collect dest files for comparison — single batch query
    dest_files_map: dict[str, str] = {}  # sha256 → path
    dest_phash_map: dict[str, str] = {}  # phash → path
    dest_name_map: dict[str, str] = {}  # filename → path

    if dest_exists:
        dest_abs_files = []
        for root, _, files in os.walk(dest_path):
            for f in files:
                dest_abs_files.append(os.path.join(root, f))
        dest_rels = [_rel(p) for p in dest_abs_files]
        with get_session() as s:
            dest_meta_rows = (
                s.execute(select(ImageMeta).where(ImageMeta.file_path.in_(dest_rels)))
                .scalars()
                .all()
            )
        dest_meta_map = {r.file_path: r for r in dest_meta_rows}
        for fpath in dest_abs_files:
            rel = _rel(fpath)
            meta = dest_meta_map.get(rel)
            if meta and meta.sha256:
                dest_files_map[meta.sha256] = fpath
            if meta and meta.phash:
                dest_phash_map[meta.phash] = fpath
            dest_name_map[os.path.basename(fpath)] = fpath

    # Batch-load source meta
    src_rels = [_rel(f) for f in source_files]
    with get_session() as s:
        src_meta_rows = (
            s.execute(select(ImageMeta).where(ImageMeta.file_path.in_(src_rels))).scalars().all()
        )
    src_meta_map = {r.file_path: r for r in src_meta_rows}

    exact_dupes = []
    phash_dupes = []
    collisions = []
    clean = []

    for src_file in source_files:
        rel = _rel(src_file)
        fname = os.path.basename(src_file)
        meta = src_meta_map.get(rel)

        src_sha256 = meta.sha256 if meta else None
        src_phash = meta.phash if meta else None

        if src_sha256 and src_sha256 in dest_files_map:
            exact_dupes.append(src_file)
        elif src_phash and (
            match_path := _find_phash_match(src_phash, dest_phash_map, phash_threshold)
        ):
            src_size = os.path.getsize(src_file)
            dest_size = os.path.getsize(match_path)
            keep = src_file if src_size > dest_size else match_path
            phash_dupes.append({"source": src_file, "dest": match_path, "keep": keep})
        elif fname in dest_name_map:
            collisions.append(src_file)
        else:
            clean.append(src_file)

    return {
        "dest_exists": dest_exists,
        "dest_sample_thumb": dest_sample_thumb,
        "exact_dupes": exact_dupes,
        "phash_dupes": phash_dupes,
        "collisions": collisions,
        "clean": clean,
    }


def _find_phash_match(phash: str, dest_phash_map: dict[str, str], threshold: int) -> str | None:
    import imagehash

    h1 = imagehash.hex_to_hash(phash)
    for dest_phash, dest_path in dest_phash_map.items():
        try:
            h2 = imagehash.hex_to_hash(dest_phash)
            if h1 - h2 <= threshold:
                return dest_path
        except Exception:
            continue
    return None


# ── merge_into ────────────────────────────────────────────────────────────────


def _purge_src_file(rel: str, abs_path: str) -> None:
    """Delete a redundant source file and remove its DB rows + thumbnail.

    Used when dest already has an identical or better copy (exact SHA256 match,
    or phash match where dest is the larger file).  Deleting the source allows
    _remove_empty_dirs to clean up the now-empty source folder so the suggestion
    card disappears after merge.
    """
    with get_session() as s:
        s.execute(delete(FaceEmbedding).where(FaceEmbedding.file_path == rel))
        s.execute(delete(ImageMeta).where(ImageMeta.file_path == rel))
        s.commit()
    thumb = _thumb_path(rel)
    if os.path.exists(thumb):
        os.remove(thumb)
    if os.path.exists(abs_path):
        os.remove(abs_path)


def merge_into(
    sources: list[str],
    dest_name: str,
    on_progress: Callable[[int, int], None] | None = None,
) -> None:
    """Move files from sources into categorized/<dest_name>/."""
    face_model = get_setting("face_model", "buffalo_s")
    cat_dir = _categorized_dir()
    dest_path = os.path.join(cat_dir, dest_name)

    # Write merge log before touching files
    with get_session() as s:
        log_row = MergeLog(sources=json.dumps(sources), dest=dest_name)
        s.add(log_row)
        s.commit()
        log_id = log_row.id

    os.makedirs(dest_path, exist_ok=True)

    # Collect all source files
    all_files: list[str] = []
    for src in sources:
        src_abs = os.path.join(DOWNLOAD_DIR, src) if not os.path.isabs(src) else src
        for root, _, files in os.walk(src_abs):
            for f in sorted(files):
                if not f.startswith("."):
                    all_files.append(os.path.join(root, f))

    # Re-run conflict detection — batch queries
    phash_threshold = int(get_setting("phash_threshold", "8"))
    dest_files_map: dict[str, str] = {}
    dest_phash_map: dict[str, str] = {}

    dest_abs_files = []
    for root, _, files in os.walk(dest_path):
        for f in files:
            dest_abs_files.append(os.path.join(root, f))
    dest_rels = [_rel(p) for p in dest_abs_files]
    with get_session() as s:
        dest_meta_rows = (
            s.execute(select(ImageMeta).where(ImageMeta.file_path.in_(dest_rels))).scalars().all()
        )
    for row in dest_meta_rows:
        abs_p = os.path.join(DOWNLOAD_DIR, row.file_path)
        if row.sha256:
            dest_files_map[row.sha256] = abs_p
        if row.phash:
            dest_phash_map[row.phash] = abs_p

    # Batch-load source meta
    all_rels = [_rel(f) for f in all_files]
    with get_session() as s:
        src_meta_rows = (
            s.execute(select(ImageMeta).where(ImageMeta.file_path.in_(all_rels))).scalars().all()
        )
    src_meta_map = {r.file_path: r for r in src_meta_rows}

    total = len(all_files)
    moved = 0
    file_map: dict[str, str | None] = {}

    for src_file in all_files:
        rel = _rel(src_file)
        fname = os.path.basename(src_file)
        meta = src_meta_map.get(rel)

        src_sha256 = meta.sha256 if meta else None
        src_phash = meta.phash if meta else None

        # Exact duplicate → delete source (dest already has identical copy)
        if src_sha256 and src_sha256 in dest_files_map:
            file_map[rel] = None
            _purge_src_file(rel, src_file)
            moved += 1
            if on_progress:
                on_progress(moved, total)
            continue

        # Perceptual duplicate → keep larger
        phash_match = (
            _find_phash_match(src_phash, dest_phash_map, phash_threshold) if src_phash else None
        )
        if phash_match:
            src_size = os.path.getsize(src_file)
            dest_size = os.path.getsize(phash_match)
            if src_size <= dest_size:
                # Dest already has an equal-or-better copy — delete inferior source
                file_map[rel] = None
                _purge_src_file(rel, src_file)
                moved += 1
                if on_progress:
                    on_progress(moved, total)
                continue
            # src is larger — replace dest file
            os.remove(phash_match)
            dest_fname = os.path.basename(phash_match)
            # Evict deleted entry so later files don't try to stat/remove it again
            dest_phash_map = {k: v for k, v in dest_phash_map.items() if v != phash_match}
            dest_files_map = {k: v for k, v in dest_files_map.items() if v != phash_match}
            # Delete DB rows for removed dest file so src rename won't hit UNIQUE conflict
            removed_rel = _rel(phash_match)
            removed_thumb = _thumb_path(removed_rel)
            with get_session() as s:
                s.execute(delete(FaceEmbedding).where(FaceEmbedding.file_path == removed_rel))
                s.execute(delete(ImageMeta).where(ImageMeta.file_path == removed_rel))
                s.commit()
            if os.path.exists(removed_thumb):
                os.remove(removed_thumb)
        else:
            dest_fname = fname

        # Handle filename collision
        dest_file = os.path.join(dest_path, dest_fname)
        if os.path.exists(dest_file) and os.path.abspath(dest_file) != os.path.abspath(src_file):
            stem, ext = os.path.splitext(dest_fname)
            counter = 2
            while os.path.exists(os.path.join(dest_path, f"{stem}_{counter}{ext}")):
                counter += 1
            dest_fname = f"{stem}_{counter}{ext}"
            dest_file = os.path.join(dest_path, dest_fname)

        import shutil

        shutil.move(src_file, dest_file)
        new_rel = _rel(dest_file)
        file_map[rel] = new_rel

        # Update DB paths
        with get_session() as s:
            if meta:
                meta_row = s.execute(
                    select(ImageMeta).where(ImageMeta.file_path == rel)
                ).scalar_one_or_none()
                if meta_row:
                    meta_row.file_path = new_rel
            fe_rows = (
                s.execute(select(FaceEmbedding).where(FaceEmbedding.file_path == rel))
                .scalars()
                .all()
            )
            for fe in fe_rows:
                fe.file_path = new_rel
            # Update thumb
            old_thumb = _thumb_path(rel)
            if os.path.exists(old_thumb):
                new_thumb = _thumb_path(new_rel)
                os.rename(old_thumb, new_thumb)
            s.commit()

        moved += 1
        if on_progress:
            on_progress(moved, total)

    # Remove dismissed_matches rows referencing source folders
    with get_session() as s:
        for src in sources:
            s.execute(
                delete(DismissedMatch).where(
                    (DismissedMatch.folder_a == src) | (DismissedMatch.folder_b == src)
                )
            )
        s.commit()

    # Remove empty source directories
    for src in sources:
        src_abs = os.path.join(DOWNLOAD_DIR, src) if not os.path.isabs(src) else src
        _remove_empty_dirs(src_abs)

    # Delete source folder_centroids
    with get_session() as s:
        for src in sources:
            s.execute(delete(FolderCentroid).where(FolderCentroid.folder == src))
        s.commit()

    # Recompute dest centroid
    dest_rel = f"categorized/{dest_name}"
    with get_session() as s:
        rows = (
            s.execute(
                select(FaceEmbedding).where(
                    FaceEmbedding.file_path.like(_like_prefix(dest_rel), escape="\\")
                )
            )
            .scalars()
            .all()
        )
        if rows:
            embeddings = [_bytes_to_embedding(r.embedding) for r in rows]
            centroid = np.mean(embeddings, axis=0).astype(np.float32)
            existing = s.execute(
                select(FolderCentroid).where(FolderCentroid.folder == dest_rel)
            ).scalar_one_or_none()
            if existing:
                existing.centroid = centroid.tobytes()
                existing.updated_at = utcnow()
            else:
                s.add(
                    FolderCentroid(
                        folder=dest_rel, centroid=centroid.tobytes(), model_name=face_model
                    )
                )
        s.commit()

    # Clear last_scan
    set_scan_meta("last_scan", None)

    # Mark merge log done; save file_map in same commit (atomic)
    with get_session() as s:
        log = s.get(MergeLog, log_id)
        if log:
            log.status = "done"
            log.completed_at = utcnow()
            log.file_map = json.dumps(file_map)
        s.commit()

    regenerate_person_gif(dest_name)


def _remove_empty_dirs(path: str) -> None:
    download_abs = os.path.abspath(DOWNLOAD_DIR)
    abs_path = os.path.abspath(path)
    if abs_path == download_abs or not abs_path.startswith(download_abs + os.sep):
        return
    for root, _dirs, _files in os.walk(path, topdown=False):
        try:
            visible = [f for f in os.listdir(root) if not f.startswith(".")]
            if not visible:
                os.rmdir(root)
        except OSError:
            pass


# ── undo_merge ────────────────────────────────────────────────────────────────


def undo_merge(log_id: int) -> str | None:
    """Reverse a completed merge. Returns error string or None on success."""
    import shutil

    if get_scan_meta("status") in ("running", "cancelling"):
        return "Cannot undo while a scan is in progress"

    with get_session() as s:
        log = s.get(MergeLog, log_id)
        if log is None or log.status != "done":
            return "Merge not found or already undone"
        if not log.file_map:
            return "This merge was recorded before undo support was added"
        dest = log.dest
        file_map = json.loads(log.file_map)

    # Pre-flight: check BOTH directions before moving anything
    missing, occupied = [], []
    for old_rel, new_rel in file_map.items():
        if new_rel is None:
            continue
        if not os.path.exists(os.path.abspath(os.path.join(DOWNLOAD_DIR, new_rel))):
            missing.append(new_rel)
        if os.path.exists(os.path.abspath(os.path.join(DOWNLOAD_DIR, old_rel))):
            occupied.append(old_rel)
    if missing:
        return f"Cannot undo: {len(missing)} file(s) no longer at destination"
    if occupied:
        return f"Cannot undo: {len(occupied)} original path(s) are occupied by newer files"

    # Move files back
    for old_rel, new_rel in file_map.items():
        if new_rel is None:
            continue
        new_abs = os.path.abspath(os.path.join(DOWNLOAD_DIR, new_rel))
        old_abs = os.path.abspath(os.path.join(DOWNLOAD_DIR, old_rel))
        os.makedirs(os.path.dirname(old_abs), exist_ok=True)
        shutil.move(new_abs, old_abs)

        with get_session() as s:
            meta = s.execute(
                select(ImageMeta).where(ImageMeta.file_path == new_rel)
            ).scalar_one_or_none()
            if meta:
                meta.file_path = old_rel
            for fe in (
                s.execute(select(FaceEmbedding).where(FaceEmbedding.file_path == new_rel))
                .scalars()
                .all()
            ):
                fe.file_path = old_rel
            s.commit()

        old_thumb = _thumb_path(new_rel)
        if os.path.exists(old_thumb):
            os.rename(old_thumb, _thumb_path(old_rel))

    # Clean up dest folder if now empty
    dest_abs = os.path.abspath(os.path.join(_categorized_dir(), dest))
    _remove_empty_dirs(dest_abs)

    _invalidate_gif(f"categorized/{dest}")
    regenerate_person_gif(dest)

    # Recompute centroids — covers both dest and restored source folders
    _recompute_centroids(get_setting("face_model", "buffalo_s"))

    # Mark undone + clear stale suggestion cache
    with get_session() as s:
        log = s.get(MergeLog, log_id)
        if log:
            log.status = "undone"
            s.commit()
    set_scan_meta("last_scan", None)
    set_scan_meta("last_suggestions", None)
    set_scan_meta("last_ambiguous", None)

    return None


# ── rename_person ─────────────────────────────────────────────────────────────


def rename_person(old_name: str, new_name: str) -> str | None:
    """Rename categorized/<old_name> → categorized/<new_name>. Returns error string or None."""
    new_name = new_name.strip()
    if not new_name:
        return "Name cannot be empty"

    if new_name.startswith("."):
        return "Name cannot start with a dot"

    illegal = set('/\\:*?"<>|')
    if any(c in illegal for c in new_name):
        return f"Name contains illegal characters: {illegal & set(new_name)}"

    cat_dir = _categorized_dir()
    old_path = os.path.join(cat_dir, old_name)
    new_path = os.path.join(cat_dir, new_name)

    if not os.path.isdir(old_path):
        return f"Folder '{old_name}' does not exist"
    if os.path.isdir(new_path):
        return f"Folder '{new_name}' already exists"

    old_rel = f"categorized/{old_name}"
    new_rel = f"categorized/{new_name}"

    os.rename(old_path, new_path)

    with get_session() as s:
        for row in (
            s.execute(
                select(ImageMeta).where(
                    ImageMeta.file_path.like(_like_prefix(old_rel), escape="\\")
                )
            )
            .scalars()
            .all()
        ):
            row.file_path = new_rel + row.file_path[len(old_rel) :]
        for row in (
            s.execute(
                select(FaceEmbedding).where(
                    FaceEmbedding.file_path.like(_like_prefix(old_rel), escape="\\")
                )
            )
            .scalars()
            .all()
        ):
            row.file_path = new_rel + row.file_path[len(old_rel) :]
        fc = s.execute(
            select(FolderCentroid).where(FolderCentroid.folder == old_rel)
        ).scalar_one_or_none()
        if fc:
            fc.folder = new_rel
        for n in (
            s.execute(select(Notification).where(Notification.person_name == old_name))
            .scalars()
            .all()
        ):
            n.person_name = new_name
        for ml in (
            s.execute(
                select(MergeLog).where(MergeLog.dest == old_name, MergeLog.status == "pending")
            )
            .scalars()
            .all()
        ):
            ml.dest = new_name
        all_dm = s.execute(select(DismissedMatch)).scalars().all()
        existing_pairs = {(r.folder_a, r.folder_b) for r in all_dm}
        to_delete = []
        to_add = []
        for dm in all_dm:
            a, b = dm.folder_a, dm.folder_b
            if a != old_rel and b != old_rel:
                continue
            new_a = new_rel if a == old_rel else a
            new_b = new_rel if b == old_rel else b
            fa, fb = _normalize_pair(new_a, new_b)
            to_delete.append(dm)
            to_add.append((fa, fb))
        for dm in to_delete:
            existing_pairs.discard((dm.folder_a, dm.folder_b))
            s.delete(dm)
        for fa, fb in to_add:
            if (fa, fb) not in existing_pairs:
                s.add(DismissedMatch(folder_a=fa, folder_b=fb))
                existing_pairs.add((fa, fb))
        s.commit()

    # Rename thumbnails: SHA256 of rel changes when the person name changes
    new_path = os.path.join(_categorized_dir(), new_name)
    old_prefix = f"categorized/{old_name}/"
    new_prefix = f"categorized/{new_name}/"
    for fname in list(os.listdir(new_path)):
        old_thumb = _thumb_path(f"{old_prefix}{fname}")
        new_thumb = _thumb_path(f"{new_prefix}{fname}")
        if os.path.exists(old_thumb):
            try:
                os.rename(old_thumb, new_thumb)
            except OSError:
                pass  # non-fatal; thumb regenerated on next scan

    _invalidate_gif(f"categorized/{old_name}")
    regenerate_person_gif(new_name)
    rename_entity_tags("person", old_name, new_name)

    return None


# ── resume_incomplete_merges ──────────────────────────────────────────────────


def resume_incomplete_merges() -> None:
    # Load pending merges then close session before calling merge_into,
    # which opens its own write sessions (avoids SQLite write-lock contention).
    with get_session() as s:
        pending = [
            (r.id, json.loads(r.sources), r.dest)
            for r in s.execute(select(MergeLog).where(MergeLog.status == "pending")).scalars().all()
        ]
    for log_id, sources, dest in pending:
        logger.info(f"Resuming incomplete merge: {sources} → {dest}")
        try:
            merge_into(sources, dest)
            with get_session() as s:
                log = s.get(MergeLog, log_id)
                if log and log.status == "pending":
                    log.status = "done"
                    log.completed_at = utcnow()
                    s.commit()
        except Exception as e:
            logger.error(f"Failed to resume merge: {e}")


# ── scan_new_download ─────────────────────────────────────────────────────────


def scan_new_download(folder: str) -> None:
    """Mini-scan triggered after a download completes."""
    cat_dir = os.path.abspath(_categorized_dir())
    folder_abs = os.path.abspath(os.path.join(DOWNLOAD_DIR, folder))

    if folder_abs == cat_dir or folder_abs.startswith(cat_dir + os.sep):
        return  # already categorized

    scan_status = get_scan_meta("status", "idle")
    if scan_status in ("running", "cancelling"):
        return  # full scan in progress

    face_model = get_setting("face_model", "buffalo_s")
    mini_scan_enabled = get_setting("mini_scan_enabled", "true") == "true"

    # Always scan files for embedding/hash/thumb
    files = []
    for root, dirs, fnames in os.walk(folder_abs):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in fnames:
            ext = os.path.splitext(f)[1].lower()
            if ext in _MEDIA_EXTS:
                files.append(os.path.join(root, f))

    for fpath in files:
        try:
            _scan_file(fpath, face_model)
        except Exception as e:
            logger.warning(f"Mini-scan skipped {fpath}: {e}")

    if not mini_scan_enabled:
        return

    # Compare against folder_centroids
    folder_rel = _rel(folder_abs)
    with get_session() as s:
        emb_rows = (
            s.execute(
                select(FaceEmbedding).where(
                    FaceEmbedding.file_path.like(_like_prefix(folder_rel), escape="\\")
                )
            )
            .scalars()
            .all()
        )
        if not emb_rows:
            return

        folder_embs = [_bytes_to_embedding(r.embedding) for r in emb_rows]
        folder_centroid = np.mean(folder_embs, axis=0).astype(np.float32)

        centroids = s.execute(select(FolderCentroid)).scalars().all()
        face_threshold = float(get_setting("face_threshold", "0.4"))
        best_match = None
        best_sim = -1.0

        for c in centroids:
            cat_centroid = _bytes_to_embedding(c.centroid)
            sim = _cosine_sim(folder_centroid, cat_centroid)
            if sim >= face_threshold and sim > best_sim:
                best_sim = sim
                best_match = c

        if best_match:
            person_name = os.path.basename(best_match.folder)
            existing = s.execute(
                select(Notification).where(
                    Notification.person_name == person_name,
                    Notification.new_folder == folder_rel,
                )
            ).scalar_one_or_none()
            if not existing:
                s.add(Notification(person_name=person_name, new_folder=folder_rel))
            s.commit()


# ── dismiss helpers ───────────────────────────────────────────────────────────


def dismiss_suggestion(folders: list[str]) -> None:
    """Store all pairs from folders as dismissed_matches rows.

    Single-folder suggestions (e.g. series) use a self-pair (f, f) as sentinel.
    """
    if not folders:
        return
    with get_session() as s:
        if len(folders) == 1:
            f = folders[0]
            existing = s.execute(
                select(DismissedMatch).where(
                    DismissedMatch.folder_a == f, DismissedMatch.folder_b == f
                )
            ).scalar_one_or_none()
            if not existing:
                s.add(DismissedMatch(folder_a=f, folder_b=f))
        else:
            for i in range(len(folders)):
                for j in range(i + 1, len(folders)):
                    a, b = _normalize_pair(folders[i], folders[j])
                    existing = s.execute(
                        select(DismissedMatch).where(
                            DismissedMatch.folder_a == a,
                            DismissedMatch.folder_b == b,
                        )
                    ).scalar_one_or_none()
                    if not existing:
                        s.add(DismissedMatch(folder_a=a, folder_b=b))
        s.commit()


def undismiss_suggestion(folders: list[str]) -> None:
    """Remove all pair rows for this group from dismissed_matches."""
    if not folders:
        return
    with get_session() as s:
        if len(folders) == 1:
            f = folders[0]
            s.execute(
                delete(DismissedMatch).where(
                    DismissedMatch.folder_a == f, DismissedMatch.folder_b == f
                )
            )
        else:
            for i in range(len(folders)):
                for j in range(i + 1, len(folders)):
                    a, b = _normalize_pair(folders[i], folders[j])
                    s.execute(
                        delete(DismissedMatch).where(
                            DismissedMatch.folder_a == a,
                            DismissedMatch.folder_b == b,
                        )
                    )
        s.commit()


def next_unknown_name() -> str:
    """Atomically increment unknown_counter and return 'unknown-N'."""
    with get_session() as s:
        row = s.get(Setting, "unknown_counter")
        current = int(row.value) if row and row.value else 0
        next_val = current + 1
        if row:
            row.value = str(next_val)
        else:
            s.add(Setting(key="unknown_counter", value=str(next_val)))
        s.commit()
    return f"unknown-{next_val}"


# ── Collections ───────────────────────────────────────────────────────────────


def list_collections() -> list[str]:
    """Return sorted list of collection names (dirs under collections/)."""
    d = _collections_dir()
    if not os.path.isdir(d):
        return []
    return sorted(
        (n for n in os.listdir(d) if os.path.isdir(os.path.join(d, n)) and not n.startswith(".")),
        key=str.casefold,
    )


def _collection_images(collection_name: str) -> list[dict]:
    """Return all media files in a collection as lightGallery-ready dicts."""
    coll_abs = os.path.abspath(_collections_dir())
    folder_abs = os.path.abspath(os.path.join(coll_abs, collection_name))
    if not folder_abs.startswith(coll_abs + os.sep):
        return []
    if not os.path.isdir(folder_abs):
        return []
    result = []
    for fname in sorted(os.listdir(folder_abs)):
        abs_path = os.path.join(folder_abs, fname)
        if not os.path.isfile(abs_path):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext not in _MEDIA_EXTS:
            continue
        rel = f"collections/{collection_name}/{fname}"
        item: dict = {
            "rel": rel,
            "src": f"/files/{rel}",
            "thumb": f"/thumbs/{hashlib.sha256(rel.encode()).hexdigest()}.jpg",
            "filename": fname,
        }
        if ext not in _IMAGE_EXTS:
            item["is_video"] = True
            item["video_type"] = _VIDEO_MIME.get(ext, "video/mp4")
        result.append(item)
    return result


def rename_collection(old_name: str, new_name: str) -> str | None:
    """Rename a collection folder and update its thumbnails. Returns error or None."""
    new_name = new_name.strip()
    if not new_name:
        return "Name cannot be empty"
    if new_name.startswith("."):
        return "Name cannot start with a dot"
    illegal = set('/\\:*?"<>|')
    if illegal & set(new_name):
        return f"Illegal characters: {''.join(sorted(illegal & set(new_name)))}"

    coll_abs = os.path.abspath(_collections_dir())
    old_abs = os.path.abspath(os.path.join(coll_abs, old_name))
    new_abs = os.path.abspath(os.path.join(coll_abs, new_name))

    if not old_abs.startswith(coll_abs + os.sep):
        return "Invalid collection name"
    if not new_abs.startswith(coll_abs + os.sep):
        return "Invalid new name"
    if not os.path.isdir(old_abs):
        return f"Collection '{old_name}' not found"
    if os.path.exists(new_abs):
        return f"'{new_name}' already exists"

    os.rename(old_abs, new_abs)

    # Rename thumbnails: SHA256 of rel changes when the collection name changes
    old_prefix = f"collections/{old_name}/"
    new_prefix = f"collections/{new_name}/"
    for fname in list(os.listdir(new_abs)):
        old_rel = f"{old_prefix}{fname}"
        new_rel = f"{new_prefix}{fname}"
        old_thumb = _thumb_path(old_rel)
        new_thumb = _thumb_path(new_rel)
        if os.path.exists(old_thumb):
            try:
                os.rename(old_thumb, new_thumb)
            except OSError:
                pass  # non-fatal; thumb stays missing until next move

    _invalidate_gif(f"collections/{old_name}")
    regenerate_collection_gif(new_name)
    rename_entity_tags("collection", old_name, new_name)

    return None


def delete_collection_file(collection_name: str, rel_path: str) -> str | None:
    """Delete a file from a collection and its thumbnail. Returns error or None."""
    coll_abs = os.path.abspath(_collections_dir())
    coll_name_abs = os.path.abspath(os.path.join(coll_abs, collection_name))
    if not coll_name_abs.startswith(coll_abs + os.sep):
        return "Invalid collection name"

    abs_path = os.path.abspath(os.path.join(DOWNLOAD_DIR, rel_path))
    if not abs_path.startswith(coll_name_abs + os.sep):
        return "Invalid file path"

    try:
        os.remove(abs_path)
    except FileNotFoundError:
        pass
    except OSError as e:
        return f"Delete failed: {e}"

    thumb = _thumb_path(rel_path)
    if os.path.exists(thumb):
        try:
            os.remove(thumb)
        except OSError:
            pass

    # Auto-remove the collection folder if it is now empty
    try:
        _remove_empty_dirs(coll_name_abs)
    except OSError:
        pass

    regenerate_collection_gif(collection_name)

    return None


def move_to_collection(source_rel: str, collection_name: str) -> str | None:
    """Move all files from source_rel folder into collections/<collection_name>.

    Moves ALL files (media + non-media) so the source dir can be fully cleaned up.
    Generates thumbnails for media files immediately.
    Cleans stale ImageMeta rows for the moved folder without a full scan.
    Returns None on success, error string on failure.
    """
    import shutil

    # Validate collection_name
    illegal = set('/\\:*?"<>|')
    collection_name = collection_name.strip()
    if not collection_name:
        return "Collection name cannot be empty"
    if collection_name.startswith("."):
        return "Collection name cannot start with a dot"
    if illegal & set(collection_name):
        return f"Illegal characters in name: {''.join(sorted(illegal & set(collection_name)))}"

    # Validate source path
    coll_abs = os.path.abspath(_collections_dir())
    cat_abs = os.path.abspath(_categorized_dir())
    dl_abs = os.path.abspath(DOWNLOAD_DIR)

    src_abs = os.path.abspath(os.path.join(DOWNLOAD_DIR, source_rel))
    if src_abs == dl_abs or not src_abs.startswith(dl_abs + os.sep):
        return "Invalid source path (cannot be the downloads root or outside it)"
    if src_abs.startswith(coll_abs + os.sep) or src_abs == coll_abs:
        return "Source is already in a collection"
    if src_abs.startswith(cat_abs + os.sep) or src_abs == cat_abs:
        return "Cannot move a categorized folder to a collection"
    if not os.path.isdir(src_abs):
        return f"Source folder not found: {source_rel}"

    # Normalize rel path for string construction (thumbnail paths, stale-cleanup prefix).
    # os.path.abspath already resolved the path, derive clean rel from that.
    source_rel = os.path.relpath(src_abs, DOWNLOAD_DIR)

    dest_dir = os.path.join(_collections_dir(), collection_name)
    os.makedirs(dest_dir, exist_ok=True)

    errors = []
    for fname in sorted(os.listdir(src_abs)):
        src_file = os.path.join(src_abs, fname)
        if not os.path.isfile(src_file):
            continue  # skip subdirs (flat move only)

        # Collision-free dest filename
        dest_fname = fname
        counter = 2
        while os.path.exists(os.path.join(dest_dir, dest_fname)):
            stem, suffix = os.path.splitext(fname)
            dest_fname = f"{stem}_{counter}{suffix}"
            counter += 1

        dest_file = os.path.join(dest_dir, dest_fname)
        try:
            shutil.move(src_file, dest_file)
        except OSError as e:
            errors.append(str(e))
            continue

        # Generate thumbnail for media files
        ext = os.path.splitext(fname)[1].lower()
        if ext in _MEDIA_EXTS:
            rel_dest = f"collections/{collection_name}/{dest_fname}"
            try:
                if ext in _IMAGE_EXTS:
                    _generate_thumb(dest_file, rel_dest)
                else:
                    _generate_video_thumb(dest_file, rel_dest)
            except Exception:
                pass  # non-fatal

        # Remove stale source thumbnail
        rel_src = f"{source_rel}/{fname}"
        old_thumb = _thumb_path(rel_src)
        if os.path.exists(old_thumb):
            try:
                os.remove(old_thumb)
            except OSError:
                pass

    if errors:
        try:
            remaining = len(
                [f for f in os.listdir(src_abs) if os.path.isfile(os.path.join(src_abs, f))]
            )
        except OSError:
            remaining = 0
        return (
            f"Errors during move: {'; '.join(errors)}. "
            f"{remaining} file(s) remain in the source folder."
        )

    _remove_empty_dirs(src_abs)

    # Targeted cleanup of stale ImageMeta rows for the moved folder.
    # Uses Python-side startswith filter (SQL LIKE is unsafe — '_' is a wildcard).
    # Only remove rows for files that no longer exist on disk — flat move skips
    # subdirectories, so subdir media files remain in place and must keep their rows.
    try:
        prefix = source_rel.rstrip("/") + "/"
        with get_session() as s:
            stale = [
                row
                for row in s.scalars(select(ImageMeta)).all()
                if row.file_path.startswith(prefix)
                and not os.path.exists(os.path.join(DOWNLOAD_DIR, row.file_path))
            ]
            for row in stale:
                s.execute(delete(FaceEmbedding).where(FaceEmbedding.file_path == row.file_path))
                old_thumb = _thumb_path(row.file_path)
                if os.path.exists(old_thumb):
                    try:
                        os.remove(old_thumb)
                    except OSError:
                        pass
                s.delete(row)
            s.commit()
    except Exception:
        pass  # stale rows cleaned on next full scan

    regenerate_collection_gif(collection_name)

    return None
