import os
from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import (
    DateTime,
    Float,
    Integer,
    LargeBinary,
    String,
    UniqueConstraint,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

DB_PATH = os.environ.get("DB_PATH", "scrap_downloader.db")
engine = create_engine(f"sqlite:///{DB_PATH}")


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


# ── Download queue ────────────────────────────────────────────────────────────


class TaskStatus(str, Enum):
    pending = "pending"
    in_progress = "in_progress"
    done = "done"
    failed = "failed"


class TaskTool(str, Enum):
    auto = "auto"
    gallery_dl = "gallery-dl"
    ytdlp = "yt-dlp"


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(primary_key=True)
    url: Mapped[str] = mapped_column(String)
    tag: Mapped[str] = mapped_column(String)
    tool: Mapped[str] = mapped_column(String, default=TaskTool.auto)
    extra_args: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default=TaskStatus.pending)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


# ── Organize feature ──────────────────────────────────────────────────────────


class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id: Mapped[int] = mapped_column(primary_key=True)
    file_path: Mapped[str] = mapped_column(String, index=True)
    face_index: Mapped[int] = mapped_column(Integer)
    embedding: Mapped[bytes] = mapped_column(LargeBinary)
    model_name: Mapped[str] = mapped_column(String)
    scanned_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    __table_args__ = (UniqueConstraint("file_path", "face_index"),)


class ImageMeta(Base):
    __tablename__ = "image_meta"

    id: Mapped[int] = mapped_column(primary_key=True)
    file_path: Mapped[str] = mapped_column(String, unique=True)
    phash: Mapped[str | None] = mapped_column(String, nullable=True)
    sha256: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    face_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    max_face_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    file_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    file_mtime: Mapped[float | None] = mapped_column(Float, nullable=True)
    scanned_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ScanMeta(Base):
    """Operational scan state — what was done (not user preferences)."""

    __tablename__ = "scan_meta"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[str | None] = mapped_column(String, nullable=True)


class Setting(Base):
    """User preferences — persisted to DB, survive Docker restart."""

    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[str | None] = mapped_column(String, nullable=True)


class DismissedMatch(Base):
    """Dismissed suggestion pairs — folder_a ≤ folder_b alphabetically (normalized)."""

    __tablename__ = "dismissed_matches"

    id: Mapped[int] = mapped_column(primary_key=True)
    folder_a: Mapped[str] = mapped_column(String)
    folder_b: Mapped[str] = mapped_column(String)
    dismissed_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    __table_args__ = (UniqueConstraint("folder_a", "folder_b"),)


class FolderCentroid(Base):
    """Mean face embedding per categorized/ folder — for scan_new_download matching."""

    __tablename__ = "folder_centroids"

    id: Mapped[int] = mapped_column(primary_key=True)
    folder: Mapped[str] = mapped_column(String, unique=True)
    centroid: Mapped[bytes] = mapped_column(LargeBinary)
    model_name: Mapped[str] = mapped_column(String)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Notification(Base):
    """Match notifications from mini-scan — polled by UI on page load."""

    __tablename__ = "notifications"

    id: Mapped[int] = mapped_column(primary_key=True)
    person_name: Mapped[str] = mapped_column(String)
    new_folder: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    seen: Mapped[int] = mapped_column(Integer, default=0)

    __table_args__ = (UniqueConstraint("person_name", "new_folder"),)


class MergeLog(Base):
    """Written before merge starts, marked done at end — enables crash recovery."""

    __tablename__ = "merge_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    sources: Mapped[str] = mapped_column(String)
    dest: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    file_map: Mapped[str | None] = mapped_column(String, nullable=True)


# ── Settings helpers ──────────────────────────────────────────────────────────

_SETTING_DEFAULTS: dict[str, str] = {
    "face_model": "buffalo_s",
    "face_threshold": "0.4",
    "phash_threshold": "8",
    "min_images_per_folder": "3",
    "mini_scan_enabled": "true",
    "unknown_counter": "0",
    "suggestions_page_size": "10",
    "auto_logout_minutes": "15",
}


def get_setting(key: str, default: str | None = None) -> str | None:
    with Session(engine) as s:
        row = s.get(Setting, key)
        if row is None:
            return _SETTING_DEFAULTS.get(key, default)
        return row.value


def set_setting(key: str, value: str) -> None:
    with Session(engine) as s:
        row = s.get(Setting, key)
        if row is None:
            s.add(Setting(key=key, value=value))
        else:
            row.value = value
        s.commit()


def get_scan_meta(key: str, default: str | None = None) -> str | None:
    with Session(engine) as s:
        row = s.get(ScanMeta, key)
        return row.value if row is not None else default


def set_scan_meta(key: str, value: str | None) -> None:
    with Session(engine) as s:
        row = s.get(ScanMeta, key)
        if row is None:
            s.add(ScanMeta(key=key, value=value))
        else:
            row.value = value
        s.commit()


# ── DB init ───────────────────────────────────────────────────────────────────


def init_db() -> None:
    Base.metadata.create_all(engine)
    with engine.connect() as conn:
        try:
            conn.execute(text("ALTER TABLE tasks ADD COLUMN completed_at DATETIME"))
            conn.commit()
        except Exception:
            pass
    with engine.connect() as conn:
        try:
            conn.execute(text("ALTER TABLE image_meta ADD COLUMN file_size INTEGER"))
            conn.commit()
        except Exception:
            pass
    with engine.connect() as conn:
        try:
            conn.execute(text("ALTER TABLE merge_log ADD COLUMN file_map TEXT"))
            conn.commit()
        except Exception:
            pass


def get_session() -> Session:
    return Session(engine)
