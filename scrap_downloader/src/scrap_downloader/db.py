import os
from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import DateTime, Float, String, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

DB_PATH = os.environ.get("DB_PATH", "scrap_downloader.db")
engine = create_engine(f"sqlite:///{DB_PATH}")


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class TaskStatus(str, Enum):
    pending = "pending"
    in_progress = "in_progress"
    done = "done"
    failed = "failed"


class Base(DeclarativeBase):
    pass


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


def init_db():
    Base.metadata.create_all(engine)
    with engine.connect() as conn:
        try:
            conn.execute(text("ALTER TABLE tasks ADD COLUMN completed_at DATETIME"))
            conn.commit()
        except Exception:
            pass  # column already exists


def get_session() -> Session:
    return Session(engine)
