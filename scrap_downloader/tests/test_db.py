import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from scrap_downloader.db import Base, Task, TaskStatus, TaskTool, utcnow


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def test_create_task(session):
    task = Task(url="https://example.com", tag="test", tool=TaskTool.auto)
    session.add(task)
    session.commit()
    assert task.id is not None
    assert task.status == TaskStatus.pending
    assert task.progress == 0.0
    assert task.created_at is not None
    assert task.completed_at is None


def test_task_status_transition(session):
    task = Task(url="https://example.com", tag="test", tool=TaskTool.auto)
    session.add(task)
    session.commit()

    task.status = TaskStatus.in_progress
    task.progress = 50.0
    session.commit()
    assert task.status == TaskStatus.in_progress

    task.status = TaskStatus.done
    task.progress = 100.0
    session.commit()
    assert task.status == TaskStatus.done


def test_task_failure(session):
    task = Task(url="https://example.com", tag="test", tool=TaskTool.auto)
    session.add(task)
    session.commit()

    task.status = TaskStatus.failed
    task.error = "Connection timeout"
    session.commit()

    fetched = session.get(Task, task.id)
    assert fetched.status == TaskStatus.failed
    assert fetched.error == "Connection timeout"


def test_gallery_dl_task_with_extra_args(session):
    task = Task(
        url="https://example.com/gallery",
        tag="pics",
        tool=TaskTool.gallery_dl,
        extra_args="--range 1-10",
    )
    session.add(task)
    session.commit()

    fetched = session.get(Task, task.id)
    assert fetched.tool == TaskTool.gallery_dl
    assert fetched.extra_args == "--range 1-10"


def test_completed_at(session):
    task = Task(url="https://example.com", tag="test", tool=TaskTool.auto)
    session.add(task)
    session.commit()

    now = utcnow()
    task.status = TaskStatus.done
    task.completed_at = now
    session.commit()

    fetched = session.get(Task, task.id)
    assert fetched.completed_at == now
