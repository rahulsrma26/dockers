import importlib.util
import logging
import os
import shlex
import subprocess
import threading
import time
from urllib.parse import urlparse

from db import Task, TaskStatus, TaskTool, get_session, utcnow
from downloader import download_image, download_video, to_download_item

logger = logging.getLogger(__name__)

DOWNLOAD_DIR = os.environ.get("DOWNLOAD_DIR", "downloads")
PLUGINS_DIR = os.environ.get("PLUGINS_DIR", "plugins")

_stop_event = threading.Event()


def _load_plugin(domain: str):
    name = domain.removeprefix("www.")
    path = os.path.join(PLUGINS_DIR, f"{name}.py")
    if not os.path.isfile(path):
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_gallery_dl(task: Task):
    save_dir = os.path.join(DOWNLOAD_DIR, task.tag)
    os.makedirs(save_dir, exist_ok=True)

    cmd = ["gallery-dl", "--dest", save_dir]
    if task.extra_args:
        cmd += shlex.split(task.extra_args)
    cmd.append(task.url)

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip() or f"gallery-dl exited with code {result.returncode}"
        )

    logger.info(result.stdout.strip())


def _run_auto(task: Task):
    domain = urlparse(task.url).hostname or ""
    plugin = _load_plugin(domain)
    save_dir = os.path.join(DOWNLOAD_DIR, task.tag)

    if plugin is None:
        logger.info(f"No plugin for {domain}, falling back to yt-dlp")
        download_video(task.url, save_dir)
        return

    items = [to_download_item(i) for i in plugin.extract(task.url)]
    total = len(items)
    errors = []

    for i, item in enumerate(items):

        def on_progress(pct, i=i):
            file_pct = (i + pct / 100) / total * 100
            with get_session() as s:
                t = s.get(Task, task.id)
                t.progress = round(file_pct, 1)
                s.commit()

        try:
            if item.type == "video":
                download_video(item.url, save_dir, on_progress=on_progress, filename=item.filename)
            else:
                download_image(item.url, save_dir, item.filename, headers=item.headers)
        except Exception as e:
            logger.error(f"Failed to download {item.url}: {e}")
            errors.append(str(e))

    if errors:
        raise RuntimeError(errors[0])


def _process(task: Task):
    with get_session() as session:
        t = session.get(Task, task.id)
        t.status = TaskStatus.in_progress
        t.progress = 0.0
        session.commit()

    try:
        if task.tool == TaskTool.gallery_dl:
            _run_gallery_dl(task)
        else:
            _run_auto(task)
    except Exception as e:
        _fail(task.id, str(e))
        return

    with get_session() as session:
        t = session.get(Task, task.id)
        t.status = TaskStatus.done
        t.progress = 100.0
        t.completed_at = utcnow()
        session.commit()


def _fail(task_id: int, error: str):
    logger.error(error)
    with get_session() as session:
        t = session.get(Task, task_id)
        t.status = TaskStatus.failed
        t.error = error
        t.completed_at = utcnow()
        session.commit()


def _next_pending() -> Task | None:
    with get_session() as session:
        task = (
            session.query(Task)
            .filter(Task.status == TaskStatus.pending)
            .order_by(Task.created_at)
            .first()
        )
        if task:
            session.expunge(task)
        return task


def run():
    logger.info("Worker started")
    while not _stop_event.is_set():
        task = _next_pending()
        if task:
            logger.info(f"Processing task {task.id}: {task.url}")
            _process(task)
        else:
            time.sleep(2)


def start() -> threading.Thread:
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def stop():
    _stop_event.set()
