"""Smoke-tests that all modules import without error and expose expected symbols."""

import importlib


def _mod(name):
    return importlib.import_module(f"scrap_downloader.{name}")


def test_main_imports():
    m = _mod("main")
    assert callable(m.main)


def test_main_has_tidy_dir():
    m = _mod("main")
    assert hasattr(m, "tidy_dir"), "tidy_dir not imported into main"


def test_main_has_threading():
    import threading as _threading

    m = _mod("main")
    import sys

    assert "threading" in sys.modules, "threading not in sys.modules"
    assert m.threading is _threading, "threading not imported into main"


def test_worker_imports():
    m = _mod("worker")
    assert callable(m.start)
    assert callable(m.stop)
    assert callable(m._load_plugin)


def test_downloader_imports():
    m = _mod("downloader")
    assert callable(m.download_video)
    assert callable(m.download_image)


def test_tidy_imports():
    m = _mod("tidy")
    assert callable(m.tidy_dir)


def test_db_imports():
    m = _mod("db")
    assert callable(m.init_db)
    assert callable(m.get_session)
    assert hasattr(m, "Task")
    assert hasattr(m, "TaskStatus")
    assert hasattr(m, "TaskTool")
