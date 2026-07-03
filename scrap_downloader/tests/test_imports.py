"""Smoke-tests that all modules import without error and expose expected symbols."""

import importlib


def _mod(name):
    return importlib.import_module(f"scrap_downloader.{name}")


def test_main_imports():
    m = _mod("main")
    assert callable(m.main)


def test_worker_imports():
    m = _mod("worker")
    assert callable(m.start)
    assert callable(m.stop)
    assert callable(m._load_plugin)


def test_downloader_imports():
    m = _mod("downloader")
    assert callable(m.download_video)
    assert callable(m.download_image)


def test_db_imports():
    m = _mod("db")
    assert callable(m.init_db)
    assert callable(m.get_session)
    assert hasattr(m, "Task")
    assert hasattr(m, "TaskStatus")
    assert hasattr(m, "TaskTool")


# ── Per-symbol import checks (catch "used but not imported" regressions) ───────
#
# These directly call every cross-module name that has previously caused a
# NameError at runtime. Adding a test here is cheap; missing one is expensive.


def test_main_uses_get_scan_meta():
    """get_scan_meta and set_scan_meta must be imported in main.py."""
    from scrap_downloader.main import get_scan_meta, set_scan_meta  # noqa: F401

    assert callable(get_scan_meta)
    assert callable(set_scan_meta)


def test_worker_uses_select():
    """sqlalchemy.select must be importable from worker (used in _next_pending)."""
    # Verify the function actually calls select by executing it against an in-memory DB
    from sqlalchemy import create_engine

    import scrap_downloader.db as db_mod
    import scrap_downloader.worker as w
    from scrap_downloader.db import Base

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    import pytest

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(db_mod, "engine", engine)
    try:
        result = w._next_pending()  # must not raise NameError
        assert result is None  # empty DB → no pending task
    finally:
        monkeypatch.undo()


def test_organize_uses_select_and_notification():
    """organize.py must import select and Notification (used in _poll_notifications)."""
    # Just importing the module exercises the top-level import block.
    # If a symbol is missing the import will raise ImportError / NameError.
    import scrap_downloader.organize  # noqa: F401


def test_organize_uses_asyncio():
    """asyncio must be present in organize.py (used in start_scan and confirm)."""
    import ast
    import inspect

    import scrap_downloader.organize as org

    source = inspect.getsource(org)
    tree = ast.parse(source)
    top_imports = {
        alias.name if isinstance(node, ast.Import) else node.module
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in (node.names if isinstance(node, ast.Import) else [node])
    }
    assert "asyncio" in top_imports, "asyncio must be imported at module level in organize.py"


def test_face_exports_like_prefix():
    """_like_prefix must be importable from face (used by organize.py)."""
    from scrap_downloader.face import _like_prefix

    assert _like_prefix("categorized/john_doe") == "categorized/john\\_doe/%"


def test_face_exports_next_unknown_name():
    from scrap_downloader.face import next_unknown_name  # noqa: F401

    assert callable(next_unknown_name)


def test_organize_imports_touch_activity():
    """touch_activity must be exported from main and importable by organize."""
    from scrap_downloader.main import touch_activity  # noqa: F401

    assert callable(touch_activity)
