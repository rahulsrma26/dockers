"""
AST-level invariant tests for NiceGUI UI handlers in organize.py.

These tests catch ordering and structural bugs that manifest only at runtime
(element-deleted errors, wrong async context, etc.) without needing a browser.

Rule of thumb for adding a test here:
  If a bug crashed the app and the fix was "reorder two lines" or
  "make this function async", add a test so it can't regress silently.
"""

import ast
import inspect
from typing import Generator

# ── Helpers ───────────────────────────────────────────────────────────────────


def _parse_organize() -> tuple[ast.Module, str]:
    import scrap_downloader.organize as org

    src = inspect.getsource(org)
    return ast.parse(src), src


def _find_async_funcs(tree: ast.Module, name: str) -> list[ast.AsyncFunctionDef]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == name
    ]


def _try_body_stmts(func: ast.AsyncFunctionDef) -> list[str]:
    """Return ast.unparse() of each statement in the first try-block body."""
    for node in ast.walk(func):
        if isinstance(node, ast.Try):
            return [ast.unparse(s) for s in node.body]
    return []


def _stmt_index(stmts: list[str], substring: str) -> int:
    """Return the index of the first statement containing `substring`, or raise."""
    for i, s in enumerate(stmts):
        if substring in s:
            return i
    raise AssertionError(f"No statement containing {substring!r} found in:\n" + "\n".join(stmts))


def _all_func_names_in(tree: ast.Module) -> Generator[str, None, None]:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node.name


# ── Structural: async / threading guards ─────────────────────────────────────


def test_confirm_is_async():
    """confirm() must be async — sync would block the event loop during merge."""
    tree, _ = _parse_organize()
    matches = _find_async_funcs(tree, "confirm")
    assert matches, "confirm() not found as async def in organize.py"


def test_start_scan_is_async():
    """start_scan() must be async — scan_dir is blocking (~minutes)."""
    tree, _ = _parse_organize()
    matches = _find_async_funcs(tree, "start_scan")
    assert matches, "start_scan() not found as async def in organize.py"


def test_no_threading_thread_in_organize():
    """organize.py must not spawn threads — use asyncio.run_in_executor instead.

    Background threads in NiceGUI cannot call ui.notify() or close dialogs
    because they lack the slot context required by the event loop.
    """
    tree, _ = _parse_organize()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        # Catch threading.Thread(...)
        if (
            node.attr == "Thread"
            and isinstance(node.value, ast.Name)
            and node.value.id == "threading"
        ):
            raise AssertionError(
                "threading.Thread found in organize.py — use asyncio.run_in_executor instead"
            )


# ── Ordering: dlg.close() / ui.notify() before any refresh ───────────────────
#
# The dialog lives inside the suggestion card's NiceGUI slot.
# refresh_suggestions() calls suggest_container.clear(), which deletes the
# suggestion card and the dialog with it.  Any UI call on a deleted element
# raises RuntimeError.  Fix: close and notify BEFORE refreshing.


def test_confirm_closes_before_refresh():
    """dlg.close() must appear before refresh_suggestions/refresh_categorized in confirm()."""
    tree, _ = _parse_organize()
    funcs = _find_async_funcs(tree, "confirm")
    assert funcs, "confirm() async def not found"
    stmts = _try_body_stmts(funcs[0])
    assert stmts, "No try block found in confirm()"

    close_idx = _stmt_index(stmts, "dlg.close()")
    refresh_sugg_idx = _stmt_index(stmts, "refresh_suggestions")
    refresh_cat_idx = _stmt_index(stmts, "refresh_categorized")

    assert close_idx < refresh_sugg_idx, (
        f"dlg.close() (line {close_idx}) must come before refresh_suggestions "
        f"(line {refresh_sugg_idx}) — refreshing first deletes the dialog's parent slot"
    )
    assert close_idx < refresh_cat_idx, (
        f"dlg.close() (line {close_idx}) must come before refresh_categorized "
        f"(line {refresh_cat_idx})"
    )


def test_notify_before_refresh_in_confirm():
    """ui.notify() must appear before refresh_suggestions/refresh_categorized in confirm()."""
    tree, _ = _parse_organize()
    funcs = _find_async_funcs(tree, "confirm")
    assert funcs, "confirm() async def not found"
    stmts = _try_body_stmts(funcs[0])
    assert stmts, "No try block found in confirm()"

    notify_idx = _stmt_index(stmts, "ui.notify(")
    refresh_sugg_idx = _stmt_index(stmts, "refresh_suggestions")
    refresh_cat_idx = _stmt_index(stmts, "refresh_categorized")

    assert notify_idx < refresh_sugg_idx, (
        f"ui.notify() (line {notify_idx}) must come before refresh_suggestions "
        f"(line {refresh_sugg_idx})"
    )
    assert notify_idx < refresh_cat_idx, (
        f"ui.notify() (line {notify_idx}) must come before refresh_categorized "
        f"(line {refresh_cat_idx})"
    )


# ── asyncio usage: run_in_executor for blocking calls ────────────────────────


def test_confirm_uses_run_in_executor():
    """confirm() must offload blocking work via run_in_executor, not call face_mod directly."""
    tree, _ = _parse_organize()
    funcs = _find_async_funcs(tree, "confirm")
    assert funcs, "confirm() async def not found"
    src = ast.unparse(funcs[0])
    assert "run_in_executor" in src, (
        "confirm() must use run_in_executor for face_mod.merge_into / analyse_downloads"
    )


def test_start_scan_uses_run_in_executor():
    """start_scan() must offload scan_dir via run_in_executor."""
    tree, _ = _parse_organize()
    funcs = _find_async_funcs(tree, "start_scan")
    assert funcs, "start_scan() async def not found"
    src = ast.unparse(funcs[0])
    assert "run_in_executor" in src, "start_scan() must use run_in_executor for face_mod.scan_dir"


# ── Name-input reliability: ui.input not ui.select ───────────────────────────
#
# ui.select(with_input=True) uses Quasar QSelect internally.  Its .value only
# updates when the user explicitly confirms a choice (presses Enter or clicks an
# option).  Typing then immediately clicking Merge leaves .value = None, which
# falls through to next_unknown_name() → "unknown-1".
#
# Fix: use ui.input for the person-name field.  ui.input.value is always the
# current text — no confirmation step required.
#
# These two tests prevent the fragile pattern from being re-introduced.


def test_suggestion_card_uses_ui_input_for_name():
    """_render_suggestion_card must use ui.input for the person-name field.

    ui.select(with_input=True) only updates .value on explicit confirmation
    (Enter / option-click).  If the user types a name and clicks Merge directly,
    .value is still None → merge falls to next_unknown_name() → 'unknown-1'.
    ui.input.value always reflects the current text.
    """
    _, src = _parse_organize()
    start = src.index("def _render_suggestion_card")
    # Slice to the next top-level def so we don't inspect unrelated code.
    rest = src[start:]
    next_def = rest.find("\ndef ", 1)
    card_src = rest[:next_def] if next_def != -1 else rest

    assert "ui.input(" in card_src, (
        "_render_suggestion_card must create a ui.input for the person name. "
        "Do not switch back to ui.select(with_input=True) — it loses typed text "
        "unless the user explicitly confirms the value."
    )


def test_organize_imports_json():
    """organize.py must import json at module level — used for suggestion persistence."""
    import ast
    import inspect

    import scrap_downloader.organize as org

    src = inspect.getsource(org)
    tree = ast.parse(src)
    imports = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert "json" in imports, (
        "organize.py must have 'import json' at module level — "
        "it is used for json.loads/json.dumps in suggestion persistence"
    )


def test_suggestion_card_no_typed_text_tracking():
    """_render_suggestion_card must NOT use typed_text event-tracking.

    The previous workaround listened to Quasar's 'input-value' event and stored
    the typed string in typed_text["value"].  The e.args structure varied across
    NiceGUI versions (string vs list) causing first-char bugs and silent drops.
    The correct fix is ui.input, which does not need event tracking at all.
    """
    _, src = _parse_organize()
    start = src.index("def _render_suggestion_card")
    rest = src[start:]
    next_def = rest.find("\ndef ", 1)
    card_src = rest[:next_def] if next_def != -1 else rest

    assert "typed_text" not in card_src, (
        "typed_text tracking has been removed from _render_suggestion_card. "
        "Use ui.input (always-synced value) instead of event-based tracking."
    )


# ── Performance: hoist _list_people / lazy chip previews ─────────────────────
#
# _render_suggestion_card is called once per visible suggestion.  Two patterns
# were costing O(N) DB queries per render cycle:
#   1. _list_people() called inside the card → N queries per render
#   2. _person_sample_thumbs() called for every chip in every card during
#      initial render, even though previews start hidden → 6N queries per render
#
# Fixes: (1) hoist _list_people() into _render_suggestions and pass it down as
# existing_people; (2) populate chip preview rows lazily inside _chip_select on
# first click.


def test_render_suggestion_card_accepts_existing_people():
    """_render_suggestion_card must accept existing_people param to allow hoisting _list_people."""
    _, src = _parse_organize()
    start = src.index("def _render_suggestion_card")
    # Capture through the closing paren+colon of the signature (skip type annotation colons)
    paren_close = src.index("):", start)
    sig_src = src[start : paren_close + 2]
    assert "existing_people" in sig_src, (
        "_render_suggestion_card must have an existing_people parameter so _render_suggestions "
        "can call _list_people() once and pass it to each card, avoiding N DB queries per render."
    )


def test_chip_previews_populated_lazily():
    """_person_sample_thumbs must only be called inside _chip_select, not at render time.

    Pre-fetching all chip preview thumbnails during initial card render costs up to
    6 DB queries per suggestion card even though previews start hidden.  The fix
    populates each preview lazily — only when the user actually clicks that chip.
    """
    _, src = _parse_organize()
    start = src.index("def _render_suggestion_card")
    rest = src[start:]
    next_def = rest.find("\ndef ", 1)
    card_src = rest[:next_def] if next_def != -1 else rest

    chip_select_start = card_src.find("def _chip_select")
    assert chip_select_start != -1, "_chip_select not found inside _render_suggestion_card"

    before_chip_select = card_src[:chip_select_start]
    assert "_person_sample_thumbs" not in before_chip_select, (
        "_person_sample_thumbs must not be called before _chip_select is defined. "
        "Chip previews must be populated lazily on click to avoid blocking the render loop."
    )


# ── Cache-control: /thumbs/ must set immutable headers ───────────────────────
#
# NiceGUI's add_static_files() uses Starlette StaticFiles which omits
# Cache-Control: max-age, so the browser re-validates thumbnails on every
# re-render after a merge.  Thumb filenames are SHA256 hashes of their rel
# path, making them safe to cache indefinitely — a content change produces a
# new hash (new URL), so immutable is correct.


def test_thumbs_route_sets_immutable_cache_header():
    """The /thumbs/ route must set Cache-Control: immutable to prevent reload on re-render."""
    import inspect

    import scrap_downloader.main as main_mod

    src = inspect.getsource(main_mod)
    assert "Cache-Control" in src, "main.py must set a Cache-Control header on /thumbs/ responses."
    assert "immutable" in src, (
        "main.py must include 'immutable' in the Cache-Control header for /thumbs/ so the "
        "browser serves thumbnails from disk cache without re-validating after each merge."
    )
