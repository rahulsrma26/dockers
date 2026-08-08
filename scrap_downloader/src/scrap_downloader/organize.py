"""Organize page — /organize — face-based folder management."""

from __future__ import annotations

import asyncio
import html as _html
import json
import os
from typing import Any, Callable
from urllib.parse import quote

from nicegui import app, ui
from sqlalchemy import delete, select

from .db import (
    DismissedMatch,
    FaceEmbedding,
    ImageMeta,
    MergeLog,
    Notification,
    get_scan_meta,
    get_session,
    get_setting,
    set_scan_meta,
    set_setting,
    utcnow,
)
from .face import _like_prefix, _thumb_path
from .main import APP_VERSION, PASSWORD, SESSION_TIMEOUT_MINUTES, check_auth, logout, touch_activity

DOWNLOAD_DIR = os.environ.get("DOWNLOAD_DIR", "downloads")

_ILLEGAL_CHARS = set('/\\:*?"<>|')

_LG_INIT_JS = (
    "(function tryInit() {"
    "  var el = document.getElementById('lg-gallery');"
    "  if (!el || !window.lightGallery) { setTimeout(tryInit, 100); return; }"
    "  window.lgInstance = lightGallery(el, {"
    "    plugins: [lgZoom, lgThumbnail, lgVideo],"
    "    speed: 300, download: true, selector: '.lg-item',"
    "  });"
    "})();"
)

_gif_warmup_task: asyncio.Task | None = None


@app.on_startup
async def _warmup_gifs_on_startup():
    global _gif_warmup_task
    from . import face as face_mod

    async def _task():
        try:
            await asyncio.get_running_loop().run_in_executor(None, face_mod.warmup_missing_gifs)
        except Exception:
            pass

    _gif_warmup_task = asyncio.create_task(_task())


def _auto_logout_js() -> str:
    """Return a <script> tag that redirects to /logout after inactivity. Empty if disabled."""
    if not PASSWORD:
        return ""
    minutes = int(get_setting("auto_logout_minutes") or SESSION_TIMEOUT_MINUTES)
    if minutes == 0:
        return ""
    ms = minutes * 60 * 1000
    return (
        "<script>(function(){"
        f"var t,ms={ms};"
        "function r(){clearTimeout(t);"
        "t=setTimeout(function(){window.location.href='/logout';},ms);}"
        "['mousemove','mousedown','keydown','touchstart','click','scroll','wheel']"
        ".forEach(function(e){document.addEventListener(e,r,true);});"
        "r();"
        "})()</script>"
    )


def _human_size(n: int | None) -> str:
    if n is None:
        return "—"
    if n == 0:
        return "0 B"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} PB"


def _get_stats_data() -> dict:
    with get_session() as s:
        rows = s.execute(select(ImageMeta.file_path, ImageMeta.file_size)).all()

    total_files = len(rows)
    total_size = sum(r.file_size or 0 for r in rows)
    has_size = any(r.file_size is not None for r in rows)

    cat_files = [r for r in rows if r.file_path.startswith("categorized/")]
    uncat_files = [r for r in rows if not r.file_path.startswith("categorized/")]

    tag_counts: dict[str, tuple[int, int]] = {}
    for r in uncat_files:
        tag = r.file_path.split("/")[0] or "(root)"
        c, sz = tag_counts.get(tag, (0, 0))
        tag_counts[tag] = (c + 1, sz + (r.file_size or 0))

    by_tag = sorted(tag_counts.items(), key=lambda kv: kv[1][0], reverse=True)

    return {
        "total_files": total_files,
        "total_size": total_size if has_size else None,
        "cat_count": len(cat_files),
        "cat_size": sum(r.file_size or 0 for r in cat_files) if has_size else None,
        "uncat_count": len(uncat_files),
        "uncat_size": sum(r.file_size or 0 for r in uncat_files) if has_size else None,
        "has_size": has_size,
        "by_tag": by_tag,
    }


def _categorized_dir() -> str:
    return os.path.join(DOWNLOAD_DIR, "categorized")


def _person_file_count(person_name: str) -> int:
    folder_rel = f"categorized/{person_name}"
    with get_session() as s:
        rows = (
            s.execute(
                select(ImageMeta).where(
                    ImageMeta.file_path.like(_like_prefix(folder_rel), escape="\\")
                )
            )
            .scalars()
            .all()
        )
        return len(rows)


def _person_thumbs(person_name: str, n: int = 4) -> list[str]:
    import hashlib

    folder_rel = f"categorized/{person_name}"
    pattern = _like_prefix(folder_rel)
    with get_session() as s:
        rows = (
            s.execute(
                select(ImageMeta)
                .where(
                    ImageMeta.file_path.like(pattern, escape="\\"),
                    ImageMeta.max_face_ratio.isnot(None),
                )
                .order_by(ImageMeta.max_face_ratio.desc())
                .limit(n)
            )
            .scalars()
            .all()
        )
        if not rows:
            rows = (
                s.execute(
                    select(ImageMeta).where(ImageMeta.file_path.like(pattern, escape="\\")).limit(n)
                )
                .scalars()
                .all()
            )

    return [f"/thumbs/{hashlib.sha256(r.file_path.encode()).hexdigest()}.jpg" for r in rows]


_VIDEO_EXTS = {".mp4", ".webm", ".mkv", ".avi", ".mov", ".flv", ".m4v", ".wmv"}
_VIDEO_MIME = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".m4v": "video/x-m4v",
    ".mov": "video/quicktime",
}


def _person_images(person_name: str) -> list[dict]:
    """Return all media files for a person with src/thumb URLs, sorted by path."""
    import hashlib

    folder_rel = f"categorized/{person_name}"
    pattern = _like_prefix(folder_rel)
    with get_session() as s:
        rows = (
            s.execute(
                select(ImageMeta)
                .where(ImageMeta.file_path.like(pattern, escape="\\"))
                .order_by(ImageMeta.file_path)
            )
            .scalars()
            .all()
        )
    result = []
    for r in rows:
        rel = r.file_path.replace("\\", "/")
        ext = os.path.splitext(rel)[1].lower()
        thumb = f"/thumbs/{hashlib.sha256(r.file_path.encode()).hexdigest()}.jpg"
        src = f"/files/{rel}"
        item: dict = {"rel": rel, "src": src, "thumb": thumb, "filename": os.path.basename(rel)}
        if ext in _VIDEO_EXTS:
            item["is_video"] = True
            item["video_type"] = _VIDEO_MIME.get(ext, "video/mp4")
        result.append(item)
    return result


def _poll_notifications() -> list[dict]:
    with get_session() as s:
        rows = s.execute(select(Notification).where(Notification.seen == 0)).scalars().all()
        data = [{"id": r.id, "person": r.person_name, "folder": r.new_folder} for r in rows]
        for r in rows:
            r.seen = 1
        s.commit()
    return data


def _list_people() -> list[str]:
    cat = _categorized_dir()
    if not os.path.isdir(cat):
        return []
    return sorted(
        (
            p
            for p in os.listdir(cat)
            if os.path.isdir(os.path.join(cat, p)) and not p.startswith(".")
        ),
        key=str.casefold,
    )


def _person_sample_thumbs(person: str, limit: int = 6) -> list[str]:
    from . import face as face_mod

    folder = os.path.join(_categorized_dir(), person)
    if not os.path.isdir(folder):
        return []
    exts = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    rel_files = [
        f"categorized/{person}/{fn}"
        for fn in sorted(os.listdir(folder))
        if os.path.splitext(fn)[1].lower() in exts
    ]
    return face_mod._pick_sample_thumbs(rel_files, n=limit)


def _validate_name(name: str) -> str | None:
    name = name.strip()
    if not name:
        return "Name cannot be empty"
    bad = _ILLEGAL_CHARS & set(name)
    if bad:
        return f"Illegal characters: {''.join(sorted(bad))}"
    return None


# ── Page ──────────────────────────────────────────────────────────────────────


@ui.page("/organize")
def organize_page(tab: str = "scan"):
    if not check_auth():
        ui.navigate.to("/login")
        return

    ui.page_title("Organize — Scrap Downloader")
    ui.add_body_html(_auto_logout_js())

    dark = ui.dark_mode()
    dark.set_value(app.storage.user.get("dark_mode", False))

    # State shared across tabs
    _saved_sugg = get_scan_meta("last_suggestions")
    _saved_amb = get_scan_meta("last_ambiguous")
    try:
        _sugg_init = json.loads(_saved_sugg) if _saved_sugg else []
        _amb_init = json.loads(_saved_amb) if _saved_amb else []
    except (json.JSONDecodeError, TypeError):
        _sugg_init, _amb_init = [], []
    state: dict[str, Any] = {
        "suggestions": _sugg_init,
        "ambiguous": _amb_init,
        "active_tab": tab,
    }

    def _toggle_dark(dark=dark):
        new_val = not dark.value
        dark.set_value(new_val)
        app.storage.user["dark_mode"] = new_val

    # ── Header ──
    with ui.column().classes("w-full max-w-5xl mx-auto p-4 gap-4"):
        with ui.row().classes("items-center justify-between w-full"):
            with ui.row().classes("items-baseline gap-2"):
                ui.label("Scrap Downloader").classes("text-2xl font-bold")
                ui.label(f"v{APP_VERSION}").classes("text-sm text-gray-400")
                ui.link("Downloads", "/").classes("text-sm text-blue-500 ml-4")
                ui.link("Organize", "/organize").classes("text-sm text-blue-500 font-semibold")
            with ui.row().classes("gap-1"):
                ui.button(icon="dark_mode", on_click=_toggle_dark).props("flat round").tooltip(
                    "Toggle dark mode"
                )
                if PASSWORD:
                    ui.button("Logout", icon="logout", on_click=logout).props("flat")

        # Notification banner
        notifs = _poll_notifications()
        for notif in notifs:
            with ui.card().classes("w-full bg-blue-50 border border-blue-200"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("notifications").classes("text-blue-500")
                    ui.label(
                        f"New files may belong to '{notif['person']}' (from {notif['folder']})"
                    ).classes("flex-1 text-sm")
                    ui.button(
                        "Review",
                        on_click=lambda: ui.navigate.to("/organize?tab=categorized"),
                    ).props("flat dense size=sm")

        # ── Tab bar ──
        with ui.tabs().classes("w-full") as tabs:
            tab_scan = ui.tab("Scan", icon="search")
            tab_suggest = ui.tab("Suggestions", icon="people")
            tab_cat = ui.tab("Categorized", icon="folder_special")
            tab_coll = ui.tab("Collections", icon="collections")
            tab_settings = ui.tab("Settings", icon="tune")

        _tab_map = {
            "scan": tab_scan,
            "suggestions": tab_suggest,
            "categorized": tab_cat,
            "collections": tab_coll,
            "settings": tab_settings,
        }
        initial_tab = _tab_map.get(tab, tab_scan)

        _label_to_slug = {
            "Scan": "scan",
            "Suggestions": "suggestions",
            "Categorized": "categorized",
            "Collections": "collections",
            "Settings": "settings",
        }

        def _sync_tab_url(e):
            slug = _label_to_slug.get(e.args, "scan")
            ui.run_javascript(f"history.replaceState(null, '', '/organize?tab={slug}')")

        tabs.on("update:model-value", _sync_tab_url)

        with ui.tab_panels(tabs, value=initial_tab).classes("w-full"):
            # ── Tab 1: Scan ──────────────────────────────────────────────────
            with ui.tab_panel(tab_scan):
                _build_scan_tab(state, tabs, tab_suggest)

            # ── Tab 2: Suggestions ───────────────────────────────────────────
            with ui.tab_panel(tab_suggest):
                suggest_container = ui.column().classes("w-full gap-4")
                with suggest_container:
                    _render_suggestions(state, suggest_container)

                def refresh_suggestions():
                    suggest_container.clear()
                    with suggest_container:
                        _render_suggestions(state, suggest_container)

                state["refresh_suggestions"] = refresh_suggestions

            # ── Tab 3: Categorized ───────────────────────────────────────────
            with ui.tab_panel(tab_cat):
                cat_container = ui.column().classes("w-full gap-4")
                with cat_container:
                    _render_categorized(cat_container, state)

                def refresh_categorized():
                    cat_container.clear()
                    with cat_container:
                        _render_categorized(cat_container, state)

                state["refresh_categorized"] = refresh_categorized

            # ── Tab 4: Collections ───────────────────────────────────────────
            with ui.tab_panel(tab_coll):
                coll_container = ui.column().classes("w-full gap-4")
                with coll_container:
                    _render_collections(coll_container, state)

                def refresh_collections():
                    coll_container.clear()
                    with coll_container:
                        _render_collections(coll_container, state)

                state["refresh_collections"] = refresh_collections

            # ── Tab 5: Settings ──────────────────────────────────────────────
            with ui.tab_panel(tab_settings):
                _build_settings_tab()


def _build_scan_tab(state: dict, tabs, tab_suggest):
    from . import face as face_mod

    scan_status = get_scan_meta("status", "idle")
    scan_model = get_scan_meta("model")
    face_model = get_setting("face_model", "buffalo_s")
    last_scan = get_scan_meta("last_scan")
    last_error = get_scan_meta("last_error")

    # First-time guide
    if not last_scan and scan_status != "running":
        with ui.card().classes("w-full bg-amber-50 border border-amber-200 mb-2"):
            ui.label("Getting started").classes("font-semibold mb-1")
            with ui.column().classes("gap-1 text-sm"):
                ui.label("1. Configure face model and thresholds in Settings")
                ui.label("2. Click 'Scan downloads' to extract face embeddings")
                ui.label("3. Review suggestions in the Suggestions tab")

    # Model mismatch warning
    if scan_model and scan_model != face_model:
        with ui.card().classes("w-full bg-orange-50 border border-orange-300"):
            ui.label(f"Model changed ({scan_model} → {face_model}) — re-scan required").classes(
                "text-orange-700 text-sm"
            )

    # Error banner
    if last_error and scan_status != "running":
        with ui.card().classes("w-full bg-red-50 border border-red-300"):
            ui.label(f"Last scan failed: {last_error}").classes("text-red-700 text-sm")

    # Scan recommended banner
    if last_scan is None and scan_model:
        with ui.card().classes("w-full bg-yellow-50 border border-yellow-200"):
            ui.label("Scan recommended — files may have changed since last merge").classes(
                "text-yellow-700 text-sm"
            )

    # Status line
    status_label = ui.label("").classes("text-sm text-gray-500")
    progress_bar = ui.linear_progress(value=0).classes("w-full").props("instant-feedback")
    counter_label = ui.label("").classes("text-xs text-gray-400")

    # ── Storage stats card ───────────────────────────────────────────────────────
    with ui.card().classes("w-full mt-2"):
        ui.label("Media storage").classes("text-sm font-semibold mb-1")
        stats_placeholder = ui.label("Scan in progress — stats will update when complete.").classes(
            "text-xs text-gray-400"
        )
        stats_body = ui.column().classes("w-full gap-1")

    def _refresh_stats():
        scan_st = get_scan_meta("status", "idle")
        if scan_st in ("running", "cancelling"):
            stats_placeholder.set_visibility(True)
            stats_body.set_visibility(False)
            return
        stats_placeholder.set_visibility(False)
        stats_body.set_visibility(True)
        stats_body.clear()
        d = _get_stats_data()
        with stats_body:
            with ui.row().classes("gap-4 text-sm"):
                ui.label(f"Total: {d['total_files']} files")
                ui.label(f"Size: {_human_size(d['total_size'])}")
            with ui.row().classes("gap-4 text-xs text-gray-500"):
                ui.label(f"Categorized: {d['cat_count']} ({_human_size(d['cat_size'])})")
                ui.label(f"Uncategorized: {d['uncat_count']} ({_human_size(d['uncat_size'])})")
            if not d["has_size"]:
                ui.label("File sizes will appear after the next scan.").classes(
                    "text-xs text-gray-400 italic"
                )
            if d["by_tag"]:
                ui.separator().classes("my-1")
                for tag, (count, size) in d["by_tag"][:10]:
                    with ui.row().classes("justify-between text-xs"):
                        ui.label(tag).classes("text-gray-600")
                        ui.label(f"{count} files · {_human_size(size) if d['has_size'] else '—'}")
            elif d["uncat_count"] == 0 and d["cat_count"] > 0:
                ui.label("All files are categorized.").classes("text-xs text-gray-400 italic")

    _refresh_stats()

    def _update_status_ui():
        s = get_scan_meta("status", "idle")
        p = int(get_scan_meta("progress") or 0)
        t = int(get_scan_meta("total") or 0)
        model = get_scan_meta("model", "—")
        err = get_scan_meta("last_error")

        if s == "running":
            status_label.set_text(f"Scanning with {model}…")
            progress_bar.set_value(p / t if t else 0)
            counter_label.set_text(f"{p} / {t} files")
            scan_btn.set_enabled(False)
            cancel_btn.set_visibility(True)
            _refresh_stats()
        elif s == "cancelling":
            status_label.set_text("Cancelling…")
            scan_btn.set_enabled(False)
            cancel_btn.set_visibility(True)
        elif s == "failed":
            status_label.set_text(f"Scan failed: {err or ''}")
            progress_bar.set_value(0)
            scan_btn.set_enabled(True)
            cancel_btn.set_visibility(False)
            _refresh_stats()
        else:
            fresh_last_scan = get_scan_meta("last_scan")
            if fresh_last_scan:
                status_label.set_text(f"Last scan: {fresh_last_scan}  |  model: {model}")
            else:
                status_label.set_text("No scan yet")
            progress_bar.set_value(1 if t and p >= t else 0)
            counter_label.set_text(f"{t} files" if t else "")
            scan_btn.set_enabled(True)
            cancel_btn.set_visibility(False)
            _refresh_stats()

    with ui.row().classes("gap-2 mt-2"):

        async def start_scan():
            touch_activity()
            set_scan_meta("status", "running")
            scan_btn.set_enabled(False)
            cancel_btn.set_visibility(True)

            loop = asyncio.get_running_loop()

            def _blocking():
                face_mod.scan_dir(DOWNLOAD_DIR)
                return face_mod.analyse_downloads(DOWNLOAD_DIR)

            try:
                res = await loop.run_in_executor(None, _blocking)
                state["suggestions"] = res["suggestions"]
                state["ambiguous"] = res["ambiguous"]
                state["suggestions_visible"] = int(
                    get_setting("suggestions_page_size", "10")
                )  # reset pagination on fresh scan
                if "refresh_suggestions" in state:
                    state["refresh_suggestions"]()
                tabs.set_value(tab_suggest)
                ui.run_javascript("history.replaceState(null, '', '/organize?tab=suggestions')")
            except Exception as e:
                set_scan_meta("status", "failed")
                set_scan_meta("last_error", str(e))

        def cancel_scan():
            set_scan_meta("status", "cancelling")
            cancel_btn.set_enabled(False)

        scan_btn = ui.button("Scan downloads", icon="search", on_click=start_scan)
        scan_btn.set_enabled(scan_status not in ("running", "cancelling"))

        cancel_btn = ui.button("Cancel", icon="cancel", on_click=cancel_scan).props(
            "flat color=negative"
        )
        cancel_btn.set_visibility(scan_status in ("running", "cancelling"))

    poll_timer = ui.timer(2.0, _update_status_ui)
    _update_status_ui()
    # Stop timer when the client navigates away; without this the timer fires
    # after NiceGUI deletes the page elements, raising "parent slot deleted".
    ui.context.client.on_disconnect(poll_timer.deactivate)


def _render_suggestions(state: dict, container):

    suggestions = state.get("suggestions", [])
    ambiguous = state.get("ambiguous", [])
    if "show_dismissed" not in state:
        state["show_dismissed"] = False

    if not suggestions and not ambiguous:
        # Check if we've ever scanned
        if not get_scan_meta("last_scan"):
            ui.label("Run a scan first to see grouping suggestions.").classes(
                "text-gray-500 text-sm"
            )
        else:
            ui.label(
                "No groupings found — all files may already be categorized, "
                "or try lowering the face threshold in Settings."
            ).classes("text-gray-500 text-sm")
        return

    # Dismissed suggestions toggle — compute once to avoid 2N event-loop DB queries
    _dismissed_map = {id(s): _is_dismissed(s) for s in suggestions}
    dismissed_active = [s for s in suggestions if _dismissed_map[id(s)]]
    active_suggestions = sorted(
        (s for s in suggestions if not _dismissed_map[id(s)]),
        key=lambda s: min(s.get("folders", [""]), default=""),
    )

    _PAGE = int(get_setting("suggestions_page_size", "10"))
    visible = state.get("suggestions_visible", _PAGE)
    existing_people = _list_people()
    for sugg in active_suggestions[:visible]:
        _render_suggestion_card(
            sugg, state, container, dismissed=False, existing_people=existing_people
        )

    remaining = len(active_suggestions) - visible
    if remaining > 0:

        def _load_more():
            state["suggestions_visible"] = state.get("suggestions_visible", _PAGE) + _PAGE
            if "refresh_suggestions" in state:
                state["refresh_suggestions"]()

        ui.button(
            f"Show more ({remaining} remaining)",
            icon="expand_more",
            on_click=_load_more,
        ).props("flat")

    if dismissed_active:

        def toggle_dismissed():
            state["show_dismissed"] = not state["show_dismissed"]
            if "refresh_suggestions" in state:
                state["refresh_suggestions"]()

        dismissed_label = (
            f"Show dismissed ({len(dismissed_active)})"
            if not state["show_dismissed"]
            else f"Hide dismissed ({len(dismissed_active)})"
        )
        ui.button(dismissed_label, icon="expand_more", on_click=toggle_dismissed).props(
            "flat dense"
        )

        if state["show_dismissed"]:
            ui.separator()
            for sugg in dismissed_active:
                _render_suggestion_card(
                    sugg, state, container, dismissed=True, existing_people=existing_people
                )

    if ambiguous:
        ui.separator()
        ui.label("Needs review").classes("text-lg font-semibold mt-2")
        for item in ambiguous:
            with ui.card().classes("w-full"):
                ui.label(f"Ambiguous match — {len(item['matched_people'])} people matched").classes(
                    "font-semibold"
                )
                with ui.row().classes("gap-1 flex-wrap mt-1"):
                    for fp in item.get("sample_images", [])[:6]:
                        ui.image(fp).classes("w-20 h-20 object-cover rounded")
                ui.label(f"Folders: {', '.join(item['folders'])}").classes("text-sm text-gray-500")
                ui.label(f"Matched people: {', '.join(item['matched_people'])}").classes(
                    "text-sm text-gray-500"
                )


def _is_dismissed(sugg: dict) -> bool:
    from sqlalchemy import select

    # For add_to cards include the categorized folder(s) in the pair check
    folders = list(sugg.get("folders", []))
    if sugg.get("type") == "add_to":
        folders = folders + list(sugg.get("cat_folders", []))

    if not folders:
        return False

    with get_session() as s:
        if len(folders) == 1:
            # Single-folder suggestions (e.g. series) use a self-pair sentinel
            f = folders[0]
            row = s.execute(
                select(DismissedMatch).where(
                    DismissedMatch.folder_a == f, DismissedMatch.folder_b == f
                )
            ).scalar_one_or_none()
            return row is not None

        for i in range(len(folders)):
            for j in range(i + 1, len(folders)):
                a, b = (
                    (folders[i], folders[j])
                    if folders[i] <= folders[j]
                    else (folders[j], folders[i])
                )
                row = s.execute(
                    select(DismissedMatch).where(
                        DismissedMatch.folder_a == a, DismissedMatch.folder_b == b
                    )
                ).scalar_one_or_none()
                if row:
                    return True
    return False


def _render_suggestion_card(
    sugg: dict,
    state: dict,
    container,
    dismissed: bool = False,
    existing_people: list[str] | None = None,
):
    from . import face as face_mod

    card_type = sugg["type"]
    folders = sugg.get("folders", [])
    dest_name = sugg.get("dest_name", "")
    confidence = sugg.get("confidence")
    sample_images = sugg.get("sample_images", [])

    type_labels = {"series": "Series", "face": "New person", "add_to": f"Add to {dest_name}"}
    badge_colors = {"series": "blue", "face": "green", "add_to": "purple"}

    with ui.card().classes(f"w-full {'opacity-60' if dismissed else ''}"):
        # ── Header row ────────────────────────────────────────────────────────
        with ui.row().classes("items-center gap-2 mb-1"):
            ui.badge(type_labels.get(card_type, card_type)).props(
                f"color={badge_colors.get(card_type, 'grey')}"
            )
            if confidence is not None:
                ui.label(f"Confidence: {confidence:.1%}").classes("text-sm text-gray-500")
            elif card_type == "series":
                ui.label("(no face data)").classes("text-sm text-gray-400 italic")

        # ── Source thumbnails — carousel (6 per page) ─────────────────────────
        if sample_images:
            page_size = 6
            carr_offset = [0]
            total_imgs = len(sample_images)
            needs_nav = total_imgs > page_size

            with ui.row().classes("items-center gap-1 mb-1"):
                if needs_nav:
                    carr_prev = ui.button(icon="chevron_left").props("flat dense round")
                carr_imgs = ui.row().classes("gap-1 flex-wrap")
                with carr_imgs:
                    for _t in sample_images[:page_size]:
                        ui.image(_t).classes("w-20 h-20 object-cover rounded")
                if needs_nav:
                    carr_next = ui.button(icon="chevron_right").props("flat dense round")

            if needs_nav:
                carr_prev.set_enabled(False)

                def _carousel_go(
                    delta,
                    imgs=sample_images,
                    cont=carr_imgs,
                    ob=carr_offset,
                    pb=carr_prev,
                    nb=carr_next,
                    ps=page_size,
                    tot=total_imgs,
                ):
                    ob[0] = max(0, min(ob[0] + delta * ps, tot - ps))
                    cont.clear()
                    with cont:
                        for _t in imgs[ob[0] : ob[0] + ps]:
                            ui.image(_t).classes("w-20 h-20 object-cover rounded")
                    pb.set_enabled(ob[0] > 0)
                    nb.set_enabled(ob[0] + ps < tot)

                carr_prev.on_click(lambda: _carousel_go(-1))
                carr_next.on_click(lambda: _carousel_go(1))

        # Source folders
        ui.label(f"Folders: {', '.join(folders)}").classes("text-xs text-gray-400 mt-1")

        # Scores pre-computed in analyse_downloads (executor thread) — use directly.
        scores: dict[str, float] = sugg.get("scores", {})
        if existing_people is None:
            existing_people = _list_people()
        scored_options: dict[str, str] = {}
        for name in existing_people:
            if name in scores:
                scored_options[name] = f"{name}  ({scores[name]:.0%})"
            else:
                scored_options[name] = name

        # Sort priority:
        #   0 – dest_name (clustering's best match, or most recently merged)
        #   1 – other recently merged names (by recency index)
        #   2 – everyone else, score descending
        recently_merged: list[str] = state.get("recently_merged", [])
        effective_dest = dest_name or (recently_merged[0] if recently_merged else "")

        def _option_sort_key(kv):
            name = kv[0]
            if name == effective_dest:
                return (0, -scores.get(name, -1), name)
            if name in recently_merged:
                return (1, recently_merged.index(name), name)
            return (2, -scores.get(name, -1), name)

        scored_options = dict(sorted(scored_options.items(), key=_option_sort_key))

        # Determine initial value for the name input:
        #   preselect → a known person name (key in scored_options)
        #   pre_typed → dest_name when it's not yet a known person (e.g. series prefix)
        preselect = (
            dest_name
            if dest_name in scored_options
            else (effective_dest if effective_dest in scored_options else None)
        )
        pre_typed = dest_name if (dest_name and dest_name not in scored_options) else ""
        init_value = preselect or pre_typed or ""

        # ui.input always syncs .value on every keystroke — no confirmation step.
        # (ui.select with with_input=True only updates .value on Enter/option-click,
        # so typing then immediately clicking Merge gave name_input.value = None.)
        name_input = (
            ui.input(label="Person name")
            .props(
                'clearable hint="Type a name or click a suggestion — leave blank to auto-assign"'
            )
            .classes("w-full mt-2")
        )
        if init_value:
            name_input.set_value(init_value)

        # Chips: positive scores only, best match first.
        # Clicking a chip sets the name input AND expands a preview of that person's images.
        chip_suggestions = sorted(
            [(k, v) for k, v in scored_options.items() if scores.get(k, 0) > 0],
            key=lambda kv: scores[kv[0]],
            reverse=True,
        )[:6]
        if chip_suggestions:
            with ui.row().classes("gap-1 flex-wrap mt-1"):
                for _person, _label in chip_suggestions:
                    ui.chip(
                        _label,
                        on_click=lambda p=_person: _chip_select(p),
                    ).props("clickable dense outline color=blue-grey").classes("text-xs")

            # Hidden preview rows per chip — populated lazily on first click
            chip_previews: dict[str, object] = {}
            chip_populated: dict[str, bool] = {}
            for _person, _ in chip_suggestions:
                preview = ui.row().classes(
                    "gap-1 flex-wrap mt-1 ml-1 border-l-2 border-blue-grey-200"
                )
                preview.set_visibility(False)
                chip_previews[_person] = preview
                chip_populated[_person] = False

            chip_active: dict = {"person": None}

            def _chip_select(
                p, previews=chip_previews, active=chip_active, populated=chip_populated
            ):
                name_input.set_value(p)
                prev = active["person"]
                if prev and prev in previews:
                    previews[prev].set_visibility(False)
                if active["person"] == p:
                    active["person"] = None
                else:
                    if not populated[p]:
                        with previews[p]:
                            thumbs = _person_sample_thumbs(p)
                            if thumbs:
                                for _t in thumbs:
                                    ui.image(_t).classes("w-16 h-16 object-cover rounded")
                            else:
                                ui.label("No images yet").classes("text-xs text-gray-400 italic")
                        populated[p] = True
                    previews[p].set_visibility(True)
                    active["person"] = p

        name_error = ui.label("").classes("text-xs text-red-500")

        # Folder checkboxes
        checked_folders: dict[str, bool] = {f: True for f in folders}
        if len(folders) > 1:
            ui.label("Include folders:").classes("text-xs text-gray-500 mt-1")
            for folder in folders:

                def make_toggle(f):
                    def toggle(e):
                        checked_folders[f] = e.value

                    return toggle

                ui.checkbox(folder, value=True, on_change=make_toggle(folder)).classes("text-xs")

        def do_merge(s=sugg):
            touch_activity()
            raw_name = (name_input.value or "").strip()
            err = _validate_name(raw_name) if raw_name else None
            if err:
                name_error.set_text(err)
                return
            name_error.set_text("")

            final_name = raw_name if raw_name else face_mod.next_unknown_name()
            sources = [f for f, checked in checked_folders.items() if checked]
            if not sources:
                ui.notify("Select at least one folder", color="negative")
                return

            # Skip dest_exists dialog for add_to cards
            if card_type == "add_to":
                _run_merge_with_preview(sources, final_name, state, container)
            else:
                # Check if dest exists and prompt
                cat = _categorized_dir()
                if os.path.isdir(os.path.join(cat, final_name)):
                    with ui.dialog() as dlg, ui.card():
                        ui.label(f"'{final_name}' already exists").classes("font-semibold")
                        ui.label("Add files to existing folder or create new?").classes(
                            "text-sm mt-1"
                        )
                        with ui.row().classes("mt-3 gap-2 justify-end"):
                            ui.button("Cancel", on_click=dlg.close).props("flat")
                            ui.button(
                                "Add to existing",
                                on_click=lambda: [
                                    dlg.close(),
                                    _run_merge_with_preview(sources, final_name, state, container),
                                ],
                            ).props("color=primary")
                    dlg.open()
                else:
                    _run_merge_with_preview(sources, final_name, state, container)

        def do_dismiss(s=sugg):
            all_folders = list(folders)
            if card_type == "add_to":
                cat_folders = sugg.get("cat_folders", [])
                all_folders.extend(cat_folders)
            face_mod.dismiss_suggestion(all_folders)
            if "refresh_suggestions" in state:
                state["refresh_suggestions"]()

        def do_undismiss(s=sugg):
            all_folders = list(folders)
            if card_type == "add_to":
                cat_folders = sugg.get("cat_folders", [])
                all_folders.extend(cat_folders)
            face_mod.undismiss_suggestion(all_folders)
            if "refresh_suggestions" in state:
                state["refresh_suggestions"]()

        # "Add to collection" — only for uncategorized source folders
        uncategorized_sources = [f for f in folders if not f.startswith("categorized/")]

        with ui.row().classes("mt-2 gap-2"):
            if not dismissed:
                ui.button("Confirm & merge", icon="merge", on_click=do_merge).props("color=primary")
                if uncategorized_sources:
                    ui.button(
                        "Add to collection",
                        icon="add_to_photos",
                        on_click=lambda s=uncategorized_sources: _show_add_to_collection_dialog(
                            s, state
                        ),
                    ).props("flat color=secondary")
                ui.button("Dismiss", icon="close", on_click=do_dismiss).props("flat color=grey")
            else:
                ui.button("Undo dismiss", icon="undo", on_click=do_undismiss).props(
                    "flat color=primary"
                )


def _run_merge_with_preview(sources: list[str], dest_name: str, state: dict, container):
    from . import face as face_mod

    report = face_mod.analyse_merge(sources, dest_name)

    n_exact = len(report["exact_dupes"])
    n_phash = len(report["phash_dupes"])
    n_coll = len(report["collisions"])
    n_clean = len(report["clean"])

    with ui.dialog() as dlg, ui.card().classes("w-full max-w-lg"):
        ui.label(f"Merge into '{dest_name}'").classes("text-lg font-semibold mb-2")

        if n_exact:
            ui.label(
                f"{n_exact} exact duplicate{'s' if n_exact > 1 else ''}"
                " → will be deleted from source"
            ).classes("text-sm text-gray-500")
        if n_phash:
            ui.label(
                f"{n_phash} perceptual duplicate{'s' if n_phash > 1 else ''} → keeping larger file"
            ).classes("text-sm text-gray-500")
        if n_coll:
            ui.label(
                f"{n_coll} filename collision{'s' if n_coll > 1 else ''} → will be renamed"
            ).classes("text-sm text-gray-500")
        ui.label(f"{n_clean} file{'s' if n_clean != 1 else ''} → will be moved cleanly").classes(
            "text-sm"
        )

        progress_bar = ui.linear_progress(value=0).classes("w-full mt-3").props("instant-feedback")
        progress_bar.set_visibility(False)

        with ui.row().classes("mt-3 gap-2 justify-end"):
            cancel_btn = ui.button("Cancel", on_click=dlg.close).props("flat")

            # Server-side guard: prevents a second click arriving at the server before
            # the browser reflects the button-disable WebSocket message.
            _in_progress: list = [False]

            async def confirm():
                if _in_progress[0]:
                    return
                _in_progress[0] = True
                cancel_btn.set_enabled(False)
                confirm_btn.set_enabled(False)
                progress_bar.set_visibility(True)

                loop = asyncio.get_running_loop()

                def on_prog(current, total):
                    loop.call_soon_threadsafe(
                        progress_bar.set_value, current / total if total else 1
                    )

                def _blocking():
                    face_mod.merge_into(sources, dest_name, on_progress=on_prog)
                    return face_mod.analyse_downloads(DOWNLOAD_DIR)

                try:
                    res = await loop.run_in_executor(None, _blocking)
                    state["suggestions"] = res["suggestions"]
                    state["ambiguous"] = res["ambiguous"]
                    # Track recently merged so the dropdown can boost this name.
                    rm = state.setdefault("recently_merged", [])
                    if dest_name not in rm:
                        rm.insert(0, dest_name)
                    # Close and notify BEFORE refresh — refresh clears suggest_container
                    # which deletes the suggestion card and this dialog along with it.
                    # UI ops on deleted elements raise RuntimeError.
                    dlg.close()
                    ui.notify(f"Merged into '{dest_name}'", color="positive")
                    if "refresh_suggestions" in state:
                        state["refresh_suggestions"]()
                    if "refresh_categorized" in state:
                        state["refresh_categorized"]()
                except Exception as e:
                    ui.notify(f"Merge failed: {e}", color="negative")
                    dlg.close()
                finally:
                    _in_progress[0] = False

            confirm_btn = ui.button("Confirm & merge", on_click=confirm).props("color=primary")

    dlg.open()


def _last_undoable_merge() -> dict | None:
    from datetime import timedelta

    cutoff = utcnow() - timedelta(minutes=30)
    with get_session() as s:
        log = s.execute(
            select(MergeLog)
            .where(
                MergeLog.status == "done",
                MergeLog.file_map.isnot(None),
                MergeLog.completed_at >= cutoff,
            )
            .order_by(MergeLog.completed_at.desc())
            .limit(1)
        ).scalar_one_or_none()
        if log is None:
            return None
        return {
            "id": log.id,
            "dest": log.dest,
            "sources": json.loads(log.sources),
            "deleted_count": sum(1 for v in json.loads(log.file_map).values() if v is None),
        }


def _show_undo_confirm(merge: dict, state: dict, container):
    from . import face as face_mod

    sources_str = ", ".join(s.removeprefix("categorized/") for s in merge["sources"])
    dest_str = merge["dest"].removeprefix("categorized/")
    deleted = merge["deleted_count"]

    with ui.dialog() as dlg, ui.card():
        ui.label(f"Undo merge into '{dest_str}'?").classes("font-semibold")
        ui.label(f"Files from: {sources_str}").classes("text-sm text-gray-500 mt-1")
        if deleted:
            ui.label(
                f"{deleted} file(s) deleted as duplicates during merge cannot be restored."
            ).classes("text-xs text-orange-500 mt-1")
        ui.label("This will move all merged files back to their original folders.").classes(
            "text-sm mt-2"
        )

        async def do_undo():
            dlg.close()
            loop = asyncio.get_running_loop()
            error = await loop.run_in_executor(None, lambda: face_mod.undo_merge(merge["id"]))
            if error:
                ui.notify(f"Undo failed: {error}", color="negative")
            else:
                ui.notify(f"Undone — '{dest_str}' restored to original folders", color="positive")
                state["suggestions"] = []
                state["ambiguous"] = []
                rm = state.get("recently_merged", [])
                if dest_str in rm:
                    rm.remove(dest_str)
                if "refresh_suggestions" in state:
                    state["refresh_suggestions"]()
                if "refresh_categorized" in state:
                    state["refresh_categorized"]()

        with ui.row().classes("mt-3 gap-2 justify-end"):
            ui.button("Cancel", on_click=dlg.close).props("flat")
            ui.button("Undo merge", on_click=do_undo).props("color=negative")
    dlg.open()


def _render_categorized(container, state: dict | None = None):
    from . import face as face_mod

    people = _list_people()

    if not people:
        with ui.column().classes("items-center gap-2 py-8"):
            ui.icon("folder_off").classes("text-4xl text-gray-300")
            ui.label("No categorized people yet.").classes("text-gray-500")
            ui.label("Merge suggestions in the Suggestions tab to create folders.").classes(
                "text-sm text-gray-400"
            )
        return

    # Pre-load all person tags in one query (avoids N DB round-trips per filter event)
    person_tags: dict[str, list[str]] = face_mod.get_all_tags("person")

    merge_state: dict[str, Any] = {"selected": [], "mode": False}
    last_merge = _last_undoable_merge()

    def enter_merge_mode():
        merge_state["mode"] = True
        merge_state["selected"] = []
        merge_bar.set_visibility(True)
        merge_btn.set_visibility(False)
        container.update()
        ui.run_javascript("document.getElementById('cat-grid').dataset.merge='1'")

    def cancel_merge_mode():
        merge_state["mode"] = False
        merge_state["selected"] = []
        merge_bar.set_visibility(False)
        merge_btn.set_visibility(True)
        ui.run_javascript("delete document.getElementById('cat-grid').dataset.merge")

    with ui.row().classes("items-center justify-between w-full mb-2 gap-2"):
        ui.label(f"{len(people)} people").classes("text-sm text-gray-500 shrink-0")
        filter_input = ui.input(placeholder="Filter...").classes("flex-1")
        tag_filter = ui.select(
            face_mod.list_all_tags("person"),
            multiple=True,
            label="Tags",
            clearable=True,
        ).classes("w-40")
        with ui.row().classes("gap-2 shrink-0"):
            if last_merge:
                ui.button(
                    "Undo last merge",
                    icon="undo",
                    on_click=lambda: _show_undo_confirm(last_merge, state or {}, container),
                ).props("flat color=negative")
            merge_btn = ui.button("Merge people", icon="merge", on_click=enter_merge_mode).props(
                "flat"
            )

    with ui.row().classes("items-center gap-2 w-full bg-blue-50 p-2 rounded") as merge_bar:
        merge_status_label = ui.label("Select two people to merge").classes("text-sm flex-1")
        ui.button("Cancel", on_click=cancel_merge_mode).props("flat dense")

    merge_bar.set_visibility(False)

    person_cards: dict[str, Any] = {}

    with ui.grid(columns=3).classes("w-full gap-4").props("id=cat-grid"):
        for person in people:
            thumbs = _person_thumbs(person, n=4)
            count = _person_file_count(person)
            entity_rel = f"categorized/{person}"

            with ui.card().classes(
                "w-full cursor-pointer hover:shadow-md transition-shadow relative"
            ) as card:
                person_cards[person] = card

                # Animated GIF thumbnail with fallback to first individual thumb
                thumb_url = face_mod.get_gif_url(entity_rel, thumbs)
                if thumb_url:
                    ui.image(thumb_url).classes("w-full aspect-square object-cover rounded").props(
                        "loading=lazy"
                    )
                else:
                    ui.element("div").classes("w-full aspect-square bg-gray-100 rounded")

                # Clickable overlay — covers the whole card; prevented in merge mode
                ui.html(
                    f'<a href="/gallery/{quote(person)}" '
                    f'class="absolute inset-0 z-0" '
                    f"onclick=\"if(this.closest('[data-merge]')){{event.preventDefault();}}\""
                    f"></a>"
                )

                ui.label(person).classes("font-semibold mt-2 truncate")
                ui.label(f"{count} file{'s' if count != 1 else ''}").classes(
                    "text-xs text-gray-400"
                )

                # Read-only tag chips — clicking propagates to card for merge-mode selection
                with ui.row().classes("gap-1 flex-wrap items-center mt-1 relative z-10"):
                    for t in person_tags.get(person, []):
                        ui.chip(t).props("dense outline").classes("text-xs")

                def make_select_handler(p):
                    def on_click():
                        if not merge_state["mode"]:
                            return  # <a> overlay handles navigation
                        if p in merge_state["selected"]:
                            merge_state["selected"].remove(p)
                        elif len(merge_state["selected"]) < 2:
                            merge_state["selected"].append(p)

                        sel = merge_state["selected"]
                        if len(sel) == 0:
                            merge_status_label.set_text("Select two people to merge")
                        elif len(sel) == 1:
                            merge_status_label.set_text(f"Selected: {sel[0]} — select one more")
                        else:
                            merge_status_label.set_text(f"Merge {sel[0]} → {sel[1]}")
                            _show_merge_people_dialog(
                                sel[0], sel[1], container, cancel_merge_mode, state
                            )

                    return on_click

                card.on("click", make_select_handler(person))

    async def on_filter_change():
        await asyncio.sleep(0.15)
        query = filter_input.value.strip().casefold()
        selected_tags = tag_filter.value or []
        if state is not None:
            state["cat_filter"] = filter_input.value
            state["cat_tag_filter"] = selected_tags
        for _p, _card in person_cards.items():
            name_ok = not query or query in _p.casefold()
            tags_ok = not selected_tags or all(t in person_tags.get(_p, []) for t in selected_tags)
            _card.set_visibility(name_ok and tags_ok)
        if merge_state["mode"]:
            merge_state["selected"] = []
            merge_status_label.set_text("Select two people to merge")

    filter_input.on("update:model-value", lambda: asyncio.ensure_future(on_filter_change()))
    tag_filter.on("update:model-value", lambda: asyncio.ensure_future(on_filter_change()))

    if state is not None and (saved_filter := state.get("cat_filter")):
        filter_input.set_value(saved_filter)
        asyncio.ensure_future(on_filter_change())
    if state is not None and (saved_tags := state.get("cat_tag_filter")):
        tag_filter.set_value(saved_tags)
        asyncio.ensure_future(on_filter_change())


def _show_add_more(person: str, container, state: dict | None = None):
    """Show folder picker from downloads tree (excluding categorized/ and collections/)."""
    from . import face as face_mod

    cat_dir = os.path.abspath(_categorized_dir())
    coll_dir = os.path.abspath(face_mod._collections_dir())

    # Build folder list
    source_folders = []
    for root, dirs, _ in os.walk(DOWNLOAD_DIR):
        abs_root = os.path.abspath(root)
        if abs_root == cat_dir or abs_root.startswith(cat_dir + os.sep):
            dirs.clear()
            continue
        if abs_root == coll_dir or abs_root.startswith(coll_dir + os.sep):
            dirs.clear()
            continue
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        if abs_root != os.path.abspath(DOWNLOAD_DIR):
            source_folders.append(os.path.relpath(abs_root, DOWNLOAD_DIR))

    source_folders.sort(key=str.casefold)

    with ui.dialog() as dlg, ui.card().classes("w-full max-w-lg"):
        ui.label(f"Add more to '{person}'").classes("font-semibold mb-2")
        if not source_folders:
            ui.label("No unorganized folders found.").classes("text-sm text-gray-500")
            ui.button("Close", on_click=dlg.close).props("flat")
        else:
            folder_select = ui.select(source_folders, label="Choose folder").classes("w-full")

            def confirm():
                chosen = folder_select.value
                if not chosen:
                    ui.notify("Select a folder", color="negative")
                    return

                sources = [chosen]
                dlg.close()
                _run_merge_with_preview(sources, person, state or {}, container)

            with ui.row().classes("mt-3 gap-2 justify-end"):
                ui.button("Cancel", on_click=dlg.close).props("flat")
                ui.button("Next", on_click=confirm).props("color=primary")

    dlg.open()


def _show_merge_people_dialog(
    person_a: str, person_b: str, container, on_cancel, state: dict | None = None
):
    with ui.dialog() as dlg, ui.card().classes("w-full max-w-2xl"):
        ui.label(f"Merge {person_a} → {person_b}").classes("font-semibold mb-2")

        with ui.row().classes("w-full gap-4"):
            with ui.column().classes("flex-1"):
                ui.label(person_a).classes("font-semibold")
                for t in _person_thumbs(person_a, 2):
                    ui.image(t).classes("w-24 h-24 object-cover rounded")
                ui.label(f"{_person_file_count(person_a)} files").classes("text-xs text-gray-400")
            ui.icon("arrow_forward").classes("self-center text-2xl")
            with ui.column().classes("flex-1"):
                ui.label(person_b).classes("font-semibold")
                for t in _person_thumbs(person_b, 2):
                    ui.image(t).classes("w-24 h-24 object-cover rounded")
                ui.label(f"{_person_file_count(person_b)} files").classes("text-xs text-gray-400")

        dest_select = ui.select(
            {person_a: person_a, person_b: person_b},
            value=person_b,
            label="Merge into",
        ).classes("w-full mt-2")

        def confirm():
            chosen_dest = dest_select.value
            src = person_a if chosen_dest == person_b else person_b
            sources = [f"categorized/{src}"]
            dlg.close()
            on_cancel()
            _run_merge_with_preview(sources, chosen_dest, state or {}, container)

        with ui.row().classes("mt-3 gap-2 justify-end"):
            ui.button("Cancel", on_click=lambda: [dlg.close(), on_cancel()]).props("flat")
            ui.button("Confirm & merge", on_click=confirm).props("color=primary")

    dlg.open()


def _build_settings_tab():
    ui.label("Face recognition settings").classes("text-lg font-semibold mb-2")

    current_model = get_setting("face_model", "buffalo_s")
    current_threshold = float(get_setting("face_threshold", "0.4"))
    current_phash = int(get_setting("phash_threshold", "8"))
    current_min_images = int(get_setting("min_images_per_folder", "3"))
    current_mini_scan = get_setting("mini_scan_enabled", "true") == "true"

    model_warning = ui.label("").classes("text-xs text-orange-500")

    def on_model_change(e):
        new_model = e.value
        old_model = get_scan_meta("model")
        set_setting("face_model", new_model)
        if old_model and old_model != new_model:
            model_warning.set_text(f"Model changed from {old_model} — re-scan required to apply")
        else:
            model_warning.set_text("")

    ui.select(
        {
            "buffalo_l": "buffalo_l (large — best accuracy)",
            "buffalo_s": "buffalo_s (small — fastest)",
        },
        value=current_model,
        label="Face model",
        on_change=on_model_change,
    ).classes("w-full")
    model_warning

    with ui.row().classes("items-center gap-4 w-full mt-3"):
        ui.label("Face similarity threshold").classes("text-sm w-48")
        threshold_label = ui.label(f"{current_threshold:.2f}").classes("text-sm w-12")

    def on_threshold_change(e):
        set_setting("face_threshold", str(round(e.value, 2)))
        threshold_label.set_text(f"{e.value:.2f}")

    ui.slider(
        min=0.1, max=0.9, step=0.05, value=current_threshold, on_change=on_threshold_change
    ).classes("w-full")

    with ui.row().classes("items-center gap-4 w-full mt-2"):
        ui.label("Perceptual hash threshold").classes("text-sm w-48")
        phash_label = ui.label(f"{current_phash} / 64 bits").classes("text-sm w-20")

    def on_phash_change(e):
        set_setting("phash_threshold", str(int(e.value)))
        phash_label.set_text(f"{int(e.value)} / 64 bits")

    ui.slider(min=0, max=20, step=1, value=current_phash, on_change=on_phash_change).classes(
        "w-full"
    )

    with ui.row().classes("items-center gap-4 w-full mt-2"):
        ui.label("Min images per folder").classes("text-sm w-48")
        min_images_label = ui.label(f"{current_min_images}").classes("text-sm w-12")

    def on_min_images_change(e):
        set_setting("min_images_per_folder", str(int(e.value)))
        min_images_label.set_text(str(int(e.value)))

    ui.slider(
        min=1, max=20, step=1, value=current_min_images, on_change=on_min_images_change
    ).classes("w-full")

    def on_mini_scan_change(e):
        set_setting("mini_scan_enabled", "true" if e.value else "false")

    ui.switch(
        "Mini-scan after download",
        value=current_mini_scan,
        on_change=on_mini_scan_change,
    ).classes("mt-3")
    ui.label("Automatically scans new downloads and checks if they match a known person.").classes(
        "text-xs text-gray-400 ml-8"
    )

    ui.separator().classes("my-4")
    ui.label("Suggestions").classes("text-lg font-semibold mb-2")

    current_page_size = int(get_setting("suggestions_page_size", "10"))

    with ui.row().classes("items-center gap-4 w-full mt-2"):
        ui.label("Suggestions per page").classes("text-sm w-48")
        page_size_label = ui.label(f"{current_page_size}").classes("text-sm w-12")

    def on_page_size_change(e):
        val = int(e.value)
        set_setting("suggestions_page_size", str(val))
        page_size_label.set_text(str(val))

    ui.slider(
        min=5, max=50, step=5, value=current_page_size, on_change=on_page_size_change
    ).classes("w-full")

    if PASSWORD:
        ui.separator().classes("my-4")
        ui.label("Security").classes("text-lg font-semibold mb-2")

        current_auto_logout = int(get_setting("auto_logout_minutes", "15"))

        with ui.row().classes("items-center gap-4 w-full mt-2"):
            ui.label("Auto-logout after inactivity").classes("text-sm w-48")
            auto_logout_label = ui.label(
                f"{current_auto_logout} min" if current_auto_logout > 0 else "disabled"
            ).classes("text-sm w-20")

        def on_auto_logout_change(e):
            val = int(e.value)
            set_setting("auto_logout_minutes", str(val))
            auto_logout_label.set_text(f"{val} min" if val > 0 else "disabled")

        ui.slider(
            min=0, max=120, step=5, value=current_auto_logout, on_change=on_auto_logout_change
        ).classes("w-full")
        ui.label("Set to 0 to disable. Change takes effect on next page load.").classes(
            "text-xs text-gray-400"
        )


# ── Collections ───────────────────────────────────────────────────────────────


def _collection_card_info(collection_name: str, thumb_n: int = 4) -> tuple[int, list[str]]:
    """Return (file_count, thumb_url_list) in one os.listdir pass. Generates missing thumbs."""
    import hashlib

    from . import face as face_mod

    folder = os.path.join(face_mod._collections_dir(), collection_name)
    if not os.path.isdir(folder):
        return 0, []
    exts = face_mod._MEDIA_EXTS
    try:
        entries = sorted(os.listdir(folder))
    except OSError:
        return 0, []
    media_files = [
        f
        for f in entries
        if os.path.isfile(os.path.join(folder, f)) and os.path.splitext(f)[1].lower() in exts
    ]
    count = len(media_files)
    thumbs = []
    for fname in media_files[:thumb_n]:
        rel = f"collections/{collection_name}/{fname}"
        thumb_path = face_mod._thumb_path(rel)
        if not os.path.exists(thumb_path):
            abs_path = os.path.join(folder, fname)
            ext = os.path.splitext(fname)[1].lower()
            try:
                if ext in face_mod._IMAGE_EXTS:
                    face_mod._generate_thumb(abs_path, rel)
                else:
                    face_mod._generate_video_thumb(abs_path, rel)
            except Exception:
                pass
        thumbs.append(f"/thumbs/{hashlib.sha256(rel.encode()).hexdigest()}.jpg")
    return count, thumbs


def _render_collections(container, state: dict | None = None) -> None:
    from . import face as face_mod

    collections = face_mod.list_collections()

    if not collections:
        with ui.column().classes("items-center gap-2 py-8"):
            ui.icon("collections").classes("text-4xl text-gray-300")
            ui.label("No collections yet.").classes("text-gray-500")
            ui.label("Use 'Add to collection' on a suggestion card to create one.").classes(
                "text-sm text-gray-400"
            )
        return

    # Pre-load all collection tags in one query
    coll_tags: dict[str, list[str]] = face_mod.get_all_tags("collection")

    with ui.row().classes("items-center justify-between w-full mb-2 gap-2"):
        ui.label(f"{len(collections)} collection{'s' if len(collections) != 1 else ''}").classes(
            "text-sm text-gray-500 shrink-0"
        )
        coll_filter_input = ui.input(placeholder="Filter...").classes("flex-1")
        coll_tag_filter = ui.select(
            face_mod.list_all_tags("collection"),
            multiple=True,
            label="Tags",
            clearable=True,
        ).classes("w-40")

    coll_cards: dict[str, Any] = {}

    with ui.grid(columns=3).classes("w-full gap-4"):
        for coll_name in collections:
            try:
                count, thumbs = _collection_card_info(coll_name)
            except OSError:
                continue  # collection deleted externally — skip card

            entity_rel = f"collections/{coll_name}"

            with ui.card().classes(
                "w-full cursor-pointer hover:shadow-md transition-shadow relative"
            ) as card:
                coll_cards[coll_name] = card

                # Animated GIF thumbnail with fallback to first individual thumb
                thumb_url = face_mod.get_gif_url(entity_rel, thumbs)
                if thumb_url:
                    ui.image(thumb_url).classes("w-full aspect-square object-cover rounded").props(
                        "loading=lazy"
                    )
                else:
                    ui.element("div").classes("w-full aspect-square bg-gray-100 rounded")

                # Clickable overlay — whole card navigates to collection gallery
                ui.html(
                    f'<a href="/collections/{quote(coll_name)}" class="absolute inset-0 z-0"></a>'
                )

                ui.label(coll_name).classes("font-semibold mt-2 truncate")
                ui.label(f"{count} file{'s' if count != 1 else ''}").classes(
                    "text-xs text-gray-400"
                )

                # Read-only tag chips (no click.stop — collections have no merge mode)
                with ui.row().classes("gap-1 flex-wrap items-center mt-1 relative z-10"):
                    for t in coll_tags.get(coll_name, []):
                        ui.chip(t).props("dense outline").classes("text-xs")

    async def on_coll_filter_change():
        await asyncio.sleep(0.15)
        query = coll_filter_input.value.strip().casefold()
        selected_tags = coll_tag_filter.value or []
        if state is not None:
            state["coll_filter"] = coll_filter_input.value
            state["coll_tag_filter"] = selected_tags
        for _c, _card in coll_cards.items():
            name_ok = not query or query in _c.casefold()
            tags_ok = not selected_tags or all(t in coll_tags.get(_c, []) for t in selected_tags)
            _card.set_visibility(name_ok and tags_ok)

    coll_filter_input.on(
        "update:model-value", lambda: asyncio.ensure_future(on_coll_filter_change())
    )
    coll_tag_filter.on("update:model-value", lambda: asyncio.ensure_future(on_coll_filter_change()))

    if state is not None and (sv := state.get("coll_filter")):
        coll_filter_input.set_value(sv)
    if state is not None and (sv := state.get("coll_tag_filter")):
        coll_tag_filter.set_value(sv)
    asyncio.ensure_future(on_coll_filter_change())


def _show_add_more_to_collection(
    coll_name: str,
    container,
    state: dict | None = None,
    on_success: Callable | None = None,
) -> None:
    """Folder picker that moves an unorganized folder into an existing collection."""
    from . import face as face_mod

    cat_dir = os.path.abspath(_categorized_dir())
    coll_dir = os.path.abspath(face_mod._collections_dir())

    source_folders = []
    for root, dirs, _ in os.walk(DOWNLOAD_DIR):
        abs_root = os.path.abspath(root)
        if abs_root == cat_dir or abs_root.startswith(cat_dir + os.sep):
            dirs.clear()
            continue
        if abs_root == coll_dir or abs_root.startswith(coll_dir + os.sep):
            dirs.clear()
            continue
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        if abs_root != os.path.abspath(DOWNLOAD_DIR):
            source_folders.append(os.path.relpath(abs_root, DOWNLOAD_DIR))

    source_folders.sort(key=str.casefold)

    with ui.dialog() as dlg, ui.card().classes("w-full max-w-lg"):
        ui.label(f"Add more to '{coll_name}'").classes("font-semibold mb-2")
        if not source_folders:
            ui.label("No unorganized folders found.").classes("text-sm text-gray-500")
            ui.button("Close", on_click=dlg.close).props("flat")
        else:
            folder_select = ui.select(source_folders, label="Choose folder").classes("w-full")
            err_lbl = ui.label("").classes("text-xs text-red-500 mt-1")
            _add_btn_ref: list = [None]
            _adding: list = [False]

            async def _confirm():
                if _adding[0]:
                    return
                chosen = folder_select.value
                if not chosen:
                    err_lbl.set_text("Select a folder first")
                    return
                _adding[0] = True
                if _add_btn_ref[0]:
                    _add_btn_ref[0].set_enabled(False)
                dlg.close()
                try:
                    loop = asyncio.get_running_loop()
                    err = await loop.run_in_executor(
                        None, face_mod.move_to_collection, chosen, coll_name
                    )
                    if err:
                        ui.notify(f"Error: {err}", color="negative")
                    else:
                        ui.notify(f"Added '{chosen}' to '{coll_name}'", color="positive")
                        if on_success is not None:
                            on_success()
                        elif container is not None:
                            if state is not None:
                                res = await loop.run_in_executor(
                                    None, face_mod.analyse_downloads, DOWNLOAD_DIR
                                )
                                state["suggestions"] = res["suggestions"]
                                state["ambiguous"] = res["ambiguous"]
                            container.clear()
                            with container:
                                _render_collections(container, state)
                finally:
                    _adding[0] = False

            with ui.row().classes("mt-3 gap-2 justify-end"):
                ui.button("Cancel", on_click=dlg.close).props("flat")
                _add_btn_ref[0] = ui.button("Add", on_click=_confirm).props("color=primary")

    dlg.open()


def _show_add_to_collection_dialog(source_folders: list[str], state: dict) -> None:
    from . import face as face_mod

    existing = face_mod.list_collections()
    _NEW = "__new__"

    with ui.dialog() as dlg, ui.card().classes("w-full max-w-md"):
        ui.label("Add to collection").classes("text-lg font-semibold mb-2")
        ui.label(f"Move {len(source_folders)} folder(s) into a collection:").classes(
            "text-sm text-gray-500 mb-2"
        )
        for f in source_folders:
            ui.label(f"• {os.path.basename(f) or f}").classes("text-xs text-gray-400")

        # Dropdown: existing collections + sentinel for creating a new one.
        if existing:
            options = {_NEW: "＋ New collection"} | {c: c for c in existing}
            collection_select = ui.select(options, value=_NEW, label="Collection").classes(
                "w-full mt-3"
            )
        else:
            collection_select = None

        new_name_input = ui.input("New collection name").classes("w-full mt-2")

        if collection_select is not None:

            def on_select_change(e):
                val = e.args if isinstance(e.args, str) else collection_select.value
                new_name_input.set_visibility(val == _NEW)

            collection_select.on("update:model-value", on_select_change)

        err_lbl = ui.label("").classes("text-xs text-red-500 mt-1")
        spinner = ui.spinner().classes("mt-2")
        spinner.set_visibility(False)

        confirm_btn_ref: list = [None]
        _in_progress: list = [False]

        async def do_add():
            if _in_progress[0]:
                return
            if collection_select is not None and collection_select.value != _NEW:
                collection_name = collection_select.value
            else:
                collection_name = new_name_input.value.strip()

            if not collection_name:
                err_lbl.set_text("Collection name cannot be empty")
                return

            _in_progress[0] = True
            if confirm_btn_ref[0]:
                confirm_btn_ref[0].set_enabled(False)
            spinner.set_visibility(True)
            err_lbl.set_text("")

            try:
                loop = asyncio.get_running_loop()
                errors = []
                succeeded = 0
                for src in source_folders:
                    err = await loop.run_in_executor(
                        None, face_mod.move_to_collection, src, collection_name
                    )
                    if err:
                        errors.append(f"{src}: {err}")
                    else:
                        succeeded += 1

                if succeeded > 0:
                    res = await loop.run_in_executor(None, face_mod.analyse_downloads, DOWNLOAD_DIR)
                    state["suggestions"] = res["suggestions"]
                    state["ambiguous"] = res["ambiguous"]

                dlg.close()

                if errors:
                    ui.notify("Some folders failed: " + "; ".join(errors), color="negative")
                else:
                    ui.notify(
                        f"Added {len(source_folders)} folder(s) to '{collection_name}'",
                        color="positive",
                    )

                if succeeded > 0:
                    if "refresh_suggestions" in state:
                        state["refresh_suggestions"]()
                    if "refresh_collections" in state:
                        state["refresh_collections"]()
            finally:
                _in_progress[0] = False

        with ui.row().classes("mt-4 gap-2 justify-end"):
            ui.button("Cancel", on_click=dlg.close).props("flat")
            confirm_btn_ref[0] = ui.button("Add to collection", on_click=do_add).props(
                "color=primary"
            )

    dlg.open()


# ── Gallery page ──────────────────────────────────────────────────────────────


def _delete_gallery_file(person: str, rel_path: str) -> str | None:
    """Delete a gallery file from disk and DB. Returns error string or None."""
    from . import face as face_mod

    cat_abs = os.path.abspath(_categorized_dir())
    person_abs = os.path.abspath(os.path.join(cat_abs, person))
    if not person_abs.startswith(cat_abs + os.sep):
        return "Invalid person name"

    abs_path = os.path.abspath(os.path.join(DOWNLOAD_DIR, rel_path))
    if not abs_path.startswith(person_abs + os.sep):
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

    with get_session() as s:
        s.execute(delete(ImageMeta).where(ImageMeta.file_path == rel_path))
        s.execute(delete(FaceEmbedding).where(FaceEmbedding.file_path == rel_path))
        s.commit()

    try:
        face_mod._recompute_centroids(get_setting("face_model", "buffalo_s"))
    except Exception:
        pass

    try:
        face_mod.regenerate_person_gif(person)
    except Exception:
        pass

    return None


def _render_gallery_grid(person: str, images: list[dict], container, count_lbl=None) -> None:
    container.clear()
    with container:
        if not images:
            ui.label("No files remaining.").classes("text-gray-500 py-8")
            return

        _thumb_style = "width:100%;aspect-ratio:1;object-fit:cover;border-radius:6px;cursor:pointer"
        _grid_style = (
            "display:grid;"
            "grid-template-columns:repeat(auto-fill,minmax(160px,1fr));"
            "gap:8px;width:100%"
        )

        with ui.element("div").style(_grid_style).props("id=lg-gallery"):
            for img in images:
                thumb_html = f'<img src="{img["thumb"]}" style="{_thumb_style}" />'
                if img.get("is_video"):
                    video_data = json.dumps(
                        {
                            "source": [{"src": img["src"], "type": img["video_type"]}],
                            "attributes": {"preload": "none", "controls": True},
                        }
                    )
                    safe_vd = _html.escape(video_data)
                    item_html = f'<a class="lg-item" data-video="{safe_vd}">{thumb_html}</a>'
                else:
                    item_html = f'<a href="{img["src"]}" class="lg-item">{thumb_html}</a>'

                with ui.element("div").classes("relative group"):
                    ui.html(item_html)

                    def make_delete_handler(rel=img["rel"]):
                        async def on_delete():
                            with ui.dialog() as dlg, ui.card():
                                ui.label(f"Delete '{os.path.basename(rel)}'?").classes(
                                    "font-semibold"
                                )
                                ui.label("This cannot be undone.").classes(
                                    "text-sm text-gray-500 mt-1"
                                )

                                async def do_delete():
                                    dlg.close()
                                    await ui.run_javascript(
                                        "if (window.lgInstance) {"
                                        "  window.lgInstance.destroy();"
                                        "  window.lgInstance = null;"
                                        "}"
                                    )
                                    loop = asyncio.get_running_loop()
                                    err = await loop.run_in_executor(
                                        None, _delete_gallery_file, person, rel
                                    )
                                    if err:
                                        ui.notify(f"Delete failed: {err}", color="negative")
                                    else:
                                        ui.notify("File deleted", color="positive")
                                    new_imgs = _person_images(person)
                                    _render_gallery_grid(person, new_imgs, container, count_lbl)
                                    if count_lbl is not None:
                                        n = len(new_imgs)
                                        count_lbl.set_text(f"({n} file{'s' if n != 1 else ''})")
                                    await ui.run_javascript(_LG_INIT_JS)

                                with ui.row().classes("mt-3 gap-2 justify-end"):
                                    ui.button("Cancel", on_click=dlg.close).props("flat")
                                    ui.button("Delete", on_click=do_delete).props("color=negative")
                            dlg.open()

                        return on_delete

                    (
                        ui.button(icon="delete_outline", on_click=make_delete_handler())
                        .props("flat round dense size=xs color=negative")
                        .classes(
                            "absolute top-1 right-1 opacity-0"
                            " group-hover:opacity-100 transition-opacity"
                        )
                        .style("background: rgba(255,255,255,0.85)")
                    )


@ui.page("/gallery/{person}")
def gallery_page(person: str):
    if not check_auth():
        ui.navigate.to("/login")
        return

    # Path traversal guard
    cat_abs = os.path.abspath(_categorized_dir())
    person_abs = os.path.abspath(os.path.join(cat_abs, person))
    if not person_abs.startswith(cat_abs + os.sep):
        ui.notify("Invalid person name", color="negative")
        ui.navigate.to("/organize?tab=categorized")
        return

    ui.page_title(f"{person} — Gallery")
    ui.add_body_html(_auto_logout_js())
    dark = ui.dark_mode()
    dark.set_value(app.storage.user.get("dark_mode", False))

    def _toggle_dark(dark=dark):
        new_val = not dark.value
        dark.set_value(new_val)
        app.storage.user["dark_mode"] = new_val

    # CDN scripts go in <head> so they execute (innerHTML scripts are blocked by browsers)
    ui.add_head_html(
        '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/lightgallery@2/css/lightgallery-bundle.min.css">'
    )
    ui.add_head_html(
        '<script src="https://cdn.jsdelivr.net/npm/lightgallery@2/lightgallery.umd.min.js"></script>'
    )
    ui.add_head_html(
        '<script src="https://cdn.jsdelivr.net/npm/lightgallery@2/plugins/zoom/lg-zoom.umd.min.js"></script>'
    )
    ui.add_head_html(
        '<script src="https://cdn.jsdelivr.net/npm/lightgallery@2/plugins/thumbnail/lg-thumbnail.umd.min.js"></script>'
    )
    ui.add_head_html(
        '<script src="https://cdn.jsdelivr.net/npm/lightgallery@2/plugins/video/lg-video.umd.min.js"></script>'
    )
    ui.add_head_html(f"<script>{_LG_INIT_JS}</script>")

    from . import face as face_mod

    images = _person_images(person)

    def _do_add_more():
        gallery_state = {"refresh_categorized": lambda: ui.navigate.to(f"/gallery/{quote(person)}")}
        _show_add_more(person, None, gallery_state)

    def _do_rename():
        with ui.dialog() as dlg, ui.card():
            ui.label(f"Rename '{person}'").classes("font-semibold")
            inp = ui.input("New name", value=person).classes("w-full mt-2")
            err_lbl = ui.label("").classes("text-xs text-red-500")

            def confirm():
                new = inp.value.strip()
                if new == person:
                    dlg.close()
                    return
                e = _validate_name(new)
                if e:
                    err_lbl.set_text(e)
                    return
                result = face_mod.rename_person(person, new)
                if result:
                    err_lbl.set_text(result)
                else:
                    dlg.close()
                    ui.navigate.to(f"/gallery/{quote(new)}")

            with ui.row().classes("mt-3 gap-2 justify-end"):
                ui.button("Cancel", on_click=dlg.close).props("flat")
                ui.button("Rename", on_click=confirm).props("color=primary")
        dlg.open()

    with ui.column().classes("w-full max-w-6xl mx-auto p-4 gap-4"):
        with ui.row().classes("items-center gap-3 w-full"):
            ui.button(
                "← Back",
                icon="arrow_back",
                on_click=lambda: ui.navigate.to("/organize?tab=categorized"),
            ).props("flat dense")
            ui.label(person).classes("text-xl font-bold flex-1")
            count_lbl = ui.label(f"({len(images)} file{'s' if len(images) != 1 else ''})").classes(
                "text-sm text-gray-400"
            )
            ui.button("Add more", icon="add", on_click=_do_add_more).props("flat dense")
            ui.button("Rename", icon="edit", on_click=_do_rename).props("flat dense")
            ui.button(icon="dark_mode", on_click=_toggle_dark).props("flat round").tooltip(
                "Toggle dark mode"
            )

        with ui.row().classes("items-center gap-2 flex-wrap mb-2"):
            ui.label("Tags:").classes("text-sm text-gray-500 shrink-0")
            tag_box = ui.row().classes("gap-1 flex-wrap items-center")

        def _render_person_tags():
            tag_box.clear()
            with tag_box:
                for t in face_mod.get_tags("person", person):
                    chip = ui.chip(t).props("dense outline removable")

                    def _make_rm(_t=t):
                        def _():
                            face_mod.remove_tag("person", person, _t)
                            _render_person_tags()

                        return _

                    chip.on("remove", _make_rm())

                tag_inp = ui.input(placeholder="+ tag").classes("w-24 text-sm")

                def _add(_, inp=tag_inp):
                    val = inp.value.strip().lower()
                    if val:
                        face_mod.add_tag("person", person, val)
                        inp.set_value("")
                        _render_person_tags()

                tag_inp.on("keydown.enter", _add)

        _render_person_tags()

        if not images:
            ui.label("No files found.").classes("text-gray-500 py-8")
            return

        gallery_container = ui.column().classes("w-full")
        _render_gallery_grid(person, images, gallery_container, count_lbl)


# ── Collection gallery page ────────────────────────────────────────────────────


def _render_collection_gallery_grid(
    name: str, images: list[dict], container, count_lbl=None
) -> None:
    from . import face as face_mod

    container.clear()
    with container:
        if not images:
            ui.label("No files remaining.").classes("text-gray-500 py-8")
            return

        _thumb_style = "width:100%;aspect-ratio:1;object-fit:cover;border-radius:6px;cursor:pointer"
        _grid_style = (
            "display:grid;"
            "grid-template-columns:repeat(auto-fill,minmax(160px,1fr));"
            "gap:8px;width:100%"
        )

        with ui.element("div").style(_grid_style).props("id=lg-gallery"):
            for img in images:
                thumb_html = f'<img src="{img["thumb"]}" style="{_thumb_style}" />'
                if img.get("is_video"):
                    video_data = json.dumps(
                        {
                            "source": [{"src": img["src"], "type": img["video_type"]}],
                            "attributes": {"preload": "none", "controls": True},
                        }
                    )
                    safe_vd = _html.escape(video_data)
                    item_html = f'<a class="lg-item" data-video="{safe_vd}">{thumb_html}</a>'
                else:
                    item_html = f'<a href="{img["src"]}" class="lg-item">{thumb_html}</a>'

                with ui.element("div").classes("relative group"):
                    ui.html(item_html)

                    def make_delete_handler(rel=img["rel"]):
                        async def on_delete():
                            with ui.dialog() as dlg, ui.card():
                                ui.label(f"Delete '{os.path.basename(rel)}'?").classes(
                                    "font-semibold"
                                )
                                ui.label("This cannot be undone.").classes(
                                    "text-sm text-gray-500 mt-1"
                                )

                                async def do_delete():
                                    dlg.close()
                                    await ui.run_javascript(
                                        "if (window.lgInstance) {"
                                        "  window.lgInstance.destroy();"
                                        "  window.lgInstance = null;"
                                        "}"
                                    )
                                    loop = asyncio.get_running_loop()
                                    err = await loop.run_in_executor(
                                        None,
                                        face_mod.delete_collection_file,
                                        name,
                                        rel,
                                    )
                                    if err:
                                        ui.notify(f"Delete failed: {err}", color="negative")
                                    else:
                                        ui.notify("File deleted", color="positive")
                                    new_imgs = face_mod._collection_images(name)
                                    _render_collection_gallery_grid(
                                        name, new_imgs, container, count_lbl
                                    )
                                    if count_lbl is not None:
                                        n = len(new_imgs)
                                        count_lbl.set_text(f"({n} file{'s' if n != 1 else ''})")
                                    if new_imgs:
                                        await ui.run_javascript(_LG_INIT_JS)

                                with ui.row().classes("mt-3 gap-2 justify-end"):
                                    ui.button("Cancel", on_click=dlg.close).props("flat")
                                    ui.button("Delete", on_click=do_delete).props("color=negative")
                            dlg.open()

                        return on_delete

                    (
                        ui.button(icon="delete_outline", on_click=make_delete_handler())
                        .props("flat round dense size=xs color=negative")
                        .classes(
                            "absolute top-1 right-1 opacity-0"
                            " group-hover:opacity-100 transition-opacity"
                        )
                        .style("background: rgba(255,255,255,0.85)")
                    )


@ui.page("/collections/{name}")
def collections_gallery_page(name: str):
    if not check_auth():
        ui.navigate.to("/login")
        return

    from . import face as face_mod

    coll_abs = os.path.abspath(face_mod._collections_dir())
    name_abs = os.path.abspath(os.path.join(coll_abs, name))
    if not name_abs.startswith(coll_abs + os.sep):
        ui.notify("Invalid collection name", color="negative")
        ui.navigate.to("/organize?tab=collections")
        return
    if not os.path.isdir(name_abs):
        ui.navigate.to("/organize?tab=collections")
        return

    ui.page_title(f"{name} — Collection")
    ui.add_body_html(_auto_logout_js())
    dark = ui.dark_mode()
    dark.set_value(app.storage.user.get("dark_mode", False))

    def _toggle_dark(dark=dark):
        new_val = not dark.value
        dark.set_value(new_val)
        app.storage.user["dark_mode"] = new_val

    ui.add_head_html(
        '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/lightgallery@2/css/lightgallery-bundle.min.css">'
    )
    ui.add_head_html(
        '<script src="https://cdn.jsdelivr.net/npm/lightgallery@2/lightgallery.umd.min.js"></script>'
    )
    ui.add_head_html(
        '<script src="https://cdn.jsdelivr.net/npm/lightgallery@2/plugins/zoom/lg-zoom.umd.min.js"></script>'
    )
    ui.add_head_html(
        '<script src="https://cdn.jsdelivr.net/npm/lightgallery@2/plugins/thumbnail/lg-thumbnail.umd.min.js"></script>'
    )
    ui.add_head_html(
        '<script src="https://cdn.jsdelivr.net/npm/lightgallery@2/plugins/video/lg-video.umd.min.js"></script>'
    )
    ui.add_head_html(f"<script>{_LG_INIT_JS}</script>")

    images = face_mod._collection_images(name)

    def _do_add_more_coll():
        _show_add_more_to_collection(
            name,
            None,
            None,
            on_success=lambda: ui.navigate.to(f"/collections/{quote(name)}"),
        )

    def _do_rename_coll():
        with ui.dialog() as dlg, ui.card():
            ui.label(f"Rename '{name}'").classes("font-semibold")
            inp = ui.input("New name", value=name).classes("w-full mt-2")
            err_lbl = ui.label("").classes("text-xs text-red-500")

            def confirm():
                new = inp.value.strip()
                if new == name:
                    dlg.close()
                    return
                e = _validate_name(new)
                if e:
                    err_lbl.set_text(e)
                    return
                result = face_mod.rename_collection(name, new)
                if result:
                    err_lbl.set_text(result)
                else:
                    dlg.close()
                    ui.navigate.to(f"/collections/{quote(new)}")

            with ui.row().classes("mt-3 gap-2 justify-end"):
                ui.button("Cancel", on_click=dlg.close).props("flat")
                ui.button("Rename", on_click=confirm).props("color=primary")
        dlg.open()

    with ui.column().classes("w-full max-w-6xl mx-auto p-4 gap-4"):
        with ui.row().classes("items-center gap-3 w-full"):
            ui.button(
                "← Back",
                icon="arrow_back",
                on_click=lambda: ui.navigate.to("/organize?tab=collections"),
            ).props("flat dense")
            ui.label(name).classes("text-xl font-bold flex-1")
            count_lbl = ui.label(f"({len(images)} file{'s' if len(images) != 1 else ''})").classes(
                "text-sm text-gray-400"
            )
            ui.button("Add more", icon="add", on_click=_do_add_more_coll).props("flat dense")
            ui.button("Rename", icon="edit", on_click=_do_rename_coll).props("flat dense")
            ui.button(icon="dark_mode", on_click=_toggle_dark).props("flat round").tooltip(
                "Toggle dark mode"
            )

        with ui.row().classes("items-center gap-2 flex-wrap mb-2"):
            ui.label("Tags:").classes("text-sm text-gray-500 shrink-0")
            coll_tag_box = ui.row().classes("gap-1 flex-wrap items-center")

        def _render_coll_tags():
            coll_tag_box.clear()
            with coll_tag_box:
                for t in face_mod.get_tags("collection", name):
                    chip = ui.chip(t).props("dense outline removable")

                    def _make_rm(_t=t):
                        def _():
                            face_mod.remove_tag("collection", name, _t)
                            _render_coll_tags()

                        return _

                    chip.on("remove", _make_rm())

                tag_inp = ui.input(placeholder="+ tag").classes("w-24 text-sm")

                def _add(_, inp=tag_inp):
                    val = inp.value.strip().lower()
                    if val:
                        face_mod.add_tag("collection", name, val)
                        inp.set_value("")
                        _render_coll_tags()

                tag_inp.on("keydown.enter", _add)

        _render_coll_tags()

        if not images:
            ui.label("No files found.").classes("text-gray-500 py-8")
            return

        gallery_container = ui.column().classes("w-full")
        _render_collection_gallery_grid(name, images, gallery_container, count_lbl)
