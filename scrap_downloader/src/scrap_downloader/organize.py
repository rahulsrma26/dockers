"""Organize page — /organize — face-based folder management."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any
from urllib.parse import quote

from nicegui import ui
from sqlalchemy import select

from .db import (
    ImageMeta,
    Notification,
    get_scan_meta,
    get_session,
    get_setting,
    set_scan_meta,
    set_setting,
)
from .face import _like_prefix
from .main import APP_VERSION, PASSWORD, check_auth, logout, touch_activity

DOWNLOAD_DIR = os.environ.get("DOWNLOAD_DIR", "downloads")

_ILLEGAL_CHARS = set('/\\:*?"<>|')


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
        item: dict = {"src": src, "thumb": thumb, "filename": os.path.basename(rel)}
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
        p for p in os.listdir(cat) if os.path.isdir(os.path.join(cat, p)) and not p.startswith(".")
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

    # ── Header ──
    with ui.column().classes("w-full max-w-5xl mx-auto p-4 gap-4"):
        with ui.row().classes("items-center justify-between w-full"):
            with ui.row().classes("items-baseline gap-2"):
                ui.label("Scrap Downloader").classes("text-2xl font-bold")
                ui.label(f"v{APP_VERSION}").classes("text-sm text-gray-400")
                ui.link("Downloads", "/").classes("text-sm text-blue-500 ml-4")
                ui.link("Organize", "/organize").classes("text-sm text-blue-500 font-semibold")
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
            tab_settings = ui.tab("Settings", icon="tune")

        _tab_map = {
            "scan": tab_scan,
            "suggestions": tab_suggest,
            "categorized": tab_cat,
            "settings": tab_settings,
        }
        initial_tab = _tab_map.get(tab, tab_scan)

        _label_to_slug = {
            "Scan": "scan",
            "Suggestions": "suggestions",
            "Categorized": "categorized",
            "Settings": "settings",
        }

        def _sync_tab_url(e):
            slug = _label_to_slug.get(e.args, "scan")
            ui.run_javascript(
                f"history.replaceState(null, '', '/organize?tab={slug}')", respond=False
            )

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

            # ── Tab 4: Settings ──────────────────────────────────────────────
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
        elif s == "cancelling":
            status_label.set_text("Cancelling…")
            scan_btn.set_enabled(False)
            cancel_btn.set_visibility(True)
        elif s == "failed":
            status_label.set_text(f"Scan failed: {err or ''}")
            progress_bar.set_value(0)
            scan_btn.set_enabled(True)
            cancel_btn.set_visibility(False)
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

    with ui.row().classes("gap-2 mt-2"):

        async def start_scan():
            touch_activity()
            set_scan_meta("status", "running")
            scan_btn.set_enabled(False)
            cancel_btn.set_visibility(True)

            loop = asyncio.get_event_loop()

            def _blocking():
                face_mod.scan_dir(DOWNLOAD_DIR)
                return face_mod.analyse_downloads(DOWNLOAD_DIR)

            try:
                res = await loop.run_in_executor(None, _blocking)
                state["suggestions"] = res["suggestions"]
                state["ambiguous"] = res["ambiguous"]
                state["suggestions_visible"] = 15  # reset pagination on fresh scan
                if "refresh_suggestions" in state:
                    state["refresh_suggestions"]()
                tabs.set_value(tab_suggest)
                ui.run_javascript(
                    "history.replaceState(null, '', '/organize?tab=suggestions')", respond=False
                )
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

    poll_timer = ui.timer(2.0, _update_status_ui)  # noqa: F841 — must stay referenced or NiceGUI GCs it
    _update_status_ui()


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

    # Dismissed suggestions toggle
    dismissed_active = [s for s in suggestions if _is_dismissed(s)]
    active_suggestions = [s for s in suggestions if not _is_dismissed(s)]

    _PAGE = 15
    visible = state.get("suggestions_visible", _PAGE)
    for sugg in active_suggestions[:visible]:
        _render_suggestion_card(sugg, state, container, dismissed=False)

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
                _render_suggestion_card(sugg, state, container, dismissed=True)

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

    from .db import DismissedMatch

    # For add_to cards include the categorized folder(s) in the pair check
    folders = list(sugg.get("folders", []))
    if sugg.get("type") == "add_to":
        folders = folders + list(sugg.get("cat_folders", []))

    if len(folders) < 2:
        return False
    with get_session() as s:
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

        # ── Source thumbnails — always visible ────────────────────────────────
        if sample_images:
            with ui.row().classes("gap-1 flex-wrap mb-1"):
                for thumb_url in sample_images[:6]:
                    ui.image(thumb_url).classes("w-20 h-20 object-cover rounded")

        # Source folders
        ui.label(f"Folders: {', '.join(folders)}").classes("text-xs text-gray-400 mt-1")

        # Scores pre-computed in analyse_downloads (executor thread) — use directly.
        scores: dict[str, float] = sugg.get("scores", {})
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

            # One hidden preview row per chip — shown accordion-style on click
            chip_previews: dict[str, object] = {}
            for _person, _ in chip_suggestions:
                preview = ui.row().classes(
                    "gap-1 flex-wrap mt-1 ml-1 border-l-2 border-blue-grey-200"
                )
                with preview:
                    thumbs = _person_sample_thumbs(_person)
                    if thumbs:
                        for _t in thumbs:
                            ui.image(_t).classes("w-16 h-16 object-cover rounded")
                    else:
                        ui.label("No images yet").classes("text-xs text-gray-400 italic")
                preview.set_visibility(False)
                chip_previews[_person] = preview

            chip_active: dict = {"person": None}

            def _chip_select(p, previews=chip_previews, active=chip_active):
                name_input.set_value(p)
                prev = active["person"]
                if prev and prev in previews:
                    previews[prev].set_visibility(False)
                if active["person"] == p:
                    active["person"] = None
                else:
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

        with ui.row().classes("mt-2 gap-2"):
            if not dismissed:
                ui.button("Confirm & merge", icon="merge", on_click=do_merge).props("color=primary")
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

            async def confirm():
                cancel_btn.set_enabled(False)
                confirm_btn.set_enabled(False)
                progress_bar.set_visibility(True)

                loop = asyncio.get_event_loop()

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

            confirm_btn = ui.button("Confirm & merge", on_click=confirm).props("color=primary")

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

    # Merge people state
    merge_state: dict[str, Any] = {"selected": [], "mode": False}

    def enter_merge_mode():
        merge_state["mode"] = True
        merge_state["selected"] = []
        merge_bar.set_visibility(True)
        merge_btn.set_visibility(False)
        container.update()

    def cancel_merge_mode():
        merge_state["mode"] = False
        merge_state["selected"] = []
        merge_bar.set_visibility(False)
        merge_btn.set_visibility(True)

    with ui.row().classes("items-center justify-between w-full mb-2"):
        ui.label(f"{len(people)} people").classes("text-sm text-gray-500")
        merge_btn = ui.button("Merge people", icon="merge", on_click=enter_merge_mode).props("flat")

    with ui.row().classes("items-center gap-2 w-full bg-blue-50 p-2 rounded") as merge_bar:
        merge_status_label = ui.label("Select two people to merge").classes("text-sm flex-1")
        ui.button("Cancel", on_click=cancel_merge_mode).props("flat dense")

    merge_bar.set_visibility(False)

    with ui.grid(columns=3).classes("w-full gap-4"):
        for person in people:
            thumbs = _person_thumbs(person, n=4)
            count = _person_file_count(person)

            with ui.card().classes(
                "w-full cursor-pointer hover:shadow-md transition-shadow"
            ) as card:
                # Thumbnail grid
                with ui.grid(columns=2).classes("w-full gap-1"):
                    for t in thumbs[:4]:
                        ui.image(t).classes("w-full aspect-square object-cover rounded")
                    for _ in range(max(0, 4 - len(thumbs))):
                        ui.element("div").classes("w-full aspect-square bg-gray-100 rounded")

                ui.label(person).classes("font-semibold mt-2 truncate")
                ui.label(f"{count} file{'s' if count != 1 else ''}").classes(
                    "text-xs text-gray-400"
                )

                with ui.row().classes("gap-1 mt-1") as btn_row:
                    ui.button(
                        "Add more",
                        icon="add",
                        on_click=lambda p=person: _show_add_more(p, container, state),
                    ).props("flat dense size=sm")

                    def make_rename_handler(p):
                        def do_rename():
                            with ui.dialog() as dlg, ui.card():
                                ui.label(f"Rename '{p}'").classes("font-semibold")
                                inp = ui.input("New name", value=p).classes("w-full mt-2")
                                err_lbl = ui.label("").classes("text-xs text-red-500")

                                def confirm():
                                    e = _validate_name(inp.value)
                                    if e:
                                        err_lbl.set_text(e)
                                        return
                                    result = face_mod.rename_person(p, inp.value.strip())
                                    if result:
                                        err_lbl.set_text(result)
                                    else:
                                        dlg.close()
                                        ui.notify(
                                            f"Renamed to '{inp.value.strip()}'", color="positive"
                                        )
                                        container.clear()
                                        with container:
                                            _render_categorized(container, state)

                                with ui.row().classes("mt-3 gap-2 justify-end"):
                                    ui.button("Cancel", on_click=dlg.close).props("flat")
                                    ui.button("Rename", on_click=confirm).props("color=primary")
                            dlg.open()

                        return do_rename

                    ui.button("Rename", icon="edit", on_click=make_rename_handler(person)).props(
                        "flat dense size=sm"
                    )
                btn_row.on("click.stop", lambda: None)

                def make_select_handler(p):
                    def on_click():
                        if not merge_state["mode"]:
                            ui.navigate.to(f"/gallery/{quote(p)}")
                            return
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


def _show_add_more(person: str, container, state: dict | None = None):
    """Show folder picker from downloads tree (excluding categorized/)."""

    cat_dir = os.path.abspath(_categorized_dir())

    # Build folder list
    source_folders = []
    for root, dirs, _ in os.walk(DOWNLOAD_DIR):
        abs_root = os.path.abspath(root)
        if abs_root == cat_dir or abs_root.startswith(cat_dir + os.sep):
            dirs.clear()
            continue
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        if abs_root != os.path.abspath(DOWNLOAD_DIR):
            source_folders.append(os.path.relpath(abs_root, DOWNLOAD_DIR))

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


# ── Gallery page ──────────────────────────────────────────────────────────────


@ui.page("/gallery/{person}")
def gallery_page(person: str):
    if not check_auth():
        ui.navigate.to("/login")
        return

    ui.page_title(f"{person} — Gallery")
    images = _person_images(person)

    with ui.column().classes("w-full max-w-6xl mx-auto p-4 gap-4"):
        with ui.row().classes("items-center gap-3 w-full"):
            ui.button(
                "← Back",
                icon="arrow_back",
                on_click=lambda: ui.navigate.to("/organize?tab=categorized"),
            ).props("flat dense")
            ui.label(person).classes("text-xl font-bold")
            ui.label(f"({len(images)} file{'s' if len(images) != 1 else ''})").classes(
                "text-sm text-gray-400"
            )

        if not images:
            ui.label("No files found.").classes("text-gray-500 py-8")
            return

        # CSS and JS libs go in <head> — scripts added there actually execute,
        # unlike <script> tags inside ui.html() which Vue renders via innerHTML
        # and the browser ignores for security reasons.
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
        # Init retries until Vue has mounted #lg-gallery in the DOM.
        ui.add_head_html(
            "<script>"
            "(function tryInit() {"
            "  var el = document.getElementById('lg-gallery');"
            "  if (!el || !window.lightGallery) { setTimeout(tryInit, 100); return; }"
            "  lightGallery(el, {"
            "    plugins: [lgZoom, lgThumbnail, lgVideo],"
            "    speed: 300,"
            "    download: true,"
            "    selector: '.lg-item',"
            "  });"
            "})();"
            "</script>"
        )

        thumb_style = "width:100%;aspect-ratio:1;object-fit:cover;border-radius:6px;cursor:pointer"
        items_html = ""
        for img in images:
            thumb_img = f'<img src="{img["thumb"]}" style="{thumb_style}" />'
            if img.get("is_video"):
                video_data = json.dumps(
                    {
                        "source": [{"src": img["src"], "type": img["video_type"]}],
                        "attributes": {"preload": "none", "controls": True},
                    }
                )
                # Use single-quoted attr so inner JSON double-quotes are safe
                items_html += f"<a class=\"lg-item\" data-video='{video_data}'>{thumb_img}</a>"
            else:
                items_html += f'<a href="{img["src"]}" class="lg-item">{thumb_img}</a>'
        _grid_style = (
            "display:grid;"
            "grid-template-columns:repeat(auto-fill,minmax(160px,1fr));"
            "gap:8px;width:100%"
        )
        ui.html(f'<div id="lg-gallery" style="{_grid_style}">{items_html}</div>').classes(
            "w-full block"
        )
