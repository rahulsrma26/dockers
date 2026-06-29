import logging
import os
import shlex
import shutil
import threading
from datetime import timedelta
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version

from nicegui import app, ui

from . import worker
from .db import Task, TaskStatus, TaskTool, get_session, init_db, utcnow
from .tidy import tidy_dir

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

try:
    APP_VERSION = pkg_version("scrap-downloader")
except PackageNotFoundError:
    APP_VERSION = "dev"

PASSWORD = os.environ.get("APP_PASSWORD", "")
SESSION_TIMEOUT_MINUTES = int(os.environ.get("SESSION_TIMEOUT_MINUTES", "60"))

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLED_PLUGINS_DIR = os.path.join(THIS_DIR, "plugins")

_DATA_DIR = os.path.dirname(os.path.abspath(os.environ.get("DB_PATH", "scrap_downloader.db")))
COOKIES_DIR = os.path.join(_DATA_DIR, "cookies")

_CELL = "max-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;"

_QUEUE_COLUMNS = [
    {"name": "id", "label": "ID", "field": "id", "align": "left", "style": "width: 40px"},
    {"name": "status", "label": "", "field": "status", "align": "left", "style": "width: 36px"},
    {"name": "tool", "label": "Tool", "field": "tool", "align": "left", "style": "width: 80px"},
    {
        "name": "tag",
        "label": "Tag",
        "field": "tag",
        "align": "left",
        "style": f"width: 160px;{_CELL}",
    },
    {
        "name": "url",
        "label": "URL",
        "field": "url",
        "align": "left",
        "style": f"width: 40%;{_CELL}",
    },
    {
        "name": "progress",
        "label": "Progress",
        "field": "progress",
        "align": "left",
        "style": "width: 70px",
    },
    {
        "name": "error",
        "label": "Error",
        "field": "error",
        "align": "left",
        "style": f"width: 20%;{_CELL}",
    },
]

_DETAIL_COLUMNS = [
    {"name": "field", "label": "Field", "field": "field", "align": "left", "style": "width: 120px"},
    {"name": "value", "label": "Value", "field": "value", "align": "left"},
]

_IMAGE_EXTS = ("jpg", "jpeg", "png", "gif", "webp", "bmp", "tiff", "avif")
_VIDEO_EXTS = ("mp4", "webm", "mkv", "avi", "mov", "flv", "m4v", "wmv")


def _build_gdl_args(
    media_type: str,
    range_val: str,
    date_after: str,
    rate_limit: str,
    sleep_val: float | None,
    filesize_min: str,
    filesize_max: str,
    cookie_path: str | None,
    extra: str,
) -> str | None:
    parts = []
    if media_type == "images":
        parts += ["--filter", f"extension in {_IMAGE_EXTS!r}"]
    elif media_type == "videos":
        parts += ["--filter", f"extension in {_VIDEO_EXTS!r}"]
    if range_val:
        parts += ["--range", range_val]
    if date_after:
        parts += ["--date-after", date_after]
    if rate_limit:
        parts += ["--limit-rate", rate_limit]
    if sleep_val and sleep_val > 0:
        parts += ["--sleep", str(sleep_val)]
    if filesize_min:
        parts += ["--filesize-min", filesize_min]
    if filesize_max:
        parts += ["--filesize-max", filesize_max]
    if cookie_path:
        parts += ["--cookies", cookie_path]
    result = shlex.join(parts)
    if extra:
        result = (result + " " + extra).strip()
    return result or None


def seed_plugins():
    plugins_dir = os.environ.get("PLUGINS_DIR", "plugins")
    if plugins_dir == BUNDLED_PLUGINS_DIR:
        return
    os.makedirs(plugins_dir, exist_ok=True)
    for fname in os.listdir(BUNDLED_PLUGINS_DIR):
        if not fname.endswith(".py"):
            continue
        dest = os.path.join(plugins_dir, fname)
        if not os.path.exists(dest):
            shutil.copy2(os.path.join(BUNDLED_PLUGINS_DIR, fname), dest)
            logger.info(f"Seeded plugin: {fname}")


def touch_activity():
    app.storage.user["last_activity"] = utcnow().isoformat()


def check_auth() -> bool:
    if not PASSWORD:
        return True
    if not app.storage.user.get("authenticated", False):
        return False
    last = app.storage.user.get("last_activity")
    if last is None:
        return False
    from datetime import datetime

    elapsed = utcnow() - datetime.fromisoformat(last)
    return elapsed < timedelta(minutes=SESSION_TIMEOUT_MINUTES)


def logout():
    app.storage.user.clear()
    ui.navigate.to("/login")


_STATUS_EMOJI = {
    TaskStatus.pending: "⏳",
    TaskStatus.in_progress: "⬇️",
    TaskStatus.done: "✅",
    TaskStatus.failed: "❌",
}


def _find_duplicate(url: str) -> dict | None:
    with get_session() as session:
        task = session.query(Task).filter(Task.url == url).order_by(Task.created_at.desc()).first()
        if task is None:
            return None
        return {"tag": task.tag, "status": task.status}


def _fetch_queue_rows() -> list[dict]:
    with get_session() as session:
        tasks = session.query(Task).order_by(Task.created_at.desc()).limit(100).all()
        session.expunge_all()
    return [
        {
            "id": t.id,
            "status": _STATUS_EMOJI[t.status],
            "tool": t.tool,
            "tag": t.tag,
            "url": t.url,
            "progress": f"{t.progress:.0f}%" if t.status == TaskStatus.in_progress else "",
            "error": t.error or "",
        }
        for t in tasks
    ]


def _fetch_task_snapshot(task_id: int) -> dict | None:
    with get_session() as session:
        task = session.get(Task, task_id)
        if task is None:
            return None
        return {
            "id": task.id,
            "status": task.status,
            "tool": task.tool,
            "tag": task.tag,
            "url": task.url,
            "extra_args": task.extra_args,
            "progress": task.progress,
            "error": task.error,
            "created_at": task.created_at,
            "completed_at": task.completed_at,
        }


def _make_detail_rows(snap: dict) -> list[dict]:
    return [
        {"field": "ID", "value": str(snap["id"]), "isDate": False},
        {"field": "Status", "value": snap["status"], "isDate": False},
        {"field": "Tool", "value": snap["tool"], "isDate": False},
        {"field": "Tag", "value": snap["tag"], "isDate": False},
        {"field": "URL", "value": snap["url"], "isDate": False},
        {"field": "Extra args", "value": snap["extra_args"] or "—", "isDate": False},
        {"field": "Progress", "value": f"{snap['progress']:.0f}%", "isDate": False},
        {"field": "Error", "value": snap["error"] or "—", "isDate": False},
        {"field": "Created at", "value": snap["created_at"].isoformat() + "Z", "isDate": True},
        {
            "field": "Completed at",
            "value": snap["completed_at"].isoformat() + "Z" if snap["completed_at"] else "—",
            "isDate": bool(snap["completed_at"]),
        },
    ]


@ui.page("/login")
def login_page():
    if not PASSWORD:
        ui.navigate.to("/")
        return

    def try_login():
        if pw_input.value == PASSWORD:
            app.storage.user["authenticated"] = True
            touch_activity()
            ui.navigate.to("/")
        else:
            ui.notify("Wrong password", color="negative")

    with ui.card().classes("absolute-center"):
        ui.label("Scrap Downloader").classes("text-xl font-bold")
        pw_input = ui.input("Password", password=True).on("keydown.enter", try_login)
        ui.button("Login", on_click=try_login)


@ui.page("/")
def main_page():
    if not check_auth():
        ui.navigate.to("/login")
        return

    ui.page_title("Scrap Downloader")
    ui.add_head_html("<style>.q-table tbody tr { cursor: pointer; }</style>")

    detail_open = {"value": False}
    queue_tbl: dict = {"ref": None}

    def refresh_queue():
        if queue_tbl["ref"] is not None:
            queue_tbl["ref"].rows = _fetch_queue_rows()
            queue_tbl["ref"].update()

    def queue_with_dup_check(url: str, on_confirmed):
        dup = _find_duplicate(url)
        if not dup:
            on_confirmed()
            return
        emoji = _STATUS_EMOJI.get(dup["status"], "")
        with ui.dialog() as dlg, ui.card():
            ui.label("Duplicate detected").classes("text-lg font-semibold")
            ui.label(f'Previously run under tag "{dup["tag"]}" — {emoji} {dup["status"]}').classes(
                "text-sm text-gray-500 mt-1"
            )
            ui.label("Add to queue anyway?").classes("mt-2")
            with ui.row().classes("mt-4 gap-2 justify-end"):
                ui.button("Cancel", on_click=dlg.close).props("flat")

                def proceed():
                    dlg.close()
                    on_confirmed()

                ui.button("Add anyway", on_click=proceed).props("color=primary")
        dlg.open()

    def show_task_detail(row: dict):
        detail_open["value"] = True
        task_id = row["id"]

        snap = _fetch_task_snapshot(task_id)
        if snap is None:
            return

        def cancel():
            with get_session() as session:
                t = session.get(Task, task_id)
                t.status = TaskStatus.failed
                t.error = "Cancelled by user"
                t.completed_at = utcnow()
                session.commit()
            d.close()
            refresh_queue()

        def retry():
            with get_session() as session:
                t = session.get(Task, task_id)
                t.status = TaskStatus.pending
                t.progress = 0.0
                t.error = None
                t.completed_at = None
                session.commit()
            d.close()
            refresh_queue()

        with ui.dialog() as d, ui.card().classes("w-full max-w-2xl"):
            ui.label(f"Task #{task_id}").classes("text-lg font-semibold mb-2")
            detail_tbl = ui.table(columns=_DETAIL_COLUMNS, rows=_make_detail_rows(snap)).classes(
                "w-full"
            )
            detail_tbl.add_slot(
                "body-cell-value",
                """
                <q-td :props="props">
                    <span v-if="props.row.isDate">
                        {{ new Date(props.row.value).toLocaleString() }}
                    </span>
                    <span v-else>{{ props.row.value }}</span>
                </q-td>
                """,
            )
            with ui.row().classes("mt-2 gap-2"):
                ui.button("Close", on_click=d.close).props("flat")
                cancel_btn = ui.button("Cancel", icon="cancel", on_click=cancel).props(
                    "flat color=negative"
                )
                retry_btn = ui.button("Retry", icon="replay", on_click=retry).props(
                    "flat color=primary"
                )
                cancel_btn.set_visibility(
                    snap["status"] in (TaskStatus.pending, TaskStatus.in_progress)
                )
                retry_btn.set_visibility(snap["status"] == TaskStatus.failed)

        def update_detail():
            s = _fetch_task_snapshot(task_id)
            if s is None:
                return
            detail_tbl.rows = _make_detail_rows(s)
            detail_tbl.update()
            cancel_btn.set_visibility(s["status"] in (TaskStatus.pending, TaskStatus.in_progress))
            retry_btn.set_visibility(s["status"] == TaskStatus.failed)
            if s["status"] not in (TaskStatus.pending, TaskStatus.in_progress):
                detail_timer.cancel()

        detail_timer = ui.timer(2.0, update_detail)

        def on_hide():
            detail_open["value"] = False
            detail_timer.cancel()

        d.on("hide", on_hide)
        d.open()

    with ui.column().classes("w-full max-w-4xl mx-auto p-4 gap-4"):
        with ui.row().classes("items-center justify-between w-full"):
            with ui.row().classes("items-baseline gap-2"):
                ui.label("Scrap Downloader").classes("text-2xl font-bold")
                ui.label(f"v{APP_VERSION}").classes("text-sm text-gray-400")
            if PASSWORD:
                ui.button("Logout", icon="logout", on_click=logout).props("flat")

        with ui.card().classes("w-full"):
            with ui.tabs().classes("w-full") as tabs:
                tab_auto = ui.tab("Auto")
                tab_gdl = ui.tab("gallery-dl")
                tab_settings = ui.tab("Settings")

            with ui.tab_panels(tabs, value=tab_auto).classes("w-full"):
                with ui.tab_panel(tab_auto):
                    url_input = ui.input("URL", placeholder="https://...").classes("w-full")
                    tag_input = ui.input("Tag", placeholder="my-album").classes("w-full")
                    auto_audio_only = ui.switch(
                        "Audio only (best quality, original format)"
                    ).classes("mt-2")

                    def submit_auto():
                        url = url_input.value.strip()
                        tag = tag_input.value.strip()
                        if not url or not tag:
                            ui.notify("URL and tag are required", color="negative")
                            return
                        touch_activity()
                        audio_only = auto_audio_only.value

                        def do_queue():
                            with get_session() as session:
                                session.add(
                                    Task(
                                        url=url,
                                        tag=tag,
                                        tool=TaskTool.ytdlp if audio_only else TaskTool.auto,
                                        extra_args="--audio-only" if audio_only else None,
                                        status=TaskStatus.pending,
                                        progress=0.0,
                                    )
                                )
                                session.commit()
                            url_input.set_value("")
                            tag_input.set_value("")
                            ui.notify(f"Queued: {url}", color="positive")
                            refresh_queue()

                        queue_with_dup_check(url, do_queue)

                    ui.button("Add to queue", on_click=submit_auto)

                with ui.tab_panel(tab_gdl):
                    gdl_url_input = ui.input("URL", placeholder="https://...").classes("w-full")
                    gdl_tag_input = ui.input("Tag", placeholder="my-album").classes("w-full")

                    ui.label("Media type").classes("text-sm text-gray-500 mt-2")
                    gdl_media = ui.toggle(
                        {"": "All", "images": "Images only", "videos": "Videos only"},
                        value="",
                    )

                    with ui.row().classes("w-full gap-4 mt-2"):
                        gdl_range = ui.input("Range", placeholder="e.g. 1-50").classes("flex-1")
                        gdl_date = (
                            ui.input("Date after", placeholder="YYYY-MM-DD")
                            .props("type=date")
                            .classes("flex-1")
                        )

                    with ui.expansion("Advanced", icon="tune").classes("w-full mt-2"):
                        with ui.row().classes("w-full gap-4"):
                            gdl_rate = ui.select(
                                {
                                    "": "Unlimited",
                                    "500k": "500 KB/s",
                                    "1M": "1 MB/s",
                                    "2M": "2 MB/s",
                                    "5M": "5 MB/s",
                                },
                                value="",
                                label="Rate limit",
                            ).classes("flex-1")
                            gdl_sleep = ui.number(
                                "Sleep (s)", value=0, min=0, step=0.5, format="%.1f"
                            ).classes("flex-1")

                        with ui.row().classes("w-full gap-4 mt-2"):
                            gdl_filesize_min = ui.input(
                                "Min file size", placeholder="e.g. 50k"
                            ).classes("flex-1")
                            gdl_filesize_min.set_value("50k")
                            gdl_filesize_max = ui.input(
                                "Max file size", placeholder="e.g. 500M"
                            ).classes("flex-1")
                            gdl_filesize_max.set_value("500M")

                        cookie_state = {"path": None}
                        ui.label("Cookies file").classes("text-sm text-gray-500 mt-3")
                        with ui.row().classes("items-center gap-2 w-full"):
                            cookie_label = ui.label("No file uploaded").classes(
                                "text-sm text-gray-400 flex-1"
                            )

                            def handle_cookie_upload(e):
                                os.makedirs(COOKIES_DIR, exist_ok=True)
                                path = os.path.join(COOKIES_DIR, e.name)
                                with open(path, "wb") as f:
                                    f.write(e.content.read())
                                cookie_state["path"] = path
                                cookie_label.set_text(e.name)
                                ui.notify(f"Cookies uploaded: {e.name}", color="positive")

                            def clear_cookie():
                                cookie_state["path"] = None
                                cookie_label.set_text("No file uploaded")

                            ui.upload(on_upload=handle_cookie_upload, auto_upload=True).props(
                                'accept=".txt,.db,.sqlite" flat dense label="Upload"'
                            )
                            ui.button(icon="close", on_click=clear_cookie).props(
                                "flat round dense"
                            ).tooltip("Clear")

                    gdl_args_input = ui.input(
                        "Additional options", placeholder="--username user --password pass"
                    ).classes("w-full mt-2")

                    def submit_gdl():
                        url = gdl_url_input.value.strip()
                        tag = gdl_tag_input.value.strip()
                        if not url or not tag:
                            ui.notify("URL and tag are required", color="negative")
                            return
                        touch_activity()

                        built_args = _build_gdl_args(
                            media_type=gdl_media.value,
                            range_val=gdl_range.value.strip(),
                            date_after=gdl_date.value.strip(),
                            rate_limit=gdl_rate.value,
                            sleep_val=gdl_sleep.value if gdl_sleep.value else None,
                            filesize_min=gdl_filesize_min.value.strip(),
                            filesize_max=gdl_filesize_max.value.strip(),
                            cookie_path=cookie_state["path"],
                            extra=gdl_args_input.value.strip(),
                        )

                        def do_queue():
                            with get_session() as session:
                                session.add(
                                    Task(
                                        url=url,
                                        tag=tag,
                                        tool=TaskTool.gallery_dl,
                                        extra_args=built_args,
                                        status=TaskStatus.pending,
                                        progress=0.0,
                                    )
                                )
                                session.commit()
                            gdl_url_input.set_value("")
                            gdl_tag_input.set_value("")
                            ui.notify(f"Queued via gallery-dl: {url}", color="positive")
                            refresh_queue()

                        queue_with_dup_check(url, do_queue)

                    ui.button("Add to queue", on_click=submit_gdl)

                with ui.tab_panel(tab_settings):
                    ui.label("Tidy downloads folder").classes("text-sm font-semibold")
                    ui.label(
                        "Groups files ending in a number (e.g. Album 001.jpg, Album 002.jpg) "
                        "into a sub-folder named after the common prefix. "
                        "Only moves when 2+ groups are present. Single files are left in place."
                    ).classes("text-xs text-gray-500 mt-1")

                    tidy_progress = (
                        ui.linear_progress(0).classes("w-full mt-3").props("instant-feedback")
                    )
                    tidy_progress.visible = False

                    def run_tidy():
                        state = {"done": 0, "total": 0, "finished": False, "count": 0}

                        def on_progress(done, total):
                            state["done"] = done
                            state["total"] = total

                        def do_work():
                            moved = tidy_dir(worker.DOWNLOAD_DIR, on_progress=on_progress)
                            state["count"] = len(moved)
                            state["finished"] = True

                        def tick():
                            if state["total"]:
                                tidy_progress.value = state["done"] / state["total"]
                            if state["finished"]:
                                poll.cancel()
                                tidy_progress.visible = False
                                tidy_btn.enable()
                                if state["count"]:
                                    ui.notify(f"Tidied {state['count']} file(s)", color="positive")
                                else:
                                    ui.notify("Nothing to tidy", color="info")

                        tidy_progress.value = 0
                        tidy_progress.visible = True
                        tidy_btn.disable()
                        poll = ui.timer(0.1, tick)
                        threading.Thread(target=do_work, daemon=True).start()

                    tidy_btn = ui.button("Tidy", icon="folder_special", on_click=run_tidy).classes(
                        "mt-3"
                    )

        with ui.card().classes("w-full"):
            with ui.row().classes("items-center justify-between w-full"):
                ui.label("Queue").classes("text-lg font-semibold")
                with ui.row().classes("gap-2"):
                    ui.button(icon="refresh", on_click=refresh_queue).props("flat round")
                    ui.button(icon="delete_sweep", on_click=lambda: clear_history()).props(
                        "flat round color=negative"
                    ).tooltip("Clear history")

            tbl = ui.table(
                columns=_QUEUE_COLUMNS,
                rows=_fetch_queue_rows(),
            ).classes("w-full table-fixed")
            tbl.on("rowClick", lambda e: show_task_detail(e.args[1]))
            queue_tbl["ref"] = tbl

    def clear_history():
        with get_session() as session:
            session.query(Task).filter(
                Task.status.in_([TaskStatus.done, TaskStatus.failed])
            ).delete()
            session.commit()
        refresh_queue()

    def maybe_refresh():
        if not check_auth():
            ui.navigate.to("/login")
            return
        if not detail_open["value"]:
            refresh_queue()

    ui.timer(5.0, maybe_refresh)


def main():
    app.on_startup(worker.start)
    app.on_shutdown(worker.stop)
    init_db()
    seed_plugins()
    ui.run(
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", 8080)),
        title="Scrap Downloader",
        reload=bool(os.environ.get("DEV", "")),
        show=False,
        storage_secret=os.environ.get("STORAGE_SECRET", "change-me-in-prod"),
    )
