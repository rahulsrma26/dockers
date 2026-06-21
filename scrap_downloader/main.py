import logging
import os
import shutil
from datetime import timedelta

from nicegui import app, ui

import worker
from db import Task, TaskStatus, TaskTool, get_session, init_db, utcnow

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

PASSWORD = os.environ.get("APP_PASSWORD", "")
SESSION_TIMEOUT_MINUTES = int(os.environ.get("SESSION_TIMEOUT_MINUTES", "60"))

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLED_PLUGINS_DIR = os.path.join(THIS_DIR, "plugins")


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


_detail_open = False


def show_task_detail(row: dict):
    global _detail_open
    _detail_open = True

    with get_session() as session:
        task = session.get(Task, row["id"])
        rows = [
            {"field": "ID", "value": str(task.id), "isDate": False},
            {"field": "Status", "value": task.status, "isDate": False},
            {"field": "Tool", "value": task.tool, "isDate": False},
            {"field": "Tag", "value": task.tag, "isDate": False},
            {"field": "URL", "value": task.url, "isDate": False},
            {"field": "Extra args", "value": task.extra_args or "—", "isDate": False},
            {"field": "Progress", "value": f"{task.progress:.0f}%", "isDate": False},
            {"field": "Error", "value": task.error or "—", "isDate": False},
            {"field": "Created at", "value": task.created_at.isoformat() + "Z", "isDate": True},
            {
                "field": "Completed at",
                "value": task.completed_at.isoformat() + "Z" if task.completed_at else "—",
                "isDate": bool(task.completed_at),
            },
        ]
        task_id = task.id
        task_status = task.status

    def cancel():
        with get_session() as session:
            t = session.get(Task, task_id)
            t.status = TaskStatus.failed
            t.error = "Cancelled by user"
            t.completed_at = utcnow()
            session.commit()
        d.close()
        queue_table.refresh()

    def retry():
        with get_session() as session:
            t = session.get(Task, task_id)
            t.status = TaskStatus.pending
            t.progress = 0.0
            t.error = None
            t.completed_at = None
            session.commit()
        d.close()
        queue_table.refresh()

    with ui.dialog() as d, ui.card().classes("w-full max-w-2xl"):
        ui.label(f"Task #{task_id}").classes("text-lg font-semibold mb-2")
        t = ui.table(
            columns=[
                {
                    "name": "field",
                    "label": "Field",
                    "field": "field",
                    "align": "left",
                    "style": "width: 120px",
                },
                {"name": "value", "label": "Value", "field": "value", "align": "left"},
            ],
            rows=rows,
        ).classes("w-full")
        t.add_slot(
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
            if task_status in (TaskStatus.pending, TaskStatus.in_progress):
                ui.button("Cancel", icon="cancel", on_click=cancel).props("flat color=negative")
            if task_status == TaskStatus.failed:
                ui.button("Retry", icon="replay", on_click=retry).props("flat color=primary")
    d.on("hide", lambda: globals().update(_detail_open=False))
    d.open()


@ui.refreshable
def queue_table():
    with get_session() as session:
        tasks = session.query(Task).order_by(Task.created_at.desc()).limit(100).all()
        session.expunge_all()

    if not tasks:
        ui.label("No tasks yet.").classes("text-gray-400")
        return

    rows = [
        {
            "id": t.id,
            "status": {
                TaskStatus.pending: "⏳",
                TaskStatus.in_progress: "⬇️",
                TaskStatus.done: "✅",
                TaskStatus.failed: "❌",
            }[t.status],
            "tool": t.tool,
            "tag": t.tag,
            "url": t.url,
            "progress": f"{t.progress:.0f}%" if t.status == TaskStatus.in_progress else "",
            "error": t.error or "",
        }
        for t in tasks
    ]

    cell = "max-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;"
    table = ui.table(
        columns=[
            {"name": "id", "label": "ID", "field": "id", "align": "left", "style": "width: 40px"},
            {
                "name": "status",
                "label": "",
                "field": "status",
                "align": "left",
                "style": "width: 36px",
            },
            {
                "name": "tool",
                "label": "Tool",
                "field": "tool",
                "align": "left",
                "style": "width: 80px",
            },
            {
                "name": "tag",
                "label": "Tag",
                "field": "tag",
                "align": "left",
                "style": "width: 160px;" + cell,
            },
            {
                "name": "url",
                "label": "URL",
                "field": "url",
                "align": "left",
                "style": "width: 40%;" + cell,
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
                "style": "width: 20%;" + cell,
            },
        ],
        rows=rows,
    ).classes("w-full table-fixed")
    table.on("rowClick", lambda e: show_task_detail(e.args[1]))


@ui.page("/")
def main_page():
    if not check_auth():
        ui.navigate.to("/login")
        return

    ui.page_title("Scrap Downloader")
    ui.add_head_html("<style>.q-table tbody tr { cursor: pointer; }</style>")

    with ui.column().classes("w-full max-w-4xl mx-auto p-4 gap-4"):
        with ui.row().classes("items-center justify-between w-full"):
            ui.label("Scrap Downloader").classes("text-2xl font-bold")
            if PASSWORD:
                ui.button("Logout", icon="logout", on_click=logout).props("flat")

        with ui.card().classes("w-full"):
            with ui.tabs().classes("w-full") as tabs:
                tab_auto = ui.tab("Auto")
                tab_gdl = ui.tab("gallery-dl")

            with ui.tab_panels(tabs, value=tab_auto).classes("w-full"):
                with ui.tab_panel(tab_auto):
                    url_input = ui.input("URL", placeholder="https://...").classes("w-full")
                    tag_input = ui.input("Tag", placeholder="my-album").classes("w-full")

                    def submit_auto():
                        url = url_input.value.strip()
                        tag = tag_input.value.strip()
                        if not url or not tag:
                            ui.notify("URL and tag are required", color="negative")
                            return
                        touch_activity()
                        with get_session() as session:
                            task = Task(
                                url=url,
                                tag=tag,
                                tool=TaskTool.auto,
                                status=TaskStatus.pending,
                                progress=0.0,
                            )
                            session.add(task)
                            session.commit()
                        url_input.set_value("")
                        tag_input.set_value("")
                        ui.notify(f"Queued: {url}", color="positive")
                        queue_table.refresh()

                    ui.button("Add to queue", on_click=submit_auto)

                with ui.tab_panel(tab_gdl):
                    gdl_url_input = ui.input("URL", placeholder="https://...").classes("w-full")
                    gdl_tag_input = ui.input("Tag", placeholder="my-album").classes("w-full")
                    gdl_args_input = ui.input(
                        "Options", placeholder="--username user --cookies cookies.txt --range 1-20"
                    ).classes("w-full")

                    def submit_gdl():
                        url = gdl_url_input.value.strip()
                        tag = gdl_tag_input.value.strip()
                        if not url or not tag:
                            ui.notify("URL and tag are required", color="negative")
                            return
                        touch_activity()
                        with get_session() as session:
                            task = Task(
                                url=url,
                                tag=tag,
                                tool=TaskTool.gallery_dl,
                                extra_args=gdl_args_input.value.strip() or None,
                                status=TaskStatus.pending,
                                progress=0.0,
                            )
                            session.add(task)
                            session.commit()
                        gdl_url_input.set_value("")
                        gdl_tag_input.set_value("")
                        gdl_args_input.set_value("")
                        ui.notify(f"Queued via gallery-dl: {url}", color="positive")
                        queue_table.refresh()

                    ui.button("Add to queue", on_click=submit_gdl)

        with ui.card().classes("w-full"):
            with ui.row().classes("items-center justify-between w-full"):
                ui.label("Queue").classes("text-lg font-semibold")
                with ui.row().classes("gap-2"):
                    ui.button(icon="refresh", on_click=queue_table.refresh).props("flat round")
                    ui.button(icon="delete_sweep", on_click=lambda: clear_history()).props(
                        "flat round color=negative"
                    ).tooltip("Clear history")
            queue_table()

    def clear_history():
        with get_session() as session:
            session.query(Task).filter(
                Task.status.in_([TaskStatus.done, TaskStatus.failed])
            ).delete()
            session.commit()
        queue_table.refresh()

    def maybe_refresh():
        if not check_auth():
            ui.navigate.to("/login")
            return
        if not _detail_open:
            queue_table.refresh()

    ui.timer(3.0, maybe_refresh)


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
