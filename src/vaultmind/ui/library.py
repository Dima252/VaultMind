"""Library page: drag-and-drop upload, per-file status, re-index and remove."""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import uuid
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from nicegui import events, ui

from vaultmind.app.engine import Engine
from vaultmind.core.models import FileRecord, file_id_for
from vaultmind.ingest.parsers import SUPPORTED_EXTENSIONS
from vaultmind.ingest.pipeline import sha256_file
from vaultmind.ingest.queue import IngestQueue, JobEvent
from vaultmind.ui.badge import off_loop
from vaultmind.ui.state import state

REFRESH_S = 1.0  # file table poll interval
INDICATOR_S = 0.5  # header queue indicator poll interval

_ACTIVE = ("queued", "indexing")
_STATUS_FILTER = {
    "all": "All",
    "queued": "Queued",
    "indexing": "Indexing",
    "ready": "Ready",
    "failed": "Failed",
}
_COLUMNS: list[dict[str, Any]] = [
    {"name": "name", "label": "Name", "field": "name", "align": "left", "sortable": True},
    {"name": "folder", "label": "Folder", "field": "folder", "align": "left", "sortable": True},
    {"name": "status", "label": "Status", "field": "status", "align": "left", "sortable": True},
    {"name": "error", "label": "Error", "field": "error", "align": "left"},
    {"name": "indexed", "label": "Indexed", "field": "indexed", "align": "left", "sortable": True},
    {"name": "actions", "label": "", "field": "key", "align": "right"},
]
_STATUS_CELL = """
<q-td :props="props">
  <q-badge :label="props.value"
    :color="{ready: 'positive', failed: 'negative', indexing: 'primary'}[props.value] || 'grey'" />
</q-td>
"""
_ERROR_CELL = """
<q-td :props="props" class="text-negative" style="white-space: normal; max-width: 28rem">
  {{ props.value }}
</q-td>
"""
_ACTIONS_CELL = """
<q-td :props="props" auto-width>
  <q-btn flat dense round size="sm" icon="refresh"
    @click="() => $parent.$emit('reindex', props.row)">
    <q-tooltip>Re-index</q-tooltip>
  </q-btn>
  <q-btn flat dense round size="sm" icon="open_in_new"
    @click="() => $parent.$emit('open', props.row)">
    <q-tooltip>Open</q-tooltip>
  </q-btn>
  <q-btn v-if="props.row.in_vault" flat dense round size="sm" color="negative" icon="delete"
    @click="() => $parent.$emit('remove', props.row)">
    <q-tooltip>Remove</q-tooltip>
  </q-btn>
</q-td>
"""


def page() -> None:
    """Build the library page. Uses `state().engine` (ready when this is called)."""
    engine = state().engine
    if engine is None:  # the shell only builds this page once the engine is ready
        ui.label("The engine isn't loaded.")
        return
    _Library(engine).build()


def queue_indicator() -> None:
    """Header element: "Indexing N files" while the ingest queue is busy, hidden otherwise."""
    with (
        ui.button(on_click=lambda: ui.navigate.to("/library"))
        .props("flat no-caps color=white")
        .mark("queue-indicator") as button
    ):
        ui.spinner(color="white", size="1.2em")
        label = ui.label().classes("ml-2")
    button.set_visibility(False)

    def poll() -> None:
        s = state()
        queue = s.engine.queue if s.engine is not None and s.phase == "ready" else None
        busy = queue is not None and queue.pending() > 0
        if busy:
            label.text = _indexing_text(s.jobs().values())
        button.set_visibility(busy)

    ui.timer(INDICATOR_S, poll)


def _indexing_text(jobs: Iterable[JobEvent]) -> str:
    n = sum(1 for job in jobs if job.state in _ACTIVE)
    if n == 0:  # a folder scan is hashing files to find what changed
        return "Checking files"
    return f"Indexing {n} file{'s' if n != 1 else ''}"


class _Library:
    """One library page: the summary, upload area, filters and the file table."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.vault = engine.cfg.paths.vault_dir.resolve()
        self.rows: list[dict[str, Any]] = []
        self.chunks = 0

    @property
    def queue(self) -> IngestQueue | None:
        return self.engine.queue

    def build(self) -> None:
        with ui.column().classes("w-full p-4 gap-4"):
            with ui.row().classes("w-full items-center gap-4"):
                ui.label("Library").classes("text-xl font-semibold")
                ui.space()
                ui.button("Re-scan vault", icon="sync", on_click=self._rescan).props("outline")
            self.summary = ui.label().mark("library-summary")
            exts = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            upload = (
                ui.upload(
                    multiple=True,
                    auto_upload=True,
                    on_upload=self._on_upload,
                    on_rejected=lambda: ui.notify(
                        f"Only these file types can be added: {exts}", type="warning"
                    ),
                    label=f"Drop files here or click + to add them to the vault ({exts})",
                )
                .props(f'accept="{",".join(sorted(SUPPORTED_EXTENSIONS))}" flat bordered')
                .classes("w-full")
            )
            upload.on("finish", upload.reset)
            with ui.row().classes("w-full items-center gap-4"):
                self.text_filter = ui.input(
                    "Filter", placeholder="name, folder or error", on_change=self._render
                ).props("clearable dense")
                self.status_filter = ui.select(
                    _STATUS_FILTER, value="all", label="Status", on_change=self._render
                ).props("dense")
            self.table = ui.table(columns=_COLUMNS, rows=[], row_key="key", pagination=50).classes(
                "w-full"
            )
            self.table.add_slot("body-cell-status", _STATUS_CELL)
            self.table.add_slot("body-cell-error", _ERROR_CELL)
            self.table.add_slot("body-cell-actions", _ACTIONS_CELL)
            self.table.on("reindex", lambda e: self._reindex(e.args))
            self.table.on("open", lambda e: self._open(e.args))
            self.table.on("remove", lambda e: self._remove(e.args))
        self.rows = self._current_rows()
        self.chunks = self.engine.index.count()
        self._render()
        ui.timer(REFRESH_S, self._poll, immediate=False)

    # -- data --

    def _current_rows(self) -> list[dict[str, Any]]:
        return file_rows(self.engine.registry.all(), state().jobs(), self.vault)

    async def _poll(self) -> None:
        """Rebuild the table only when a file changed state."""
        rows = await off_loop(self._current_rows)
        if rows is None or rows == self.rows:
            return
        chunks = await off_loop(self.engine.index.count)
        self.rows = rows
        if chunks is not None:
            self.chunks = chunks
        self._render()

    def _render(self) -> None:
        total = len(self.rows)
        counts = {
            status: sum(r["status"] == status for r in self.rows) for status in _STATUS_FILTER
        }
        parts = [
            f"{total} file{'s' if total != 1 else ''}",
            f"{counts['ready']} ready",
            f"{counts['failed']} failed",
        ]
        if busy := counts["queued"] + counts["indexing"]:
            parts.append(f"{busy} in progress")
        parts.append(f"{self.chunks} chunks in the index")
        self.summary.text = ", ".join(parts) + f". Vault folder: {self.vault}"
        self.table.rows = filter_rows(
            self.rows, self.text_filter.value or "", self.status_filter.value or "all"
        )

    # -- actions --

    async def _on_upload(self, e: events.UploadEventArguments) -> None:
        name = e.file.name
        if not name.strip(". ") or Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
            exts = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            ui.notify(f"{name}: not a supported file type ({exts})", type="warning")
            return
        # Written under a hidden .tmp name first: the watcher and the indexer ignore it.
        tmp = self.vault / f".{uuid.uuid4().hex}.upload.tmp"
        try:
            self.vault.mkdir(parents=True, exist_ok=True)
            await e.file.save(tmp)
            target = await off_loop(place_upload, tmp, name, self.vault)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            ui.notify(f"Couldn't save {name}: {exc}", type="negative")
            return
        if target is None:  # the app is shutting down
            tmp.unlink(missing_ok=True)
            return
        if self.queue is not None:
            self.queue.submit(target)
        ui.notify(f"Added {target.name}", type="positive")

    def _reindex(self, row: Mapping[str, Any]) -> None:
        path = Path(row["path"])
        if self.queue is None:
            return
        if not path.exists():
            ui.notify(f"{path.name} no longer exists", type="warning")
            return
        # Unchanged `ready` files are skipped, so mark it first. A file without a registry
        # row (e.g. it couldn't be read) just gets submitted.
        with contextlib.suppress(KeyError):
            self.engine.registry.set_status(file_id_for(path), "queued")
        self.queue.submit(path)
        ui.notify(f"Re-indexing {path.name}")

    async def _open(self, row: Mapping[str, Any]) -> None:
        path = Path(row["path"])
        try:
            await off_loop(open_with_os, path)
        except OSError as exc:
            ui.notify(f"Couldn't open {path.name}: {exc}", type="negative")

    async def _remove(self, row: Mapping[str, Any]) -> None:
        path = Path(row["path"]).resolve()  # the row comes from the browser: no ".." tricks
        if not _is_within(path, self.vault):
            ui.notify("Only files inside the vault folder can be removed here", type="warning")
            return
        with ui.dialog() as dialog, ui.card().classes("max-w-lg"):
            ui.label(f"Delete {path.name}?").classes("text-lg font-semibold")
            ui.label(
                f"The file will be deleted from the vault folder ({path.parent}) and removed "
                "from the index. This can't be undone."
            )
            with ui.row().classes("w-full justify-end"):
                ui.button("Cancel", on_click=lambda: dialog.submit(False)).props("flat")
                ui.button("Delete", color="negative", on_click=lambda: dialog.submit(True)).mark(
                    "confirm"
                )
        confirmed = await dialog
        dialog.delete()
        if not confirmed:
            return
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            ui.notify(f"Couldn't delete {path.name}: {exc}", type="negative")
            return
        if self.queue is not None:
            self.queue.forget(path)
        ui.notify(f"Deleted {path.name}")

    def _rescan(self) -> None:
        if self.queue is not None:
            self.queue.scan()
            ui.notify("Checking the vault folder for changes")


def file_rows(
    records: Iterable[FileRecord], jobs: Mapping[Path, JobEvent], vault: Path
) -> list[dict[str, Any]]:
    """Table rows: the registry, overlaid with live ingestion events since startup.

    The worker writes the registry before it emits `ready` or `failed`, so for those the
    registry is at least as new; `queued` and `indexing` are only known from the events.
    """
    entries: dict[str, tuple[Path, str, str | None, datetime | None]] = {
        _key(r.path): (r.path, r.status, r.error, r.indexed_at) for r in records
    }
    for path, event in jobs.items():
        key = _key(path)
        known = entries.get(key)
        if event.state in _ACTIVE:
            entries[key] = (path, event.state, None, known[3] if known else None)
        elif event.state == "failed" and known is None:  # failed before it got a registry row
            entries[key] = (path, "failed", event.error, None)
    rows = [_row(key, *entry, vault=vault) for key, entry in entries.items()]
    return sorted(rows, key=lambda r: (r["folder"].lower(), r["name"].lower()))


def filter_rows(rows: list[dict[str, Any]], text: str, status: str) -> list[dict[str, Any]]:
    needle = text.strip().lower()
    return [
        r
        for r in rows
        if (status == "all" or r["status"] == status)
        and (not needle or any(needle in r[f].lower() for f in ("name", "folder", "error")))
    ]


def place_upload(tmp: Path, name: str, vault: Path) -> Path:
    """Move an uploaded temp file into the vault as `name`. An identical file already there is
    reused; a different one keeps its name and the upload becomes `name (2)`, `name (3)`..."""
    stem, suffix = Path(name).stem, Path(name).suffix
    digest = sha256_file(tmp)
    n = 1
    while True:
        target = vault / (name if n == 1 else f"{stem} ({n}){suffix}")
        if not target.exists():
            os.replace(tmp, target)
            return target
        if target.is_file() and sha256_file(target) == digest:
            tmp.unlink()
            return target
        n += 1


def open_with_os(path: Path) -> None:
    """Open a file in the OS default app."""
    if sys.platform == "win32":
        os.startfile(path)
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])


def _row(
    key: str,
    path: Path,
    status: str,
    error: str | None,
    indexed_at: datetime | None,
    *,
    vault: Path,
) -> dict[str, Any]:
    folder = _relative_folder(path, vault)
    return {
        "key": key,
        "name": path.name,
        "folder": str(path.parent) if folder is None else folder,
        "status": status,
        "error": (error or "") if status == "failed" else "",
        "indexed": indexed_at.astimezone().strftime("%Y-%m-%d %H:%M") if indexed_at else "",
        "path": str(path),
        "in_vault": folder is not None,
    }


def _key(path: Path) -> str:
    """Registry paths and event paths are both resolved; only the case may differ (Windows)."""
    return os.path.normcase(str(path))


def _relative_folder(path: Path, vault: Path) -> str | None:
    """The file's folder relative to the vault ("" at the top), None when it's outside."""
    if not _is_within(path, vault):
        return None
    relative = Path(os.path.normcase(path.parent)).relative_to(os.path.normcase(vault))
    if relative == Path("."):
        return ""
    return str(path.parent)[-len(str(relative)) :]  # same length, original case


def _is_within(path: Path, vault: Path) -> bool:
    parent, root = Path(os.path.normcase(path.parent)), Path(os.path.normcase(vault))
    return parent == root or parent.is_relative_to(root)
