"""Live folder watching: file changes under the vault go to the IngestQueue once they settle."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from watchdog.events import (
    DirCreatedEvent,
    DirDeletedEvent,
    DirModifiedEvent,
    DirMovedEvent,
    FileClosedEvent,
    FileClosedNoWriteEvent,
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
    FileOpenedEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

from vaultmind.ingest.parsers import SUPPORTED_EXTENSIONS
from vaultmind.ingest.queue import IngestQueue

log = logging.getLogger(__name__)

_TEMP_SUFFIXES = frozenset({".tmp", ".part", ".crdownload", ".swp"})

_AnyEvent = (
    DirCreatedEvent
    | DirDeletedEvent
    | DirModifiedEvent
    | DirMovedEvent
    | FileClosedEvent
    | FileClosedNoWriteEvent
    | FileCreatedEvent
    | FileDeletedEvent
    | FileModifiedEvent
    | FileMovedEvent
    | FileOpenedEvent
)


class VaultWatcher:
    """A watchdog observer on `root` (recursive) that feeds `queue`.

    Created or modified supported files are submitted after `debounce_s` without further
    changes; deleted ones are forgotten; a move is a forget plus a submit. Temporary and
    hidden files are ignored.
    """

    def __init__(self, root: Path, queue: IngestQueue, *, debounce_s: float = 2.0) -> None:
        self._root = root.resolve()
        self._queue = queue
        self._debounce_s = debounce_s
        self._lock = threading.Lock()
        self._timers: dict[str, threading.Timer] = {}
        # (mtime, size) at the last submit: Windows also reports reads and attribute changes
        # as modifications, and the worker reading a file must not resubmit it forever.
        self._submitted: dict[str, tuple[int, int]] = {}
        self._observer: Observer | None = None  # type: ignore[valid-type]
        self._stopped = False

    def start(self) -> None:
        with self._lock:
            if self._observer is not None or self._stopped:
                return
            self._root.mkdir(parents=True, exist_ok=True)
            self._remember_existing()
            observer = Observer()
            observer.daemon = True
            observer.schedule(_Handler(self), str(self._root), recursive=True)
            observer.start()
            self._observer = observer

    def stop(self) -> None:
        """Idempotent."""
        with self._lock:
            self._stopped = True
            for timer in self._timers.values():
                timer.cancel()
            self._timers.clear()
            observer, self._observer = self._observer, None
        if observer is not None:
            observer.stop()
            observer.join()

    def _remember_existing(self) -> None:
        """Record the files already here (the startup scan handles them). NTFS updates access
        times about once an hour, so the scan's own reads would otherwise come back as
        "modified" events and resubmit every file."""
        for path in self._root.rglob("*"):
            if not self._wanted(path):
                continue
            try:
                st = path.stat()
            except OSError:
                continue
            if path.is_file():
                self._submitted[_key(path)] = (st.st_mtime_ns, st.st_size)

    # -- events (on the observer thread) --

    def handle(self, event: _AnyEvent) -> None:
        try:
            self._handle(event)
        except Exception:  # an exception here would kill the observer thread
            log.exception("watcher failed to handle %s", event)

    def _handle(self, event: _AnyEvent) -> None:
        src = Path(os.fsdecode(event.src_path))
        if isinstance(event, FileMovedEvent | DirMovedEvent):
            dest = Path(os.fsdecode(event.dest_path))
            self._removed(src)
            if event.is_directory:
                if self._inside(dest) and not self._ignored(dest):
                    self._schedule(dest, fresh=True)
            elif self._wanted(dest):
                self._schedule(dest, fresh=True)
        elif isinstance(event, FileDeletedEvent | DirDeletedEvent):
            # On Windows a deleted folder arrives as one FileDeletedEvent: forget handles both.
            self._removed(src)
        elif isinstance(event, FileCreatedEvent):
            if self._wanted(src):
                self._schedule(src, fresh=True)
        elif isinstance(event, FileModifiedEvent | FileClosedEvent):
            if self._wanted(src):
                self._schedule(src)
        elif isinstance(event, DirCreatedEvent):
            # A folder moved in from outside the vault; its files may not get events of their own.
            if self._inside(src) and not self._ignored(src):
                self._schedule(src, fresh=True)

    def _removed(self, path: Path) -> None:
        if not self._inside(path) or self._ignored(path):
            return
        self._cancel_under(path)
        self._queue.forget(path)

    # -- filters --

    def _inside(self, path: Path) -> bool:
        resolved = _norm(path)
        root = _norm(self._root)
        return resolved == root or resolved.is_relative_to(root)

    def _ignored(self, path: Path) -> bool:
        """Hidden (any part under root starts with '.') or an editor/download temp file."""
        try:
            parts = _norm(path).relative_to(_norm(self._root)).parts
        except ValueError:
            return True
        if any(part.startswith(".") for part in parts):
            return True
        name = path.name.lower()
        return name.startswith("~$") or Path(name).suffix in _TEMP_SUFFIXES

    def _wanted(self, path: Path) -> bool:
        return (
            path.suffix.lower() in SUPPORTED_EXTENSIONS
            and self._inside(path)
            and not self._ignored(path)
        )

    # -- debounce --

    def _schedule(self, path: Path, *, fresh: bool = False) -> None:
        """Submit `path` after `debounce_s` of quiet. `fresh`: new at this path, always submit."""
        key = _key(path)
        with self._lock:
            if self._stopped:
                return
            if fresh:
                self._submitted.pop(key, None)
            old = self._timers.pop(key, None)
            if old is not None:
                old.cancel()
            timer = threading.Timer(self._debounce_s, self._fire, args=(key, path))
            timer.daemon = True
            self._timers[key] = timer
            timer.start()

    def _fire(self, key: str, path: Path) -> None:
        with self._lock:
            if self._stopped or self._timers.get(key) is not threading.current_thread():
                return  # replaced by a newer event, or stopped
            del self._timers[key]
            try:
                st = path.stat()
            except OSError:
                return  # gone again; its delete event does the rest
            stamp = (st.st_mtime_ns, st.st_size)
            if path.is_file():
                if self._submitted.get(key) == stamp:
                    return  # touched or read, not changed
                self._submitted[key] = stamp
        try:
            self._queue.submit(path)
        except Exception:
            log.exception("submitting %s failed", path)

    def _cancel_under(self, path: Path) -> None:
        """A path (file or folder) is gone: drop pending submits for it and anything under it."""
        prefix = _key(path)
        with self._lock:
            for key in [k for k in self._timers if _under(k, prefix)]:
                self._timers.pop(key).cancel()
            for key in [k for k in self._submitted if _under(k, prefix)]:
                del self._submitted[key]


class _Handler(FileSystemEventHandler):
    def __init__(self, watcher: VaultWatcher) -> None:
        self._watcher = watcher

    def on_any_event(self, event: _AnyEvent) -> None:  # type: ignore[override]
        self._watcher.handle(event)


def _norm(path: Path) -> Path:
    return Path(os.path.normcase(os.path.abspath(path)))


def _key(path: Path) -> str:
    return str(_norm(path))


def _under(key: str, prefix: str) -> bool:
    return key == prefix or key.startswith(prefix.rstrip(os.sep) + os.sep)
