"""Background ingestion: one worker thread indexes files one at a time, so the UI never blocks."""

from __future__ import annotations

import logging
import os
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from vaultmind.core.config import Config
from vaultmind.core.models import file_id_for
from vaultmind.core.providers import Embedder
from vaultmind.ingest.parsers import SUPPORTED_EXTENSIONS
from vaultmind.ingest.pipeline import IngestSummary, forget_file, ingest_path
from vaultmind.store.index import ChunkIndex
from vaultmind.store.registry import Registry

log = logging.getLogger(__name__)

JobState = Literal["queued", "indexing", "ready", "failed", "removed"]
_JobKind = Literal["index", "forget", "prune"]  # prune: forget everything outside a folder
_Job = tuple[_JobKind, Path]


class _Stopped(Exception):
    """Raised from `on_file` to end a folder job between files once `stop()` was called."""


@dataclass(frozen=True)
class JobEvent:
    """A file changed state. `error` is set for `failed`."""

    path: Path
    state: JobState
    error: str | None = None


class IngestQueue:
    """Feeds `ingest_path` / `forget_file` from a queue on one worker thread.

    `on_event` is called from the worker thread (a UI must hop to its own loop). The one
    exception is `queued`, which is emitted by `submit` on the caller's thread, before the
    job becomes visible to the worker, so it always precedes that file's other events.
    A failing file, including Ollama being unreachable, emits `failed` and the worker keeps
    running.
    """

    def __init__(
        self,
        cfg: Config,
        *,
        embedder: Embedder,
        index: ChunkIndex,
        registry: Registry,
        on_event: Callable[[JobEvent], None] | None = None,
    ) -> None:
        self._cfg = cfg
        self._embedder = embedder
        self._index = index
        self._registry = registry
        self._on_event = on_event
        # RLock: an `on_event` callback for `queued` runs under it and may call `submit`.
        self._cond = threading.Condition(threading.RLock())
        self._jobs: deque[_Job] = deque()
        self._waiting: set[tuple[_JobKind, str]] = set()
        self._running: _Job | None = None
        self._stopped = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        with self._cond:
            if self._thread is not None or self._stopped:
                return
            self._thread = threading.Thread(target=self._work, name="vaultmind-ingest", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        """Finish the current file, drop what's still waiting, join the worker. Idempotent."""
        with self._cond:
            self._stopped = True
            self._jobs.clear()
            self._waiting.clear()
            self._cond.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)

    def submit(self, path: Path) -> None:
        """Index a file or folder (new or changed files only). A path already waiting is skipped."""
        path = path.resolve()
        with self._cond:
            if not self._enqueue("index", path):
                return
            # Emitted under the lock so the worker can't report `indexing` before `queued`.
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                self._emit(JobEvent(path, "queued"))

    def forget(self, path: Path) -> None:
        """A file was deleted or moved away: drop it from the index and registry (on the worker).

        A folder path forgets every registered file under it (a deleted folder gives the
        watcher a single event on Windows).
        """
        with self._cond:
            self._enqueue("forget", path.resolve())

    def scan(self) -> None:
        """Catch up with `paths.vault_dir`: index what changed while the app was closed, forget
        files that are gone, and forget registry files outside the vault (the vault moved)."""
        vault = self._cfg.paths.vault_dir
        vault.mkdir(parents=True, exist_ok=True)
        vault = vault.resolve()
        with self._cond:
            self._enqueue("prune", vault)
            self._enqueue("index", vault)

    def pending(self) -> int:
        """Jobs waiting or running."""
        with self._cond:
            return self._pending()

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Block until nothing is pending; False on timeout."""
        with self._cond:
            return self._cond.wait_for(lambda: self._pending() == 0, timeout)

    # -- queue internals (callers hold the lock) --

    def _pending(self) -> int:
        return len(self._jobs) + (self._running is not None)

    def _enqueue(self, kind: _JobKind, path: Path) -> bool:
        key = (kind, file_id_for(path))
        if self._stopped or key in self._waiting:
            return False
        self._waiting.add(key)
        self._jobs.append((kind, path))
        self._cond.notify_all()
        return True

    # -- worker --

    def _work(self) -> None:
        while True:
            with self._cond:
                while not self._jobs and not self._stopped:
                    self._cond.wait()
                if self._stopped:
                    self._cond.notify_all()
                    return
                job = self._jobs.popleft()
                self._waiting.discard((job[0], file_id_for(job[1])))
                self._running = job
            try:
                self._run(job)
            except Exception:  # never let one job kill the worker
                log.exception("ingest job %s failed", job)
            finally:
                with self._cond:
                    self._running = None
                    self._cond.notify_all()

    def _run(self, job: _Job) -> None:
        kind, path = job
        if kind == "index":
            self._index_path(path)
        elif kind == "forget":
            self._forget(path)
        else:
            self._prune(path)

    def _index_path(self, path: Path) -> None:
        reported: set[Path] = set()
        current: list[Path] = []  # the file ingest_path is working on

        def on_file(file: Path) -> None:
            if current:
                self._report_from_registry(current[-1], reported)
            if self._stopped:
                raise _Stopped  # nothing of `file` is written yet
            current.append(file)
            self._emit(JobEvent(file, "indexing"))

        try:
            summary = ingest_path(
                path,
                self._cfg,
                embedder=self._embedder,
                index=self._index,
                registry=self._registry,
                on_file=on_file,
            )
        except _Stopped:
            return
        except Exception as e:
            message = _describe(e)
            log.warning("indexing %s failed: %s", path, message)
            # ingest_path marks the file it was indexing as failed before raising; if that
            # file isn't marked, the error came from outside it (e.g. reading a locked file).
            failed = current[-1] if current else path
            record = self._registry.get(file_id_for(failed))
            if record is None or record.status != "failed":
                failed = path
            if current and current[-1] != failed:
                self._report_from_registry(current[-1], reported)
            self._emit(JobEvent(failed, "failed", message))
            return
        self._report_summary(path, summary, reported)

    def _report_from_registry(self, file: Path, reported: set[Path]) -> None:
        """The file before this one is done: say how it ended without waiting for the folder."""
        record = self._registry.get(file_id_for(file))
        if record is None or record.status not in ("ready", "failed"):
            return
        reported.add(file)
        if record.status == "ready":
            self._emit(JobEvent(file, "ready"))
        else:
            self._emit(JobEvent(file, "failed", record.error))

    def _report_summary(self, path: Path, summary: IngestSummary, reported: set[Path]) -> None:
        for file in summary.added + summary.updated:
            if file not in reported:
                self._emit(JobEvent(file, "ready"))
        for file, reason in summary.failed:
            if file not in reported:
                self._emit(JobEvent(file, "failed", reason))
        for file in summary.removed:
            self._emit(JobEvent(file, "removed"))
        # A single file got `queued` on submit; close that state even when nothing ran.
        if path.suffix.lower() in SUPPORTED_EXTENSIONS and not path.is_dir():
            if path in summary.skipped:
                self._emit(JobEvent(path, "ready"))
            elif not path.exists() and path not in summary.removed:
                self._emit(JobEvent(path, "removed"))

    def _forget(self, path: Path) -> None:
        if forget_file(path, index=self._index, registry=self._registry):
            self._emit(JobEvent(path, "removed"))
            return
        root = _norm(path)
        self._drop([r.path for r in self._registry.all() if _is_within(_norm(r.path), root)])

    def _prune(self, vault: Path) -> None:
        root = _norm(vault)
        self._drop([r.path for r in self._registry.all() if not _is_within(_norm(r.path), root)])

    def _drop(self, files: list[Path]) -> None:
        """Forget several files with one FTS rebuild."""
        for file in files:
            file_id = file_id_for(file)
            self._index.delete_file(file_id)
            self._registry.remove(file_id)
        if files:
            self._index.refresh_fts()
        for file in files:
            self._emit(JobEvent(file, "removed"))

    def _emit(self, event: JobEvent) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(event)
        except Exception:
            log.exception("on_event callback failed for %s", event)


def _describe(exc: Exception) -> str:
    text = str(exc).strip()
    return text if text else type(exc).__name__


def _norm(path: Path) -> Path:
    """Resolved and case-folded the way `file_id_for` does, for prefix comparisons."""
    return Path(os.path.normcase(str(path.resolve())))


def _is_within(path: Path, root: Path) -> bool:
    return path == root or path.is_relative_to(root)
