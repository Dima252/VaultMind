"""Background ingestion: one worker thread indexes files one at a time, so the UI never blocks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from vaultmind.core.config import Config
from vaultmind.core.providers import Embedder
from vaultmind.store.index import ChunkIndex
from vaultmind.store.registry import Registry

JobState = Literal["queued", "indexing", "ready", "failed", "removed"]


@dataclass(frozen=True)
class JobEvent:
    """A file changed state. `error` is set for `failed`."""

    path: Path
    state: JobState
    error: str | None = None


class IngestQueue:
    """Feeds `ingest_path` / `forget_file` from a queue on one worker thread.

    `on_event` is called from the worker thread (a UI must hop to its own loop). A failing
    file, including Ollama being unreachable, emits `failed` and the worker keeps running.
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
        raise NotImplementedError

    def start(self) -> None:
        raise NotImplementedError

    def stop(self, timeout: float = 10.0) -> None:
        """Finish the current file, drop what's still waiting, join the worker. Idempotent."""
        raise NotImplementedError

    def submit(self, path: Path) -> None:
        """Index a file or folder (new or changed files only). A path already waiting is skipped."""
        raise NotImplementedError

    def forget(self, path: Path) -> None:
        """A file was deleted or moved away: drop it from the index and registry (on the worker)."""
        raise NotImplementedError

    def scan(self) -> None:
        """Catch up with `paths.vault_dir`: index what changed while the app was closed, forget
        files that are gone, and forget registry files outside the vault (the vault moved)."""
        raise NotImplementedError

    def pending(self) -> int:
        """Jobs waiting or running."""
        raise NotImplementedError

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Block until nothing is pending; False on timeout."""
        raise NotImplementedError
