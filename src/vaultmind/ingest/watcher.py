"""Live folder watching: file changes under the vault go to the IngestQueue once they settle."""

from __future__ import annotations

from pathlib import Path

from vaultmind.ingest.queue import IngestQueue


class VaultWatcher:
    """A watchdog observer on `root` (recursive) that feeds `queue`.

    Created or modified supported files are submitted after `debounce_s` without further
    changes; deleted ones are forgotten; a move is a forget plus a submit. Temporary and
    hidden files are ignored.
    """

    def __init__(self, root: Path, queue: IngestQueue, *, debounce_s: float = 2.0) -> None:
        raise NotImplementedError

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        """Idempotent."""
        raise NotImplementedError
