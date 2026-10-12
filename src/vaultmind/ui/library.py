"""Library page: drag-and-drop upload, per-file status, re-index and remove."""

from __future__ import annotations

from nicegui import ui


def page() -> None:
    """Build the library page. Uses `state().engine` (ready when this is called)."""
    ui.label("Library (wave 2)")


def queue_indicator() -> None:
    """Header element: "Indexing N files" while the ingest queue is busy, hidden otherwise."""
