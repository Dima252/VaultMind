"""Chat page: chat list, streamed answers with citation chips, trace and export per answer."""

from __future__ import annotations

from nicegui import ui


def page() -> None:
    """Build the chat page. Uses `state().engine` (ready when this is called)."""
    ui.label("Chat (wave 2)")
