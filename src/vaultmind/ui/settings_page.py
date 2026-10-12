"""Settings page: vault folder, models, retrieval/agent switches, saved to config/local.yaml."""

from __future__ import annotations

from nicegui import ui


def page() -> None:
    """Build the settings page. Works without an engine (e.g. to fix a model name)."""
    ui.label("Settings (wave 2)")
