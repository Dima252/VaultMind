"""Collapsible trace timeline under an answer: queries, passage scores, per-step timing."""

from __future__ import annotations

from collections.abc import Sequence

from nicegui import ui

from vaultmind.agent.types import TraceEvent


def trace_view(events: Sequence[TraceEvent]) -> None:
    """Build the trace for one answer in the current container (collapsed by default)."""
    ui.label(f"{len(events)} trace events (wave 2)")
