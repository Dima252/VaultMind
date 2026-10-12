"""Collapsible trace timeline under an answer: queries, passage scores, per-step timing."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from nicegui import ui

from vaultmind.agent.trace_text import summarize
from vaultmind.agent.types import TraceEvent


def trace_view(events: Sequence[TraceEvent]) -> None:
    """Build the trace for one answer in the current container (collapsed by default)."""
    if not events:
        ui.label("No trace events were recorded for this answer.").classes("text-sm text-grey")
        return
    with ui.column().classes("w-full gap-1 text-sm"):
        for event in events:
            _event_row(event)
            if event.node == "grade":
                _grade_table(event.data.get("scores") or [])
        total = sum(e.duration_ms for e in events)
        with ui.row().classes("w-full no-wrap gap-3 border-t pt-1 font-semibold"):
            ui.label("total").classes("w-32 shrink-0 font-mono")
            ui.label(f"{total:.0f} ms").classes("w-20 shrink-0 text-right font-mono")


def _event_row(event: TraceEvent) -> None:
    with ui.row().classes("w-full no-wrap gap-3 items-baseline"):
        ui.label(event.node).classes("w-32 shrink-0 font-mono")
        ui.label(f"{event.duration_ms:.0f} ms").classes("w-20 shrink-0 text-right font-mono")
        ui.label(_summary(event)).classes("min-w-0 break-words").props('dir="auto"')
    if "error" in event.data:
        ui.label(f"error: {event.data['error']}").classes("pl-36 text-negative")


def _summary(event: TraceEvent) -> str:
    try:
        return summarize(event.node, event.data)
    except (AttributeError, KeyError, TypeError, ValueError):  # unexpected data shape
        return json.dumps(event.data, ensure_ascii=False, separators=(",", ":"))


def _grade_table(scores: Sequence[Any]) -> None:
    """Passage scores, best first: label, a bar relative to the best score, passed or not."""
    rows = [s for s in scores if isinstance(s, dict) and "score" in s]
    if not rows:
        return
    rows.sort(key=lambda s: float(s["score"]), reverse=True)
    top = max(abs(float(s["score"])) for s in rows) or 1.0
    with ui.element("table").classes("ml-36 my-1 text-xs border-collapse"):
        for s in rows:
            score = float(s["score"])
            passed = bool(s.get("passed"))
            with ui.element("tr").classes("align-middle"):
                with ui.element("td").classes("pr-2"):
                    ui.icon("check" if passed else "close", size="xs").classes(
                        "text-positive" if passed else "text-negative"
                    ).tooltip("passed" if passed else "failed")
                with ui.element("td").classes("pr-3 max-w-md break-words"):
                    ui.label(str(s.get("label", ""))).props('dir="auto"')
                with ui.element("td").classes("pr-2 w-24"):
                    width = max(0.0, min(1.0, score / top)) * 100
                    with ui.element("div").classes("h-2 rounded bg-grey-3 w-24"):
                        ui.element("div").classes(
                            "h-2 rounded " + ("bg-positive" if passed else "bg-grey-6")
                        ).style(f"width: {width:.0f}%")
                with ui.element("td").classes("font-mono text-right"):
                    ui.label(f"{score:.3f}")
