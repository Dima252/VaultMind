"""Plain-text trace summaries, shared by `vaultmind ask --trace` / `trace` and the app."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from vaultmind.agent.types import TraceEvent


def short(text: object, limit: int = 80) -> str:
    """`text` on one line, quoted, cut to about `limit` characters."""
    flat = " ".join(str(text).split())
    if len(flat) > limit:
        flat = flat[: limit - 3] + "..."
    return f'"{flat}"'


def _compact(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def summarize(node: str, data: dict[str, Any]) -> str:
    """One-line summary of a node's trace data (keys per `agent.types`; extras ignored)."""
    if node == "analyze":
        parts = [f"query {short(data.get('standalone_query', ''))}"]
        variants = data.get("variants") or []
        if variants:
            parts.append("variants " + " | ".join(short(v, 50) for v in variants))
        if data.get("needs_retrieval") is False:
            parts.append("no retrieval needed")
        return ", ".join(parts)
    if node == "retrieve":
        queries = data.get("queries") or []
        return f"{len(queries)} queries -> {data.get('candidates', '?')} candidates"
    if node == "grade":
        scores = data.get("scores") or []
        threshold = data.get("threshold")
        limit = "no threshold" if threshold is None else f"threshold {float(threshold):.2f}"
        top = sorted((float(s["score"]) for s in scores), reverse=True)[:3]
        text = f"{data.get('passed', '?')}/{len(scores)} passed ({limit})"
        return text + (", top " + " ".join(f"{v:.2f}" for v in top) if top else "")
    if node == "rewrite":
        return f"attempt {data.get('attempt', '?')}: {short(data.get('query', ''))}"
    if node == "contradictions":
        found = data.get("found") or []
        pairs = " ".join(f"[{c.get('source_a')}] vs [{c.get('source_b')}]" for c in found)
        return f"{len(found)} found" + (f": {pairs}" if pairs else "")
    if node == "generate":
        shown = "".join(f"[{n}]" for n in data.get("sources") or []) or "none"
        return f"sources {shown}" + (", abstained" if data.get("abstained") else "")
    if node == "not_found":
        closest = data.get("closest") or []
        return "closest: " + ("; ".join(str(c) for c in closest) or "none")
    if node == "groundedness":
        return f"{len(data.get('unsupported') or [])} unsupported claim(s)"
    if node == "direct":
        return f"no retrieval: {short(data.get('reason', ''))}"
    return _compact(data)


def format_trace(events: Sequence[TraceEvent]) -> list[str]:
    """One line per event (node, duration, summary of its data), then the total."""
    lines: list[str] = []
    for e in events:
        try:
            summary = summarize(e.node, e.data)
        except (AttributeError, KeyError, TypeError, ValueError):  # unexpected data shape
            summary = _compact(e.data)
        if "error" in e.data:
            summary += f" [error: {e.data['error']}]"
        lines.append(f"  {e.node:<14} {e.duration_ms:>7.0f} ms  {summary}".rstrip())
    total = sum(e.duration_ms for e in events)
    lines.append(f"  {'total':<14} {total:>7.0f} ms")
    return lines
