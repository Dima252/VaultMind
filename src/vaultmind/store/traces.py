"""Agent runs and their trace events in the app database (`runs`, `trace_events` tables).

The schema lives in `store/registry.py` (migration 2); opening a TraceStore applies `migrate`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from vaultmind.agent.types import AgentResult, TraceEvent


@dataclass(frozen=True)
class StoredRun:
    id: str
    created_at: datetime  # UTC
    question: str
    answer: str
    abstained: bool
    events: list[TraceEvent]  # in the order they happened


class TraceStore:
    def __init__(self, db_path: Path) -> None:
        raise NotImplementedError

    def save(self, result: AgentResult) -> str:
        """Store a run with its trace; returns the new run id."""
        raise NotImplementedError

    def get(self, run_id: str) -> StoredRun | None:
        raise NotImplementedError

    def recent(self, limit: int = 20) -> list[StoredRun]:
        """Newest first."""
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
