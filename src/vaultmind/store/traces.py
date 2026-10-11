"""Agent runs and their trace events in the app database (`runs`, `trace_events` tables).

The schema lives in `store/registry.py` (migration 2); opening a TraceStore applies `migrate`.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from vaultmind.agent.types import AgentResult, TraceEvent
from vaultmind.store.registry import migrate


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
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = sqlite3.connect(
            db_path, check_same_thread=False, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            migrate(self._conn)
        except BaseException:
            self.close()
            raise

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("trace store is closed")
        return self._conn

    def save(self, result: AgentResult) -> str:
        """Store a run with its trace; returns the new run id."""
        run_id = uuid.uuid4().hex
        created_at = datetime.now(UTC).isoformat()
        events = [
            (
                run_id,
                seq,
                e.node,
                e.duration_ms,
                json.dumps(e.data, ensure_ascii=False, separators=(",", ":")),
            )
            for seq, e in enumerate(result.trace)
        ]
        with self._lock:
            db = self._db()
            db.execute("BEGIN")
            try:
                db.execute(
                    "INSERT INTO runs (id, created_at, question, answer, abstained) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (run_id, created_at, result.question, result.answer, int(result.abstained)),
                )
                db.executemany(
                    "INSERT INTO trace_events (run_id, seq, node, duration_ms, data) "
                    "VALUES (?, ?, ?, ?, ?)",
                    events,
                )
            except BaseException:
                db.execute("ROLLBACK")
                raise
            db.execute("COMMIT")
        return run_id

    def get(self, run_id: str) -> StoredRun | None:
        with self._lock:
            db = self._db()
            row = db.execute(
                "SELECT id, created_at, question, answer, abstained FROM runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            events = self._events(db, [run_id])
        return _to_run(row, events.get(run_id, []))

    def recent(self, limit: int = 20) -> list[StoredRun]:
        """Newest first."""
        with self._lock:
            db = self._db()
            rows = db.execute(
                "SELECT id, created_at, question, answer, abstained FROM runs "
                "ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
            events = self._events(db, [r["id"] for r in rows])
        return [_to_run(r, events.get(r["id"], [])) for r in rows]

    @staticmethod
    def _events(db: sqlite3.Connection, run_ids: list[str]) -> dict[str, list[TraceEvent]]:
        if not run_ids:
            return {}
        marks = ", ".join("?" * len(run_ids))
        rows = db.execute(
            "SELECT run_id, node, duration_ms, data FROM trace_events "
            f"WHERE run_id IN ({marks}) ORDER BY run_id, seq",
            run_ids,
        ).fetchall()
        events: dict[str, list[TraceEvent]] = {}
        for r in rows:
            event = TraceEvent(r["node"], float(r["duration_ms"]), json.loads(r["data"]))
            events.setdefault(r["run_id"], []).append(event)
        return events

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


def _to_run(row: sqlite3.Row, events: list[TraceEvent]) -> StoredRun:
    return StoredRun(
        id=row["id"],
        created_at=datetime.fromisoformat(row["created_at"]),
        question=row["question"],
        answer=row["answer"],
        abstained=bool(row["abstained"]),
        events=events,
    )
