import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from vaultmind.agent.types import AgentResult, TraceEvent
from vaultmind.core.models import FileRecord
from vaultmind.store.registry import Registry
from vaultmind.store.traces import TraceStore

HEBREW_Q = "מה תקופת ההודעה המוקדמת?"
HEBREW_A = "שלושים יום [1]."


def _result(question: str = "q", answer: str = "a [1].", *, abstained: bool = False) -> AgentResult:
    return AgentResult(
        question=question,
        answer=answer,
        sources=[],
        abstained=abstained,
        queries=[question],
        rewrites=0,
        trace=[
            TraceEvent("analyze", 12.5, {"standalone_query": question, "variants": ["v1"]}),
            TraceEvent("retrieve", 3.0, {"queries": [question, "v1"], "candidates": 7}),
            TraceEvent(
                "grade",
                40.25,
                {"threshold": 0.3, "passed": 1, "scores": [{"label": "x", "score": 0.9}]},
            ),
            TraceEvent("generate", 800.0, {"sources": [1], "abstained": abstained}),
        ],
    )


@pytest.fixture
def store(tmp_path: Path) -> Iterator[TraceStore]:
    s = TraceStore(tmp_path / "data" / "vaultmind.sqlite")
    yield s
    s.close()


def test_round_trip(store: TraceStore) -> None:
    result = _result()
    before = datetime.now(UTC)
    run_id = store.save(result)
    after = datetime.now(UTC)

    assert len(run_id) == 32 and int(run_id, 16) >= 0
    run = store.get(run_id)
    assert run is not None
    assert run.id == run_id
    assert (run.question, run.answer, run.abstained) == ("q", "a [1].", False)
    assert run.created_at.tzinfo is not None
    assert before - timedelta(seconds=1) <= run.created_at <= after + timedelta(seconds=1)
    assert run.events == result.trace  # same order, durations and data


def test_abstained_and_empty_trace(store: TraceStore) -> None:
    result = AgentResult("q", "not found", [], True, ["q"], 2)
    run = store.get(store.save(result))
    assert run is not None and run.abstained and run.events == []


def test_unknown_id(store: TraceStore) -> None:
    assert store.get("0" * 32) is None
    assert store.get("") is None


def test_recent_newest_first_with_events(store: TraceStore) -> None:
    ids = [store.save(_result(f"q{i}")) for i in range(4)]
    recent = store.recent()
    assert [r.id for r in recent] == ids[::-1]
    assert all(len(r.events) == 4 for r in recent)
    assert recent[0].events[0].data["standalone_query"] == "q3"
    assert [r.id for r in store.recent(2)] == ids[:1:-1]
    assert store.recent(0) == []


def test_unicode_text(store: TraceStore) -> None:
    result = _result(HEBREW_Q, HEBREW_A)
    run = store.get(store.save(result))
    assert run is not None
    assert (run.question, run.answer) == (HEBREW_Q, HEBREW_A)
    assert run.events[0].data["standalone_query"] == HEBREW_Q


def test_data_stored_as_compact_utf8_json(tmp_path: Path) -> None:
    db = tmp_path / "vaultmind.sqlite"
    store = TraceStore(db)
    store.save(_result(HEBREW_Q))
    store.close()
    conn = sqlite3.connect(db)
    try:
        data = conn.execute("SELECT data FROM trace_events ORDER BY seq LIMIT 1").fetchone()[0]
    finally:
        conn.close()
    assert HEBREW_Q in data and ", " not in data and ": " not in data


def test_persists_across_instances(tmp_path: Path) -> None:
    db = tmp_path / "vaultmind.sqlite"
    first = TraceStore(db)
    run_id = first.save(_result())
    first.close()
    second = TraceStore(db)
    try:
        run = second.get(run_id)
        assert run is not None and len(run.events) == 4
    finally:
        second.close()


def test_close_is_idempotent(tmp_path: Path) -> None:
    store = TraceStore(tmp_path / "vaultmind.sqlite")
    store.close()
    store.close()
    with pytest.raises(RuntimeError, match="closed"):
        store.recent()


def test_failed_save_writes_nothing(store: TraceStore) -> None:
    # The second event violates NOT NULL after the run row went in: the whole save rolls back.
    trace = [TraceEvent("grade", 1.0), TraceEvent(None, 2.0)]  # type: ignore[arg-type]
    with pytest.raises(sqlite3.IntegrityError):
        store.save(AgentResult("q", "a", [], False, ["q"], 0, trace=trace))
    assert store.recent() == []
    store.save(_result())  # the connection is usable afterwards
    assert len(store.recent()) == 1


def test_side_by_side_with_registry(tmp_path: Path) -> None:
    db = tmp_path / "vaultmind.sqlite"
    registry = Registry(db)
    store = TraceStore(db)
    try:
        registry.upsert(FileRecord("f1", tmp_path / "a.md", "sha", "ready"))
        run_id = store.save(_result())
        registry.set_status("f1", "failed", "boom")
        assert store.get(run_id) is not None
        record = registry.get("f1")
        assert record is not None and record.status == "failed"
    finally:
        store.close()
        registry.close()

    # Either order of opening works on an existing file.
    store = TraceStore(db)
    registry = Registry(db)
    try:
        assert [r.id for r in store.recent()] == [run_id]
        assert [r.id for r in registry.all()] == ["f1"]
    finally:
        registry.close()
        store.close()
