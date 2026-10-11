"""Registry: SQLite record of ingested files and their status."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from vaultmind.core.models import FileRecord, FileStatus
from vaultmind.store.registry import SCHEMA_VERSION, Registry


@pytest.fixture
def registry(tmp_path: Path) -> Iterator[Registry]:
    reg = Registry(tmp_path / "nested" / "dir" / "registry.db")
    yield reg
    reg.close()


def _record(file_id: str = "f1", path: str = "docs/a.md") -> FileRecord:
    return FileRecord(id=file_id, path=Path(path), sha256="abc123", status="queued")


def test_upsert_get_round_trip(registry: Registry) -> None:
    record = FileRecord(
        id="f1",
        path=Path("docs/a.md"),
        sha256="abc",
        status="ready",
        error=None,
        embedder="fake-embedder",
        indexed_at=datetime(2026, 1, 2, 3, 4, 5, 678901, tzinfo=UTC),
    )
    registry.upsert(record)
    assert registry.get("f1") == record


def test_none_fields_round_trip(registry: Registry) -> None:
    record = _record()
    registry.upsert(record)
    got = registry.get("f1")
    assert got == record
    assert got is not None and got.indexed_at is None and got.embedder is None


def test_naive_datetime_is_stored_as_utc(registry: Registry) -> None:
    registry.upsert(replace(_record(), indexed_at=datetime(2026, 5, 6, 7, 8, 9)))
    got = registry.get("f1")
    assert got is not None
    assert got.indexed_at == datetime(2026, 5, 6, 7, 8, 9, tzinfo=UTC)


def test_upsert_replaces(registry: Registry) -> None:
    registry.upsert(_record())
    registry.upsert(replace(_record(), sha256="new"))
    got = registry.get("f1")
    assert got is not None and got.sha256 == "new"
    assert len(registry.all()) == 1


def test_get_unknown_is_none(registry: Registry) -> None:
    assert registry.get("missing") is None


def test_set_status(registry: Registry) -> None:
    registry.upsert(_record())
    registry.set_status("f1", "failed", "parse error")
    got = registry.get("f1")
    assert got is not None and (got.status, got.error) == ("failed", "parse error")

    registry.set_status("f1", "ready")
    got = registry.get("f1")
    assert got is not None and (got.status, got.error) == ("ready", None)


def test_set_status_unknown_raises(registry: Registry) -> None:
    with pytest.raises(KeyError):
        registry.set_status("missing", "ready")


def test_invalid_status_rejected(registry: Registry) -> None:
    registry.upsert(_record())
    with pytest.raises(sqlite3.IntegrityError):
        registry.set_status("f1", "bogus")  # type: ignore[arg-type]


def test_all_ordered_by_path(registry: Registry) -> None:
    registry.upsert(_record("f1", "c.md"))
    registry.upsert(_record("f2", "a.md"))
    registry.upsert(_record("f3", "b.md"))
    assert [r.path for r in registry.all()] == [Path("a.md"), Path("b.md"), Path("c.md")]


def test_remove(registry: Registry) -> None:
    registry.upsert(_record())
    registry.remove("f1")
    registry.remove("f1")  # removing twice is fine
    assert registry.get("f1") is None


def test_persists_across_reopen(tmp_path: Path) -> None:
    db = tmp_path / "registry.db"
    reg = Registry(db)
    reg.upsert(_record())
    reg.close()

    reopened = Registry(db)
    try:
        assert reopened.get("f1") == _record()
    finally:
        reopened.close()


def test_schema_version_and_wal(tmp_path: Path) -> None:
    db = tmp_path / "registry.db"
    Registry(db).close()
    conn = sqlite3.connect(db)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        conn.close()


def test_newer_schema_refused(tmp_path: Path) -> None:
    db = tmp_path / "registry.db"
    conn = sqlite3.connect(db)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    conn.close()
    with pytest.raises(RuntimeError, match="newer"):
        Registry(db)


def test_close_is_idempotent(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "registry.db")
    reg.close()
    reg.close()
    with pytest.raises(RuntimeError, match="closed"):
        reg.get("f1")


def test_concurrent_set_status(registry: Registry) -> None:
    ids = [f"f{i}" for i in range(8)]
    for i, file_id in enumerate(ids):
        registry.upsert(_record(file_id, f"{i}.md"))
    statuses: list[FileStatus] = ["queued", "indexing", "ready", "failed"]
    errors: list[BaseException] = []
    start = threading.Barrier(len(ids))

    def worker(file_id: str) -> None:
        try:
            start.wait()
            for n in range(50):
                registry.set_status(file_id, statuses[n % 4], f"err {n}")
            registry.set_status(file_id, "ready")
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert all(r.status == "ready" and r.error is None for r in registry.all())
