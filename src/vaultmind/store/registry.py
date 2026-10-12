"""SQLite registry of ingested files and their indexing status."""

from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from vaultmind.core.models import FileRecord, FileStatus

# Migration N brings the schema from user_version N-1 to N. Append new steps; never edit old ones.
_MIGRATIONS: dict[int, str] = {
    1: """
    CREATE TABLE files (
        id         TEXT PRIMARY KEY,
        path       TEXT NOT NULL,
        sha256     TEXT NOT NULL,
        status     TEXT NOT NULL CHECK (status IN ('queued', 'indexing', 'ready', 'failed')),
        error      TEXT,
        embedder   TEXT,
        indexed_at TEXT
    );
    CREATE INDEX files_path ON files (path);
    """,
    2: """
    CREATE TABLE runs (
        id         TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        question   TEXT NOT NULL,
        answer     TEXT NOT NULL,
        abstained  INTEGER NOT NULL
    );
    CREATE INDEX runs_created ON runs (created_at);
    CREATE TABLE trace_events (
        run_id      TEXT NOT NULL REFERENCES runs (id) ON DELETE CASCADE,
        seq         INTEGER NOT NULL,
        node        TEXT NOT NULL,
        duration_ms REAL NOT NULL,
        data        TEXT NOT NULL,
        PRIMARY KEY (run_id, seq)
    );
    """,
    3: """
    CREATE TABLE chats (
        id         TEXT PRIMARY KEY,
        title      TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE INDEX chats_updated ON chats (updated_at);
    CREATE TABLE messages (
        chat_id    TEXT NOT NULL REFERENCES chats (id) ON DELETE CASCADE,
        seq        INTEGER NOT NULL,
        role       TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
        content    TEXT NOT NULL,
        created_at TEXT NOT NULL,
        run_id     TEXT REFERENCES runs (id) ON DELETE SET NULL,
        meta       TEXT NOT NULL DEFAULT '{}',
        PRIMARY KEY (chat_id, seq)
    );
    """,
    4: """
    CREATE TABLE doc_cards (
        sha256         TEXT NOT NULL,
        model          TEXT NOT NULL,
        prompt_version INTEGER NOT NULL,
        created_at     TEXT NOT NULL,
        card           TEXT NOT NULL,
        PRIMARY KEY (sha256, model, prompt_version)
    );
    """,
}
SCHEMA_VERSION = max(_MIGRATIONS)

_COLUMNS = "id, path, sha256, status, error, embedder, indexed_at"


def _to_iso(value: datetime | None) -> str | None:
    """ISO-8601 in UTC; naive datetimes are taken to be UTC already."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def _from_row(row: sqlite3.Row) -> FileRecord:
    indexed_at = row["indexed_at"]
    return FileRecord(
        id=row["id"],
        path=Path(row["path"]),
        sha256=row["sha256"],
        status=cast(FileStatus, row["status"]),
        error=row["error"],
        embedder=row["embedder"],
        indexed_at=datetime.fromisoformat(indexed_at) if indexed_at else None,
    )


def migrate(conn: sqlite3.Connection) -> None:
    """Bring the app database to SCHEMA_VERSION. Every store on `vaultmind.sqlite` calls this."""
    version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if version > SCHEMA_VERSION:
        raise RuntimeError(
            f"registry schema version {version} is newer than this app supports "
            f"({SCHEMA_VERSION}); upgrade VaultMind"
        )
    for target in range(version + 1, SCHEMA_VERSION + 1):
        # executescript commits any open transaction first, so wrap the step explicitly.
        conn.executescript(
            f"BEGIN;\n{_MIGRATIONS[target]}\nPRAGMA user_version = {target};\nCOMMIT;"
        )


class Registry:
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
            raise RuntimeError("registry is closed")
        return self._conn

    def get(self, file_id: str) -> FileRecord | None:
        with self._lock:
            row = self._db().execute(f"SELECT {_COLUMNS} FROM files WHERE id = ?", (file_id,))
            found = row.fetchone()
        return _from_row(found) if found else None

    def upsert(self, record: FileRecord) -> None:
        values = (
            record.id,
            str(record.path),
            record.sha256,
            record.status,
            record.error,
            record.embedder,
            _to_iso(record.indexed_at),
        )
        with self._lock:
            self._db().execute(
                f"""
                INSERT INTO files ({_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                    path = excluded.path,
                    sha256 = excluded.sha256,
                    status = excluded.status,
                    error = excluded.error,
                    embedder = excluded.embedder,
                    indexed_at = excluded.indexed_at
                """,
                values,
            )

    def set_status(self, file_id: str, status: FileStatus, error: str | None = None) -> None:
        """Set the status and error together; a status without an error clears the old one."""
        with self._lock:
            cursor = self._db().execute(
                "UPDATE files SET status = ?, error = ? WHERE id = ?", (status, error, file_id)
            )
        if cursor.rowcount == 0:
            raise KeyError(file_id)

    def all(self) -> list[FileRecord]:
        with self._lock:
            rows = self._db().execute(f"SELECT {_COLUMNS} FROM files ORDER BY path, id").fetchall()
        return [_from_row(r) for r in rows]

    def remove(self, file_id: str) -> None:
        with self._lock:
            self._db().execute("DELETE FROM files WHERE id = ?", (file_id,))

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
