"""SQLite registry of ingested files and their indexing status."""

from __future__ import annotations

from pathlib import Path

from vaultmind.core.models import FileRecord, FileStatus


class Registry:
    def __init__(self, db_path: Path) -> None:
        raise NotImplementedError

    def get(self, file_id: str) -> FileRecord | None:
        raise NotImplementedError

    def upsert(self, record: FileRecord) -> None:
        raise NotImplementedError

    def set_status(self, file_id: str, status: FileStatus, error: str | None = None) -> None:
        raise NotImplementedError

    def all(self) -> list[FileRecord]:
        raise NotImplementedError

    def remove(self, file_id: str) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
