"""Data types shared across ingestion, storage, retrieval and evaluation.

These are the contracts between modules: change them deliberately, not in passing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

FileStatus = Literal["queued", "indexing", "ready", "failed"]


def file_id_for(path: Path) -> str:
    """Stable id for a file, derived from its resolved absolute path."""
    return hashlib.sha256(str(path.resolve()).lower().encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Block:
    """A contiguous piece of parsed text (Markdown) with its location in the source."""

    text: str
    page: int | None  # 1-based; None for formats without pages (md, txt)
    section_path: tuple[str, ...]  # heading hierarchy, outermost first; () if none


@dataclass(frozen=True)
class ParsedDocument:
    path: Path
    title: str  # first H1, else PDF metadata title, else file stem
    blocks: list[Block]


@dataclass(frozen=True)
class ParentSection:
    """A large section (about `parent_tokens`) the LLM reads when any child matches."""

    id: str  # f"{file_id}:p{index}"
    file_id: str
    text: str
    section_path: tuple[str, ...]
    page_start: int | None
    page_end: int | None


@dataclass(frozen=True)
class ChildChunk:
    """A small chunk (about `child_tokens`) that is embedded and searched."""

    id: str  # f"{parent_id}:c{index}"
    parent_id: str
    file_id: str
    text: str  # raw chunk text, shown in citations
    embed_text: str  # "Title > Section > Sub\n\n" + text (title not repeated), what gets embedded
    page: int | None
    section_path: tuple[str, ...]


@dataclass(frozen=True)
class SearchHit:
    chunk: ChildChunk
    score: float  # higher is better, regardless of the underlying distance metric


@dataclass(frozen=True)
class FileRecord:
    id: str
    path: Path
    sha256: str
    status: FileStatus
    error: str | None = None
    embedder: str | None = None
    indexed_at: datetime | None = None
