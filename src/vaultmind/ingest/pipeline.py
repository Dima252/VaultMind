"""Files -> index. Incremental: unchanged files are skipped by content hash."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from vaultmind.core.config import Config
from vaultmind.core.models import FileRecord, file_id_for
from vaultmind.core.providers import Embedder
from vaultmind.ingest.chunker import chunk_document
from vaultmind.ingest.parsers import SUPPORTED_EXTENSIONS, ParseError, parse
from vaultmind.store.index import ChunkIndex
from vaultmind.store.registry import Registry


@dataclass
class IngestSummary:
    added: list[Path] = field(default_factory=list)
    updated: list[Path] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)
    failed: list[tuple[Path, str]] = field(default_factory=list)
    removed: list[Path] = field(default_factory=list)


def iter_files(root: Path) -> list[Path]:
    """Supported files under `root` (or `root` itself), sorted for a stable order."""
    if root.is_file():
        return [root] if root.suffix.lower() in SUPPORTED_EXTENSIONS else []
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ingest_path(
    path: Path,
    cfg: Config,
    *,
    embedder: Embedder,
    index: ChunkIndex,
    registry: Registry,
    on_file: Callable[[Path], None] | None = None,
) -> IngestSummary:
    """Index new and changed files under `path`; drop files that were deleted from it.

    Files that fail to parse are recorded as `failed` and retried on the next run.
    Any other error (e.g. Ollama unreachable) marks the file failed and stops the run.
    """
    summary = IngestSummary()
    root = path.resolve()
    seen: set[str] = set()

    for file in iter_files(root):
        file_id = file_id_for(file)
        seen.add(file_id)
        digest = sha256_file(file)
        existing = registry.get(file_id)
        if (
            existing is not None
            and existing.status == "ready"
            and existing.sha256 == digest
            and existing.embedder == embedder.model_name
        ):
            summary.skipped.append(file)
            continue

        if on_file is not None:
            on_file(file)
        registry.upsert(FileRecord(id=file_id, path=file, sha256=digest, status="indexing"))
        try:
            doc = parse(file)
            parents, children = chunk_document(doc, file_id, cfg.chunking)
            vectors = embedder.embed([c.embed_text for c in children])
            index.delete_file(file_id)
            index.add(parents, children, vectors)
        except ParseError as e:
            index.delete_file(file_id)  # don't keep answering from an old version
            registry.set_status(file_id, "failed", str(e))
            summary.failed.append((file, str(e)))
            continue
        except Exception as e:
            registry.set_status(file_id, "failed", str(e))
            raise

        registry.upsert(
            FileRecord(
                id=file_id,
                path=file,
                sha256=digest,
                status="ready",
                embedder=embedder.model_name,
                indexed_at=datetime.now(UTC),
            )
        )
        (summary.added if existing is None else summary.updated).append(file)

    for record in registry.all():
        if record.id in seen or record.path.exists() or not _is_within(record.path, root):
            continue
        index.delete_file(record.id)
        registry.remove(record.id)
        summary.removed.append(record.path)

    return summary


def _is_within(path: Path, root: Path) -> bool:
    return path == root or path.is_relative_to(root)
