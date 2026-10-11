"""ParsedDocument -> heading-aware parent sections, each split into overlapping child chunks."""

from __future__ import annotations

from vaultmind.core.config import ChunkingConfig
from vaultmind.core.models import ChildChunk, ParentSection, ParsedDocument


def estimate_tokens(text: str) -> int:
    """Approximate token count (no tokenizer download needed)."""
    raise NotImplementedError


def chunk_document(
    doc: ParsedDocument, file_id: str, cfg: ChunkingConfig
) -> tuple[list[ParentSection], list[ChildChunk]]:
    raise NotImplementedError
