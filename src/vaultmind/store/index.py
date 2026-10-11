"""LanceDB index of child chunks (vectors + text) and their parent sections."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from vaultmind.core.models import ChildChunk, ParentSection, SearchHit


class EmbedderMismatchError(Exception):
    """The index was built with a different embedding model; it must be rebuilt."""


class ChunkIndex:
    def __init__(self, db_dir: Path, *, embedder_name: str) -> None:
        raise NotImplementedError

    def add(
        self,
        parents: Sequence[ParentSection],
        children: Sequence[ChildChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        """Insert chunks; `vectors[i]` is the embedding of `children[i].embed_text`."""
        raise NotImplementedError

    def delete_file(self, file_id: str) -> None:
        raise NotImplementedError

    def vector_search(self, vector: Sequence[float], k: int) -> list[SearchHit]:
        raise NotImplementedError

    def get_parents(self, parent_ids: Sequence[str]) -> dict[str, ParentSection]:
        raise NotImplementedError

    def count(self) -> int:
        """Number of child chunks in the index."""
        raise NotImplementedError
