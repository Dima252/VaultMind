"""The document catalog: which files the vault has, what each is, and which one a request means.

Backed by the registry (ready files) and the document cards. The planner reads `describe()`;
the tools and recipes turn a mention ("the Workato worksheet", "הסילבוס") into a file with
`resolve()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vaultmind.core.models import DocCard, FileRecord
from vaultmind.core.providers import Reranker
from vaultmind.store.cards import CardStore
from vaultmind.store.registry import Registry


@dataclass(frozen=True)
class CatalogEntry:
    file: FileRecord
    name: str  # path relative to the vault, forward slashes (absolute if outside it)
    card: DocCard | None  # the newest card for the file's content, if one was written


class Catalog:
    def __init__(
        self,
        registry: Registry,
        cards: CardStore,
        vault_dir: Path,
        *,
        reranker: Reranker | None = None,
    ) -> None:
        raise NotImplementedError

    def entries(self) -> list[CatalogEntry]:
        """Ready files, sorted by name. Read fresh on every call (files come and go)."""
        raise NotImplementedError

    def resolve(self, mention: str) -> CatalogEntry:
        """The one document `mention` means: a file id, a file name or path, a title, or a
        description ("the AI syllabus"), in any language. Raises `DocumentNotFound` when
        nothing matches and `AmbiguousDocument` when several match about equally well."""
        raise NotImplementedError

    def describe(self, limit: int) -> str:
        """One line per document for the planner prompt (name, type, language, a short
        summary), at most `limit` lines; the rest are counted, not listed."""
        raise NotImplementedError
