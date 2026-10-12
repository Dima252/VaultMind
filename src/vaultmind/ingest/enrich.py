"""Document cards: what each indexed file is about as a whole (Phase 5.5, M1).

After a file is indexed, a background worker reads its parent sections and asks the LLM for a
card (`core/models.DocCard`): a 1-2 sentence summary per section, then one call over those for
type, language, summary, outline, entities and dates. Cards are stored by content hash, so each
file version costs its ~10-60 s of GPU once. The file is searchable before its card exists.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

from vaultmind.core.config import Config
from vaultmind.core.models import DocCard, FileRecord, ParentSection
from vaultmind.core.providers import LLM
from vaultmind.store.cards import CardStore
from vaultmind.store.index import ChunkIndex
from vaultmind.store.registry import Registry

PROMPT_VERSION = 1  # bump when a card prompt changes: every card is rewritten once

CardStatus = Literal["writing", "ready", "failed"]


class EnrichError(Exception):
    """The LLM's card output could not be used, even after a retry."""


@dataclass(frozen=True)
class CardEvent:
    file_id: str
    status: CardStatus
    error: str | None = None


def build_card(
    record: FileRecord, parents: Sequence[ParentSection], *, llm: LLM, cfg: Config
) -> DocCard:
    """Write the card for one indexed file from its parent sections (document order).

    Every LLM call fits `models.num_ctx`: sections are summarized in batches, and the final
    call sees the section summaries, not the full text. The title is the document's own title
    when it states one, else the file name without extension. Raises `EnrichError`.
    """
    raise NotImplementedError


class Enricher:
    """Background worker: writes the missing cards of ready files, one file at a time.

    `submit` is called when a file becomes ready; `scan` queues every ready file whose card
    for the current model and `PROMPT_VERSION` is missing. Questions come first: while
    `paused()` is held, the worker finishes its current LLM call and then waits.
    """

    def __init__(
        self,
        cfg: Config,
        *,
        llm: LLM,
        index: ChunkIndex,
        registry: Registry,
        cards: CardStore,
        on_event: Callable[[CardEvent], None] | None = None,
    ) -> None:
        raise NotImplementedError

    def start(self) -> None:
        raise NotImplementedError

    def stop(self, timeout: float = 10.0) -> None:
        """Stop after the current LLM call. Idempotent."""
        raise NotImplementedError

    def submit(self, file_id: str) -> None:
        raise NotImplementedError

    def scan(self) -> None:
        raise NotImplementedError

    def pending(self) -> int:
        """Files queued or being enriched."""
        raise NotImplementedError

    def wait_idle(self, timeout: float | None = None) -> bool:
        raise NotImplementedError

    @contextmanager
    def paused(self) -> Iterator[None]:
        """Hold enrichment while a question runs (the engine wraps `ask` in this)."""
        raise NotImplementedError
        yield
