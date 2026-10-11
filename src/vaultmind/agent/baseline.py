"""Linear baseline RAG: embed -> vector search -> parent sections -> cited answer.

This is the Phase 1 reference point the eval harness measures later phases against.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from vaultmind.core.config import Config
from vaultmind.core.models import SearchHit
from vaultmind.core.providers import LLM, Embedder, Message
from vaultmind.store.index import ChunkIndex
from vaultmind.store.registry import Registry

NOT_FOUND = "I couldn't find this in your documents."

SYSTEM_PROMPT = f"""You answer questions using only the numbered sources provided.
- Cite every claim with the source number in square brackets, e.g. [1] or [2][3].
- Do not use outside knowledge. Do not invent sources.
- If the sources do not contain the answer, reply with exactly: {NOT_FOUND}
- Be concise. Answer in the language of the question."""

_CITATION = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class Source:
    """One numbered source shown to the LLM: a parent section, cited at its best child's page."""

    n: int
    file_path: Path | None
    page: int | None
    section_path: tuple[str, ...]
    text: str
    score: float

    @property
    def label(self) -> str:
        name = self.file_path.name if self.file_path else "unknown file"
        parts = [name]
        if self.page is not None:
            parts.append(f"p. {self.page}")
        if self.section_path:
            parts.append(" > ".join(self.section_path))
        return ", ".join(parts)


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[Source]

    @property
    def abstained(self) -> bool:
        return is_abstention(self.text)

    @property
    def cited(self) -> list[Source]:
        return cited_sources(self.text, self.sources)


def retrieve(
    question: str, cfg: Config, *, embedder: Embedder, index: ChunkIndex, registry: Registry
) -> list[Source]:
    """Top-k child hits, collapsed to their parent sections in rank order."""
    [vector] = embedder.embed([question])
    hits = index.vector_search(vector, k=cfg.retrieval.top_k)
    best: dict[str, SearchHit] = {}
    for hit in hits:  # hits are sorted best-first; keep each parent's best child
        best.setdefault(hit.chunk.parent_id, hit)
    parents = index.get_parents(list(best))

    sources: list[Source] = []
    for parent_id, hit in best.items():
        parent = parents.get(parent_id)
        if parent is None:
            continue
        record = registry.get(hit.chunk.file_id)
        sources.append(
            Source(
                n=len(sources) + 1,
                file_path=record.path if record else None,
                page=hit.chunk.page,
                section_path=parent.section_path,
                text=parent.text,
                score=hit.score,
            )
        )
    return sources


def build_messages(question: str, sources: Sequence[Source]) -> list[Message]:
    context = "\n\n".join(f"[{s.n}] ({s.label})\n{s.text}" for s in sources)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Sources:\n\n{context}\n\nQuestion: {question}"},
    ]


def stream_answer(
    question: str,
    cfg: Config,
    *,
    llm: LLM,
    embedder: Embedder,
    index: ChunkIndex,
    registry: Registry,
) -> tuple[list[Source], Iterator[str]]:
    """Retrieve, then return the sources and a token stream of the answer."""
    sources = retrieve(question, cfg, embedder=embedder, index=index, registry=registry)
    if not sources:
        return [], iter([NOT_FOUND])
    return sources, llm.stream(build_messages(question, sources))


def answer(
    question: str,
    cfg: Config,
    *,
    llm: LLM,
    embedder: Embedder,
    index: ChunkIndex,
    registry: Registry,
) -> Answer:
    sources = retrieve(question, cfg, embedder=embedder, index=index, registry=registry)
    if not sources:
        return Answer(NOT_FOUND, [])
    return Answer(llm.generate(build_messages(question, sources)).strip(), sources)


def is_abstention(text: str) -> bool:
    return text.strip().startswith(NOT_FOUND.rstrip("."))


def cited_sources(text: str, sources: Sequence[Source]) -> list[Source]:
    """Sources referenced as [n] in `text`, in source order; unknown numbers are ignored."""
    numbers = {int(m) for m in _CITATION.findall(text)}
    return [s for s in sources if s.n in numbers]
