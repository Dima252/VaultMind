"""Linear RAG: retrieve child chunks -> parent sections -> cited answer.

With hybrid search and reranking off this is the Phase 1 baseline the eval measures later
phases against; `agent.retrieval.search` adds both behind config switches.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from vaultmind.agent.retrieval import search
from vaultmind.core.config import Config
from vaultmind.core.models import SearchHit
from vaultmind.core.providers import LLM, Embedder, Message, Reranker
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
    """A numbered source shown to the LLM: a parent section, cited where its best child sits."""

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
    question: str,
    cfg: Config,
    *,
    embedder: Embedder,
    index: ChunkIndex,
    registry: Registry,
    reranker: Reranker | None = None,
) -> list[Source]:
    """Top-k child hits as numbered sources in rank order.

    With `retrieval.small_to_big`, hits are collapsed to their parent sections (each cited at
    its best child); without it, each child chunk is its own source.
    """
    hits = search(question, cfg, embedder=embedder, index=index, reranker=reranker)
    if not cfg.retrieval.small_to_big:
        return [_source(i + 1, hit, hit.chunk.text, registry) for i, hit in enumerate(hits)]

    best: dict[str, SearchHit] = {}
    for hit in hits:  # hits are sorted best-first; keep each parent's best child
        best.setdefault(hit.chunk.parent_id, hit)
    parents = index.get_parents(list(best))

    sources: list[Source] = []
    for parent_id, hit in best.items():
        parent = parents.get(parent_id)
        if parent is not None:
            sources.append(_source(len(sources) + 1, hit, parent.text, registry))
    return sources


def _source(n: int, hit: SearchHit, text: str, registry: Registry) -> Source:
    record = registry.get(hit.chunk.file_id)
    return Source(
        n=n,
        file_path=record.path if record else None,
        page=hit.chunk.page,
        section_path=hit.chunk.section_path,  # a parent can span several sections
        text=text,
        score=hit.score,
    )


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
    reranker: Reranker | None = None,
) -> tuple[list[Source], Iterator[str]]:
    """Retrieve, then return the sources and a token stream of the answer."""
    sources = retrieve(
        question, cfg, embedder=embedder, index=index, registry=registry, reranker=reranker
    )
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
    reranker: Reranker | None = None,
) -> Answer:
    sources = retrieve(
        question, cfg, embedder=embedder, index=index, registry=registry, reranker=reranker
    )
    if not sources:
        return Answer(NOT_FOUND, [])
    return Answer(llm.generate(build_messages(question, sources)).strip(), sources)


def is_abstention(text: str) -> bool:
    return text.strip().startswith(NOT_FOUND.rstrip("."))


def cited_sources(text: str, sources: Sequence[Source]) -> list[Source]:
    """Sources referenced as [n] in `text`, in source order; unknown numbers are ignored."""
    numbers = {int(m) for m in _CITATION.findall(text)}
    return [s for s in sources if s.n in numbers]
