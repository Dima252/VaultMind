"""Corrective RAG loop as a LangGraph state machine (Phase 4).

analyze -> retrieve -> grade -> (generate | rewrite -> retrieve ... | not_found), with optional
contradiction and groundedness checks. Grading uses reranker scores, never per-chunk LLM calls.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence

from vaultmind.agent.types import AgentResult
from vaultmind.core.config import Config
from vaultmind.core.providers import LLM, Embedder, Message, Reranker
from vaultmind.store.index import ChunkIndex
from vaultmind.store.registry import Registry

# langchain-core ships a LangSmith client that uploads traces when these are set. Force them off:
# no network except localhost, whatever the user's environment says.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"


def stream_agent(
    question: str,
    cfg: Config,
    *,
    llm: LLM,
    embedder: Embedder,
    index: ChunkIndex,
    registry: Registry,
    reranker: Reranker | None,
    history: Sequence[Message] = (),
) -> Iterator[str | AgentResult]:
    """Run the agent: yield answer tokens (str) as they are generated, then one AgentResult last.

    `history` holds earlier chat turns (user/assistant), oldest first. `reranker` may be None
    only when `retrieval.rerank` is off; then grading keeps the top_k candidates unfiltered.
    The not-found answer is yielded as text too, so a consumer can print every str it gets.
    """
    raise NotImplementedError


def run_agent(
    question: str,
    cfg: Config,
    *,
    llm: LLM,
    embedder: Embedder,
    index: ChunkIndex,
    registry: Registry,
    reranker: Reranker | None,
    history: Sequence[Message] = (),
) -> AgentResult:
    """`stream_agent` without streaming."""
    for event in stream_agent(
        question,
        cfg,
        llm=llm,
        embedder=embedder,
        index=index,
        registry=registry,
        reranker=reranker,
        history=history,
    ):
        if isinstance(event, AgentResult):
            return event
    raise RuntimeError("stream_agent ended without an AgentResult")
