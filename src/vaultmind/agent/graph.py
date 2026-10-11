"""Corrective RAG loop as a LangGraph state machine (Phase 4).

analyze -> retrieve -> grade -> (generate | rewrite -> retrieve ... | not_found), with optional
contradiction and groundedness checks. Grading uses reranker scores, never per-chunk LLM calls.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from functools import partial
from typing import cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from vaultmind.agent import nodes
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
    if cfg.retrieval.rerank and reranker is None:
        raise ValueError("retrieval.rerank is on but no reranker was given")
    deps = nodes.Deps(cfg, llm, embedder, index, registry, reranker)
    app = _build(deps)
    # Each rewrite round is three steps (rewrite, retrieve, grade); the rest fit in ten.
    config: RunnableConfig = {"recursion_limit": 10 + 3 * cfg.agent.max_rewrites}
    final: nodes.AgentState | None = None
    for mode, chunk in app.stream(
        nodes.initial_state(question, list(history)),
        config,
        stream_mode=["custom", "values"],
    ):
        if mode == "custom":
            yield str(chunk)  # nodes write answer text only
        else:
            final = cast(nodes.AgentState, chunk)
    if final is None:
        raise RuntimeError("the agent graph produced no state")
    yield nodes.to_result(final)


def _build(deps: nodes.Deps) -> CompiledStateGraph[nodes.AgentState]:
    cfg = deps.cfg
    g = StateGraph(nodes.AgentState)
    for name, node in (
        ("analyze", nodes.analyze),
        ("direct", nodes.direct),
        ("retrieve", nodes.retrieve),
        ("grade", nodes.grade),
        ("rewrite", nodes.rewrite),
        ("contradictions", nodes.contradictions),
        ("generate", nodes.generate),
        ("not_found", nodes.not_found),
        ("groundedness", nodes.groundedness),
    ):
        g.add_node(name, partial(node, deps=deps))
    g.add_edge(START, "analyze")
    g.add_conditional_edges("analyze", nodes.after_analyze)
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", partial(nodes.after_grade, cfg=cfg))
    g.add_conditional_edges("rewrite", nodes.after_rewrite)
    g.add_edge("contradictions", "generate")
    g.add_conditional_edges("generate", partial(nodes.after_generate, cfg=cfg))
    for last in ("direct", "not_found", "groundedness"):
        g.add_edge(last, END)
    return g.compile()


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
