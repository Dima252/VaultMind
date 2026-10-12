"""Agent tools: plain Python functions over the index, the registry and the document cards.

Read-only unless `write` is set; a write tool runs only after the user approves it, and the
agent has no tool that deletes anything. Tools never see the network. The gateway
(`agent/gateway.py`) validates arguments, enforces budgets and guards every observation, so a
tool returns its raw text and the passages it read.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from vaultmind.agent.baseline import Source
from vaultmind.agent.catalog import Catalog
from vaultmind.core.config import Config
from vaultmind.core.providers import LLM, Embedder, Reranker, ToolSpec
from vaultmind.store.cards import CardStore
from vaultmind.store.index import ChunkIndex
from vaultmind.store.registry import Registry


@dataclass(frozen=True)
class ToolContext:
    """What a tool may use."""

    cfg: Config
    llm: LLM
    embedder: Embedder
    index: ChunkIndex
    registry: Registry
    reranker: Reranker | None
    cards: CardStore
    catalog: Catalog


@dataclass(frozen=True)
class ToolOutput:
    text: str  # the raw observation; the gateway budgets, guards and wraps it
    sources: tuple[Source, ...] = ()  # the text read, for citations (`n` = 0)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str  # for the model: what it does and when to use it, one or two sentences
    args: type[BaseModel]  # validated arguments; its JSON schema is the tool's parameters
    run: Callable[[ToolContext, Any], ToolOutput]  # gets an instance of `args`
    write: bool = False  # changes something outside the chat: needs the user's approval

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.args.model_json_schema())


def default_tools() -> list[Tool]:
    """The read-only tools: list_documents, document_card, outline, read, search, summarize,
    extract, calc, date_diff (see docs/EXECUTION_PLAN.md, Phase 5.5, component 4)."""
    raise NotImplementedError
