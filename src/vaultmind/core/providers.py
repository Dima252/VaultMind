"""Model provider interfaces.

Everything that touches a model goes through these protocols, so the pipeline can swap
Ollama for llama.cpp, compare models in evals, and run tests with fakes (no GPU in CI).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, NotRequired, Protocol, TypedDict


class ToolCallData(TypedDict):
    """A tool call as it travels in a message (the assistant turn that made it)."""

    name: str
    arguments: dict[str, Any]


class Message(TypedDict):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: NotRequired[list[ToolCallData]]  # assistant turns that called tools
    tool_name: NotRequired[str]  # tool messages: the tool whose result `content` is


@dataclass(frozen=True)
class ToolSpec:
    """A tool the model may call. `parameters` is a JSON schema object."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def as_data(self) -> ToolCallData:
        return {"name": self.name, "arguments": self.arguments}


@dataclass(frozen=True)
class AssistantTurn:
    """One model turn: text, tool calls, or both."""

    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    thinking: str = ""  # the model's reasoning when thinking was on (for traces only)


class LLM(Protocol):
    model_name: str

    def generate(self, messages: Sequence[Message], *, schema: dict[str, Any] | None = None) -> str:
        """Return the full completion. With `schema`, the output is JSON matching it."""
        ...

    def stream(self, messages: Sequence[Message]) -> Iterator[str]:
        """Yield completion tokens as they are produced."""
        ...

    def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] = (),
        think: bool | None = None,
    ) -> AssistantTurn:
        """One assistant turn that may call `tools`. `think` overrides the model's default."""
        ...


class Embedder(Protocol):
    model_name: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class Reranker(Protocol):
    model_name: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Relevance score per passage, same order as `passages`. Higher is more relevant."""
        ...
