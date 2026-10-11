"""Model provider interfaces.

Everything that touches a model goes through these protocols, so the pipeline can swap
Ollama for llama.cpp, compare models in evals, and run tests with fakes (no GPU in CI).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, Literal, Protocol, TypedDict


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class LLM(Protocol):
    model_name: str

    def generate(self, messages: Sequence[Message], *, schema: dict[str, Any] | None = None) -> str:
        """Return the full completion. With `schema`, the output is JSON matching it."""
        ...

    def stream(self, messages: Sequence[Message]) -> Iterator[str]:
        """Yield completion tokens as they are produced."""
        ...


class Embedder(Protocol):
    model_name: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class Reranker(Protocol):
    model_name: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Relevance score per passage, same order as `passages`. Higher is more relevant."""
        ...
