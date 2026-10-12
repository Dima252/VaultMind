"""Deterministic fake providers so the pipeline can be tested without Ollama or a GPU."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from typing import Any

from vaultmind.core.providers import AssistantTurn, Message, ToolCall, ToolSpec


class FakeLLM:
    """Returns queued responses in order; with a schema, returns JSON-encoded dicts.

    `chat` plays back `turns` (a str is a text-only turn), so a test can script tool calls:
    `FakeLLM(turns=[AssistantTurn("", (ToolCall("list_documents"),)), "Done."])`.
    """

    model_name = "fake-llm"

    def __init__(
        self,
        responses: Sequence[str | dict[str, Any]] = (),
        *,
        turns: Sequence[str | AssistantTurn] = (),
    ) -> None:
        self._responses = list(responses)
        self._turns = list(turns)
        self.calls: list[Sequence[Message]] = []
        self.chat_calls: list[tuple[Sequence[Message], Sequence[ToolSpec], bool | None]] = []

    def generate(self, messages: Sequence[Message], *, schema: dict[str, Any] | None = None) -> str:
        self.calls.append(messages)
        response = self._responses.pop(0) if self._responses else ""
        return json.dumps(response) if isinstance(response, dict) else response

    def stream(self, messages: Sequence[Message]) -> Iterator[str]:
        yield from self.generate(messages).split(" ")

    def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] = (),
        think: bool | None = None,
    ) -> AssistantTurn:
        self.chat_calls.append((list(messages), list(tools), think))
        turn = self._turns.pop(0) if self._turns else ""
        return AssistantTurn(turn) if isinstance(turn, str) else turn


def tool_turn(name: str, **arguments: Any) -> AssistantTurn:
    """A scripted assistant turn that makes one tool call."""
    return AssistantTurn("", (ToolCall(name, arguments),))


class FakeEmbedder:
    """Hash-based vectors: identical text gives identical vectors, no model needed."""

    model_name = "fake-embedder"

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [b / 255 for b in digest[: self.dim]]


class FakeReranker:
    """Scores by word overlap with the query."""

    model_name = "fake-reranker"

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        terms = set(query.lower().split())
        return [len(terms & set(p.lower().split())) / max(len(terms), 1) for p in passages]
