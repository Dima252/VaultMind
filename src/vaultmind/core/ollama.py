"""Ollama-backed implementations of the provider protocols."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from vaultmind.core.config import Config
from vaultmind.core.providers import Message


class OllamaLLM:
    def __init__(self, model: str, *, host: str, num_ctx: int, keep_alive: str) -> None:
        raise NotImplementedError

    @classmethod
    def from_config(cls, cfg: Config) -> OllamaLLM:
        raise NotImplementedError

    def generate(self, messages: Sequence[Message], *, schema: dict[str, Any] | None = None) -> str:
        raise NotImplementedError

    def stream(self, messages: Sequence[Message]) -> Iterator[str]:
        raise NotImplementedError


class OllamaEmbedder:
    def __init__(self, model: str, *, host: str, keep_alive: str, batch_size: int = 32) -> None:
        raise NotImplementedError

    @classmethod
    def from_config(cls, cfg: Config) -> OllamaEmbedder:
        raise NotImplementedError

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError
