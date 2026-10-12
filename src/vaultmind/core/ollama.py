"""Ollama-backed implementations of the provider protocols."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import ollama

from vaultmind.core.config import Config
from vaultmind.core.providers import AssistantTurn, Message, ToolCall, ToolSpec

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_THINK_BLOCK = re.compile(r"<think>.*?(?:</think>|\Z)", re.DOTALL)


class OllamaUnavailableError(RuntimeError):
    """Ollama is not reachable, or the requested model is not pulled."""


def _is_connection_error(exc: BaseException) -> bool:
    # The client maps httpx.ConnectError to ConnectionError for normal requests, but
    # streaming requests let the raw httpx error through. Match httpx by name so we
    # don't import a transitive dependency directly.
    if isinstance(exc, ConnectionError):
        return True
    return any(
        cls.__module__.startswith("httpx") and cls.__name__ in {"ConnectError", "ConnectTimeout"}
        for cls in type(exc).__mro__
    )


def _is_missing_model(exc: BaseException) -> bool:
    return isinstance(exc, ollama.ResponseError) and exc.status_code == 404


@contextmanager
def _ollama_errors(*, host: str, model: str) -> Iterator[None]:
    """Turn connection and missing-model failures into `OllamaUnavailableError`."""
    try:
        yield
    except Exception as exc:
        if _is_connection_error(exc):
            raise OllamaUnavailableError(
                f"Cannot reach Ollama: is Ollama running at {host}?"
            ) from exc
        if _is_missing_model(exc):
            raise OllamaUnavailableError(
                f"Model {model!r} is not available in Ollama at {host}. Run: ollama pull {model}"
            ) from exc
        raise


def _content(chunk: Any) -> str:
    """Message content from a chat response, ignoring any separate thinking field."""
    return chunk["message"]["content"] or ""


def _wire(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Messages in Ollama's format: tool calls nest their name and arguments under `function`."""
    wired: list[dict[str, Any]] = []
    for message in messages:
        item: dict[str, Any] = dict(message)
        if "tool_calls" in message:
            item["tool_calls"] = [{"function": dict(call)} for call in message["tool_calls"]]
        wired.append(item)
    return wired


def _arguments(raw: Any) -> dict[str, Any]:
    """Tool-call arguments as a dict; a model that sends a JSON string gets it parsed."""
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"_unparsed": raw}
        return parsed if isinstance(parsed, dict) else {"_unparsed": raw}
    return {}


def _turn(response: Any) -> AssistantTurn:
    message = response["message"]
    calls = tuple(
        ToolCall(call["function"]["name"], _arguments(call["function"]["arguments"]))
        for call in message.get("tool_calls") or ()
    )
    content = _THINK_BLOCK.sub("", message.get("content") or "").strip()
    return AssistantTurn(content, calls, message.get("thinking") or "")


def _strip_thinking(stream: Iterable[str]) -> Iterator[str]:
    """Drop `<think>...</think>` spans from a token stream, even if tags span deltas."""
    buffer = ""
    thinking = False
    for delta in stream:
        buffer += delta
        while buffer:
            tag = _THINK_CLOSE if thinking else _THINK_OPEN
            idx = buffer.find(tag)
            if idx >= 0:
                if not thinking and idx > 0:
                    yield buffer[:idx]
                buffer = buffer[idx + len(tag) :]
                thinking = not thinking
                continue
            # Hold back a suffix that could be the start of the tag.
            keep = next((n for n in range(len(tag) - 1, 0, -1) if buffer.endswith(tag[:n])), 0)
            emit, buffer = buffer[: len(buffer) - keep], buffer[len(buffer) - keep :]
            if emit and not thinking:
                yield emit
            break
    if buffer and not thinking:
        yield buffer


class OllamaLLM:
    def __init__(
        self,
        model: str,
        *,
        host: str,
        num_ctx: int,
        keep_alive: str,
        think: bool = False,
        client: Any | None = None,
    ) -> None:
        self.model_name = model
        self.host = host
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.think = think
        self._client = client if client is not None else ollama.Client(host=host)

    @classmethod
    def from_config(cls, cfg: Config) -> OllamaLLM:
        return cls(
            cfg.models.llm,
            host=cfg.ollama.host,
            num_ctx=cfg.models.num_ctx,
            keep_alive=cfg.ollama.keep_alive,
            think=cfg.models.think,
        )

    def generate(self, messages: Sequence[Message], *, schema: dict[str, Any] | None = None) -> str:
        options: dict[str, Any] = {"num_ctx": self.num_ctx}
        kwargs: dict[str, Any] = {"think": self.think}
        if schema is not None:
            options["temperature"] = 0
            kwargs["format"] = schema
            kwargs["think"] = False
        with _ollama_errors(host=self.host, model=self.model_name):
            response = self._client.chat(
                model=self.model_name,
                messages=_wire(messages),
                options=options,
                keep_alive=self.keep_alive,
                **kwargs,
            )
        content = _content(response)
        return content if schema is not None else _THINK_BLOCK.sub("", content).strip()

    def stream(self, messages: Sequence[Message]) -> Iterator[str]:
        with _ollama_errors(host=self.host, model=self.model_name):
            chunks = self._client.chat(
                model=self.model_name,
                messages=_wire(messages),
                stream=True,
                options={"num_ctx": self.num_ctx},
                keep_alive=self.keep_alive,
                think=self.think,
            )
            for delta in _strip_thinking(_content(c) for c in chunks):
                if delta:
                    yield delta

    def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] = (),
        think: bool | None = None,
    ) -> AssistantTurn:
        wired_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]
        with _ollama_errors(host=self.host, model=self.model_name):
            response = self._client.chat(
                model=self.model_name,
                messages=_wire(messages),
                tools=wired_tools or None,
                options={"num_ctx": self.num_ctx},
                keep_alive=self.keep_alive,
                think=self.think if think is None else think,
            )
        return _turn(response)


class OllamaEmbedder:
    def __init__(
        self,
        model: str,
        *,
        host: str,
        keep_alive: str,
        batch_size: int = 32,
        client: Any | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.model_name = model
        self.host = host
        self.keep_alive = keep_alive
        self.batch_size = batch_size
        self._client = client if client is not None else ollama.Client(host=host)

    @classmethod
    def from_config(cls, cfg: Config) -> OllamaEmbedder:
        return cls(cfg.models.embedder, host=cfg.ollama.host, keep_alive=cfg.ollama.keep_alive)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            with _ollama_errors(host=self.host, model=self.model_name):
                response = self._client.embed(
                    model=self.model_name, input=batch, keep_alive=self.keep_alive
                )
            embeddings = response["embeddings"]
            if len(embeddings) != len(batch):
                raise RuntimeError(
                    f"Ollama returned {len(embeddings)} embeddings for {len(batch)} inputs"
                )
            vectors.extend([float(x) for x in vec] for vec in embeddings)
        return vectors
