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

# Scripts qwen3:14b drops words from when it writes Hebrew, mostly from English sources
# (measured 2026-10-14): Arabic, Cyrillic, Thai, CJK, kana and Hangul. With `script_guard`
# a free-form answer is requested as one JSON string whose pattern excludes them, so the
# grammar never lets the model emit them; the string is decoded back as it streams.
_FOREIGN_SCRIPTS = "Ѐ-ӿ؀-ۿ฀-๿ᄀ-ᇿ぀-ヿ㐀-鿿가-힯"
GUARDED_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {"type": "string", "pattern": f"^[^{_FOREIGN_SCRIPTS}]*$"}},
    "required": ["answer"],
}
_ANSWER_START = re.compile(r'\s*\{\s*"answer"\s*:\s*"')
_JSON_STRING = json.JSONDecoder(strict=False)


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


def _is_dropped_connection(exc: BaseException) -> bool:
    """Ollama closed the connection mid-response (its model runner crashed or restarted)."""
    return any(
        cls.__module__.startswith("httpx") and cls.__name__ in {"ReadError", "RemoteProtocolError"}
        for cls in type(exc).__mro__
    )


def _is_repeat_abort(exc: BaseException) -> bool:
    """Ollama aborted a constrained answer that kept repeating one token."""
    return isinstance(exc, ollama.ResponseError) and "repeat limit" in str(exc)


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
        if _is_dropped_connection(exc):
            raise OllamaUnavailableError(
                f"Ollama at {host} stopped responding in the middle of an answer (its model "
                "runner may have crashed). Try again."
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


def _guarded_answer(content: str) -> str:
    """The `answer` of a script-guarded response, even one cut off by the token limit; the raw
    text if it isn't that JSON."""
    return "".join(_guarded_stream([content]))


def _guarded_stream(stream: Iterable[str]) -> Iterator[str]:
    """Decode the `answer` string of a streamed `{"answer": "..."}` as it arrives.

    Output that doesn't start like that object passes through unchanged."""
    buffer = ""
    state = "start"  # start -> answer -> done, or raw
    for delta in stream:
        if state == "done":
            continue
        if state == "raw":
            yield delta
            continue
        buffer += delta
        if state == "start":
            match = _ANSWER_START.match(buffer)
            if match is None:
                if not buffer.lstrip() or _could_start_answer(buffer.lstrip()):
                    continue  # still a prefix of the opening
                state = "raw"
                yield buffer
                buffer = ""
                continue
            buffer = buffer[match.end() :]
            state = "answer"
        text, buffer, closed = _decode_partial(buffer)
        if text:
            yield text
        if closed:
            state = "done"
    if state == "start" and buffer:
        yield buffer


def _could_start_answer(text: str) -> bool:
    opening = '{"answer":"'
    compact = re.sub(r"\s+", "", text)
    return opening.startswith(compact) or compact.startswith(opening)


def _decode_partial(raw: str) -> tuple[str, str, bool]:
    """Decode a JSON string body up to the last complete escape: (text, rest, closed)."""
    i = 0
    while i < len(raw):
        ch = raw[i]
        if ch == '"':
            return _unescape(raw[:i]), "", True
        if ch == "\\":
            if i + 1 >= len(raw):
                break
            if raw[i + 1] != "u":
                i += 2
                continue
            if i + 6 > len(raw):
                break
            if raw[i + 2 : i + 4].lower() in ("d8", "d9", "da", "db"):  # high surrogate
                if i + 12 > len(raw):
                    break
                i += 12
                continue
            i += 6
            continue
        i += 1
    return _unescape(raw[:i]), raw[i:], False


def _unescape(raw: str) -> str:
    if not raw:
        return ""
    try:
        text, _ = _JSON_STRING.raw_decode(f'"{raw}"')
    except json.JSONDecodeError:
        return raw
    return str(text)


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
        script_guard: bool = False,
        max_answer_tokens: int | None = None,
        client: Any | None = None,
    ) -> None:
        self.model_name = model
        self.host = host
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.think = think
        self.script_guard = script_guard
        # Free-form answers only: a looping model otherwise writes until the context is full
        # (one compare ran 43 minutes, ~14k tokens, under the script guard).
        self._answer_options = {"num_predict": max_answer_tokens} if max_answer_tokens else {}
        self._client = client if client is not None else ollama.Client(host=host)

    @classmethod
    def from_config(cls, cfg: Config) -> OllamaLLM:
        return cls(
            cfg.models.llm,
            host=cfg.ollama.host,
            num_ctx=cfg.models.num_ctx,
            keep_alive=cfg.ollama.keep_alive,
            think=cfg.models.think,
            script_guard=cfg.models.script_guard,
            max_answer_tokens=cfg.models.max_answer_tokens,
        )

    def generate(self, messages: Sequence[Message], *, schema: dict[str, Any] | None = None) -> str:
        if schema is not None:
            return self._generate(messages, schema, {"temperature": 0}, think=False)
        if self.script_guard:
            try:
                content = self._generate(
                    messages, GUARDED_SCHEMA, self._answer_options, think=self.think
                )
            except ollama.ResponseError as exc:
                if not _is_repeat_abort(exc):
                    raise
                # The grammar kept the model in a loop; the unguarded answer beats none.
                return self._generate_text(messages)
            return _guarded_answer(_THINK_BLOCK.sub("", content).strip()).strip()
        return self._generate_text(messages)

    def _generate_text(self, messages: Sequence[Message]) -> str:
        content = self._generate(messages, None, self._answer_options, think=self.think)
        return _THINK_BLOCK.sub("", content).strip()

    def _generate(
        self,
        messages: Sequence[Message],
        schema: dict[str, Any] | None,
        options: dict[str, Any],
        *,
        think: bool,
    ) -> str:
        kwargs: dict[str, Any] = {"format": schema} if schema is not None else {}
        with _ollama_errors(host=self.host, model=self.model_name):
            response = self._client.chat(
                model=self.model_name,
                messages=_wire(messages),
                options={"num_ctx": self.num_ctx, **options},
                keep_alive=self.keep_alive,
                think=think,
                **kwargs,
            )
        return _content(response)

    def stream(self, messages: Sequence[Message]) -> Iterator[str]:
        if not self.script_guard:
            yield from self._stream(messages, guarded=False)
            return
        sent = False
        try:
            for delta in self._stream(messages, guarded=True):
                sent = True
                yield delta
        except ollama.ResponseError as exc:
            if not _is_repeat_abort(exc):
                raise
            # Ollama stops a constrained answer that keeps repeating a token. Mid-answer, the
            # text so far stands (the repeats were junk); before any text, retry unguarded.
            if not sent:
                yield from self._stream(messages, guarded=False)

    def _stream(self, messages: Sequence[Message], *, guarded: bool) -> Iterator[str]:
        kwargs: dict[str, Any] = {"format": GUARDED_SCHEMA} if guarded else {}
        with _ollama_errors(host=self.host, model=self.model_name):
            chunks = self._client.chat(
                model=self.model_name,
                messages=_wire(messages),
                stream=True,
                options={"num_ctx": self.num_ctx, **self._answer_options},
                keep_alive=self.keep_alive,
                think=self.think,
                **kwargs,
            )
            deltas = _strip_thinking(_content(c) for c in chunks)
            for delta in _guarded_stream(deltas) if guarded else deltas:
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
