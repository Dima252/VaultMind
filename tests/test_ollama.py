"""Unit tests for the Ollama providers, using a fake client that records calls."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import ollama
import pytest

from vaultmind.core.config import Config
from vaultmind.core.ollama import OllamaEmbedder, OllamaLLM, OllamaUnavailableError
from vaultmind.core.providers import LLM, AssistantTurn, Embedder, Message, ToolCall, ToolSpec

HOST = "http://127.0.0.1:11434"
MESSAGES: list[Message] = [{"role": "user", "content": "hi"}]


class FakeClient:
    """Stands in for `ollama.Client`; responses are plain dicts like the real ones."""

    def __init__(
        self,
        *,
        content: str = "hello",
        deltas: Sequence[str] = (),
        error: Exception | None = None,
        dim: int = 3,
    ) -> None:
        self.content = content
        self.deltas = list(deltas)
        self.error = error
        self.dim = dim
        self.chat_calls: list[dict[str, Any]] = []
        self.embed_calls: list[dict[str, Any]] = []

    def chat(self, **kwargs: Any) -> Any:
        self.chat_calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if kwargs.get("stream"):
            return self._stream()
        return {"message": {"role": "assistant", "content": self.content, "thinking": "hmm"}}

    def _stream(self) -> Iterator[dict[str, Any]]:
        for delta in self.deltas:
            yield {"message": {"role": "assistant", "content": delta}}

    def embed(self, **kwargs: Any) -> dict[str, Any]:
        self.embed_calls.append(kwargs)
        if self.error is not None:
            raise self.error
        # Encode the input text in the vector so order can be checked.
        return {"embeddings": [[float(len(t))] * self.dim for t in kwargs["input"]]}


def make_llm(client: FakeClient) -> OllamaLLM:
    return OllamaLLM("qwen3:14b", host=HOST, num_ctx=8192, keep_alive="5m", client=client)


def make_embedder(client: FakeClient, batch_size: int = 2) -> OllamaEmbedder:
    return OllamaEmbedder(
        "qwen3-embedding:0.6b", host=HOST, keep_alive="5m", batch_size=batch_size, client=client
    )


def test_satisfy_protocols() -> None:
    llm: LLM = make_llm(FakeClient())
    embedder: Embedder = make_embedder(FakeClient())
    assert llm.model_name == "qwen3:14b"
    assert embedder.model_name == "qwen3-embedding:0.6b"


def test_from_config_reads_models_and_ollama_sections() -> None:
    cfg = Config.model_validate(
        {
            "ollama": {"host": "http://localhost:9999", "keep_alive": "1h"},
            "models": {"llm": "llama3:8b", "embedder": "nomic-embed-text", "num_ctx": 4096},
        }
    )
    llm = OllamaLLM.from_config(cfg)
    embedder = OllamaEmbedder.from_config(cfg)
    assert (llm.model_name, llm.host, llm.num_ctx, llm.keep_alive) == (
        "llama3:8b",
        "http://localhost:9999",
        4096,
        "1h",
    )
    assert (embedder.model_name, embedder.host, embedder.keep_alive) == (
        "nomic-embed-text",
        "http://localhost:9999",
        "1h",
    )


def test_generate_sends_num_ctx_and_keep_alive_without_schema_options() -> None:
    client = FakeClient(content="hello")
    assert make_llm(client).generate(MESSAGES) == "hello"
    (call,) = client.chat_calls
    assert call["model"] == "qwen3:14b"
    assert call["messages"] == MESSAGES
    assert call["options"] == {"num_ctx": 8192}
    assert call["keep_alive"] == "5m"
    assert "format" not in call
    assert "temperature" not in call["options"]
    assert call["think"] is False


def test_generate_with_schema_sets_format_temperature_and_disables_thinking() -> None:
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    client = FakeClient(content='{"x": 1}')
    assert make_llm(client).generate(MESSAGES, schema=schema) == '{"x": 1}'
    (call,) = client.chat_calls
    assert call["format"] == schema
    assert call["options"] == {"num_ctx": 8192, "temperature": 0}
    assert call["keep_alive"] == "5m"
    assert call["think"] is False


def test_think_flag_reaches_free_form_calls_but_never_schema_calls() -> None:
    client = FakeClient(content="{}")
    llm = OllamaLLM("m", host=HOST, num_ctx=8192, keep_alive="5m", think=True, client=client)
    llm.generate(MESSAGES)
    llm.generate(MESSAGES, schema={"type": "object"})
    list(llm.stream(MESSAGES))
    assert [c["think"] for c in client.chat_calls] == [True, False, True]


def test_generate_drops_inline_thinking() -> None:
    client = FakeClient(content="<think>\nlet me see\n</think>\n\nThe answer.")
    assert make_llm(client).generate(MESSAGES) == "The answer."


def test_stream_skips_empty_deltas_and_sends_options() -> None:
    client = FakeClient(deltas=["Hel", "", "lo", "", " world"])
    assert list(make_llm(client).stream(MESSAGES)) == ["Hel", "lo", " world"]
    (call,) = client.chat_calls
    assert call["stream"] is True
    assert call["options"] == {"num_ctx": 8192}
    assert call["keep_alive"] == "5m"
    assert call["think"] is False


def test_stream_drops_thinking_split_across_deltas() -> None:
    client = FakeClient(deltas=["<th", "ink>", "secret", "</thi", "nk>", "Hi", " <", "3"])
    assert "".join(make_llm(client).stream(MESSAGES)) == "Hi <3"


def test_embed_batches_in_order() -> None:
    client = FakeClient()
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]
    vectors = make_embedder(client, batch_size=2).embed(texts)
    assert [c["input"] for c in client.embed_calls] == [["a", "bb"], ["ccc", "dddd"], ["eeeee"]]
    assert all(c["keep_alive"] == "5m" for c in client.embed_calls)
    assert all(c["model"] == "qwen3-embedding:0.6b" for c in client.embed_calls)
    assert [v[0] for v in vectors] == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_embed_empty_input_makes_no_call() -> None:
    client = FakeClient()
    assert make_embedder(client).embed([]) == []
    assert client.embed_calls == []


def test_embed_rejects_non_positive_batch_size() -> None:
    with pytest.raises(ValueError):
        make_embedder(FakeClient(), batch_size=0)


@pytest.mark.parametrize("call", ["generate", "stream", "embed"])
def test_connection_error_maps_to_unavailable(call: str) -> None:
    client = FakeClient(error=ConnectionError("Failed to connect to Ollama."))
    with pytest.raises(OllamaUnavailableError, match="is Ollama running at"):
        if call == "generate":
            make_llm(client).generate(MESSAGES)
        elif call == "stream":
            list(make_llm(client).stream(MESSAGES))
        else:
            make_embedder(client).embed(["a"])


def test_missing_model_suggests_pull() -> None:
    client = FakeClient(error=ollama.ResponseError('model "qwen3:14b" not found', 404))
    with pytest.raises(OllamaUnavailableError, match="ollama pull qwen3:14b"):
        make_llm(client).generate(MESSAGES)


def test_other_errors_propagate() -> None:
    client = FakeClient(error=ollama.ResponseError("bad request", 400))
    with pytest.raises(ollama.ResponseError):
        make_llm(client).generate(MESSAGES)


class ToolClient(FakeClient):
    """Answers every non-streaming chat with the given raw message."""

    def __init__(self, message: dict[str, Any]) -> None:
        super().__init__()
        self.message = message

    def chat(self, **kwargs: Any) -> Any:
        self.chat_calls.append(kwargs)
        return {"message": self.message}


SPEC = ToolSpec("search", "Search the documents.", {"type": "object", "properties": {}})


def test_chat_sends_tools_in_ollama_format_and_parses_calls() -> None:
    client = ToolClient(
        {
            "role": "assistant",
            "content": "",
            "thinking": "I should search.",
            "tool_calls": [
                {"function": {"name": "search", "arguments": {"query": "deadline"}}},
                {"function": {"name": "calc", "arguments": '{"expr": "1+1"}'}},
                {"function": {"name": "calc", "arguments": "not json"}},
            ],
        }
    )
    turn = make_llm(client).chat(MESSAGES, tools=[SPEC])
    assert turn == AssistantTurn(
        "",
        (
            ToolCall("search", {"query": "deadline"}),
            ToolCall("calc", {"expr": "1+1"}),
            ToolCall("calc", {"_unparsed": "not json"}),
        ),
        "I should search.",
    )
    (call,) = client.chat_calls
    assert call["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": SPEC.description,
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    assert call["options"] == {"num_ctx": 8192}
    assert call["keep_alive"] == "5m"
    assert call["think"] is False


def test_chat_without_tools_sends_none_and_think_overrides_default() -> None:
    client = ToolClient({"role": "assistant", "content": "<think>x</think>Hello."})
    turn = make_llm(client).chat(MESSAGES, think=True)
    assert turn == AssistantTurn("Hello.")
    assert client.chat_calls[0]["tools"] is None
    assert client.chat_calls[0]["think"] is True


def test_tool_messages_are_sent_in_ollama_format() -> None:
    history: list[Message] = [
        {"role": "user", "content": "find it"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [ToolCall("search", {"query": "x"}).as_data()],
        },
        {"role": "tool", "content": "result", "tool_name": "search"},
    ]
    client = ToolClient({"role": "assistant", "content": "Done."})
    make_llm(client).chat(history)
    sent = client.chat_calls[0]["messages"]
    assert sent[1]["tool_calls"] == [{"function": {"name": "search", "arguments": {"query": "x"}}}]
    assert sent[2] == {"role": "tool", "content": "result", "tool_name": "search"}
    assert history[1]["tool_calls"] == [{"name": "search", "arguments": {"query": "x"}}]
