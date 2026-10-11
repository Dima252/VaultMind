"""Integration tests against a real Ollama daemon with the configured models pulled."""

from __future__ import annotations

import json

import pytest

from vaultmind.core.config import load_config
from vaultmind.core.ollama import OllamaEmbedder, OllamaLLM

pytestmark = pytest.mark.ollama


def test_generate_short_reply() -> None:
    llm = OllamaLLM.from_config(load_config())
    reply = llm.generate([{"role": "user", "content": "Reply with the single word: pong"}])
    assert reply.strip()
    assert "<think>" not in reply


def test_generate_with_schema_returns_matching_json() -> None:
    llm = OllamaLLM.from_config(load_config())
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}, "population_millions": {"type": "number"}},
        "required": ["city", "population_millions"],
    }
    raw = llm.generate(
        [{"role": "user", "content": "Give the capital of France and its population in millions."}],
        schema=schema,
    )
    data = json.loads(raw)
    assert isinstance(data["city"], str)
    assert isinstance(data["population_millions"], int | float)


def test_stream_yields_text() -> None:
    llm = OllamaLLM.from_config(load_config())
    deltas = list(llm.stream([{"role": "user", "content": "Count from 1 to 3."}]))
    assert all(deltas)
    assert "".join(deltas).strip()


def test_embed_three_texts_equal_length() -> None:
    embedder = OllamaEmbedder.from_config(load_config())
    vectors = embedder.embed(["local first", "privacy matters", "retrieval augmented generation"])
    assert len(vectors) == 3
    assert len({len(v) for v in vectors}) == 1
    assert len(vectors[0]) > 0
