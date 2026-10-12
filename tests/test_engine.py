"""Engine: one loaded set of models and stores; ask() streams and stores the exchange."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import pytest

from vaultmind.app.engine import Engine
from vaultmind.core.config import Config
from vaultmind.core.ollama import OllamaUnavailableError
from vaultmind.core.providers import Message
from vaultmind.ingest.pipeline import ingest_path
from vaultmind.store.chats import ChatMessage, ChatStore
from vaultmind.store.index import ChunkIndex
from vaultmind.store.registry import Registry
from vaultmind.store.traces import TraceStore

from .fakes import FakeEmbedder, FakeLLM


class LLM(FakeLLM):
    """Streams with the spaces kept, like a real model."""

    def stream(self, messages: Sequence[Message]) -> Iterator[str]:
        yield from re.findall(r"\S+\s*", self.generate(messages))


class DownLLM(FakeLLM):
    def stream(self, messages: Sequence[Message]) -> Iterator[str]:
        raise OllamaUnavailableError("Ollama is not running")


MakeEngine = Callable[[FakeLLM], Engine]


def _config(tmp_path: Path) -> Config:
    return Config.model_validate(
        {
            "paths": {"vault_dir": str(tmp_path / "vault"), "data_dir": str(tmp_path / "data")},
            "retrieval": {"rerank": False},
        }
    )


@pytest.fixture
def make_engine(tmp_path: Path) -> Iterator[MakeEngine]:
    cfg = _config(tmp_path)
    vault = cfg.paths.vault_dir
    vault.mkdir()
    (vault / "fees.md").write_text("# Fees\n\nPayment is due within sixty days.\n", "utf-8")
    engines: list[Engine] = []

    def make(llm: FakeLLM) -> Engine:
        embedder = FakeEmbedder()
        db = cfg.paths.data_dir / "vaultmind.sqlite"
        index = ChunkIndex(cfg.paths.data_dir / "index", embedder_name=embedder.model_name)
        registry = Registry(db)
        ingest_path(vault, cfg, embedder=embedder, index=index, registry=registry)
        engine = Engine(
            cfg,
            llm=llm,
            embedder=embedder,
            reranker=None,
            index=index,
            registry=registry,
            traces=TraceStore(db),
            chats=ChatStore(db),
        )
        engines.append(engine)
        return engine

    yield make
    for engine in engines:
        engine.close()


def test_ask_streams_then_stores_the_exchange(make_engine: MakeEngine) -> None:
    engine = make_engine(LLM(["Within sixty days [1]."]))
    chat = engine.chats.create()
    events = list(engine.ask(chat.id, "When is payment due?"))

    tokens, final = events[:-1], events[-1]
    assert "".join(t for t in tokens if isinstance(t, str)) == "Within sixty days [1]."
    assert isinstance(final, ChatMessage) and final.role == "assistant"
    assert final.run_id is not None and engine.traces.get(final.run_id) is not None
    assert [m.role for m in engine.chats.messages(chat.id)] == ["user", "assistant"]
    assert final.sources and final.sources[0].file_path is not None


def test_follow_up_passes_chat_history(make_engine: MakeEngine) -> None:
    analyze = {"standalone_query": "When is payment due?", "variants": [], "needs_retrieval": True}
    llm = LLM(["Within sixty days [1].", analyze, "Sixty days [1]."])
    engine = make_engine(llm)
    chat = engine.chats.create()
    list(engine.ask(chat.id, "When is payment due?"))
    list(engine.ask(chat.id, "And how many days is that?"))

    analyze_call = llm.calls[1]
    assert any("When is payment due?" in m["content"] for m in analyze_call)
    assert len(engine.chats.messages(chat.id)) == 4


def test_failed_ask_stores_nothing(make_engine: MakeEngine) -> None:
    engine = make_engine(DownLLM())
    chat = engine.chats.create()
    with pytest.raises(OllamaUnavailableError):
        list(engine.ask(chat.id, "When is payment due?"))
    assert engine.chats.messages(chat.id) == []
    assert engine.traces.recent() == []


def test_close_is_idempotent(make_engine: MakeEngine) -> None:
    engine = make_engine(LLM())
    engine.close()
    engine.close()
