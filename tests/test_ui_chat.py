"""Chat page, trace view and citation viewer, driven with NiceGUI's simulated user (no models)."""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pymupdf as _pymupdf
import pytest
from nicegui import ui
from nicegui.testing.user import User
from nicegui.testing.user_simulation import user_simulation

from vaultmind.agent.baseline import Source
from vaultmind.agent.types import AgentResult, Contradiction, TraceEvent
from vaultmind.app.engine import Engine
from vaultmind.core.config import Config
from vaultmind.core.ollama import OllamaUnavailableError
from vaultmind.core.providers import Message
from vaultmind.ingest.pipeline import ingest_path
from vaultmind.store.chats import ChatStore
from vaultmind.store.index import ChunkIndex
from vaultmind.store.registry import Registry
from vaultmind.store.traces import TraceStore
from vaultmind.ui import state as ui_state
from vaultmind.ui.app import root
from vaultmind.ui.citation_view import marked_html, open_citation
from vaultmind.ui.trace_view import trace_view

from .fakes import FakeEmbedder, FakeLLM

pymupdf: Any = _pymupdf  # partially typed upstream

QUESTION = "When is payment due?"
ANSWER = "Payment is due within sixty days [1]."
WAIT = 50  # should_see retries (0.1 s each) for work done on worker threads


class LLM(FakeLLM):
    """Streams with the spaces kept, like a real model."""

    def stream(self, messages: Sequence[Message]) -> Iterator[str]:
        yield from re.findall(r"\S+\s*", self.generate(messages))


class DownLLM(FakeLLM):
    def stream(self, messages: Sequence[Message]) -> Iterator[str]:
        raise OllamaUnavailableError("Ollama is not running")


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    cfg = Config.model_validate(
        {
            "paths": {"vault_dir": str(tmp_path / "vault"), "data_dir": str(tmp_path / "data")},
            "retrieval": {"rerank": False},
        }
    )
    cfg.paths.vault_dir.mkdir()
    (cfg.paths.vault_dir / "fees.md").write_text(
        "# Fees\n\nPayment is due within sixty days of the invoice date.\n", "utf-8"
    )
    return cfg


def _engine(cfg: Config, llm: FakeLLM) -> Engine:
    embedder = FakeEmbedder()
    db = cfg.paths.data_dir / "vaultmind.sqlite"
    index = ChunkIndex(cfg.paths.data_dir / "index", embedder_name=embedder.model_name)
    registry = Registry(db)
    ingest_path(cfg.paths.vault_dir, cfg, embedder=embedder, index=index, registry=registry)
    return Engine(
        cfg,
        llm=llm,
        embedder=embedder,
        reranker=None,
        index=index,
        registry=registry,
        traces=TraceStore(db),
        chats=ChatStore(db),
    )


@pytest.fixture
def make_app(cfg: Config) -> Iterator[Any]:
    """Make the engine with a given LLM and mark the app ready, as after startup."""
    engines: list[Engine] = []

    def make(llm: FakeLLM) -> Engine:
        engine = _engine(cfg, llm)
        engines.append(engine)
        app_state = ui_state.init_state(cfg)
        app_state.engine, app_state.phase = engine, "ready"
        return engine

    yield make
    for engine in engines:
        engine.close()
    ui_state._state = None


async def _ask(user: User, question: str) -> None:
    user.find(marker="question").type(question)
    user.find(marker="send").click()


def _input_value(user: User) -> str:
    (field,) = user.find(marker="question").elements
    assert isinstance(field, ui.textarea)
    return str(field.value or "")


@pytest.mark.anyio
async def test_answer_streams_with_citation_chip_and_titles_the_chat(make_app: Any) -> None:
    engine = make_app(LLM([ANSWER]))
    async with user_simulation(root=root) as user:
        await user.open("/")
        await _ask(user, QUESTION)
        await user.should_see("within sixty days", retries=WAIT)
        await user.should_see(kind=ui.chip, content="[1] fees.md", retries=WAIT)
        (chat,) = engine.chats.recent()
        assert chat.title == QUESTION
        with user.scope(marker=f"chat-{chat.id}"):
            await user.should_see(QUESTION)
        assert _input_value(user) == ""


@pytest.mark.anyio
async def test_new_chat_starts_empty_and_switching_shows_its_messages(make_app: Any) -> None:
    engine = make_app(LLM([ANSWER, "No idea [1]."]))
    async with user_simulation(root=root) as user:
        await user.open("/")
        await _ask(user, QUESTION)
        await user.should_see(kind=ui.chip, retries=WAIT)

        user.find(marker="new-chat").click()
        with user.scope(marker="messages"):
            await user.should_not_see(QUESTION)
            await user.should_see("Ask a question about your documents")

        await _ask(user, "What color is the sky?")
        await user.should_see("No idea", retries=WAIT)
        with user.scope(marker="messages"):
            await user.should_not_see(QUESTION)

        first = next(c for c in engine.chats.recent() if c.title == QUESTION)
        user.find(marker=f"chat-{first.id}").click()
        with user.scope(marker="messages"):
            await user.should_see(QUESTION)
            await user.should_see("within sixty days")
            await user.should_not_see("No idea")


@pytest.mark.anyio
async def test_open_chat_is_kept_across_navigation(make_app: Any) -> None:
    engine = make_app(LLM([ANSWER]))
    older = engine.chats.create("Older chat")
    async with user_simulation(root=root) as user:
        await user.open("/")
        user.find(marker="new-chat").click()
        await _ask(user, QUESTION)
        await user.should_see(kind=ui.chip, retries=WAIT)
        user.find(marker=f"chat-{older.id}").click()
        await user.should_not_see("within sixty days")

        user.find("Library").click()
        await user.should_not_see(marker="question")
        user.find("Chat").click()
        await user.should_see(marker="question")
        await user.should_not_see("within sixty days")  # still the older chat, not the newest


@pytest.mark.anyio
async def test_ollama_down_notifies_and_restores_the_question(make_app: Any) -> None:
    engine = make_app(DownLLM())
    async with user_simulation(root=root) as user:
        await user.open("/")
        await _ask(user, QUESTION)
        await user.should_see("Ollama is not running", retries=WAIT)
        assert _input_value(user) == QUESTION
        await user.should_not_see(kind=ui.chip)
        (field,) = user.find(marker="question").elements
        assert isinstance(field, ui.textarea) and field.enabled
        assert all(not engine.chats.messages(c.id) for c in engine.chats.recent())


@pytest.mark.anyio
async def test_enter_sends(make_app: Any) -> None:
    make_app(LLM([ANSWER]))
    async with user_simulation(root=root) as user:
        await user.open("/")
        user.find(marker="question").type(QUESTION).trigger("keydown.enter.exact.prevent")
        await user.should_see("within sixty days", retries=WAIT)


@pytest.mark.anyio
async def test_trace_expansion_loads_the_trace_when_opened(make_app: Any) -> None:
    make_app(LLM([ANSWER]))
    async with user_simulation(root=root) as user:
        await user.open("/")
        await _ask(user, QUESTION)
        await user.should_see(kind=ui.chip, retries=WAIT)
        await user.should_not_see("total")
        (expansion,) = user.find(marker="trace").elements
        assert isinstance(expansion, ui.expansion)
        with user:
            expansion.value = True
        await user.should_see("retrieve", retries=WAIT)
        await user.should_see("generate")
        await user.should_see("total")


@pytest.mark.anyio
async def test_stored_answer_shows_notes_and_closest_matches(make_app: Any, cfg: Config) -> None:
    engine = make_app(LLM())
    chat = engine.chats.create()
    fees = cfg.paths.vault_dir / "fees.md"
    sources = [
        Source(1, fees, None, ("Fees",), "Payment is due within sixty days.", 0.9),
        Source(2, fees, None, ("Fees",), "Payment is due within thirty days.", 0.8, None, ("x",)),
    ]
    result = AgentResult(
        question=QUESTION,
        answer="Not found in your documents.",
        sources=sources,
        abstained=True,
        queries=[QUESTION],
        rewrites=0,
        contradictions=[Contradiction("sixty days", 1, "thirty days", 2)],
    )
    engine.chats.add_exchange(chat.id, result, None)
    async with user_simulation(root=root) as user:
        await user.open("/")
        await user.should_see("Not found in your documents")
        await user.should_see("Closest matches")
        await user.should_see("Sources disagree")
        await user.should_see(kind=ui.chip, content="[2] fees.md")


@pytest.mark.anyio
async def test_export_chat_as_markdown(make_app: Any) -> None:
    make_app(LLM([ANSWER]))
    async with user_simulation(root=root) as user:
        await user.open("/")
        await _ask(user, QUESTION)
        await user.should_see(kind=ui.chip, retries=WAIT)
        user.find(marker="export-chat-md").click()
        response = await user.download.next()
        text = response.content.decode("utf-8")
        assert f"# {QUESTION}" in text and "[1] fees.md" in text


@pytest.mark.anyio
async def test_delete_chat_after_confirmation(make_app: Any) -> None:
    engine = make_app(LLM())
    chat = engine.chats.create("Doomed chat")
    async with user_simulation(root=root) as user:
        await user.open("/")
        user.find(marker=f"delete-{chat.id}").click()
        await user.should_see("Its messages are removed for good.")
        user.find(marker="delete-confirm").click()
        await user.should_not_see(marker=f"chat-{chat.id}", retries=WAIT)
        assert engine.chats.recent() == []


@pytest.mark.anyio
async def test_citation_chip_opens_markdown_with_the_passage_marked(make_app: Any) -> None:
    make_app(LLM([ANSWER]))
    async with user_simulation(root=root) as user:
        await user.open("/")
        await _ask(user, QUESTION)
        await user.should_see(kind=ui.chip, retries=WAIT)
        user.find(kind=ui.chip).click()
        await user.should_see(kind=ui.html, content="<mark", retries=WAIT)
        (shown,) = user.find(kind=ui.html, content="<mark").elements
        assert isinstance(shown, ui.html)
        marked = re.search(r"<mark[^>]*>(.*?)</mark>", shown.content, re.DOTALL)
        assert marked is not None and "sixty days" in marked.group(1)


@pytest.mark.anyio
async def test_citation_of_a_missing_file_says_so(make_app: Any, tmp_path: Path) -> None:
    make_app(LLM())
    source = Source(1, tmp_path / "gone.md", None, (), "Some passage.", 1.0)
    async with user_simulation(root=root) as user:
        await user.open("/")
        with user:
            open_citation(source)
        await user.should_see("The file is no longer at", retries=WAIT)
        await user.should_see("Some passage.")


@pytest.mark.anyio
async def test_citation_of_a_pdf_renders_the_page(make_app: Any, tmp_path: Path) -> None:
    make_app(LLM())
    pdf = tmp_path / "terms.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "The fee is payable within sixty days of invoicing.")
    doc.save(pdf)
    doc.close()
    source = Source(1, pdf, 1, (), "The fee is payable within sixty days of invoicing.", 1.0)
    async with user_simulation(root=root) as user:
        await user.open("/")
        with user:
            open_citation(source)
        await user.should_see("Page 1 of 1", retries=WAIT)
        await user.should_see(kind=ui.image)
        await user.should_not_see("Passage not found")


def test_marked_html_escapes_the_text() -> None:
    text = "a <b> c PASSAGE & d"
    start = text.index("PASSAGE")
    out = marked_html(text, (start, start + 7), "m1")
    assert out == 'a &lt;b&gt; c <mark id="m1">PASSAGE</mark> &amp; d'


@pytest.mark.anyio
async def test_trace_view_shows_grade_scores_sorted() -> None:
    events = [
        TraceEvent("retrieve", 12.0, {"queries": ["q"], "candidates": 3}),
        TraceEvent(
            "grade",
            5.0,
            {
                "threshold": 0.5,
                "passed": 1,
                "scores": [
                    {"label": "low.md, p. 1", "score": 0.125, "passed": False},
                    {"label": "high.md, p. 2", "score": 0.912, "passed": True},
                ],
            },
        ),
        TraceEvent("generate", 800.0, {"sources": [1], "abstained": False}),
    ]

    def page() -> None:
        trace_view(events)

    async with user_simulation(root=page) as user:
        await user.open("/")
        await user.should_see("1/2 passed")
        await user.should_see("0.912")
        await user.should_see("817 ms")
        in_page_order = [
            e.text
            for e in user.current_layout.descendants()
            if isinstance(e, ui.label)
            and e.text.endswith(".md, p. 1") | e.text.endswith(".md, p. 2")
        ]
        assert in_page_order == ["high.md, p. 2", "low.md, p. 1"]
