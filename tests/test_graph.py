"""Corrective agent loop: routing, trace contract, streaming. Fakes only, no Ollama."""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from vaultmind.agent import guard, nodes
from vaultmind.agent.baseline import NOT_FOUND, Source
from vaultmind.agent.checks import CheckError
from vaultmind.agent.graph import run_agent, stream_agent
from vaultmind.agent.types import AgentResult, Contradiction, UnsupportedClaim
from vaultmind.core.config import Config
from vaultmind.core.models import ChildChunk, FileRecord, ParentSection
from vaultmind.core.providers import Message, Reranker
from vaultmind.store.index import ChunkIndex
from vaultmind.store.registry import Registry

from .fakes import FakeEmbedder, FakeLLM, FakeReranker

FILE_A = "0123456789abcdef"
FILE_B = "fedcba9876543210"
PARENT_A = ParentSection(f"{FILE_A}:p0", FILE_A, "Whole contract section.", ("Terms",), 1, 1)
PARENT_B = ParentSection(f"{FILE_B}:p0", FILE_B, "Whole amendment section.", ("Fees",), 2, 2)


def _child(parent: ParentSection, i: int, text: str) -> ChildChunk:
    section = parent.section_path[0]
    return ChildChunk(
        f"{parent.id}:c{i}",
        parent.id,
        parent.file_id,
        text,
        f"Doc > {section}\n\n{text}",
        parent.page_start,
        parent.section_path,
    )


GOV = _child(PARENT_A, 0, "governing law is the state of delaware")
PAY_A = _child(PARENT_A, 1, "payment is due within sixty days")
TERM = _child(PARENT_A, 2, "either party may terminate for convenience")
PAY_B = _child(PARENT_B, 0, "payment is due within thirty days")
CHILDREN = [GOV, PAY_A, TERM, PAY_B]

QUESTION = "when is payment due"  # passes PAY_A and PAY_B (0.75), fails GOV (0.25) and TERM (0)
ANSWER = "Payment is due within sixty days [1]."
ORDER_HAPPY = ["analyze", "retrieve", "grade", "generate"]


class LLM(FakeLLM):
    """FakeLLM whose stream keeps the spaces, so streamed answers read like real ones."""

    def stream(self, messages: Sequence[Message]) -> Iterator[str]:
        yield from re.findall(r"\S+\s*", self.generate(messages))


class AliasEmbedder(FakeEmbedder):
    """Embeds an aliased query as the given chunk text: its vector search finds that chunk."""

    def __init__(self, aliases: dict[str, ChildChunk]) -> None:
        super().__init__()
        self.aliases = aliases

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return super().embed([a.embed_text if (a := self.aliases.get(t)) else t for t in texts])


@pytest.fixture
def store(tmp_path: Path) -> Iterator[tuple[ChunkIndex, Registry]]:
    embedder = FakeEmbedder()
    index = ChunkIndex(tmp_path / "index", embedder_name=embedder.model_name)
    registry = Registry(tmp_path / "r.sqlite")
    index.add([PARENT_A, PARENT_B], CHILDREN, embedder.embed([c.embed_text for c in CHILDREN]))
    registry.upsert(FileRecord(FILE_A, tmp_path / "contract.txt", "sha", "ready"))
    registry.upsert(FileRecord(FILE_B, tmp_path / "amendment.txt", "sha", "ready"))
    yield index, registry
    registry.close()
    index.close()


def _cfg(**sections: dict[str, Any]) -> Config:
    """Small, deterministic defaults: vector search over the whole tiny index, then rerank."""
    retrieval = {"hybrid": False, "candidates": 20, "rerank": True} | sections.get("retrieval", {})
    agent = {"query_variants": 0, "rewrite": True, "detect_contradictions": False} | sections.get(
        "agent", {}
    )
    return Config.model_validate({"retrieval": retrieval, "agent": agent})


def _stream(
    store: tuple[ChunkIndex, Registry],
    llm: Any,
    cfg: Config,
    question: str = QUESTION,
    *,
    embedder: FakeEmbedder | None = None,
    reranker: Reranker | None = None,
    history: Sequence[Message] = (),
) -> list[str | AgentResult]:
    index, registry = store
    return list(
        stream_agent(
            question,
            cfg,
            llm=llm,
            embedder=embedder or FakeEmbedder(),
            index=index,
            registry=registry,
            reranker=reranker if reranker is not None else FakeReranker(),
            history=history,
        )
    )


def _run(*args: Any, **kwargs: Any) -> AgentResult:
    events = _stream(*args, **kwargs)
    result = events[-1]
    assert isinstance(result, AgentResult)
    assert all(isinstance(e, str) for e in events[:-1])
    assert "".join(e for e in events[:-1] if isinstance(e, str)).strip() == result.answer
    for event in result.trace:  # the trace store serializes every event
        json.dumps(event.data)
        assert event.duration_ms >= 0
    return result


def _nodes(result: AgentResult) -> list[str]:
    return [e.node for e in result.trace]


def _data(result: AgentResult, node: str) -> dict[str, Any]:
    [event] = [e for e in result.trace if e.node == node]
    return event.data


def _chunks(sources: Sequence[Source]) -> set[str | None]:
    return {s.chunk_id for s in sources}


def _analysis(standalone: str = QUESTION, variants: Sequence[str] = ()) -> dict[str, Any]:
    return {"standalone_query": standalone, "variants": list(variants), "needs_retrieval": True}


# --- happy path ----------------------------------------------------------------------------


def test_happy_path_trace_and_data_keys(store: tuple[ChunkIndex, Registry]) -> None:
    variants = ["payment deadline", "when must invoices be paid"]
    llm = LLM([_analysis("rephrased by the model", variants), ANSWER])
    result = _run(store, llm, _cfg(agent={"query_variants": 2}))

    assert _nodes(result) == ORDER_HAPPY
    assert result.answer == ANSWER
    assert not result.abstained
    assert result.rewrites == 0
    # Without history the question already stands alone, so it is kept verbatim.
    assert result.queries == [QUESTION, *variants]

    analyze = _data(result, "analyze")
    assert analyze["standalone_query"] == QUESTION
    assert analyze["variants"] == variants
    assert analyze["needs_retrieval"] is True

    retrieve = _data(result, "retrieve")
    assert retrieve["queries"] == [QUESTION, *variants]
    assert retrieve["candidates"] == 4

    grade = _data(result, "grade")
    assert grade["threshold"] == pytest.approx(0.3)
    assert grade["passed"] == 2
    assert len(grade["scores"]) == 4
    assert {s["label"] for s in grade["scores"] if s["passed"]} == {
        "contract.txt, p. 1, Terms",
        "amendment.txt, p. 2, Fees",
    }
    assert all(set(s) == {"label", "score", "passed"} for s in grade["scores"])
    assert [s["score"] for s in grade["scores"]] == sorted(
        (s["score"] for s in grade["scores"]), reverse=True
    )

    assert _chunks(result.sources) == {PAY_A.id, PAY_B.id}
    assert [s.n for s in result.sources] == [1, 2]
    assert _data(result, "generate") == {"sources": [1, 2], "abstained": False}
    assert [s.n for s in result.cited] == [1]


def test_variants_are_deduplicated_and_capped(store: tuple[ChunkIndex, Registry]) -> None:
    variants = ["WHEN is  payment due", "payment deadline", "Payment Deadline", "", "third", "x"]
    llm = LLM([_analysis(variants=variants), ANSWER])
    result = _run(store, llm, _cfg(agent={"query_variants": 2}))
    assert result.queries == [QUESTION, "payment deadline", "third"]


def test_grading_scores_the_standalone_question_not_the_variants(
    store: tuple[ChunkIndex, Registry],
) -> None:
    # The variant matches the governing-law clause word for word, but grading is against the
    # question, so that clause still fails.
    llm = LLM([_analysis(variants=["governing law state delaware"]), ANSWER])
    result = _run(store, llm, _cfg(agent={"query_variants": 1}))
    assert GOV.id not in _chunks(result.sources)


# --- analyze -------------------------------------------------------------------------------


def test_analyze_skipped_without_history_or_variants(store: tuple[ChunkIndex, Registry]) -> None:
    llm = LLM([ANSWER])
    result = _run(store, llm, _cfg(agent={"query_variants": 0}))
    assert len(llm.calls) == 1  # only the answer
    assert _nodes(result) == ORDER_HAPPY
    assert _data(result, "analyze")["standalone_query"] == QUESTION
    assert _data(result, "analyze")["variants"] == []
    assert result.queries == [QUESTION]


def test_analyze_falls_back_on_bad_json(store: tuple[ChunkIndex, Registry]) -> None:
    llm = LLM(["this is not json", ANSWER])
    result = _run(store, llm, _cfg(agent={"query_variants": 2}))
    analyze = _data(result, "analyze")
    assert "error" in analyze
    assert analyze["standalone_query"] == QUESTION
    assert analyze["variants"] == []
    assert analyze["needs_retrieval"] is True
    assert _nodes(result) == ORDER_HAPPY
    assert result.answer == ANSWER


def test_history_reaches_analyze_and_answer(store: tuple[ChunkIndex, Registry]) -> None:
    history: list[Message] = [
        {"role": "user", "content": "what does the payment clause say"},
        {"role": "assistant", "content": "It sets a deadline [1]."},
    ]
    follow_up = "and when is it?"
    llm = LLM([_analysis(QUESTION), ANSWER])
    result = _run(store, llm, _cfg(agent={"query_variants": 0}), follow_up, history=history)

    analyze_prompt = "\n".join(m["content"] for m in llm.calls[0])
    assert "what does the payment clause say" in analyze_prompt
    assert follow_up in analyze_prompt
    assert result.queries == [QUESTION]  # the standalone query is searched and graded
    assert _chunks(result.sources) == {PAY_A.id, PAY_B.id}

    answer_messages = list(llm.calls[1])
    assert all(m in answer_messages for m in history)
    assert follow_up in answer_messages[-1]["content"]  # the user's own words are answered
    assert result.question == follow_up


def test_direct_path_skips_retrieval(store: tuple[ChunkIndex, Registry]) -> None:
    analysis = {"standalone_query": "hi there", "variants": [], "needs_retrieval": False}
    llm = LLM([analysis, "Hello! Ask me about your documents."])
    result = _run(store, llm, _cfg(agent={"query_variants": 2}), "hi there")
    assert _nodes(result) == ["analyze", "direct"]
    assert result.answer == "Hello! Ask me about your documents."
    assert result.sources == []
    assert result.queries == []
    assert not result.abstained
    assert _data(result, "direct")["reason"]
    assert "Sources" not in "\n".join(m["content"] for m in llm.calls[1])


def test_unclassified_greeting_with_no_match_gets_a_reply(
    store: tuple[ChunkIndex, Registry],
) -> None:
    llm = LLM(["Hello! Ask me about your documents."])
    result = _run(store, llm, _cfg(agent={"rewrite": False}), "hi")
    assert _nodes(result) == ["analyze", "retrieve", "grade", "direct"]
    assert result.answer == "Hello! Ask me about your documents."
    assert not result.abstained
    assert result.sources == []
    assert "closest" not in _data(result, "direct")


def test_unclassified_question_with_no_match_is_not_found_with_closest(
    store: tuple[ChunkIndex, Registry],
) -> None:
    embedder = AliasEmbedder({QUESTION: GOV})
    result = _run(store, LLM([NOT_FOUND]), _loop_cfg(rewrite=False), embedder=embedder)
    assert _nodes(result) == ["analyze", "retrieve", "grade", "direct"]
    assert result.abstained
    assert [s.chunk_id for s in result.sources] == [GOV.id]
    assert _data(result, "direct")["closest"] == [s.label for s in result.sources]


def test_classified_question_with_no_match_skips_the_reply(
    store: tuple[ChunkIndex, Registry],
) -> None:
    embedder = AliasEmbedder({QUESTION: GOV})
    history: list[Message] = [{"role": "user", "content": "earlier"}]
    llm = LLM([_analysis()])
    result = _run(store, llm, _loop_cfg(rewrite=False), embedder=embedder, history=history)
    assert _nodes(result) == ["analyze", "retrieve", "grade", "not_found"]
    assert len(llm.calls) == 1  # analyze only


# --- rewrite loop --------------------------------------------------------------------------


def _loop_cfg(**agent: Any) -> Config:
    # One candidate per query, so each query reaches exactly the chunk its alias names.
    return _cfg(retrieval={"candidates": 1}, agent=agent)


def test_rewrite_recovers(store: tuple[ChunkIndex, Registry]) -> None:
    embedder = AliasEmbedder({QUESTION: GOV, "payment terms": PAY_A})
    llm = LLM([{"query": "payment terms"}, ANSWER])
    result = _run(store, llm, _loop_cfg(), embedder=embedder)

    assert _nodes(result) == [
        "analyze",
        "retrieve",
        "grade",
        "rewrite",
        "retrieve",
        "grade",
        "generate",
    ]
    assert result.rewrites == 1
    assert result.queries == [QUESTION, "payment terms"]
    assert _data(result, "rewrite") == {"attempt": 1, "query": "payment terms"}
    assert _chunks(result.sources) == {PAY_A.id}
    assert not result.abstained

    rewrite_prompt = llm.calls[0][-1]["content"]
    assert QUESTION in rewrite_prompt
    assert GOV.text in rewrite_prompt  # what the failed search found


def test_rewrites_exhausted_gives_not_found_with_closest(
    store: tuple[ChunkIndex, Registry],
) -> None:
    embedder = AliasEmbedder({QUESTION: GOV, "r1": TERM, "r2": GOV})
    llm = LLM([{"query": "r1"}, {"query": "r2"}])
    result = _run(store, llm, _loop_cfg(max_rewrites=2), embedder=embedder)

    assert _nodes(result) == [
        "analyze",
        *["retrieve", "grade", "rewrite"] * 2,
        "retrieve",
        "grade",
        "not_found",
    ]
    assert result.rewrites == 2
    assert result.queries == [QUESTION, "r1", "r2"]
    assert result.answer == NOT_FOUND
    assert result.abstained
    assert [s.chunk_id for s in result.sources] == [GOV.id, TERM.id]  # best first, no repeats
    assert result.sources[0].score == pytest.approx(0.25)
    assert _data(result, "not_found") == {"closest": [s.label for s in result.sources]}
    assert [e.data["attempt"] for e in result.trace if e.node == "rewrite"] == [1, 2]
    assert len(llm.calls) == 2  # two rewrites, no answer call


def test_rewrite_off_goes_straight_to_not_found(store: tuple[ChunkIndex, Registry]) -> None:
    embedder = AliasEmbedder({QUESTION: GOV})
    llm = LLM([])
    result = _run(store, llm, _loop_cfg(rewrite=False, smalltalk=False), embedder=embedder)
    assert _nodes(result) == ["analyze", "retrieve", "grade", "not_found"]
    assert llm.calls == []
    assert result.abstained
    assert result.rewrites == 0


def test_repeated_rewrite_query_ends_the_loop(store: tuple[ChunkIndex, Registry]) -> None:
    embedder = AliasEmbedder({QUESTION: GOV})
    llm = LLM([{"query": "  When IS payment   due "}])
    result = _run(store, llm, _loop_cfg(max_rewrites=2), embedder=embedder)
    assert _nodes(result) == ["analyze", "retrieve", "grade", "rewrite", "not_found"]
    assert _data(result, "rewrite")["repeated"] is True
    assert result.queries == [QUESTION]
    assert result.abstained


def test_unparseable_rewrite_ends_the_loop(store: tuple[ChunkIndex, Registry]) -> None:
    embedder = AliasEmbedder({QUESTION: GOV})
    result = _run(store, LLM(["nope"]), _loop_cfg(), embedder=embedder)
    assert _nodes(result) == ["analyze", "retrieve", "grade", "rewrite", "not_found"]
    assert "error" in _data(result, "rewrite")
    assert _data(result, "rewrite")["query"] == ""


def test_generator_abstention_is_detected(store: tuple[ChunkIndex, Registry]) -> None:
    result = _run(store, LLM([NOT_FOUND]), _cfg())
    assert result.abstained
    assert _data(result, "generate")["abstained"] is True
    assert result.sources  # the graded passages it was shown


# --- grading options -----------------------------------------------------------------------


def test_rerank_off_keeps_top_k(store: tuple[ChunkIndex, Registry]) -> None:
    index, registry = store
    cfg = _cfg(retrieval={"rerank": False, "top_k": 2})
    result = run_agent(
        QUESTION,
        cfg,
        llm=LLM([ANSWER]),
        embedder=FakeEmbedder(),
        index=index,
        registry=registry,
        reranker=None,
    )
    grade = _data(result, "grade")
    assert grade["threshold"] is None
    assert grade["passed"] == 2
    assert all(s["passed"] for s in grade["scores"])
    assert len(result.sources) == 2


def test_rerank_on_without_reranker_raises(store: tuple[ChunkIndex, Registry]) -> None:
    index, registry = store
    with pytest.raises(ValueError, match="no reranker"):
        run_agent(
            QUESTION,
            _cfg(),
            llm=LLM([]),
            embedder=FakeEmbedder(),
            index=index,
            registry=registry,
            reranker=None,
        )


def test_injection_warnings_on_sources(
    store: tuple[ChunkIndex, Registry], monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_find(text: str) -> list[str]:
        return ["instruction-like text"] if "sixty" in text else []

    monkeypatch.setattr(nodes, "find_injections", fake_find)
    result = _run(store, LLM([ANSWER]), _cfg())
    warned = {s.chunk_id: s.warnings for s in result.sources}
    assert warned == {PAY_A.id: ("instruction-like text",), PAY_B.id: ()}
    assert _data(result, "grade")["flagged"] == [
        {"label": "contract.txt, p. 1, Terms", "warnings": ["instruction-like text"]}
    ]

    off = _run(store, LLM([ANSWER]), _cfg(agent={"injection_guard": False}))
    assert all(s.warnings == () for s in off.sources)
    assert _data(off, "grade")["flagged"] == []


def test_guard_flag_reaches_the_answer_prompt(
    store: tuple[ChunkIndex, Registry], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[dict[str, Any]] = []

    def spy(question: str, sources: Sequence[Source], **kw: Any) -> list[Message]:
        seen.append({"question": question, "n": len(sources), **kw})
        return [{"role": "user", "content": question}]

    monkeypatch.setattr(nodes, "build_answer_messages", spy)
    _run(store, LLM([ANSWER]), _cfg(agent={"injection_guard": False}))
    assert seen[0]["guard"] is False
    assert seen[0]["question"] == QUESTION
    assert seen[0]["n"] == 2


# --- optional checks -----------------------------------------------------------------------

CONFLICT = Contradiction("sixty days", 1, "thirty days", 2)


def test_contradictions_feed_the_answer(
    store: tuple[ChunkIndex, Registry], monkeypatch: pytest.MonkeyPatch
) -> None:
    passed: list[Any] = []
    real = guard.build_answer_messages

    def spy(*args: Any, **kw: Any) -> list[Message]:
        passed.append(kw["contradictions"])
        return real(*args, **kw)

    monkeypatch.setattr(nodes, "detect_contradictions", lambda q, s, *, llm: [CONFLICT])
    monkeypatch.setattr(nodes, "build_answer_messages", spy)
    result = _run(store, LLM([ANSWER]), _cfg(agent={"detect_contradictions": True}))
    assert _nodes(result) == ["analyze", "retrieve", "grade", "contradictions", "generate"]
    assert result.contradictions == [CONFLICT]
    assert passed == [[CONFLICT]]
    assert _data(result, "contradictions")["found"] == [
        {"claim_a": "sixty days", "source_a": 1, "claim_b": "thirty days", "source_b": 2}
    ]


def test_contradictions_need_two_files(
    store: tuple[ChunkIndex, Registry], monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*a: Any, **kw: Any) -> list[Contradiction]:
        raise AssertionError("should not run")

    monkeypatch.setattr(nodes, "detect_contradictions", boom)
    cfg = _cfg(agent={"detect_contradictions": True}, retrieval={"top_k": 1})
    result = _run(store, LLM([ANSWER]), cfg)
    assert _nodes(result) == ORDER_HAPPY


def test_contradictions_off_by_config(
    store: tuple[ChunkIndex, Registry], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(nodes, "detect_contradictions", lambda *a, **kw: [CONFLICT])
    result = _run(store, LLM([ANSWER]), _cfg(agent={"detect_contradictions": False}))
    assert _nodes(result) == ORDER_HAPPY
    assert result.contradictions == []


def test_contradiction_check_error_is_recorded(
    store: tuple[ChunkIndex, Registry], monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*a: Any, **kw: Any) -> list[Contradiction]:
        raise CheckError("bad output")

    monkeypatch.setattr(nodes, "detect_contradictions", fail)
    result = _run(store, LLM([ANSWER]), _cfg(agent={"detect_contradictions": True}))
    assert _data(result, "contradictions") == {"error": "bad output", "found": []}
    assert result.answer == ANSWER


def test_groundedness_reports_unsupported_claims(
    store: tuple[ChunkIndex, Registry], monkeypatch: pytest.MonkeyPatch
) -> None:
    claim = UnsupportedClaim("sixty days", (1,), "the passage says thirty")
    seen: list[str] = []

    def check(answer: str, sources: Sequence[Source], *, llm: Any) -> list[UnsupportedClaim]:
        seen.append(answer)
        return [claim]

    monkeypatch.setattr(nodes, "check_groundedness", check)
    result = _run(store, LLM([ANSWER]), _cfg(agent={"groundedness_check": True}))
    assert _nodes(result) == [*ORDER_HAPPY, "groundedness"]
    assert seen == [ANSWER]
    assert result.unsupported == [claim]
    assert _data(result, "groundedness") == {
        "unsupported": [{"claim": "sixty days", "cited": [1], "reason": "the passage says thirty"}]
    }

    off = _run(store, LLM([ANSWER]), _cfg(agent={"groundedness_check": False}))
    assert _nodes(off) == ORDER_HAPPY
    assert off.unsupported == []


def test_groundedness_skipped_on_abstention(
    store: tuple[ChunkIndex, Registry], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(nodes, "check_groundedness", lambda *a, **kw: [])
    cfg = _cfg(agent={"groundedness_check": True})
    assert _nodes(_run(store, LLM([NOT_FOUND]), cfg)) == ORDER_HAPPY
    loop = _loop_cfg(groundedness_check=True, rewrite=False, smalltalk=False)
    embedder = AliasEmbedder({QUESTION: GOV})
    assert _nodes(_run(store, LLM([]), loop, embedder=embedder))[-1] == "not_found"


def test_groundedness_error_is_recorded(
    store: tuple[ChunkIndex, Registry], monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*a: Any, **kw: Any) -> list[UnsupportedClaim]:
        raise CheckError("unparseable")

    monkeypatch.setattr(nodes, "check_groundedness", fail)
    result = _run(store, LLM([ANSWER]), _cfg(agent={"groundedness_check": True}))
    assert _data(result, "groundedness") == {"error": "unparseable", "unsupported": []}
    assert result.unsupported == []
    assert result.answer == ANSWER


# --- streaming -----------------------------------------------------------------------------


class GatedLLM(LLM):
    """Streams one token, then waits until the consumer has received it."""

    def __init__(self) -> None:
        super().__init__([])
        self.received = threading.Event()
        self.streamed_live = False

    def stream(self, messages: Sequence[Message]) -> Iterator[str]:
        yield "Payment "
        self.streamed_live = self.received.wait(timeout=5)
        yield "is due [1]."


def test_tokens_stream_before_the_node_finishes(store: tuple[ChunkIndex, Registry]) -> None:
    index, registry = store
    llm = GatedLLM()
    events: list[str | AgentResult] = []
    for event in stream_agent(
        QUESTION,
        _cfg(),
        llm=llm,
        embedder=FakeEmbedder(),
        index=index,
        registry=registry,
        reranker=FakeReranker(),
    ):
        events.append(event)
        if isinstance(event, str):
            llm.received.set()
    assert llm.streamed_live
    assert events[:2] == ["Payment ", "is due [1]."]
    assert isinstance(events[-1], AgentResult)
    assert events[-1].answer == "Payment is due [1]."
    assert sum(isinstance(e, AgentResult) for e in events) == 1


def test_not_found_is_yielded_as_text(store: tuple[ChunkIndex, Registry]) -> None:
    embedder = AliasEmbedder({QUESTION: GOV})
    cfg = _loop_cfg(rewrite=False, smalltalk=False)
    events = _stream(store, LLM([]), cfg, embedder=embedder)
    assert events[0] == NOT_FOUND
    assert isinstance(events[-1], AgentResult)


def test_run_agent_matches_stream(store: tuple[ChunkIndex, Registry]) -> None:
    index, registry = store
    cfg = _cfg(agent={"query_variants": 1})
    responses: list[str | dict[str, Any]] = [_analysis(variants=["payment deadline"]), ANSWER]
    streamed = _run(store, LLM(responses), cfg)
    ran = run_agent(
        QUESTION,
        cfg,
        llm=LLM(responses),
        embedder=FakeEmbedder(),
        index=index,
        registry=registry,
        reranker=FakeReranker(),
    )
    assert ran.answer == streamed.answer
    assert ran.sources == streamed.sources
    assert ran.queries == streamed.queries
    assert [(e.node, e.data) for e in ran.trace] == [(e.node, e.data) for e in streamed.trace]
