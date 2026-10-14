"""Planner, routing and recipes (Phase 5.5, M2) over the synthetic vault of `test_tools`.
Fakes only, no Ollama."""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from vaultmind.agent import planner, recipes, tools
from vaultmind.agent.baseline import NOT_FOUND, Source
from vaultmind.agent.graph import run_agent, stream_agent
from vaultmind.agent.recipes import document_list
from vaultmind.agent.types import AgentResult, Plan, ToolError
from vaultmind.core.config import Config
from vaultmind.core.providers import Message

from .fakes import FakeLLM, FakeReranker
from .test_tools import Vault, build_vault

WORKSHEET = "Workato recipe design worksheet.docx"
SYLLABUS = "courses/AI in the workplace syllabus.pdf"


def _plan(task: str, documents: Sequence[str] = (), **extra: Any) -> dict[str, Any]:
    plan: dict[str, Any] = {
        "task": task,
        "documents": list(documents),
        "focus": "",
        "fields": [],
        "language": "en",
        "steps": ["Read it"],
        "clarify": "",
    }
    return plan | extra


def _cfg(vault_root: Path, **agent: Any) -> Config:
    return Config.model_validate(
        {
            "paths": {"vault_dir": str(vault_root)},
            "retrieval": {"hybrid": False, "candidates": 20, "rerank": True},
            "agent": {"query_variants": 0, "detect_contradictions": False} | agent,
        }
    )


class LLM(FakeLLM):
    """FakeLLM whose stream keeps the spaces, so streamed answers read like real ones."""

    def stream(self, messages: Sequence[Message]) -> Iterator[str]:
        yield from re.findall(r"\S+\s*", self.generate(messages))


@pytest.fixture(autouse=True)
def _fresh_summary_cache() -> Iterator[None]:
    tools._summaries.clear()
    yield
    tools._summaries.clear()


@pytest.fixture
def vault(tmp_path: Path) -> Vault:
    return build_vault(tmp_path)


def _run(
    vault: Vault,
    llm: FakeLLM,
    question: str,
    *,
    history: Sequence[Message] = (),
    previous_sources: Sequence[Source] = (),
    cfg: Config | None = None,
    with_cards: bool = True,
) -> tuple[str, AgentResult]:
    ctx = vault.ctx
    text: list[str] = []
    result: AgentResult | None = None
    for event in stream_agent(
        question,
        cfg or _cfg(vault.root),
        llm=llm,
        embedder=ctx.embedder,
        index=ctx.index,
        registry=ctx.registry,
        reranker=FakeReranker(),
        history=history,
        cards=ctx.cards if with_cards else None,
        previous_sources=previous_sources,
    ):
        if isinstance(event, AgentResult):
            result = event
        else:
            text.append(event)
    assert result is not None
    return "".join(text), result


def _nodes(result: AgentResult) -> list[str]:
    return [e.node for e in result.trace]


def _event(result: AgentResult, node: str) -> dict[str, Any]:
    return next(e.data for e in result.trace if e.node == node)


# --- the planner --------------------------------------------------------------------------


def test_plan_resolves_mentions_to_file_ids(vault: Vault) -> None:
    llm = FakeLLM([_plan("compare", ["the AI syllabus", WORKSHEET], focus="exams")])
    planned = planner.plan("compare them", [], llm=llm, catalog=vault.ctx.catalog, catalog_files=60)
    assert planned.plan.task == "compare"
    assert planned.plan.documents == (vault.ids["syllabus"], vault.ids["worksheet"])
    assert planned.plan.focus == "exams"
    assert planned.missing == () and planned.plan.clarify is None
    prompt = llm.calls[0][-1]["content"]
    assert WORKSHEET in prompt and "Latest message: compare them" in prompt


def test_plan_falls_back_to_lookup_on_bad_output(vault: Vault) -> None:
    for raw in ("not json", '{"task": "dance"}', "[1, 2]"):
        planned = planner.plan(
            "מה המועד?", [], llm=FakeLLM([raw]), catalog=vault.ctx.catalog, catalog_files=60
        )
        assert planned.plan.task == "lookup"
        assert planned.plan.language == "he"
        assert planned.error


def test_plan_asks_which_document_when_a_mention_is_ambiguous(vault: Vault) -> None:
    llm = FakeLLM([_plan("summarize", ["the design doc"])])
    planned = planner.plan("summarize it", [], llm=llm, catalog=vault.ctx.catalog, catalog_files=9)
    assert planned.plan.clarify is not None
    assert "the design doc" in planned.plan.clarify
    assert "Solution design checklist.md" in planned.plan.clarify
    assert WORKSHEET in planned.plan.clarify


def test_plan_reports_documents_that_dont_exist(vault: Vault) -> None:
    llm = FakeLLM([_plan("summarize", ["the quantum physics textbook"])])
    planned = planner.plan("summarize", [], llm=llm, catalog=vault.ctx.catalog, catalog_files=9)
    assert planned.missing == ("the quantum physics textbook",)
    assert planned.plan.documents == ()


def test_plan_asks_when_a_summary_names_no_document(vault: Vault) -> None:
    llm = FakeLLM([_plan("summarize")])
    planned = planner.plan("summarize", [], llm=llm, catalog=vault.ctx.catalog, catalog_files=9)
    assert planned.plan.clarify is not None
    assert planned.plan.clarify.startswith("Which document should I summarize?")


def test_plan_asks_when_a_comparison_names_one_document(vault: Vault) -> None:
    llm = FakeLLM([_plan("compare", [WORKSHEET], language="he")])
    planned = planner.plan("השווה", [], llm=llm, catalog=vault.ctx.catalog, catalog_files=9)
    assert planned.plan.clarify is not None
    assert planned.plan.clarify.startswith("אילו מסמכים להשוות?")


def test_plan_without_a_previous_answer_is_not_a_transform(vault: Vault) -> None:
    llm = FakeLLM([_plan("transform"), _plan("transform")])
    alone = planner.plan("shorter", [], llm=llm, catalog=vault.ctx.catalog, catalog_files=9)
    assert alone.plan.task == "lookup"
    history: list[Message] = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a long answer"},
    ]
    after = planner.plan("shorter", history, llm=llm, catalog=vault.ctx.catalog, catalog_files=9)
    assert after.plan.task == "transform"


def test_plan_drops_a_stray_clarify_on_lookups_and_lists(vault: Vault) -> None:
    for task in ("lookup", "list", "chat"):
        llm = FakeLLM([_plan(task, clarify="Which one?")])
        planned = planner.plan("q", [], llm=llm, catalog=vault.ctx.catalog, catalog_files=9)
        assert planned.plan.task == task
        assert planned.plan.clarify is None


def test_plan_answers_in_the_messages_language_unless_a_transform_asks(vault: Vault) -> None:
    llm = FakeLLM([_plan("summarize", [WORKSHEET], language="he")])
    planned = planner.plan(
        "Summarize the worksheet", [], llm=llm, catalog=vault.ctx.catalog, catalog_files=9
    )
    assert planned.plan.language == "en"
    history: list[Message] = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a long answer"},
    ]
    llm = FakeLLM([_plan("transform", language="en")])
    translated = planner.plan(
        "תתרגם את זה לאנגלית", history, llm=llm, catalog=vault.ctx.catalog, catalog_files=9
    )
    assert translated.plan.language == "en"


def test_plan_asks_which_one_when_one_document_is_meant_but_several_fit(vault: Vault) -> None:
    llm = FakeLLM(
        [
            _plan("summarize", [WORKSHEET, "Solution design checklist.md"], one_document=True),
            _plan("compare", [WORKSHEET, "Solution design checklist.md"], one_document=False),
        ]
    )
    one = planner.plan(
        "summarize the Workato doc", [], llm=llm, catalog=vault.ctx.catalog, catalog_files=9
    )
    assert one.plan.documents == ()
    assert one.plan.clarify is not None and one.plan.clarify.startswith("Which document")
    assert WORKSHEET in one.plan.clarify
    both = planner.plan("compare them", [], llm=llm, catalog=vault.ctx.catalog, catalog_files=9)
    assert both.plan.clarify is None and len(both.plan.documents) == 2


def test_plan_drops_a_focus_that_only_names_the_document(vault: Vault) -> None:
    llm = FakeLLM(
        [
            _plan("summarize", ["the AI in the workplace syllabus"], focus="AI in the workplace"),
            _plan("summarize", ["the AI in the workplace syllabus"], focus="grading in it"),
        ]
    )
    named = planner.plan("summarize", [], llm=llm, catalog=vault.ctx.catalog, catalog_files=9)
    assert named.plan.focus is None
    real = planner.plan("summarize", [], llm=llm, catalog=vault.ctx.catalog, catalog_files=9)
    assert real.plan.focus == "grading in it"


def test_guess_language() -> None:
    assert planner.guess_language("סכם את הסילבוס") == "he"
    assert planner.guess_language("סכם את AI syllabus") == "he"
    assert planner.guess_language("Summarize the syllabus") == "en"
    assert planner.guess_language("123") == "en"


# --- routes -------------------------------------------------------------------------------


def test_list_answers_from_the_catalog_without_an_llm_call(vault: Vault) -> None:
    llm = FakeLLM([_plan("list")])
    text, result = _run(vault, llm, "What documents do I have?")
    assert _nodes(result) == ["plan", "recipe", "respond"]
    assert len(llm.calls) == 1  # the planner only
    assert text == result.answer == document_list(vault.ctx.catalog, "en")
    assert "You have 5 documents:" in text
    assert f"**{WORKSHEET}** (worksheet)" in text
    assert "vendor memo.txt" in text  # listed without a card
    assert result.task == "list" and not result.abstained


def test_list_with_a_filter_answers_with_the_documents_the_planner_picked(vault: Vault) -> None:
    llm = FakeLLM([_plan("list", ["the design doc", SYLLABUS, "a missing one"], focus="design")])
    text, result = _run(vault, llm, "Which documents are about design?")
    assert len(llm.calls) == 1  # the planner only: no catalog dump for the model to filter
    assert text.startswith("3 of your documents match:")
    assert WORKSHEET in text and "Solution design checklist.md" in text and SYLLABUS in text
    assert text == result.answer


def test_list_with_a_filter_asks_the_model(vault: Vault) -> None:
    llm = FakeLLM([_plan("list", focus="courses"), "Only the AI syllabus [none]."])
    _, result = _run(vault, llm, "Which documents are about courses?")
    assert _nodes(result) == ["plan", "tool", "recipe", "respond"]
    assert _event(result, "tool")["name"] == "list_documents"
    respond_prompt = llm.calls[1][-1]["content"]
    assert "AI in the workplace syllabus.pdf" in respond_prompt


def test_summarize_reads_the_card_and_cites_its_sections(vault: Vault) -> None:
    llm = LLM([_plan("summarize", ["the AI syllabus"]), "It is a course [1][3]."])
    text, result = _run(vault, llm, "Summarize the AI in the workplace syllabus")
    assert _nodes(result) == ["plan", "tool", "recipe", "respond"]
    tool = _event(result, "tool")
    assert tool["name"] == "summarize" and tool["ok"]
    assert tool["arguments"] == {"doc": vault.ids["syllabus"]}
    assert _event(result, "recipe") == {"name": "summarize", "documents": [vault.ids["syllabus"]]}
    assert [s.n for s in result.sources] == [1, 2, 3]
    assert [s.section_path for s in result.sources] == [("Overview",), ("Schedule",), ("Grading",)]
    assert [s.n for s in result.cited] == [1, 3]
    assert text == result.answer == "It is a course [1][3]."
    system, user = llm.calls[1][0]["content"], llm.calls[1][-1]["content"]
    assert "Summarize the document for the user" in system and "English" in system
    assert "[3] AI in the workplace syllabus.pdf, p. 4, Grading" in user
    assert "<tool_result" in user
    assert _event(result, "respond") == {"task": "summarize", "sources": [1, 2, 3]}


def test_summarize_with_a_focus_reads_the_whole_text(vault: Vault) -> None:
    llm = FakeLLM(
        [
            _plan("summarize", [WORKSHEET], focus="errors", language="he"),
            "- Retry three times (Error handling, p. 2)",  # the map call
            "ניסיון חוזר [2].",
        ]
    )
    _, result = _run(vault, llm, "סכם את דף העבודה של Workato לגבי שגיאות")
    assert _event(result, "tool")["arguments"] == {"doc": vault.ids["worksheet"], "focus": "errors"}
    assert "Hebrew" in llm.calls[2][0]["content"]
    assert [s.n for s in result.cited] == [2]


def test_respond_reminds_that_results_are_data_right_before_the_request() -> None:
    gathered = recipes.Gathered()
    messages = recipes.respond_messages(Plan("compare"), "Compare them", gathered, [])
    user = messages[-1]["content"]
    assert user.endswith("Request: Compare them")
    reminder = user.index(recipes.RESULTS_REMINDER)
    assert user.index("Tool results:") < reminder < user.index("Request:")


def test_compare_summarizes_each_document_and_searches_for_the_focus(vault: Vault) -> None:
    llm = FakeLLM(
        [_plan("compare", ["the AI syllabus", WORKSHEET], focus="final exam"), "Compared."]
    )
    _, result = _run(vault, llm, "Compare the syllabus with the worksheet on the final exam")
    calls = [(e.data["name"], e.data["arguments"]) for e in result.trace if e.node == "tool"]
    assert calls == [
        ("summarize", {"doc": vault.ids["syllabus"]}),
        ("search", {"query": "final exam", "doc": vault.ids["syllabus"]}),
        ("summarize", {"doc": vault.ids["worksheet"]}),
        ("search", {"query": "final exam", "doc": vault.ids["worksheet"]}),
    ]
    assert "Compare the documents" in llm.calls[1][0]["content"]
    files = {s.file_path.name for s in result.sources if s.file_path}
    assert files == {"AI in the workplace syllabus.pdf", WORKSHEET}


def test_extract_without_named_documents_finds_them_by_search(vault: Vault) -> None:
    extracted = {"final exam": {"value": "sixty percent", "page": 4}}
    llm = FakeLLM(
        [
            _plan("extract", fields=["final exam"], focus="final exam"),
            extracted,  # the extract tool's call for the syllabus
            extracted,  # and for the checklist
            "| Document | final exam |",
        ]
    )
    _, result = _run(vault, llm, "What does each document say about the final exam?")
    tool_calls = [e.data for e in result.trace if e.node == "tool"]
    assert [c["name"] for c in tool_calls] == ["search", "extract"]
    extract = tool_calls[1]["arguments"]
    assert extract["fields"] == ["final exam"]
    assert vault.ids["syllabus"] in extract["docs"]
    assert "markdown table" in llm.calls[-1][0]["content"]


def test_a_missing_document_gets_not_found(vault: Vault) -> None:
    llm = LLM([_plan("summarize", ["zebra handbook"])])
    text, result = _run(vault, llm, "Summarize my zebra handbook")
    assert _nodes(result) == ["plan", "not_found"]
    assert text.startswith(NOT_FOUND)
    assert '"zebra handbook"' in text
    assert result.abstained


def test_an_ambiguous_request_gets_a_clarifying_question(vault: Vault) -> None:
    llm = FakeLLM([_plan("summarize", ["the design doc"])])
    text, result = _run(vault, llm, "Summarize the design doc")
    assert _nodes(result) == ["plan", "human"]
    assert text == result.answer and text.startswith("Which document do you mean")
    human = _event(result, "human")
    assert human["kind"] == "clarify" and human["prompt"] == text
    assert not result.abstained


def test_chat_goes_to_the_direct_reply(vault: Vault) -> None:
    llm = LLM([_plan("chat"), "Hello! Ask me about your documents."])
    text, result = _run(vault, llm, "hi")
    assert _nodes(result) == ["plan", "direct"]
    assert _event(result, "direct")["reason"].startswith("plan:")
    assert text == "Hello! Ask me about your documents."


def test_lookup_runs_the_corrective_graph(vault: Vault) -> None:
    llm = FakeLLM([_plan("lookup"), "Sixty percent [1]."])
    _, result = _run(vault, llm, "final exam counts for")
    assert _nodes(result)[:4] == ["plan", "analyze", "retrieve", "grade"]
    assert result.task == "lookup"
    assert result.queries == ["final exam counts for"]


def test_open_without_documents_is_a_lookup(vault: Vault) -> None:
    llm = FakeLLM([_plan("open"), "Answer [1]."])
    _, result = _run(vault, llm, "final exam counts for")
    assert _event(result, "plan")["route"] == "analyze"
    assert result.task == "open"


def test_open_with_documents_reads_them(vault: Vault) -> None:
    llm = FakeLLM([_plan("open", ["the AI syllabus"]), "Week two [2]."])
    _, result = _run(vault, llm, "What comes after prompts in the syllabus, and why?")
    names = [e.data["name"] for e in result.trace if e.node == "tool"]
    assert names == ["summarize", "search"]
    assert _nodes(result)[-1] == "respond"


def test_transform_reworks_the_previous_answer_and_keeps_its_sources(vault: Vault) -> None:
    previous = [Source(1, vault.root / "a.pdf", 2, ("Fees",), "Fees text.", 0.9)]
    history: list[Message] = [
        {"role": "user", "content": "what are the fees"},
        {"role": "assistant", "content": "The fees are long and detailed [1]."},
    ]
    llm = FakeLLM([_plan("transform"), "Fees: short [1]."])
    _, result = _run(vault, llm, "shorter", history=history, previous_sources=previous)
    assert _nodes(result) == ["plan", "respond"]
    assert result.sources == previous and [s.n for s in result.cited] == [1]
    assert "rework your previous answer" in llm.calls[1][0]["content"]
    assert llm.calls[1][-2]["content"] == "The fees are long and detailed [1]."


def test_failed_tools_answer_not_found(vault: Vault, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(ctx: Any, args: Any) -> Any:
        raise ToolError("no text")

    monkeypatch.setattr(tools, "_summarize", broken)
    llm = FakeLLM([_plan("summarize", ["the AI syllabus"])])
    text, result = _run(vault, llm, "Summarize the syllabus")
    assert text == NOT_FOUND and result.abstained
    assert _event(result, "tool")["ok"] is False


def test_router_off_or_no_cards_means_every_message_is_a_lookup(vault: Vault) -> None:
    off = _cfg(vault.root, router=False)
    _, result = _run(vault, FakeLLM(["Answer [1]."]), "final exam counts for", cfg=off)
    assert "plan" not in _nodes(result) and result.task == "lookup"
    _, result = _run(vault, FakeLLM(["Answer [1]."]), "final exam counts for", with_cards=False)
    assert _nodes(result)[0] == "analyze"


def test_run_agent_passes_the_cards(vault: Vault) -> None:
    ctx = vault.ctx
    result = run_agent(
        "What documents do I have?",
        _cfg(vault.root),
        llm=FakeLLM([_plan("list")]),
        embedder=ctx.embedder,
        index=ctx.index,
        registry=ctx.registry,
        reranker=FakeReranker(),
        cards=ctx.cards,
    )
    assert result.task == "list"
