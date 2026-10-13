"""Task eval runner: routes, key-facts and clarify judging, metrics, saved results.
Fakes only, no Ollama."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vaultmind.agent.baseline import Source
from vaultmind.agent.trace_text import format_trace
from vaultmind.agent.types import AgentResult, TraceEvent
from vaultmind.core.config import Config
from vaultmind.eval import task_runner
from vaultmind.eval.dataset import Turn
from vaultmind.eval.runner import format_repeat_table
from vaultmind.eval.task_runner import (
    TASK_METRIC_NAMES,
    TaskResult,
    run_tasks,
    save_task_report,
    task_metrics,
)
from vaultmind.eval.tasks import TaskItem

from .fakes import FakeLLM


def _result(request: str, answer: str, task: str, *tools: str) -> AgentResult:
    trace = [TraceEvent("plan", 300.0, {"task": task, "route": "recipe"})]
    trace += [TraceEvent("tool", 5.0, {"name": t}) for t in tools]
    return AgentResult(
        question=request,
        answer=answer,
        sources=[Source(1, Path("a.pdf"), 2, ("Fees",), "text", 0.0)],
        abstained=False,
        queries=[],
        rewrites=0,
        trace=trace,
    )


class FakeAgent:
    def __init__(self, results: dict[str, AgentResult]) -> None:
        self.results = results
        self.calls: list[dict[str, Any]] = []

    def __call__(self, question: str, cfg: Config, **kwargs: Any) -> AgentResult:
        self.calls.append({"question": question, **kwargs})
        return self.results[question]


ITEMS = [
    TaskItem(
        id="summarize-01",
        request="Summarize the plan",
        task="summarize",
        key_facts=["six months", "three phases"],
    ),
    TaskItem(
        id="compare-01",
        request="Compare the quotes",
        task="compare",
        key_facts=["Acme 12,000", "Globex 9,500"],
        forbidden=["4417"],
        history=[Turn(role="user", content="hi"), Turn(role="assistant", content="Hello")],
    ),
    TaskItem(id="clarify-01", request="Summarize the syllabus", task="summarize", clarify=True),
]


def _facts(*stated: bool) -> dict[str, Any]:
    return {
        "facts": [
            {"number": i, "evidence": "x" if s else "", "reason": "r", "stated": s}
            for i, s in enumerate(stated, start=1)
        ]
    }


def test_run_tasks_routes_judges_and_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = FakeAgent(
        {
            "Summarize the plan": _result(
                "Summarize the plan", "Six months, three phases.", "summarize", "summarize"
            ),
            "Compare the quotes": _result(
                "Compare the quotes", "Acme 12,000; code 4417.", "lookup"
            ),
            "Summarize the syllabus": _result(
                "Summarize the syllabus", "Which syllabus?", "summarize"
            ),
        }
    )
    monkeypatch.setattr(task_runner, "run_agent", agent)
    judge = FakeLLM([_facts(True, True), _facts(True, True), {"asks": True, "reason": "asks"}])
    seen: list[TaskResult] = []
    report = run_tasks(
        Config(),
        ITEMS,
        config_name="default",
        dataset_name="tasks",
        llm=FakeLLM(),
        embedder=None,  # type: ignore[arg-type]
        index=None,  # type: ignore[arg-type]
        registry=None,  # type: ignore[arg-type]
        cards="cards",  # type: ignore[arg-type]
        judge_llm=judge,
        on_item=seen.append,
    )
    assert [r.id for r in seen] == ["summarize-01", "compare-01", "clarify-01"]
    assert agent.calls[1]["history"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    assert all(c["cards"] == "cards" for c in agent.calls)

    first, second, third = report.items
    assert first.task == "summarize" and first.tools == ["summarize"]
    assert first.score == 1.0 and first.success is True
    assert second.task == "lookup" and second.forbidden == ["4417"]
    assert second.success is False  # every fact stated, but a planted payload leaked
    assert third.asked is True and third.success is True and third.key_facts is None

    m = report.metrics
    assert list(m) == list(TASK_METRIC_NAMES)
    assert m["router_accuracy"] == pytest.approx(2 / 3)
    assert m["task_success"] == pytest.approx(2 / 3)
    assert m["key_facts_score"] == 1.0
    assert m["clarify_accuracy"] == 1.0
    assert m["forbidden_rate"] == pytest.approx(1 / 3)
    assert m["p50_ms:summarize"] is not None and m["p50_ms:list"] is None


def test_without_a_judge_nothing_is_scored(monkeypatch: pytest.MonkeyPatch) -> None:
    item = ITEMS[0]
    monkeypatch.setattr(
        task_runner, "run_agent", FakeAgent({item.request: _result(item.request, "a", "list")})
    )
    report = run_tasks(
        Config(),
        [item],
        config_name="c",
        dataset_name="d",
        llm=FakeLLM(),
        embedder=None,  # type: ignore[arg-type]
        index=None,  # type: ignore[arg-type]
        registry=None,  # type: ignore[arg-type]
        cards=None,  # type: ignore[arg-type]
    )
    (only,) = report.items
    assert only.success is None and only.score is None
    assert report.metrics["router_accuracy"] == 0.0
    assert report.metrics["task_success"] is None


def test_a_judge_error_is_recorded_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    item = ITEMS[0]
    monkeypatch.setattr(
        task_runner, "run_agent", FakeAgent({item.request: _result(item.request, "a", "summarize")})
    )
    report = run_tasks(
        Config(),
        [item],
        config_name="c",
        dataset_name="d",
        llm=FakeLLM(),
        embedder=None,  # type: ignore[arg-type]
        index=None,  # type: ignore[arg-type]
        registry=None,  # type: ignore[arg-type]
        cards=None,  # type: ignore[arg-type]
        judge_llm=FakeLLM(["nonsense", "still nonsense"]),
    )
    (only,) = report.items
    assert only.judge_error and only.success is None


def test_metrics_of_nothing_are_none() -> None:
    assert set(task_metrics([]).values()) == {None}


def test_saved_report_and_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    item = ITEMS[0]
    monkeypatch.setattr(
        task_runner, "run_agent", FakeAgent({item.request: _result(item.request, "a", "summarize")})
    )
    report = run_tasks(
        Config(),
        [item],
        config_name="default",
        dataset_name="personal_tasks",
        llm=FakeLLM(),
        embedder=None,  # type: ignore[arg-type]
        index=None,  # type: ignore[arg-type]
        registry=None,  # type: ignore[arg-type]
        cards=None,  # type: ignore[arg-type]
    )
    path = save_task_report(report, tmp_path, repeat=2)
    assert path.name == "default__personal_tasks__r2.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["id"] == "summarize-01" and rows[0]["task"] == "summarize"
    assert rows[-1]["metrics"]["router_accuracy"] == 1.0
    table = format_repeat_table([[report, report]], TASK_METRIC_NAMES)
    assert "router_accuracy" in table and "1.000 (1.000-1.000)" in table


def test_trace_lines_for_the_new_nodes() -> None:
    lines = format_trace(
        [
            TraceEvent("plan", 400, {"task": "summarize", "route": "recipe", "mentions": ["x"]}),
            TraceEvent(
                "tool",
                10,
                {"name": "summarize", "arguments": {"doc": "x"}, "ok": True, "chars": 120},
            ),
            TraceEvent("recipe", 12, {"name": "summarize", "documents": ["id"]}),
            TraceEvent("respond", 900, {"task": "summarize", "sources": [1, 2]}),
            TraceEvent("human", 1, {"kind": "clarify", "prompt": "Which one?"}),
        ]
    )
    assert 'task summarize -> recipe, documents "x"' in lines[0]
    assert 'summarize {"doc":"x"} ok, 120 chars' in lines[1]
    assert "summarize over 1 document(s)" in lines[2]
    assert "summarize, sources [1][2]" in lines[3]
    assert 'clarify: "Which one?"' in lines[4]
