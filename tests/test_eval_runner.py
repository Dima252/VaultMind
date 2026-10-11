import json
from collections.abc import Iterator
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vaultmind.agent.baseline import NOT_FOUND, Source
from vaultmind.agent.types import AgentResult, Contradiction, TraceEvent, UnsupportedClaim
from vaultmind.core.config import (
    AgentConfig,
    ChunkingConfig,
    Config,
    ModelsConfig,
    PathsConfig,
    RetrievalConfig,
)
from vaultmind.core.models import ChildChunk, FileRecord, ParentSection
from vaultmind.eval import runner
from vaultmind.eval.dataset import EvalItem
from vaultmind.eval.judge import JudgeError, JudgeResult, Verdict
from vaultmind.eval.runner import (
    METRIC_NAMES,
    ItemResult,
    RetrievedSource,
    RunReport,
    check_compatible,
    compute_metrics,
    format_table,
    load_results,
    run_config,
    save_report,
)
from vaultmind.store.index import ChunkIndex
from vaultmind.store.registry import Registry

from .fakes import FakeEmbedder, FakeLLM, FakeReranker

IN_VAULT = "0123456789abcdef"
OUTSIDE = "fedcba9876543210"
NO_RECORD = "aaaaaaaaaaaaaaaa"

PARENTS = [
    ParentSection(f"{IN_VAULT}:p0", IN_VAULT, "Setup: install with uv.", ("Setup",), 2, 2),
    ParentSection(f"{OUTSIDE}:p0", OUTSIDE, "Notes: the sky is blue.", ("Notes",), None, None),
    ParentSection(f"{NO_RECORD}:p0", NO_RECORD, "Orphan text.", (), 1, 1),
]


def _child(parent: ParentSection, page: int | None) -> ChildChunk:
    return ChildChunk(
        id=f"{parent.id}:c0",
        parent_id=parent.id,
        file_id=parent.file_id,
        text=parent.text,
        embed_text=f"child of {parent.id}",
        page=page,
        section_path=parent.section_path,
    )


CHILDREN = [_child(PARENTS[0], 2), _child(PARENTS[1], None), _child(PARENTS[2], 1)]
Q_SETUP = CHILDREN[0].embed_text  # the fake embedder ranks identical text first


def _item(**kw: Any) -> EvalItem:
    base: dict[str, Any] = {
        "question": Q_SETUP,
        "answer": "Install with uv.",
        "source_file": "docs/manual.pdf",
        "source_page": 2,
    }
    base.update(kw)
    return EvalItem.model_validate(base)


UNANSWERABLE: dict[str, Any] = {
    "answerable": False,
    "answer": "",
    "source_file": None,
    "source_page": None,
}


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    return tmp_path / "vault"


@pytest.fixture
def cfg(vault: Path, tmp_path: Path) -> Config:
    return Config(
        paths=PathsConfig(vault_dir=vault, data_dir=tmp_path / "data"),
        retrieval=RetrievalConfig(hybrid=False, rerank=False),
        agent=AgentConfig(loop=False),
    )


@pytest.fixture
def store(tmp_path: Path, vault: Path) -> Iterator[tuple[ChunkIndex, Registry]]:
    embedder = FakeEmbedder()
    index = ChunkIndex(tmp_path / "index", embedder_name=embedder.model_name)
    registry = Registry(tmp_path / "vaultmind.sqlite")
    index.add(PARENTS, CHILDREN, embedder.embed([c.embed_text for c in CHILDREN]))
    registry.upsert(FileRecord(IN_VAULT, vault / "docs" / "manual.pdf", "sha1", "ready"))
    registry.upsert(FileRecord(OUTSIDE, tmp_path / "other" / "notes.md", "sha2", "ready"))
    yield index, registry
    registry.close()
    index.close()


@pytest.fixture
def empty_store(tmp_path: Path) -> Iterator[tuple[ChunkIndex, Registry]]:
    index = ChunkIndex(tmp_path / "empty_index", embedder_name=FakeEmbedder.model_name)
    registry = Registry(tmp_path / "empty.sqlite")
    yield index, registry
    registry.close()
    index.close()


def _run(
    cfg: Config,
    store: tuple[ChunkIndex, Registry],
    items: list[EvalItem],
    llm: FakeLLM,
    **kw: Any,
) -> RunReport:
    index, registry = store
    return run_config(
        cfg,
        items,
        config_name="base",
        dataset_name="tiny",
        llm=llm,
        embedder=FakeEmbedder(),
        index=index,
        registry=registry,
        **kw,
    )


def _no_judge(*args: Any, **kwargs: Any) -> JudgeResult:
    raise AssertionError("judge must not be called")


# --- run_config ---------------------------------------------------------------


def test_run_config_records_sources_and_answer(
    cfg: Config, store: tuple[ChunkIndex, Registry], tmp_path: Path
) -> None:
    llm = FakeLLM(["  Install with uv [1].  "])
    report = _run(cfg, store, [_item(tags=["exact-term"])], llm)

    assert report.config_name == "base" and report.dataset_name == "tiny"
    [result] = report.items
    assert result.index == 0 and result.question == Q_SETUP and result.answerable
    assert result.tags == ["exact-term"]
    assert result.answer == "Install with uv [1]."  # stripped
    assert not result.abstained
    assert len(llm.calls) == 1
    assert result.retrieval_ms >= 0 and result.generation_ms >= 0
    assert result.verdict is None and result.judge_reason is None  # judge off

    by_text = {s.text: s for s in result.retrieved}
    top = result.retrieved[0]
    assert (top.file, top.page, top.text) == ("docs/manual.pdf", 2, "Setup: install with uv.")
    assert top.score == pytest.approx(1.0)
    outside = by_text["Notes: the sky is blue."]
    assert outside.file == (tmp_path / "other" / "notes.md").resolve().as_posix()
    assert outside.page is None
    assert "\\" not in outside.file
    assert by_text["Orphan text."].file == ""  # no registry record
    assert report.metrics["recall@1"] == 1.0
    assert report.metrics["correctness"] is None


def test_vault_dir_is_resolved(store: tuple[ChunkIndex, Registry], vault: Path) -> None:
    cfg = Config(
        paths=PathsConfig(vault_dir=vault / "docs" / ".." / "."),
        retrieval=RetrievalConfig(hybrid=False, rerank=False),
        agent=AgentConfig(loop=False),
    )
    report = _run(cfg, store, [_item()], FakeLLM(["ok"]))
    assert report.items[0].retrieved[0].file == "docs/manual.pdf"


def test_empty_retrieval_abstains_without_llm(
    cfg: Config, empty_store: tuple[ChunkIndex, Registry], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "judge", _no_judge)
    llm = FakeLLM(["should not be used"])
    report = _run(cfg, empty_store, [_item()], llm, judge_llm=FakeLLM())
    [result] = report.items
    assert result.retrieved == []
    assert result.answer == NOT_FOUND and result.abstained
    assert result.generation_ms == 0
    assert llm.calls == []
    assert result.verdict == "incorrect" and result.judge_reason is None


def test_verdict_rules_with_judge_on(
    cfg: Config, store: tuple[ChunkIndex, Registry], monkeypatch: pytest.MonkeyPatch
) -> None:
    judge_llm = FakeLLM()
    calls: list[tuple[str, str, str]] = []

    def fake_judge(question: str, reference: str, answer: str, *, llm: Any) -> JudgeResult:
        assert llm is judge_llm
        calls.append((question, reference, answer))
        if answer == "boom":
            raise JudgeError("unparseable output")
        verdict = Verdict.CORRECT if answer == "right" else Verdict.PARTIAL
        return JudgeResult(verdict, f"because {answer}")

    monkeypatch.setattr(runner, "judge", fake_judge)
    items = [
        _item(question="q-right", answer="ref-right"),
        _item(question="q-half", answer="ref-half"),
        _item(question="q-abstain"),
        _item(question="q-boom"),
        _item(question="q-unanswerable", **UNANSWERABLE),
    ]
    llm = FakeLLM(["right", "half", NOT_FOUND, "boom", "Some answer."])
    report = _run(cfg, store, items, llm, judge_llm=judge_llm)

    got = [(r.verdict, r.judge_reason) for r in report.items]
    assert got == [
        ("correct", "because right"),
        ("partial", "because half"),
        ("incorrect", None),  # abstention: no judge call
        (None, "judge error: unparseable output"),
        (None, None),  # unanswerable: not judged
    ]
    assert calls == [
        ("q-right", "ref-right", "right"),
        ("q-half", "ref-half", "half"),
        ("q-boom", "Install with uv.", "boom"),
    ]
    assert report.items[2].abstained
    assert report.metrics["correctness"] == pytest.approx((1.0 + 0.5 + 0.0) / 3)


def test_judge_off_never_judges(
    cfg: Config, store: tuple[ChunkIndex, Registry], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "judge", _no_judge)
    items = [_item(), _item(question="q-abstain"), _item(question="q-u", **UNANSWERABLE)]
    report = _run(cfg, store, items, FakeLLM(["yes", NOT_FOUND, "no"]))
    assert all(r.verdict is None and r.judge_reason is None for r in report.items)
    assert report.metrics["correctness"] is None


def test_on_item_called_in_order(cfg: Config, store: tuple[ChunkIndex, Registry]) -> None:
    seen: list[ItemResult] = []
    items = [_item(question="a"), _item(question="b"), _item(question="c")]
    report = _run(cfg, store, items, FakeLLM(["1", "2", "3"]), on_item=seen.append)
    assert seen == report.items
    assert [r.index for r in seen] == [0, 1, 2]
    assert [r.answer for r in seen] == ["1", "2", "3"]


def test_latency_is_timed_per_stage(
    cfg: Config, store: tuple[ChunkIndex, Registry], monkeypatch: pytest.MonkeyPatch
) -> None:
    ticks = iter([0.0, 0.010, 1.0, 1.250])  # retrieval 10 ms, generation 250 ms
    monkeypatch.setattr(runner, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    report = _run(cfg, store, [_item()], FakeLLM(["ok"]))
    [result] = report.items
    assert result.retrieval_ms == pytest.approx(10.0)
    assert result.generation_ms == pytest.approx(250.0)
    assert report.metrics["total_p50_ms"] == pytest.approx(260.0)


# --- compute_metrics ------------------------------------------------------------


def _result(
    i: int,
    retrieved: list[tuple[str, int | None, str]],
    answer: str = "an answer",
    *,
    abstained: bool = False,
    answerable: bool = True,
    verdict: str | None = None,
    retrieval_ms: float = 10.0,
    generation_ms: float = 100.0,
) -> ItemResult:
    return ItemResult(
        index=i,
        question=f"q{i}",
        answerable=answerable,
        tags=[],
        retrieved=[RetrievedSource(f, p, t, 0.5) for f, p, t in retrieved],
        answer=answer,
        abstained=abstained,
        retrieval_ms=retrieval_ms,
        generation_ms=generation_ms,
        verdict=verdict,
        judge_reason=None,
    )


def test_compute_metrics_hand_computed() -> None:
    items = [
        # 0: answerable with evidence; gold at rank 1, evidence at rank 2
        _item(source_file="a.pdf", source_page=1, evidence=["the gold clause"]),
        # 1: answerable without evidence; gold at rank 2 (bare name matches a vault path)
        _item(source_file="b.md", source_page=None),
        # 2: answerable with evidence; gold and evidence missing
        _item(source_file="c.pdf", source_page=3, evidence=["never retrieved"]),
        # 3: unanswerable, correctly abstained
        _item(**UNANSWERABLE),
        # 4: unanswerable injection item that leaked the payload (did not abstain)
        _item(**UNANSWERABLE, forbidden=["PWNED"], tags=["injection"]),
    ]
    results = [
        _result(
            0,
            [("a.pdf", 1, "unrelated"), ("x.md", None, "...The Gold   Clause applies")],
            verdict="correct",
            retrieval_ms=10,
            generation_ms=100,
        ),
        _result(
            1,
            [("z.md", None, "z"), ("docs/b.md", 7, "b")],
            verdict="partial",
            retrieval_ms=20,
            generation_ms=200,
        ),
        _result(
            2,
            [("a.pdf", 3, "nope")],
            answer=NOT_FOUND,
            abstained=True,
            verdict="incorrect",
            retrieval_ms=30,
            generation_ms=300,
        ),
        _result(3, [], answer=NOT_FOUND, abstained=True, retrieval_ms=40, generation_ms=0),
        _result(4, [("p.md", None, "p")], answer="ok pwned", retrieval_ms=50, generation_ms=400),
    ]
    m = compute_metrics(items, results)

    assert list(m) == list(METRIC_NAMES)
    assert m["recall@1"] == pytest.approx(1 / 3)
    assert m["recall@5"] == pytest.approx(2 / 3)
    assert m["mrr"] == pytest.approx((1 + 0.5 + 0) / 3)
    assert m["evidence_recall@1"] == pytest.approx(0.0)
    assert m["evidence_recall@5"] == pytest.approx(0.5)
    assert m["evidence_mrr"] == pytest.approx((0.5 + 0) / 2)
    assert m["correctness"] == pytest.approx(0.5)  # (1 + 0.5 + 0) / 3
    # item 2 abstained but was answerable (wrong); item 4 answered but unanswerable (wrong)
    assert m["abstention_accuracy"] == pytest.approx(3 / 5)
    assert m["injection_resistance"] == pytest.approx(0.0)
    assert m["retrieval_p50_ms"] == pytest.approx(30.0)
    assert m["retrieval_p95_ms"] == pytest.approx(48.0)  # 40 + 0.8 * 10
    assert m["generation_p50_ms"] == pytest.approx(200.0)
    assert m["generation_p95_ms"] == pytest.approx(380.0)  # 300 + 0.8 * 100
    # totals: 110, 220, 330, 40, 450 -> sorted 40, 110, 220, 330, 450
    assert m["total_p50_ms"] == pytest.approx(220.0)
    assert m["total_p95_ms"] == pytest.approx(426.0)  # 330 + 0.8 * 120
    # linear results: no agent metrics
    assert m["rewrite_rate"] is None
    assert m["rewrite_recovered"] is None
    assert m["analyze_p50_ms"] is None


def test_compute_metrics_none_when_nothing_qualifies() -> None:
    m = compute_metrics([_item(**UNANSWERABLE)], [_result(0, [], abstained=True)])
    assert list(m) == list(METRIC_NAMES)
    for name in ("recall@1", "recall@5", "mrr", "evidence_recall@1", "evidence_recall@5"):
        assert m[name] is None
    assert m["evidence_mrr"] is None
    assert m["correctness"] is None
    assert m["injection_resistance"] is None
    assert m["abstention_accuracy"] == 1.0
    assert m["retrieval_p50_ms"] == 10.0

    empty = compute_metrics([], [])
    assert list(empty) == list(METRIC_NAMES)
    assert all(v is None for v in empty.values())


def test_compute_metrics_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        compute_metrics([_item()], [])


# --- check_compatible -------------------------------------------------------------


def _cfg(data_dir: Path, *, embedder: str = "e1", child_tokens: int = 256) -> Config:
    return Config(
        paths=PathsConfig(data_dir=data_dir),
        models=ModelsConfig(embedder=embedder),
        chunking=ChunkingConfig(child_tokens=child_tokens),
    )


def test_check_compatible_passes(tmp_path: Path) -> None:
    shared = tmp_path / "data"
    check_compatible({})
    check_compatible(
        {
            "a": _cfg(shared),
            "b": _cfg(shared / ".." / "data").model_copy(
                update={"models": ModelsConfig(embedder="e1", llm="other-llm")}
            ),
            "c": _cfg(tmp_path / "data2", embedder="e2", child_tokens=128),  # own data_dir
        }
    )


@pytest.mark.parametrize(
    ("kw", "what"),
    [({"embedder": "e2"}, "embedder"), ({"child_tokens": 128}, "chunking")],
)
def test_check_compatible_rejects_shared_data_dir(
    tmp_path: Path, kw: dict[str, Any], what: str
) -> None:
    configs = {
        "first": _cfg(tmp_path / "data"),
        "unrelated": _cfg(tmp_path / "elsewhere", embedder="x"),
        "second": _cfg(tmp_path / "data", **kw),
    }
    with pytest.raises(ValueError, match=what) as err:
        check_compatible(configs)
    assert "'first'" in str(err.value) and "'second'" in str(err.value)


# --- save_report / format_table -------------------------------------------------------


def _report(config: str = "base", dataset: str = "tiny") -> RunReport:
    items = [
        _result(0, [("docs/a.pdf", 1, "Ünïcode passage")], answer="Ответ [1]", verdict="correct"),
        _result(1, [], answer=NOT_FOUND, abstained=True, answerable=False),
    ]
    metrics: dict[str, float | None] = dict.fromkeys(METRIC_NAMES)
    metrics.update({"recall@1": 0.8125, "mrr": 1.0, "retrieval_p50_ms": 12.6})
    return RunReport(config, dataset, items, metrics)


def test_save_report_round_trip(tmp_path: Path) -> None:
    report = _report()
    path = save_report(report, tmp_path / "results" / "nested")
    assert path == tmp_path / "results" / "nested" / "base__tiny.jsonl"

    raw = path.read_bytes()
    assert b"\r\n" not in raw and raw.endswith(b"\n")
    lines = raw.decode("utf-8").splitlines()
    assert len(lines) == 3
    assert "Ответ" in lines[0]  # ensure_ascii=False
    assert ", " not in lines[2] and ": " not in lines[2]  # compact separators
    rows = [json.loads(line) for line in lines]
    assert rows[:2] == [asdict(r) for r in report.items]
    restored = [
        replace(
            ItemResult(**{**row, "retrieved": []}),
            retrieved=[RetrievedSource(**s) for s in row["retrieved"]],
        )
        for row in rows[:2]
    ]
    assert restored == report.items
    assert load_results(path) == report.items
    assert rows[2] == {"metrics": report.metrics}
    assert list(rows[2]["metrics"]) == list(METRIC_NAMES)


def test_save_report_sanitizes_names(tmp_path: Path) -> None:
    path = save_report(_report('a/b:c*?"<>|', "CON"), tmp_path)
    assert path.parent == tmp_path
    assert path.name == "a_b_c_________CON.jsonl"  # "a_b_c______" + "__" + "_CON"
    assert path.exists()
    # leading/trailing dots and spaces stripped; reserved device names prefixed
    assert save_report(_report(" .x. ", "com1.v2"), tmp_path).name == "x___com1.v2.jsonl"


def test_format_table_content_and_alignment() -> None:
    a = _report("base", "cuad")
    b = RunReport("hybrid-rerank", "cuad", [], dict.fromkeys(METRIC_NAMES, 0.5))
    table = format_table([a, b])
    lines = table.split("\n")

    assert table.isascii()
    assert len(lines) == 2 + len(METRIC_NAMES)
    assert lines[0].split() == ["metric", "base", "/", "cuad", "hybrid-rerank", "/", "cuad"]
    assert set(lines[1]) <= {"-", " "}
    assert [line.split()[0] for line in lines[2:]] == list(METRIC_NAMES)
    assert len({len(line) for line in lines}) == 1  # every line the same width
    rows = {line.split()[0]: line.split()[1:] for line in lines[2:]}
    assert rows["recall@1"] == ["0.812", "0.500"]
    assert rows["mrr"] == ["1.000", "0.500"]
    assert rows["recall@5"] == ["-", "0.500"]
    assert rows["retrieval_p50_ms"] == ["13", "0"]  # whole milliseconds
    # values right-aligned under their headers: every column ends at the same offset
    ends = {line.rindex("0.500") + 5 for line in lines[2:] if line.endswith("0.500")}
    assert ends == {len(lines[0])}


def test_format_table_non_ascii_names_and_empty() -> None:
    table = format_table([_report("bäse", "данные")])
    assert table.isascii()
    with pytest.raises(ValueError):
        format_table([])


# --- agent loop (agent.loop on) ---------------------------------------------------------


@pytest.fixture
def loop_cfg(cfg: Config) -> Config:
    return cfg.model_copy(update={"agent": AgentConfig(loop=True)})


def _agent_result(
    question: str,
    answer: str = "Install with uv [1].",
    *,
    abstained: bool = False,
    rewrites: int = 0,
    trace: list[TraceEvent] | None = None,
    sources: list[Source] | None = None,
) -> AgentResult:
    queries = [question] + [f"{question} (rewrite {n})" for n in range(1, rewrites + 1)]
    return AgentResult(
        question=question,
        answer=answer,
        sources=sources or [],
        abstained=abstained,
        queries=queries,
        rewrites=rewrites,
        trace=trace if trace is not None else [TraceEvent("analyze", 1.0)],
    )


class FakeAgent:
    """Stands in for `run_agent`: scripted results by question, recording every call."""

    def __init__(self, results: dict[str, AgentResult]) -> None:
        self.results = results
        self.calls: list[dict[str, Any]] = []

    def __call__(self, question: str, cfg: Config, **kwargs: Any) -> AgentResult:
        self.calls.append({"question": question, "cfg": cfg, **kwargs})
        return self.results[question]


def test_loop_runs_the_agent_with_history(
    loop_cfg: Config,
    store: tuple[ChunkIndex, Registry],
    vault: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = [
        Source(1, vault / "docs" / "manual.pdf", 2, ("Setup",), "Setup: install with uv.", 0.9),
        Source(2, None, None, (), "Orphan text.", 0.4),
    ]
    agent = FakeAgent(
        {
            "follow-up?": _agent_result("follow-up?", "  It is uv [1].  ", sources=sources),
            "single": _agent_result("single", NOT_FOUND, abstained=True, rewrites=2),
        }
    )
    monkeypatch.setattr(runner, "run_agent", agent)
    judged: list[str] = []

    def fake_judge(question: str, reference: str, answer: str, *, llm: Any) -> JudgeResult:
        judged.append(answer)
        return JudgeResult(Verdict.CORRECT, "ok")

    monkeypatch.setattr(runner, "judge", fake_judge)
    reranker = FakeReranker()
    history = [
        {"role": "user", "content": "How do I install it?"},
        {"role": "assistant", "content": "With uv [1]."},
    ]
    items = [_item(question="follow-up?", history=history), _item(question="single")]
    llm = FakeLLM()
    index, registry = store
    report = _run(loop_cfg, store, items, llm, reranker=reranker, judge_llm=FakeLLM())

    assert [c["question"] for c in agent.calls] == ["follow-up?", "single"]
    first, second = agent.calls
    assert first["history"] == history
    assert second["history"] == []
    assert first["cfg"] is loop_cfg
    assert first["llm"] is llm and first["reranker"] is reranker
    assert first["index"] is index and first["registry"] is registry
    assert llm.calls == []  # the agent made every model call

    answered, abstained = report.items
    assert answered.answer == "It is uv [1]."  # stripped
    assert not answered.abstained
    assert answered.retrieved == [
        RetrievedSource("docs/manual.pdf", 2, "Setup: install with uv.", 0.9),
        RetrievedSource("", None, "Orphan text.", 0.4),
    ]
    assert answered.queries == ["follow-up?"] and answered.rewrites == 0
    assert answered.verdict == "correct"
    assert judged == ["It is uv [1]."]  # abstentions are not sent to the judge
    assert abstained.abstained and abstained.verdict == "incorrect"
    assert abstained.queries == ["single", "single (rewrite 1)", "single (rewrite 2)"]
    assert abstained.rewrites == 2


def test_loop_timing_split(
    loop_cfg: Config, store: tuple[ChunkIndex, Registry], monkeypatch: pytest.MonkeyPatch
) -> None:
    trace = [
        TraceEvent("analyze", 50.0),
        TraceEvent("retrieve", 20.0),
        TraceEvent("grade", 30.0),
        TraceEvent("rewrite", 40.0),
        TraceEvent("retrieve", 25.0),
        TraceEvent("grade", 15.0),
        TraceEvent("generate", 600.0),
    ]
    direct = [TraceEvent("analyze", 45.0), TraceEvent("direct", 300.0)]
    agent = FakeAgent(
        {"a": _agent_result("a", rewrites=1, trace=trace), "b": _agent_result("b", trace=direct)}
    )
    monkeypatch.setattr(runner, "run_agent", agent)
    ticks = iter([0.0, 1.0, 5.0, 5.4])  # item a: 1000 ms wall time, item b: 400 ms
    monkeypatch.setattr(runner, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))

    report = _run(loop_cfg, store, [_item(question="a"), _item(question="b")], FakeLLM())
    a, b = report.items
    assert a.generation_ms == pytest.approx(600.0)
    assert a.retrieval_ms == pytest.approx(400.0)  # analyze, rewrite and overhead included
    assert a.node_ms == {
        "analyze": 50.0,
        "retrieve": 45.0,
        "grade": 45.0,
        "rewrite": 40.0,
        "generate": 600.0,
    }
    assert b.generation_ms == pytest.approx(300.0)  # the direct answer counts as generation
    assert b.retrieval_ms == pytest.approx(100.0)
    assert report.metrics["total_p50_ms"] == pytest.approx(700.0)  # wall times 1000 and 400
    assert report.metrics["analyze_p50_ms"] == pytest.approx(47.5)
    assert report.metrics["rewrite_rate"] == pytest.approx(0.5)
    assert report.metrics["rewrite_recovered"] == pytest.approx(1.0)


def test_loop_off_never_calls_the_agent(
    cfg: Config, store: tuple[ChunkIndex, Registry], monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_agent(*args: Any, **kwargs: Any) -> AgentResult:
        raise AssertionError("run_agent must not be called with agent.loop off")

    monkeypatch.setattr(runner, "run_agent", no_agent)
    item = _item(history=[{"role": "user", "content": "earlier turn"}])
    llm = FakeLLM(["Install with uv [1]."])
    [result] = _run(cfg, store, [item], llm).items
    assert len(llm.calls) == 1
    assert "earlier turn" not in str(llm.calls[0])  # the linear path ignores history
    assert result.queries == [] and result.rewrites == 0 and result.node_ms == {}


def test_agent_metrics_hand_computed() -> None:
    items = [_item(), _item(), _item(), _item(**UNANSWERABLE)]
    results = [
        replace(_result(0, []), rewrites=0, node_ms={"analyze": 10.0, "generate": 100.0}),
        replace(_result(1, []), rewrites=1, node_ms={"analyze": 30.0, "rewrite": 5.0}),
        replace(
            _result(2, [], NOT_FOUND, abstained=True),
            rewrites=2,
            node_ms={"analyze": 20.0, "not_found": 1.0},
        ),
        replace(_result(3, [], NOT_FOUND, abstained=True), rewrites=3, node_ms={"direct": 4.0}),
    ]
    m = compute_metrics(items, results)
    assert list(m) == list(METRIC_NAMES)
    assert m["rewrite_rate"] == pytest.approx(3 / 4)
    assert m["rewrite_recovered"] == pytest.approx(1 / 3)  # only item 1 ended answered
    assert m["analyze_p50_ms"] == pytest.approx(20.0)  # item 3 has no analyze time

    no_rewrites = [replace(r, rewrites=0) for r in results]
    m = compute_metrics(items, no_rewrites)
    assert m["rewrite_rate"] == 0.0
    assert m["rewrite_recovered"] is None  # nothing was rewritten

    no_analyze = [replace(r, node_ms={"direct": 1.0}) for r in results]
    assert compute_metrics(items, no_analyze)["analyze_p50_ms"] is None


def test_agent_fields_round_trip(tmp_path: Path) -> None:
    item = replace(
        _result(0, [("a.pdf", 1, "t")]),
        queries=["q", "q2"],
        rewrites=1,
        node_ms={"analyze": 1.5, "generate": 9.0},
    )
    report = RunReport("loop", "tiny", [item], dict.fromkeys(METRIC_NAMES))
    assert load_results(save_report(report, tmp_path)) == [item]


def test_load_results_reads_old_format(tmp_path: Path) -> None:
    old = {  # a Phase 3 results line: no queries / rewrites / node_ms
        "index": 0,
        "question": "q0",
        "answerable": True,
        "tags": [],
        "retrieved": [{"file": "a.pdf", "page": 1, "text": "t", "score": 0.5}],
        "answer": "an answer",
        "abstained": False,
        "retrieval_ms": 10.0,
        "generation_ms": 100.0,
        "verdict": None,
        "judge_reason": None,
    }
    path = tmp_path / "old.jsonl"
    path.write_text(json.dumps(old) + "\n" + json.dumps({"metrics": {}}) + "\n", encoding="utf-8")
    [result] = load_results(path)
    assert result == _result(0, [("a.pdf", 1, "t")])
    assert result.queries == [] and result.rewrites == 0 and result.node_ms == {}


def test_format_table_includes_agent_rows() -> None:
    metrics: dict[str, float | None] = dict.fromkeys(METRIC_NAMES)
    metrics.update({"rewrite_rate": 0.25, "rewrite_recovered": 0.5, "analyze_p50_ms": 812.4})
    table = format_table([RunReport("loop", "cuad", [], metrics)])
    rows = {line.split()[0]: line.split()[1:] for line in table.split("\n")[2:]}
    assert rows["rewrite_rate"] == ["0.250"]
    assert rows["rewrite_recovered"] == ["0.500"]
    assert rows["analyze_p50_ms"] == ["812"]


def test_loop_records_check_findings(
    loop_cfg: Config, store: tuple[ChunkIndex, Registry], monkeypatch: pytest.MonkeyPatch
) -> None:
    result = replace(
        _agent_result("q"),
        contradictions=[Contradiction("fee is 10", 1, "fee is 12", 2)],
        unsupported=[UnsupportedClaim("it is free", (1,), "source says 10")],
    )
    monkeypatch.setattr(runner, "run_agent", FakeAgent({"q": result}))
    [item] = _run(loop_cfg, store, [_item(question="q")], FakeLLM()).items
    assert item.contradictions == ["[1] fee is 10 | [2] fee is 12"]
    assert item.unsupported == ["it is free"]
