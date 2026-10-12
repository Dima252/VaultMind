import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from vaultmind import cli
from vaultmind.agent.baseline import NOT_FOUND, Source
from vaultmind.agent.trace_text import format_trace
from vaultmind.agent.types import AgentResult, Contradiction, TraceEvent, UnsupportedClaim
from vaultmind.core import ollama
from vaultmind.core.config import load_config
from vaultmind.core.models import ChildChunk, FileRecord, ParentSection
from vaultmind.eval import runner
from vaultmind.store.index import ChunkIndex
from vaultmind.store.registry import Registry
from vaultmind.store.traces import TraceStore

from .fakes import FakeEmbedder, FakeLLM

FILE_ID = "0123456789abcdef"
PARENT = ParentSection(f"{FILE_ID}:p0", FILE_ID, "Setup: install with uv.", ("Setup",), 2, 2)
CHILD = ChildChunk(
    id=f"{PARENT.id}:c0",
    parent_id=PARENT.id,
    file_id=FILE_ID,
    text=PARENT.text,
    embed_text="Setup: install with uv.",
    page=2,
    section_path=("Setup",),
)
HEBREW_Q = "מה תקופת ההודעה?"
RUN_LINE = re.compile(r"^run ([0-9a-f]{32})$", re.MULTILINE)

app_runner = CliRunner()


def _write_config(tmp_path: Path, *, loop: bool) -> Path:
    path = tmp_path / f"config-{'loop' if loop else 'linear'}.yaml"
    data = {
        "paths": {"vault_dir": str(tmp_path / "vault"), "data_dir": str(tmp_path / "data")},
        "retrieval": {"rerank": False, "hybrid": False},
        "agent": {"loop": loop},
    }
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    """A loop-on config whose data_dir holds a one-chunk index."""
    path = _write_config(tmp_path, loop=True)
    cfg = load_config(path)
    index = ChunkIndex(cfg.paths.data_dir / "index", embedder_name=cfg.models.embedder)
    registry = Registry(cfg.paths.data_dir / "vaultmind.sqlite")
    try:
        index.add([PARENT], [CHILD], FakeEmbedder().embed([CHILD.embed_text]))
        registry.upsert(FileRecord(FILE_ID, tmp_path / "vault" / "manual.md", "sha", "ready"))
    finally:
        registry.close()
        index.close()
    return path


@pytest.fixture
def llm(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    """No Ollama: the CLI gets fakes from the providers' from_config."""
    fake = FakeLLM(["Install with uv [1]."])
    monkeypatch.setattr(ollama.OllamaLLM, "from_config", lambda cfg: fake)
    monkeypatch.setattr(ollama.OllamaEmbedder, "from_config", lambda cfg: FakeEmbedder())
    return fake


def _source(n: int, name: str, *warnings: str) -> Source:
    return Source(n, Path("/vault") / name, n, ("Terms",), f"passage {n}", 0.5, None, warnings)


TRACE = [
    TraceEvent("analyze", 120.4, {"standalone_query": "notice period?", "variants": ["a", "b"]}),
    TraceEvent("retrieve", 35.0, {"queries": ["notice period?", "a", "b"], "candidates": 12}),
    TraceEvent(
        "grade",
        80.0,
        {
            "threshold": 0.3,
            "passed": 0,
            "scores": [
                {"label": "x.md", "score": 0.12, "passed": False},
                {"label": "y.md", "score": 0.25, "passed": False},
            ],
        },
    ),
    TraceEvent("rewrite", 400.0, {"attempt": 1, "query": "termination notice days"}),
    TraceEvent("generate", 950.0, {"sources": [1, 2], "abstained": False}),
]


def _answer(**kw: Any) -> AgentResult:
    base: dict[str, Any] = {
        "question": "notice period?",
        "answer": "It is 30 days [1].",
        "sources": [_source(1, "lease.md", "instruction-like text"), _source(2, "other.md")],
        "abstained": False,
        "queries": ["notice period?", "termination notice days"],
        "rewrites": 1,
        "trace": TRACE,
    }
    base.update(kw)
    return AgentResult(**base)


class FakeStream:
    """Stands in for `stream_agent`: yields tokens, then the result; records the call."""

    def __init__(self, result: AgentResult, tokens: list[str]) -> None:
        self.result = result
        self.tokens = tokens
        self.calls: list[dict[str, Any]] = []

    def __call__(self, question: str, cfg: Any, **kwargs: Any) -> Iterator[str | AgentResult]:
        self.calls.append({"question": question, **kwargs})
        yield from self.tokens
        yield self.result


def _stream(monkeypatch: pytest.MonkeyPatch, result: AgentResult, *tokens: str) -> FakeStream:
    fake = FakeStream(result, list(tokens))
    monkeypatch.setattr(cli, "stream_agent", fake)
    return fake


def _invoke(*args: str) -> Any:
    result = app_runner.invoke(cli.app, list(args))
    assert result.exception is None or isinstance(result.exception, SystemExit), result.output
    return result


def _stored(config_path: Path) -> TraceStore:
    return TraceStore(load_config(config_path).paths.data_dir / "vaultmind.sqlite")


# --- ask with the agent ------------------------------------------------------------------


def test_ask_streams_tokens_prints_sources_and_saves_the_run(
    config_path: Path, llm: FakeLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _answer(
        contradictions=[Contradiction("30 days", 1, "60 days", 2)],
        unsupported=[UnsupportedClaim("Renewal is automatic.", (2,), "not stated")],
    )
    stream = _stream(monkeypatch, result, "It is ", "30 days ", "[1].")

    out = _invoke("ask", "notice period?", "-c", str(config_path))
    assert out.exit_code == 0, out.output
    lines = out.output.splitlines()
    assert lines[0] == "It is 30 days [1]."
    assert "Sources:" in lines
    assert "  [1] lease.md, p. 1, Terms" in lines
    assert "[2] other.md" not in out.output  # only cited sources are listed
    assert "      [!] instruction-like text" in lines
    assert "  [1] 30 days" in lines and "  [2] 60 days" in lines
    assert "  Renewal is automatic. [2]: not stated" in lines
    assert "Trace:" not in out.output

    [call] = stream.calls
    assert call["question"] == "notice period?"
    assert call["llm"] is llm and call["reranker"] is None
    assert isinstance(call["embedder"], FakeEmbedder)
    assert isinstance(call["index"], ChunkIndex) and isinstance(call["registry"], Registry)

    match = RUN_LINE.search(out.output)
    assert match is not None
    store = _stored(config_path)
    try:
        run = store.get(match.group(1))
    finally:
        store.close()
    assert run is not None
    assert (run.question, run.answer, run.abstained) == ("notice period?", result.answer, False)
    assert run.events == TRACE


def test_ask_abstention_lists_closest_matches(
    config_path: Path, llm: FakeLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _answer(answer=NOT_FOUND, abstained=True, trace=[TraceEvent("not_found", 1.0)])
    _stream(monkeypatch, result, NOT_FOUND)
    out = _invoke("ask", "notice period?", "-c", str(config_path))
    assert out.exit_code == 0, out.output
    lines = out.output.splitlines()
    assert "Closest matches:" in lines and "Sources:" not in lines
    assert "  [1] lease.md, p. 1, Terms" in lines and "  [2] other.md, p. 2, Terms" in lines
    assert RUN_LINE.search(out.output)


def test_ask_trace_prints_one_line_per_event(
    config_path: Path, llm: FakeLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stream(monkeypatch, _answer(), "It is 30 days [1].")
    out = _invoke("ask", "notice period?", "--trace", "-c", str(config_path))
    assert out.exit_code == 0, out.output
    trace = out.output.split("Trace:\n", 1)[1].split("\n\n", 1)[0].splitlines()
    assert [line.split()[0] for line in trace] == [
        "analyze",
        "retrieve",
        "grade",
        "rewrite",
        "generate",
        "total",
    ]
    assert all(line.isascii() for line in trace)
    by_node = {line.split()[0]: line for line in trace}
    assert "120 ms" in by_node["analyze"]
    assert 'query "notice period?", variants "a" | "b"' in by_node["analyze"]
    assert "3 queries -> 12 candidates" in by_node["retrieve"]
    assert "0/2 passed (threshold 0.30), top 0.25 0.12" in by_node["grade"]
    assert 'attempt 1: "termination notice days"' in by_node["rewrite"]
    assert "sources [1][2]" in by_node["generate"]
    assert "1585 ms" in by_node["total"]
    assert RUN_LINE.search(out.output)


def test_ask_empty_index_fails(tmp_path: Path, llm: FakeLLM) -> None:
    out = _invoke("ask", "q", "-c", str(_write_config(tmp_path, loop=True)))
    assert out.exit_code == 1
    assert "index is empty" in out.output


def test_ask_linear_path_unchanged(
    config_path: Path, tmp_path: Path, llm: FakeLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_agent(*args: Any, **kwargs: Any) -> Iterator[str | AgentResult]:
        raise AssertionError("stream_agent must not be called with agent.loop off")

    monkeypatch.setattr(cli, "stream_agent", no_agent)
    linear = _write_config(tmp_path, loop=False)  # same data_dir as config_path
    out = _invoke("ask", "Setup: install with uv.", "-c", str(linear))
    assert out.exit_code == 0, out.output
    assert len(llm.calls) == 1
    assert out.output.splitlines()[-2:] == ["Sources:", "  [1] manual.md, p. 2, Setup"]
    assert not RUN_LINE.search(out.output)
    store = _stored(linear)
    try:
        assert store.recent() == []  # linear runs are not traced
    finally:
        store.close()


# --- trace command -------------------------------------------------------------------------


def test_trace_shows_a_stored_run(config_path: Path) -> None:
    store = _stored(config_path)
    try:
        run_id = store.save(_answer(question=HEBREW_Q))
    finally:
        store.close()

    out = _invoke("trace", run_id, "-c", str(config_path))
    assert out.exit_code == 0, out.output
    lines = out.output.splitlines()
    assert lines[0].startswith(f"Run {run_id}") and lines[0].endswith("answered")
    assert lines[1] == f"Question: {HEBREW_Q}"
    assert lines[2:4] == ["Answer:", "It is 30 days [1]."]
    assert "Trace:" in lines
    assert any(line.split()[:1] == ["rewrite"] for line in lines)


def test_trace_lists_recent_runs(config_path: Path) -> None:
    store = _stored(config_path)
    try:
        first = store.save(_answer(question="first question"))
        second = store.save(_answer(question=HEBREW_Q, answer=NOT_FOUND, abstained=True))
    finally:
        store.close()

    out = _invoke("trace", "-c", str(config_path))
    assert out.exit_code == 0, out.output
    lines = out.output.splitlines()
    assert [line.split()[0] for line in lines] == [second, first]
    assert "abstained" in lines[0] and f'"{HEBREW_Q}"' in lines[0]
    assert "answered" in lines[1] and '"first question"' in lines[1]

    out = _invoke("trace", "--last", "1", "-c", str(config_path))
    assert [line.split()[0] for line in out.output.splitlines()] == [second]


def test_trace_empty_and_unknown(tmp_path: Path) -> None:
    config = _write_config(tmp_path, loop=True)
    out = _invoke("trace", "-c", str(config))
    assert out.exit_code == 0 and "No runs stored yet" in out.output
    out = _invoke("trace", "f" * 32, "-c", str(config))
    assert out.exit_code == 1 and "No run with id" in out.output


# --- format_trace ----------------------------------------------------------------------------


def test_format_trace_tolerates_odd_data() -> None:
    lines = format_trace(
        [
            TraceEvent("custom_node", 1.0, {"k": "v"}),
            TraceEvent("grade", 2.0, {"scores": [{"no_score": 1}]}),  # malformed: falls back
            TraceEvent("groundedness", 3.0, {"error": "unparseable output"}),
            TraceEvent("direct", 4.0, {"reason": "greeting", "extra": 1}),
            TraceEvent("not_found", 5.0, {"closest": ["a.md, p. 1", "b.md"]}),
            TraceEvent("contradictions", 6.0, {"found": [{"source_a": 1, "source_b": 3}]}),
        ]
    )
    assert lines[0].endswith('{"k":"v"}')
    assert lines[1].endswith('{"scores":[{"no_score":1}]}')
    assert lines[2].endswith("0 unsupported claim(s) [error: unparseable output]")
    assert lines[3].endswith('no retrieval: "greeting"')
    assert lines[4].endswith("closest: a.md, p. 1; b.md")
    assert lines[5].endswith("1 found: [1] vs [3]")
    assert lines[-1].split() == ["total", "21", "ms"]


# --- eval run ----------------------------------------------------------------------------------


@pytest.mark.parametrize("loop", [True, False])
def test_eval_run_progress_shows_rewrites(
    config_path: Path, tmp_path: Path, llm: FakeLLM, monkeypatch: pytest.MonkeyPatch, loop: bool
) -> None:
    config = config_path if loop else _write_config(tmp_path, loop=False)
    dataset = tmp_path / "tiny.jsonl"
    item = {"question": "Setup: install with uv.", "answer": "uv", "source_file": "manual.md"}
    dataset.write_text(json.dumps({**item, "source_page": 2}) + "\n", encoding="utf-8")
    agent_result = _answer(question=item["question"], rewrites=2, sources=[])
    monkeypatch.setattr(runner, "run_agent", lambda *a, **kw: agent_result)

    out = _invoke(
        "eval",
        "run",
        "-c",
        str(config),
        "-d",
        str(dataset),
        "--no-judge",
        "--out-dir",
        str(tmp_path / "results"),
    )
    assert out.exit_code == 0, out.output
    expected = "  [1/1] answered rewrites=2" if loop else "  [1/1] answered"
    assert expected in out.output.splitlines()
    assert "rewrite_rate" in out.output
