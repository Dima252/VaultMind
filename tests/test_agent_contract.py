"""The Phase 4 contract pieces that exist before the agents implement them."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from vaultmind.agent import graph
from vaultmind.agent.baseline import Source
from vaultmind.agent.types import AgentResult, TraceEvent
from vaultmind.core.config import Config


def _result() -> AgentResult:
    return AgentResult(
        question="q",
        answer="It is 4 [2].",
        sources=[Source(n, None, None, (), "t", 0.9) for n in (1, 2)],
        abstained=False,
        queries=["q"],
        rewrites=0,
        trace=[TraceEvent("grade", 5.0), TraceEvent("generate", 7.5), TraceEvent("grade", 1.0)],
    )


def test_run_agent_returns_the_final_result(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_stream(*args: Any, **kwargs: Any) -> Iterator[str | AgentResult]:
        yield "It is "
        yield "4 [2]."
        yield _result()

    monkeypatch.setattr(graph, "stream_agent", fake_stream)
    deps: dict[str, Any] = dict.fromkeys(("llm", "embedder", "index", "registry", "reranker"))
    result = graph.run_agent("q", Config(), **deps)
    assert result == _result()
    assert [s.n for s in result.cited] == [2]
    assert result.node_ms("grade") == pytest.approx(6.0)


def test_langsmith_tracing_forced_off() -> None:
    import os

    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"
