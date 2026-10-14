"""Calculations before the answer (agent/calculate.py), with fakes."""

from __future__ import annotations

from pathlib import Path

from vaultmind.agent.calculate import CALC_TOOLS, MAX_ROUNDS, NOTE_HEADER, calculate, note
from vaultmind.agent.gateway import Gateway
from vaultmind.agent.tools import default_tools
from vaultmind.agent.types import ToolResult
from vaultmind.core.providers import AssistantTurn, ToolCall

from .fakes import FakeLLM, tool_turn
from .test_tools import build_vault


def _gateway(tmp_path: Path) -> Gateway:
    tools = [t for t in default_tools() if t.name in CALC_TOOLS]
    return Gateway(build_vault(tmp_path).ctx, tools)


def test_calculate_runs_the_calls_and_notes_the_results(tmp_path: Path) -> None:
    two = AssistantTurn(
        "",
        (
            ToolCall("calc", {"expression": "84000 + 30 * 360"}),
            ToolCall("date_diff", {"a": "28.5.2025", "b": "11.6.2025"}),
        ),
    )
    llm = FakeLLM(turns=[two, "done"])
    seen: list[ToolResult] = []
    results = calculate(llm, "Prices... Request: cost?", _gateway(tmp_path), on_result=seen.append)
    assert seen == results and all(r.ok for r in results)
    assert note(results) == (
        f"{NOTE_HEADER}\n- 84000 + 30 * 360 = 94800\n- From 2025-05-28 to 2025-06-11: 14 days."
    )
    messages, tools, think = llm.chat_calls[1]
    assert [t.name for t in tools] == ["calc", "date_diff"] and think is False
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "tool"]
    assert messages[2]["tool_calls"][0]["name"] == "calc"
    assert messages[3]["tool_name"] == "calc"


def test_calculate_with_nothing_to_compute(tmp_path: Path) -> None:
    llm = FakeLLM(turns=["done"])
    assert calculate(llm, "Request: what is the deadline?", _gateway(tmp_path)) == []
    assert note([]) == ""


def test_calculate_stops_after_its_rounds_and_only_has_calc_tools(tmp_path: Path) -> None:
    llm = FakeLLM(turns=[tool_turn("read", doc="syllabus")] * (MAX_ROUNDS + 2))
    results = calculate(llm, "x", _gateway(tmp_path))
    assert len(llm.chat_calls) == MAX_ROUNDS
    assert all(not r.ok and "Unknown tool" in (r.error or "") for r in results)
    assert note(results) == ""
