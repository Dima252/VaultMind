"""LLM judge: prompt construction, parsing, retry, and a live check against Ollama."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from vaultmind.core.config import load_config
from vaultmind.core.ollama import OllamaLLM
from vaultmind.core.providers import Message
from vaultmind.eval.judge import (
    JUDGE_RETRY_PROMPT,
    JUDGE_SCHEMA,
    JUDGE_SYSTEM_PROMPT,
    MAX_REASON_CHARS,
    MAX_REFERENCE_CHARS,
    JudgeError,
    JudgeResult,
    Verdict,
    judge,
)

from .fakes import FakeLLM

QUESTION = "Which state's law governs the agreement?"
REFERENCE = "This Agreement shall be governed by the laws of the State of Delaware."
ANSWER = "It is governed by Delaware law [1]."


class SchemaRecordingLLM(FakeLLM):
    """FakeLLM that also records the schema passed to each call."""

    def __init__(self, responses: Sequence[str | dict[str, Any]] = ()) -> None:
        super().__init__(responses)
        self.schemas: list[dict[str, Any] | None] = []

    def generate(self, messages: Sequence[Message], *, schema: dict[str, Any] | None = None) -> str:
        self.schemas.append(schema)
        return super().generate(messages, schema=schema)


def _user_prompt(llm: FakeLLM, call: int = 0) -> str:
    return llm.calls[call][1]["content"]


@pytest.mark.parametrize("verdict", list(Verdict))
def test_each_verdict(verdict: Verdict) -> None:
    llm = FakeLLM([{"verdict": verdict.value, "reason": "Because."}])
    assert judge(QUESTION, REFERENCE, ANSWER, llm=llm) == JudgeResult(verdict, "Because.")
    assert len(llm.calls) == 1


def test_verdict_case_and_whitespace_tolerated() -> None:
    llm = FakeLLM([{"verdict": "  Correct\n", "reason": "Matches."}])
    assert judge(QUESTION, REFERENCE, ANSWER, llm=llm).verdict is Verdict.CORRECT


def test_multiline_reason_collapsed_and_capped() -> None:
    llm = FakeLLM([{"verdict": "partial", "reason": "Names one party.\n\n  Misses   the other."}])
    assert (
        judge(QUESTION, REFERENCE, ANSWER, llm=llm).reason == "Names one party. Misses the other."
    )

    llm = FakeLLM([{"verdict": "partial", "reason": "word " * 200}])
    reason = judge(QUESTION, REFERENCE, ANSWER, llm=llm).reason
    assert len(reason) <= MAX_REASON_CHARS
    assert "\n" not in reason
    assert reason.endswith("...")


def test_schema_passed_to_generate() -> None:
    llm = SchemaRecordingLLM([{"verdict": "correct", "reason": "ok"}])
    judge(QUESTION, REFERENCE, ANSWER, llm=llm)
    assert llm.schemas == [JUDGE_SCHEMA]
    assert JUDGE_SCHEMA["required"] == ["verdict", "reason"]
    assert JUDGE_SCHEMA["additionalProperties"] is False
    assert JUDGE_SCHEMA["properties"]["verdict"]["enum"] == ["correct", "partial", "incorrect"]


def test_prompt_contains_question_reference_and_answer() -> None:
    llm = FakeLLM([{"verdict": "correct", "reason": "ok"}])
    judge(QUESTION, REFERENCE, ANSWER, llm=llm)
    system, user = llm.calls[0]
    assert system == {"role": "system", "content": JUDGE_SYSTEM_PROMPT}
    assert user["role"] == "user"
    prompt = user["content"]
    assert f"<question>\n{QUESTION}\n</question>" in prompt
    assert f"<reference>\n{REFERENCE}\n</reference>" in prompt
    assert f"<answer>\n{ANSWER}\n</answer>" in prompt
    assert prompt.index("<question>") < prompt.index("<reference>") < prompt.index("<answer>")


def test_retry_then_success() -> None:
    llm = SchemaRecordingLLM(["not json at all", {"verdict": "incorrect", "reason": "Wrong."}])
    assert judge(QUESTION, REFERENCE, ANSWER, llm=llm) == JudgeResult(Verdict.INCORRECT, "Wrong.")
    assert len(llm.calls) == 2
    assert llm.schemas == [JUDGE_SCHEMA, JUDGE_SCHEMA]
    first, retry = llm.calls
    assert len(first) == 2
    assert list(retry[:2]) == list(first)
    assert retry[2] == {"role": "assistant", "content": "not json at all"}
    assert retry[3] == {"role": "user", "content": JUDGE_RETRY_PROMPT}


def test_retry_then_failure_raises_with_excerpt() -> None:
    llm = FakeLLM(["garbage one", "garbage two"])
    with pytest.raises(JudgeError, match="garbage two"):
        judge(QUESTION, REFERENCE, ANSWER, llm=llm)
    assert len(llm.calls) == 2


def test_error_excerpt_is_short() -> None:
    llm = FakeLLM(["x", "y" * 5000])
    with pytest.raises(JudgeError) as info:
        judge(QUESTION, REFERENCE, ANSWER, llm=llm)
    assert len(str(info.value)) < 400


def test_invalid_verdict_value_retried() -> None:
    llm = FakeLLM([{"verdict": "mostly right", "reason": "?"}, {"verdict": "maybe", "reason": "?"}])
    with pytest.raises(JudgeError, match="invalid verdict"):
        judge(QUESTION, REFERENCE, ANSWER, llm=llm)
    assert len(llm.calls) == 2


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "{not json",
        '["correct", "ok"]',
        '{"verdict": "correct"}',
        '{"reason": "ok"}',
        '{"verdict": 1, "reason": "ok"}',
        '{"verdict": "correct", "reason": null}',
    ],
)
def test_malformed_outputs_trigger_retry(bad: str) -> None:
    llm = FakeLLM([bad, {"verdict": "partial", "reason": "ok"}])
    assert judge(QUESTION, REFERENCE, ANSWER, llm=llm).verdict is Verdict.PARTIAL
    assert len(llm.calls) == 2


@pytest.mark.parametrize("answer", ["", "   ", "\n\t"])
def test_empty_answer_skips_llm(answer: str) -> None:
    llm = FakeLLM()
    assert judge(QUESTION, REFERENCE, answer, llm=llm) == JudgeResult(
        Verdict.INCORRECT, "empty answer"
    )
    assert llm.calls == []


def test_long_reference_truncated_with_marker() -> None:
    reference = "A" * MAX_REFERENCE_CHARS + "TAIL_SHOULD_NOT_APPEAR"
    llm = FakeLLM([{"verdict": "correct", "reason": "ok"}])
    judge(QUESTION, reference, ANSWER, llm=llm)
    prompt = _user_prompt(llm)
    assert "A" * MAX_REFERENCE_CHARS in prompt
    assert "TAIL_SHOULD_NOT_APPEAR" not in prompt
    assert "[reference truncated" in prompt
    assert str(len(reference)) in prompt


def test_short_reference_not_truncated() -> None:
    llm = FakeLLM([{"verdict": "correct", "reason": "ok"}])
    judge(QUESTION, REFERENCE, ANSWER, llm=llm)
    assert "truncated" not in _user_prompt(llm)


# --- live -----------------------------------------------------------------------------------

_LICENSE_REFERENCE = (
    'This License Agreement is entered into by and between Acme Robotics, Inc. ("Licensor") '
    'and Borealis Software LLC ("Licensee"). ... This Agreement shall be governed by and '
    "construed in accordance with the laws of the State of New York, without regard to its "
    "conflict of laws principles. ... Licensee shall not assign this Agreement without the "
    "prior written consent of Licensor."
)
_TERMINATION_REFERENCE = (
    "Either party may terminate this Agreement for convenience upon ninety (90) days' prior "
    "written notice to the other party. ... Upon termination, Licensee shall cease all use of "
    "the Licensed Software and return or destroy all copies within thirty (30) days."
)


@pytest.mark.ollama
def test_live_judge_three_cases() -> None:
    llm = OllamaLLM.from_config(load_config())

    correct = judge(
        "How can the agreement be terminated, and what happens afterwards?",
        _TERMINATION_REFERENCE,
        "Either side can end it for convenience by giving 90 days' written notice [1]. After "
        "termination the licensee has to stop using the software and return or destroy every "
        "copy within 30 days [1].",
        llm=llm,
    )
    assert correct.verdict is Verdict.CORRECT, correct.reason

    wrong = judge(
        "Which state's law governs the agreement?",
        _LICENSE_REFERENCE,
        "The agreement is governed by the laws of the State of California [1].",
        llm=llm,
    )
    assert wrong.verdict is Verdict.INCORRECT, wrong.reason

    partial = judge(
        "Who are the parties to the agreement?",
        _LICENSE_REFERENCE,
        "The agreement is entered into by Acme Robotics, Inc., the licensor [1].",
        llm=llm,
    )
    assert partial.verdict in {Verdict.PARTIAL, Verdict.INCORRECT}, partial.reason
