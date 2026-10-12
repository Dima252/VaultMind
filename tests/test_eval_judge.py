"""LLM judge: prompt construction, parsing, retry, and a live check against Ollama."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from vaultmind.core.config import load_config
from vaultmind.core.ollama import OllamaLLM
from vaultmind.core.providers import Message
from vaultmind.eval.judge import (
    CLARIFY_RETRY_PROMPT,
    CLARIFY_SCHEMA,
    CLARIFY_SYSTEM_PROMPT,
    JUDGE_RETRY_PROMPT,
    JUDGE_SCHEMA,
    JUDGE_SYSTEM_PROMPT,
    KEY_FACTS_RETRY_PROMPT,
    KEY_FACTS_SYSTEM_PROMPT,
    MAX_REASON_CHARS,
    MAX_REFERENCE_CHARS,
    JudgeError,
    JudgeResult,
    KeyFactsResult,
    Verdict,
    judge,
    judge_clarify,
    judge_key_facts,
    key_facts_schema,
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


# --- key-facts judge --------------------------------------------------------------------------

REQUEST = "Summarize the supply agreement."
FACTS = ["The term is 3 years.", "Vermont law governs.", "Invoices are due in 45 days."]
SUMMARY = "A three-year supply deal under Vermont law [1]."


def _facts(*stated: bool, numbers: Sequence[int] | None = None) -> dict[str, Any]:
    nums = list(numbers) if numbers is not None else list(range(1, len(stated) + 1))
    return {
        "facts": [
            {"number": n, "reason": f"reason {n}", "stated": s}
            for n, s in zip(nums, stated, strict=True)
        ]
    }


def test_key_facts_mixed_found_and_score() -> None:
    llm = SchemaRecordingLLM([_facts(True, True, False)])
    result = judge_key_facts(REQUEST, FACTS, SUMMARY, llm=llm)
    assert result == KeyFactsResult((True, True, False), ("reason 1", "reason 2", "reason 3"))
    assert result.score == pytest.approx(2 / 3)
    assert llm.schemas == [key_facts_schema(3)]
    facts_schema = key_facts_schema(3)["properties"]["facts"]
    assert facts_schema["minItems"] == facts_schema["maxItems"] == 3
    # evidence and reason come before the verdict, so the model quotes before it decides
    assert facts_schema["items"]["required"] == ["number", "evidence", "reason", "stated"]


def test_key_facts_prompt_numbers_facts_inside_delimiters() -> None:
    llm = FakeLLM([_facts(True, True, True)])
    judge_key_facts(REQUEST, [*FACTS[:2], "Multi\n  line   fact."], SUMMARY, llm=llm)
    system, user = llm.calls[0]
    assert system == {"role": "system", "content": KEY_FACTS_SYSTEM_PROMPT}
    prompt = user["content"]
    assert f"<request>\n{REQUEST}\n</request>" in prompt
    numbered = f"1. {FACTS[0]}\n2. {FACTS[1]}\n3. Multi line fact."
    assert f"<key_facts>\n{numbered}\n</key_facts>" in prompt
    assert f"<answer>\n{SUMMARY}\n</answer>" in prompt
    assert prompt.index("<request>") < prompt.index("<key_facts>") < prompt.index("<answer>")


def test_key_facts_entries_in_any_order() -> None:
    llm = FakeLLM([_facts(False, True, True, numbers=[3, 1, 2])])
    result = judge_key_facts(REQUEST, FACTS, SUMMARY, llm=llm)
    assert result.found == (True, True, False)
    assert result.reasons == ("reason 1", "reason 2", "reason 3")


def test_key_facts_all_found_and_hebrew_answer() -> None:
    llm = FakeLLM([_facts(True, True, True)])
    result = judge_key_facts(REQUEST, FACTS, "הסכם לשלוש שנים לפי דיני ורמונט [1].", llm=llm)
    assert result.score == 1.0
    assert "ורמונט" in llm.calls[0][1]["content"]


def test_key_facts_no_facts_and_empty_answer_skip_llm() -> None:
    llm = FakeLLM()
    assert judge_key_facts(REQUEST, [], SUMMARY, llm=llm) == KeyFactsResult((), ())
    assert judge_key_facts(REQUEST, [], SUMMARY, llm=llm).score == 1.0
    empty = judge_key_facts(REQUEST, FACTS, "  \n", llm=llm)
    assert empty.found == (False, False, False) and empty.score == 0.0
    assert llm.calls == []


@pytest.mark.parametrize(
    "bad",
    [
        "not json",
        '["facts"]',
        '{"facts": "none"}',
        _facts(True, True),  # too few: length mismatch
        _facts(True, True, True, True),  # too many
        _facts(True, True, True, numbers=[1, 1, 2]),  # repeated number
        _facts(True, True, True, numbers=[0, 1, 2]),  # out of range
        {"facts": [{"number": 1, "reason": "r", "stated": "yes"}] * 3},
        {"facts": [{"number": "1", "reason": "r", "stated": True}] * 3},
        {"facts": [{"number": n, "stated": True} for n in (1, 2, 3)]},
        {"facts": [1, 2, 3]},
    ],
)
def test_key_facts_malformed_outputs_trigger_retry(bad: str | dict[str, Any]) -> None:
    llm = FakeLLM([bad, _facts(True, False, True)])
    assert judge_key_facts(REQUEST, FACTS, SUMMARY, llm=llm).found == (True, False, True)
    assert len(llm.calls) == 2
    retry = llm.calls[1]
    assert retry[3] == {"role": "user", "content": KEY_FACTS_RETRY_PROMPT.format(count=3)}
    assert "exactly 3 entries" in retry[3]["content"]


def test_key_facts_length_mismatch_twice_raises() -> None:
    llm = FakeLLM([_facts(True), _facts(True, True)])
    with pytest.raises(JudgeError, match="expected 3 facts, got 2"):
        judge_key_facts(REQUEST, FACTS, SUMMARY, llm=llm)
    assert len(llm.calls) == 2


def test_key_facts_reason_collapsed_and_capped() -> None:
    entry = {"number": 1, "reason": "Says it.\n\n  Twice. " + "x " * 300, "stated": True}
    llm = FakeLLM([{"facts": [entry]}])
    [reason] = judge_key_facts(REQUEST, FACTS[:1], SUMMARY, llm=llm).reasons
    assert reason.startswith("Says it. Twice.") and len(reason) <= MAX_REASON_CHARS


# --- clarify judge ----------------------------------------------------------------------------


@pytest.mark.parametrize("asks", [True, False])
def test_clarify_yes_and_no(asks: bool) -> None:
    llm = SchemaRecordingLLM([{"reason": "r", "asks": asks}])
    assert judge_clarify("Summarize the syllabus", "Which one: A or B?", llm=llm) is asks
    assert llm.schemas == [CLARIFY_SCHEMA]
    system, user = llm.calls[0]
    assert system == {"role": "system", "content": CLARIFY_SYSTEM_PROMPT}
    assert "<request>\nSummarize the syllabus\n</request>" in user["content"]
    assert "<response>\nWhich one: A or B?\n</response>" in user["content"]


def test_clarify_retry_then_failure() -> None:
    llm = FakeLLM(['{"asks": "yes", "reason": "r"}', {"reason": "r", "asks": True}])
    assert judge_clarify("q", "a", llm=llm) is True
    assert llm.calls[1][3] == {"role": "user", "content": CLARIFY_RETRY_PROMPT}

    llm = FakeLLM(["nope", '{"asks": true}'])
    with pytest.raises(JudgeError, match="missing reason"):
        judge_clarify("q", "a", llm=llm)


def test_clarify_empty_answer_skips_llm() -> None:
    llm = FakeLLM()
    assert judge_clarify("q", "   ", llm=llm) is False
    assert llm.calls == []


# --- live: key facts ---------------------------------------------------------------------------

_LIVE_FACTS = [
    "The monthly fee rises from USD 12,500 to USD 14,200.",
    "The change takes effect on July 1, 2025.",
    "Three compressors are added.",
]


@pytest.mark.ollama
def test_live_key_facts_paraphrase_hebrew_and_contradiction() -> None:
    llm = OllamaLLM.from_config(load_config())
    request = "What does the amendment change?"

    english = judge_key_facts(
        request,
        _LIVE_FACTS,
        "From July 1, 2025 the client pays $14,200 a month instead of $12,500 [1]. The emergency "
        "response time stays the same [1].",
        llm=llm,
    )
    assert english.found == (True, True, False), english.reasons

    hebrew = judge_key_facts(
        request,
        _LIVE_FACTS,
        "התוספת מוסיפה שלושה מדחסים, ומה-1 ביולי 2025 התשלום החודשי עולה מ-12,500 ל-14,200 "
        "דולר [1].",
        llm=llm,
    )
    assert hebrew.found == (True, True, True), hebrew.reasons

    wrong = judge_key_facts(
        request,
        _LIVE_FACTS,
        "The monthly fee drops from USD 14,200 to USD 12,500 starting July 1, 2025 [1].",
        llm=llm,
    )
    assert wrong.found[0] is False, wrong.reasons
