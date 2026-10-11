"""LLM judge: grades a generated answer against the reference answer (Phase 2 eval).

One structured-output call per answer. Abstentions are scored by the runner, never judged.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from vaultmind.core.providers import LLM


class Verdict(StrEnum):
    CORRECT = "correct"
    PARTIAL = "partial"
    INCORRECT = "incorrect"


@dataclass(frozen=True)
class JudgeResult:
    verdict: Verdict
    reason: str  # one sentence


class JudgeError(Exception):
    """The judge's output could not be parsed, even after a retry."""


def judge(question: str, reference: str, answer: str, *, llm: LLM) -> JudgeResult:
    """Grade ``answer`` to ``question`` against ``reference``.

    ``reference`` may be a written answer or excerpts of the source text (CUAD clause spans,
    joined with `` ... ``). Raises JudgeError when the output can't be parsed after one retry.
    """
    raise NotImplementedError
