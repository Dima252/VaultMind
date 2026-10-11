"""Optional LLM checks around the answer (Phase 4): contradictions and groundedness.

Each is one structured LLM call (`schema=`), off by default in config, and evaluated with its
own eval tags (`contradiction` items; answers to unanswerable questions for groundedness).
"""

from __future__ import annotations

from collections.abc import Sequence

from vaultmind.agent.baseline import Source
from vaultmind.agent.types import Contradiction, UnsupportedClaim
from vaultmind.core.providers import LLM


class CheckError(Exception):
    """The check's LLM output could not be parsed, even after one retry."""


def detect_contradictions(
    question: str, sources: Sequence[Source], *, llm: LLM
) -> list[Contradiction]:
    """Pairs of sources from different files that disagree on a fact relevant to the question."""
    raise NotImplementedError


def check_groundedness(
    answer: str, sources: Sequence[Source], *, llm: LLM
) -> list[UnsupportedClaim]:
    """Claims in `answer` that the sources it cites (`[n]`) don't support. Empty when grounded."""
    raise NotImplementedError
