"""Prompt-injection guard and the answer prompt (Phase 4).

With the guard on, passages go into the prompt as clearly delimited data that is never to be
followed as instructions, and instruction-like text in them is detected so it can be flagged.
The same patterns will back the ingest-time scanner (Phase 6).

PLACEHOLDERS: `find_injections` finds nothing and `build_answer_messages` ignores `guard` and
`contradictions` until the safety agent implements them. Their signatures are the contract.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from vaultmind.agent.baseline import Source, build_messages
from vaultmind.agent.types import Contradiction
from vaultmind.core.providers import Message

INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = ()


def find_injections(text: str) -> list[str]:
    """Short descriptions of instruction-like text found in `text` (empty when clean)."""
    return []


def build_answer_messages(
    question: str,
    sources: Sequence[Source],
    *,
    guard: bool,
    history: Sequence[Message] = (),
    contradictions: Sequence[Contradiction] = (),
) -> list[Message]:
    """Messages for the answer: system prompt, earlier turns, then sources + question.

    The system prompt keeps the `[n]` citation rule and the exact `NOT_FOUND` reply from
    `agent.baseline`. With `contradictions`, the answer must state both claims and cite each.
    """
    system, user = build_messages(question, sources)
    return [system, *history, user]
