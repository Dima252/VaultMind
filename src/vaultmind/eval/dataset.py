"""Evaluation dataset: one JSON object per line in eval/datasets/<name>.jsonl."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator


class EvalItem(BaseModel):
    """One question. Answerable items name a gold source; unanswerable ones name none.

    Rules: if ``answerable`` is false, ``source_file`` and ``source_page`` must be None and
    ``evidence`` empty.
    If ``answerable`` is true, ``source_file`` is required (non-blank). ``source_page``,
    when given, is 1-based (>= 1). Unknown keys are rejected so typos surface early.
    """

    model_config = ConfigDict(extra="forbid")

    question: str
    answer: str  # reference answer; empty for unanswerable items
    source_file: str | None  # file name relative to the vault; None if unanswerable
    source_page: int | None
    answerable: bool = True
    tags: list[str] = []  # e.g. "exact-term", "multi-hop", "follow-up"
    evidence: list[str] = []  # gold text spans from the source, when known (span-level recall)
    forbidden: list[str] = []  # strings the answer must not contain (planted injection payloads)

    @model_validator(mode="after")
    def _check_source(self) -> EvalItem:
        if self.source_page is not None and self.source_page < 1:
            raise ValueError(f"source_page must be >= 1, got {self.source_page}")
        if self.answerable:
            if self.source_file is None or not self.source_file.strip():
                raise ValueError("answerable items require source_file")
        elif self.source_file is not None or self.source_page is not None:
            raise ValueError("unanswerable items must have source_file and source_page set to None")
        elif self.evidence:
            raise ValueError("unanswerable items must have no evidence")
        return self


def _describe(error: ValidationError) -> str:
    """One-line summary of a pydantic error, e.g. ``source_page: Value error, ...``."""
    parts = []
    for err in error.errors():
        loc = ".".join(str(p) for p in err["loc"])
        parts.append(f"{loc}: {err['msg']}" if loc else err["msg"])
    return "; ".join(parts)


def load_dataset(path: Path) -> list[EvalItem]:
    """Read a UTF-8 JSONL dataset. Blank lines are skipped.

    Raises ValueError naming the file and 1-based line number on invalid JSON or schema errors.
    """
    items: list[EvalItem] = []
    # utf-8-sig tolerates a BOM written by Windows editors; it is otherwise plain UTF-8.
    text = path.read_text(encoding="utf-8-sig")
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{lineno}: invalid JSON: {e.msg}") from e
        try:
            items.append(EvalItem.model_validate(data))
        except ValidationError as e:
            raise ValueError(f"{path}:{lineno}: invalid eval item: {_describe(e)}") from e
    return items


def save_dataset(items: Sequence[EvalItem], path: Path) -> None:
    """Write one compact JSON object per line (UTF-8, non-ASCII kept as is, trailing newline)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(item.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
        for item in items
    ]
    content = "".join(line + "\n" for line in lines)
    path.write_text(content, encoding="utf-8", newline="\n")
