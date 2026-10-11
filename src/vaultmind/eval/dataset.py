"""Evaluation dataset: one JSON object per line in eval/datasets/<name>.jsonl."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel


class EvalItem(BaseModel):
    question: str
    answer: str  # reference answer; empty for unanswerable items
    source_file: str | None  # file name relative to the vault; None if unanswerable
    source_page: int | None
    answerable: bool = True
    tags: list[str] = []  # e.g. "exact-term", "multi-hop", "follow-up"


def load_dataset(path: Path) -> list[EvalItem]:
    raise NotImplementedError


def save_dataset(items: Sequence[EvalItem], path: Path) -> None:
    raise NotImplementedError
