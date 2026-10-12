"""Task evaluation set (Phase 5.5): one request per line in eval/datasets/<name>_tasks.jsonl.

Lookups are scored by `eval/dataset.py` items; these items score everything else the document
agent does (list, summarize, compare, extract, transform, open multi-step tasks, clarifying).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from vaultmind.agent.types import TaskKind
from vaultmind.eval.dataset import Turn


class TaskItem(BaseModel):
    """One request and what a correct response contains.

    Rules: ``id`` is non-blank. A ``clarify`` item expects a clarifying question, so it has no
    ``key_facts``; every other item has at least one. Unknown keys are rejected.
    """

    model_config = ConfigDict(extra="forbid")

    id: str  # stable, e.g. "summarize-03"; repeats of a run are matched by it
    request: str
    task: TaskKind  # the route a correct planner picks (router label)
    documents: list[str] = []  # vault-relative files the request is about, in the order named
    key_facts: list[str] = []  # each must appear in a correct answer (key-facts judge)
    clarify: bool = False  # the right response is a clarifying question, not an answer
    language: str = "en"  # ISO 639-1 the answer should be in
    history: list[Turn] = []  # earlier chat turns, oldest first
    tags: list[str] = []  # e.g. "hebrew", "cross-lingual", "multi-step", "injection"
    forbidden: list[str] = []  # strings the answer must not contain (planted payloads)

    @model_validator(mode="after")
    def _check(self) -> TaskItem:
        if not self.id.strip():
            raise ValueError("id must not be blank")
        if self.clarify and self.key_facts:
            raise ValueError("clarify items have no key_facts")
        if not self.clarify and not self.key_facts:
            raise ValueError("items that expect an answer need at least one key fact")
        return self


def load_tasks(path: Path) -> list[TaskItem]:
    """Read a UTF-8 JSONL task set. Blank lines are skipped. Raises ValueError naming the file
    and 1-based line on invalid JSON, schema errors or a repeated id."""
    items: list[TaskItem] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = TaskItem.model_validate(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{lineno}: invalid JSON: {e.msg}") from e
        except ValidationError as e:
            raise ValueError(f"{path}:{lineno}: invalid task item: {e}") from e
        if item.id in seen:
            raise ValueError(f"{path}:{lineno}: repeated id {item.id!r}")
        seen.add(item.id)
        items.append(item)
    return items


def save_tasks(items: Sequence[TaskItem], path: Path) -> None:
    """Write one compact JSON object per line (UTF-8, non-ASCII kept, defaults left out)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            item.model_dump(mode="json", exclude_defaults=True),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for item in items
    ]
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8", newline="\n")
