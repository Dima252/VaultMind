"""Task eval set format (Phase 5.5)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from vaultmind.eval.dataset import Turn
from vaultmind.eval.tasks import TaskItem, load_tasks, save_tasks

TASKS_EXAMPLE = Path(__file__).resolve().parent.parent / "eval" / "datasets" / "tasks_example.jsonl"


def test_tasks_example_loads_and_covers_every_field() -> None:
    items = load_tasks(TASKS_EXAMPLE)
    assert len(items) == 8
    for name in TaskItem.model_fields:
        assert any(name in item.model_fields_set for item in items), name
    assert {item.task for item in items} >= {"list", "summarize", "compare", "extract", "open"}
    assert any(item.clarify for item in items) and any(item.forbidden for item in items)
    assert any(item.language == "he" for item in items)


def test_round_trip_keeps_hebrew_and_drops_defaults(tmp_path: Path) -> None:
    items = [
        TaskItem(
            id="summarize-01",
            request="סכם את הסילבוס",
            task="summarize",
            documents=["personal/syllabus.pdf"],
            key_facts=["three assignments"],
            language="he",
            tags=["hebrew"],
        ),
        TaskItem(
            id="clarify-01",
            request="Summarize the syllabus",
            task="summarize",
            clarify=True,
            history=[Turn(role="user", content="hi"), Turn(role="assistant", content="Hello!")],
        ),
    ]
    path = tmp_path / "t_tasks.jsonl"
    save_tasks(items, path)
    text = path.read_text(encoding="utf-8")
    assert "סכם" in text
    assert '"forbidden"' not in text
    assert load_tasks(path) == items


@pytest.mark.parametrize(
    "fields",
    [
        {"id": " ", "key_facts": ["x"]},
        {"id": "a", "clarify": True, "key_facts": ["x"]},
        {"id": "a"},
        {"id": "a", "key_facts": ["x"], "task": "dance"},
        {"id": "a", "key_facts": ["x"], "extra": 1},
    ],
)
def test_invalid_items(fields: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        TaskItem.model_validate({"request": "r", "task": "list", **fields})


def test_load_reports_line_and_repeated_ids(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    line = '{"id":"a","request":"r","task":"list","key_facts":["x"]}'
    path.write_text(f"{line}\n\n{line}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"t\.jsonl:3: repeated id 'a'"):
        load_tasks(path)
    path.write_text("{oops\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r":1: invalid JSON"):
        load_tasks(path)
