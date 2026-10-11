from pathlib import Path

import pytest
from pydantic import ValidationError

from vaultmind.eval.dataset import EvalItem, load_dataset, save_dataset

EXAMPLE = Path(__file__).resolve().parent.parent / "eval" / "datasets" / "example.jsonl"


def _items() -> list[EvalItem]:
    return [
        EvalItem(
            question="What is the refund window?",
            answer="30 days.",
            source_file="policies/refunds.pdf",
            source_page=2,
            tags=["fact"],
        ),
        EvalItem(
            question="מה שעות הפתיחה של הספרייה?",
            answer="הספרייה פתוחה מ-9:00 עד 17:00.",
            source_file="מסמכים/ספרייה.pdf",
            source_page=None,
            tags=["hebrew", "exact-term"],
        ),
        EvalItem(
            question="Who won the 2090 world cup?",
            answer="",
            source_file=None,
            source_page=None,
            answerable=False,
        ),
    ]


def test_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "ds.jsonl"
    items = _items()
    save_dataset(items, path)
    assert load_dataset(path) == items


def test_save_format(tmp_path: Path) -> None:
    path = tmp_path / "ds.jsonl"
    save_dataset(_items(), path)
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    assert text.endswith("\n")
    assert b"\r\n" not in raw
    lines = text.splitlines()
    assert len(lines) == 3
    assert "מה שעות" in lines[1]  # non-ASCII kept as is, not \u-escaped
    assert "\\u" not in text
    assert lines[0] == (
        '{"question":"What is the refund window?","answer":"30 days.",'
        '"source_file":"policies/refunds.pdf","source_page":2,"answerable":true,"tags":["fact"],'
        '"evidence":[],"forbidden":[]}'
    )


def test_save_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    save_dataset([], path)
    assert load_dataset(path) == []


def test_blank_lines_skipped(tmp_path: Path) -> None:
    path = tmp_path / "ds.jsonl"
    save_dataset(_items()[:1], path)
    line = path.read_text(encoding="utf-8").strip()
    path.write_text(f"\n{line}\n   \n\n{line}\n\n", encoding="utf-8")
    assert len(load_dataset(path)) == 2


def test_invalid_json_names_file_and_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    save_dataset(_items()[:1], path)
    good = path.read_text(encoding="utf-8").strip()
    path.write_text(f"{good}\n\n{{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.jsonl:3: invalid JSON") as exc:
        load_dataset(path)
    assert str(path) in str(exc.value)


def test_schema_error_names_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    save_dataset(_items()[:1], path)
    good = path.read_text(encoding="utf-8").strip()
    bad = '{"question":"q","answer":"","source_file":"a.pdf","source_page":0}'
    path.write_text(f"{good}\n{bad}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.jsonl:2: invalid eval item.*source_page"):
        load_dataset(path)


def test_non_object_line_is_schema_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.jsonl:1: invalid eval item"):
        load_dataset(path)


def test_bom_tolerated(tmp_path: Path) -> None:
    path = tmp_path / "bom.jsonl"
    save_dataset(_items(), path)
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    assert load_dataset(path) == _items()


def test_example_dataset_loads() -> None:
    items = load_dataset(EXAMPLE)
    assert len(items) == 5
    assert any(not item.answerable for item in items)


# --- validator --------------------------------------------------------------


def test_unanswerable_rejects_source_file() -> None:
    with pytest.raises(ValidationError, match="unanswerable"):
        EvalItem(question="q", answer="", source_file="a.pdf", source_page=None, answerable=False)


def test_unanswerable_rejects_source_page() -> None:
    with pytest.raises(ValidationError, match="unanswerable"):
        EvalItem(question="q", answer="", source_file=None, source_page=1, answerable=False)


def test_unanswerable_rejects_evidence() -> None:
    with pytest.raises(ValidationError, match="no evidence"):
        EvalItem(
            question="q",
            answer="",
            source_file=None,
            source_page=None,
            answerable=False,
            evidence=["span"],
        )


@pytest.mark.parametrize("source_file", [None, "", "   "])
def test_answerable_requires_source_file(source_file: str | None) -> None:
    with pytest.raises(ValidationError, match="require source_file"):
        EvalItem(question="q", answer="a", source_file=source_file, source_page=1)


@pytest.mark.parametrize("page", [0, -3])
def test_source_page_must_be_positive(page: int) -> None:
    with pytest.raises(ValidationError, match="source_page must be >= 1"):
        EvalItem(question="q", answer="a", source_file="a.pdf", source_page=page)


def test_valid_items() -> None:
    item = EvalItem(question="q", answer="a", source_file="a.pdf", source_page=1)
    assert item.answerable and item.tags == []
    assert EvalItem(question="q", answer="a", source_file="a.pdf", source_page=None)
    assert EvalItem(question="q", answer="", source_file=None, source_page=None, answerable=False)


def test_unknown_key_rejected() -> None:
    with pytest.raises(ValidationError):
        EvalItem.model_validate(
            {"question": "q", "answer": "a", "source_file": "a.pdf", "source_page": 1, "tag": []}
        )
