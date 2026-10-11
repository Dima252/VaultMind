"""Threshold sweep replayed over saved results."""

from __future__ import annotations

import pytest

from vaultmind.eval.dataset import EvalItem
from vaultmind.eval.runner import ItemResult, RetrievedSource
from vaultmind.eval.sweep import format_sweep, sweep


def _item(question: str, evidence: list[str] | None = None) -> EvalItem:
    """Answerable when `evidence` is given."""
    answerable = evidence is not None
    return EvalItem(
        question=question,
        answer="ref" if answerable else "",
        source_file="c.txt" if answerable else None,
        source_page=None,
        answerable=answerable,
        evidence=evidence or [],
    )


def _result(
    i: int, item: EvalItem, scored: list[tuple[str, float]], *, abstained: bool = False
) -> ItemResult:
    return ItemResult(
        index=i,
        question=item.question,
        answerable=item.answerable,
        tags=[],
        retrieved=[RetrievedSource("c.txt", None, text, score) for text, score in scored],
        answer="",
        abstained=abstained,
        retrieval_ms=0.0,
        generation_ms=0.0,
        verdict=None,
        judge_reason=None,
    )


ITEMS = [
    _item("governing law?", ["laws of delaware"]),
    _item("termination?", ["terminate on notice"]),
    _item("non-compete?"),
    _item("audit rights?"),
]
RESULTS = [
    _result(0, ITEMS[0], [("the laws of Delaware apply", 0.9), ("filler", 0.2)]),
    _result(1, ITEMS[1], [("filler", 0.5), ("may terminate on notice", 0.15)]),
    _result(2, ITEMS[2], [("unrelated", 0.4)]),
    _result(3, ITEMS[3], [("unrelated", 0.05)], abstained=True),  # the LLM abstained itself
]


def test_threshold_zero_replays_the_run() -> None:
    [row] = sweep(ITEMS, RESULTS, [0.0])
    assert row.evidence_recall == pytest.approx(1.0)
    assert row.answered == pytest.approx(1.0)
    assert row.unanswerable_abstained == pytest.approx(0.5)  # only the LLM's own abstention
    assert row.abstention_accuracy == pytest.approx(0.75)


def test_raising_the_threshold_trades_recall_for_abstention() -> None:
    low, mid, high = sweep(ITEMS, RESULTS, [0.1, 0.3, 0.95])
    assert (low.evidence_recall, low.unanswerable_abstained) == (1.0, 0.5)
    # 0.3 drops the weakly scored termination clause, but item 1 still has "filler" (0.5),
    # so it is answered, without its evidence.
    assert (mid.evidence_recall, mid.answered) == (0.5, 1.0)
    assert mid.unanswerable_abstained == pytest.approx(0.5)
    # 0.95 drops everything: every item abstains.
    assert (high.evidence_recall, high.answered, high.unanswerable_abstained) == (0.0, 0.0, 1.0)
    assert high.abstention_accuracy == pytest.approx(0.5)


def test_mismatched_results_rejected() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        sweep(ITEMS, RESULTS[:2], [0.0])
    with pytest.raises(ValueError, match="item 2"):
        sweep(ITEMS, [RESULTS[0], RESULTS[0], *RESULTS[2:]], [0.0])


def test_format_sweep() -> None:
    table = format_sweep(sweep(ITEMS, RESULTS, [0.0, 0.5]))
    lines = table.splitlines()
    assert lines[0].startswith("threshold  evidence_recall@5")
    assert len(lines) == 4
    assert lines[2].split() == ["0.00", "1.000", "1.000", "0.500", "0.750"]
    assert table.isascii()
