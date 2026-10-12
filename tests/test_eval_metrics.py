import math
from collections.abc import Callable

import pytest

from vaultmind.eval.metrics import (
    Source,
    abstention_accuracy,
    mean_and_spread,
    mrr,
    percentile,
    recall_at_k,
    router_accuracy,
    router_confusion,
    source_matches,
    task_success,
)

# --- source_matches ---------------------------------------------------------


@pytest.mark.parametrize(
    ("retrieved", "gold"),
    [
        ("docs/a.pdf", "docs/a.pdf"),
        ("docs\\a.pdf", "docs/a.pdf"),
        ("Docs/A.PDF", "docs/a.pdf"),
        ("./docs/a.pdf", "docs/a.pdf"),
        ("docs/a.pdf", ".\\docs\\a.pdf"),
        ("a.pdf", "docs/sub/a.pdf"),
        ("docs\\sub\\A.pdf", "a.pdf"),
        ("./a.pdf", "a.pdf"),
    ],
)
def test_source_matches_path_normalization(retrieved: str, gold: str) -> None:
    assert source_matches((retrieved, 3), (gold, 3))


@pytest.mark.parametrize(
    ("retrieved", "gold"),
    [
        ("x/a.pdf", "y/a.pdf"),
        ("docs/a.pdf", "docs/b.pdf"),
        ("a.pdf", "docs/ba.pdf"),
        ("docs/a.pdf", "a.pdf.bak"),
        ("docs", "docs/a.pdf"),
    ],
)
def test_source_matches_different_files(retrieved: str, gold: str) -> None:
    assert not source_matches((retrieved, 1), (gold, 1))


def test_source_matches_pages() -> None:
    assert source_matches(("a.pdf", 2), ("a.pdf", 2))
    assert not source_matches(("a.pdf", 2), ("a.pdf", 3))
    # Gold page None matches any page, including None.
    assert source_matches(("a.pdf", 7), ("a.pdf", None))
    assert source_matches(("a.pdf", None), ("a.pdf", None))
    # A retrieved page of None does not satisfy a specific gold page.
    assert not source_matches(("a.pdf", None), ("a.pdf", 1))


# --- recall_at_k ------------------------------------------------------------

G: Source = ("g.pdf", 5)
X: Source = ("x.pdf", 1)
Y: Source = ("y.pdf", 2)


def test_recall_hit_exactly_at_k_and_just_past() -> None:
    retrieved = [[X, Y, G]]
    assert recall_at_k(retrieved, [G], k=3) == 1.0
    assert recall_at_k(retrieved, [G], k=2) == 0.0
    # k larger than the list is fine.
    assert recall_at_k(retrieved, [G], k=10) == 1.0


def test_recall_hand_computed() -> None:
    retrieved = [
        [G, X],  # hit at rank 1
        [X, ("G.PDF", 5)],  # hit at rank 2 (case-insensitive)
        [X, Y],  # miss
        [],  # nothing retrieved: miss
    ]
    gold = [G, G, G, G]
    assert recall_at_k(retrieved, gold, k=1) == 1 / 4
    assert recall_at_k(retrieved, gold, k=2) == 2 / 4


def test_recall_gold_page_none() -> None:
    retrieved = [[X, ("docs/g.pdf", 99)]]
    assert recall_at_k(retrieved, [("g.pdf", None)], k=2) == 1.0
    assert recall_at_k(retrieved, [("g.pdf", 5)], k=2) == 0.0


def test_recall_duplicates_count_once() -> None:
    retrieved = [[G, G, G], [X, X, X]]
    assert recall_at_k(retrieved, [G, G], k=3) == 0.5
    # Duplicate non-matching sources still occupy ranks.
    assert recall_at_k([[X, X, G]], [G], k=2) == 0.0


@pytest.mark.parametrize("k", [0, -1])
def test_recall_rejects_bad_k(k: int) -> None:
    with pytest.raises(ValueError, match="k must be"):
        recall_at_k([[G]], [G], k=k)


def test_recall_rejects_empty_and_mismatch() -> None:
    with pytest.raises(ValueError):
        recall_at_k([], [], k=1)
    with pytest.raises(ValueError, match="length mismatch"):
        recall_at_k([[G], [G]], [G], k=1)


# --- mrr --------------------------------------------------------------------


def test_mrr_hand_computed() -> None:
    retrieved = [
        [G, X, Y],  # 1/1
        [X, G, Y],  # 1/2
        [X, Y, ("./g.pdf", 5)],  # 1/3
        [X, Y],  # 0
    ]
    expected = (1 + 1 / 2 + 1 / 3 + 0) / 4
    assert mrr(retrieved, [G, G, G, G]) == pytest.approx(expected)


def test_mrr_duplicates_use_first_hit_only() -> None:
    # First hit at rank 2; later duplicates add nothing.
    assert mrr([[X, G, G, G]], [G]) == 0.5
    # Duplicated misses before the hit push its rank back.
    assert mrr([[X, X, X, G]], [G]) == 0.25


def test_mrr_gold_page_none() -> None:
    assert mrr([[("g.pdf", 1), X]], [("g.pdf", None)]) == 1.0


def test_mrr_all_miss_and_empty_lists() -> None:
    assert mrr([[X], []], [G, G]) == 0.0


def test_mrr_rejects_empty_and_mismatch() -> None:
    with pytest.raises(ValueError):
        mrr([], [])
    with pytest.raises(ValueError, match="length mismatch"):
        mrr([[G]], [G, G])


# --- abstention_accuracy ----------------------------------------------------


def test_abstention_all_correct() -> None:
    assert abstention_accuracy([False, True, False], [True, False, True]) == 1.0


def test_abstention_all_wrong() -> None:
    assert abstention_accuracy([True, False, True], [True, False, True]) == 0.0


def test_abstention_mixed() -> None:
    # Correct: item 0 (answered answerable), item 3 (abstained unanswerable).
    # Wrong: item 1 (abstained answerable), item 2 (answered unanswerable).
    abstained = [False, True, False, True]
    answerable = [True, True, False, False]
    assert abstention_accuracy(abstained, answerable) == 0.5


def test_abstention_rejects_empty_and_mismatch() -> None:
    with pytest.raises(ValueError):
        abstention_accuracy([], [])
    with pytest.raises(ValueError, match="length mismatch"):
        abstention_accuracy([True], [True, False])


# --- percentile -------------------------------------------------------------

VALUES = [15.0, 1.0, 10.0, 5.0, 3.0]  # sorted: 1, 3, 5, 10, 15


@pytest.mark.parametrize(
    ("q", "expected"),
    [
        (0, 1.0),
        (25, 3.0),  # h = 1
        (50, 5.0),  # h = 2
        (60, 7.0),  # h = 2.4 -> 5 + 0.4 * (10 - 5)
        (95, 14.0),  # h = 3.8 -> 10 + 0.8 * (15 - 10)
        (100, 15.0),
    ],
)
def test_percentile_hand_computed(q: float, expected: float) -> None:
    assert percentile(VALUES, q) == pytest.approx(expected)


def test_percentile_even_count_median() -> None:
    assert percentile([4.0, 1.0, 2.0, 3.0], 50) == 2.5  # h = 1.5


def test_percentile_single_value() -> None:
    for q in (0, 50, 95, 100):
        assert percentile([42.0], q) == 42.0


def test_percentile_does_not_mutate_input() -> None:
    values = [3.0, 1.0, 2.0]
    percentile(values, 50)
    assert values == [3.0, 1.0, 2.0]


@pytest.mark.parametrize("q", [-0.1, 100.1, math.nan])
def test_percentile_rejects_bad_q(q: float) -> None:
    with pytest.raises(ValueError, match="q must be"):
        percentile([1.0, 2.0], q)


def test_percentile_rejects_empty() -> None:
    with pytest.raises(ValueError):
        percentile([], 50)


# --- task metrics -------------------------------------------------------------

PREDICTED = ["list", "summarize", "lookup", "compare", "summarize", "open"]
GOLD = ["list", "summarize", "summarize", "compare", "compare", "open"]


def test_router_accuracy_hand_computed() -> None:
    assert router_accuracy(PREDICTED, GOLD) == pytest.approx(4 / 6)
    assert router_accuracy(["chat"], ["chat"]) == 1.0
    assert router_accuracy(["chat"], ["lookup"]) == 0.0


def test_router_confusion_hand_computed() -> None:
    confusion = router_confusion(PREDICTED, GOLD)
    assert confusion == {
        "compare": {"compare": 1, "summarize": 1},
        "list": {"list": 1},
        "open": {"open": 1},
        "summarize": {"lookup": 1, "summarize": 1},
    }
    assert list(confusion) == sorted(confusion)  # deterministic order
    assert sum(n for row in confusion.values() for n in row.values()) == len(GOLD)
    diagonal = sum(row.get(gold, 0) for gold, row in confusion.items())
    assert diagonal / len(GOLD) == router_accuracy(PREDICTED, GOLD)


@pytest.mark.parametrize("fn", [router_accuracy, router_confusion])
def test_router_metrics_reject_bad_input(fn: Callable[[list[str], list[str]], object]) -> None:
    with pytest.raises(ValueError, match="no items"):
        fn([], [])
    with pytest.raises(ValueError, match="length mismatch"):
        fn(["a"], ["a", "b"])


def test_task_success_threshold() -> None:
    scores = [1.0, 0.75, 2 / 3, 0.8, 0.0]
    assert task_success(scores) == pytest.approx(3 / 5)  # 1.0, 0.75 and 0.8 pass
    assert task_success(scores, threshold=1.0) == pytest.approx(1 / 5)
    assert task_success(scores, threshold=0.0) == 1.0
    assert task_success([0.5], threshold=0.5) == 1.0  # at the threshold counts


@pytest.mark.parametrize("threshold", [-0.1, 1.5, math.nan])
def test_task_success_rejects_bad_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        task_success([1.0], threshold=threshold)


def test_task_success_rejects_empty() -> None:
    with pytest.raises(ValueError):
        task_success([])


def test_mean_and_spread() -> None:
    assert mean_and_spread([0.8, 0.77, 0.83]) == pytest.approx((0.8, 0.77, 0.83))
    assert mean_and_spread([5.0]) == (5.0, 5.0, 5.0)
    assert mean_and_spread((1.0, 3.0)) == (2.0, 1.0, 3.0)
    with pytest.raises(ValueError):
        mean_and_spread([])
