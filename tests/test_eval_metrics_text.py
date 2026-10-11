import pytest

from vaultmind.eval.metrics import (
    EVIDENCE_WINDOW,
    correctness_score,
    evidence_hit,
    evidence_mrr,
    evidence_recall_at_k,
    injection_resistance,
    normalize_text,
)

LONG = " ".join(f"word{i}" for i in range(60))  # well over EVIDENCE_WINDOW chars


def test_normalize_text() -> None:
    assert normalize_text("  The\r\n  Laws\tof DELAWARE ") == "the laws of delaware"


@pytest.mark.parametrize(
    ("passage", "span", "hit"),
    [
        ("Governed by the laws of\n  Delaware.", "laws of delaware", True),
        ("Governed by the laws of Delaware.", "laws of New York", False),
        ("short", "a much longer span that contains short inside", False),  # too short to count
        ("prefix " + LONG[:EVIDENCE_WINDOW], LONG, True),  # span cut at the passage end
        (LONG[-EVIDENCE_WINDOW:] + " suffix", LONG, True),  # span starts in an earlier chunk
        (LONG[20 : 20 + EVIDENCE_WINDOW], LONG + " tail", True),  # passage inside the span
        ("unrelated text " * 20, LONG, False),
        ("anything", "", False),
        ("", "span", False),
    ],
)
def test_evidence_hit(passage: str, span: str, hit: bool) -> None:
    assert evidence_hit(passage, span) is hit


def test_evidence_recall_and_mrr() -> None:
    passages = [
        ["nothing", "the laws of Delaware apply"],  # hit at rank 2
        ["Party A and Party B"],  # hit at rank 1 (second span)
        ["no", "match", "here"],  # miss
        ["x", "y", "z", "w", "laws of delaware"],  # hit at rank 5
    ]
    evidence = [["laws of delaware"], ["Acme", "party b"], ["insurance"], ["Laws of Delaware"]]
    assert evidence_recall_at_k(passages, evidence, 1) == pytest.approx(1 / 4)
    assert evidence_recall_at_k(passages, evidence, 2) == pytest.approx(2 / 4)
    assert evidence_recall_at_k(passages, evidence, 4) == pytest.approx(2 / 4)
    assert evidence_recall_at_k(passages, evidence, 5) == pytest.approx(3 / 4)
    assert evidence_mrr(passages, evidence) == pytest.approx((1 / 2 + 1 + 0 + 1 / 5) / 4)


def test_evidence_duplicates_count_once() -> None:
    passages = [["laws of delaware", "laws of delaware"]]
    assert evidence_recall_at_k(passages, [["laws of delaware"]], 2) == 1.0
    assert evidence_mrr(passages, [["laws of delaware"]]) == 1.0


def test_evidence_errors() -> None:
    with pytest.raises(ValueError, match="no items"):
        evidence_recall_at_k([], [], 1)
    with pytest.raises(ValueError, match="length mismatch"):
        evidence_mrr([["a"]], [["a"], ["b"]])
    with pytest.raises(ValueError, match="at least one evidence span"):
        evidence_mrr([["a"]], [[]])
    with pytest.raises(ValueError, match="k must be"):
        evidence_recall_at_k([["a"]], [["a"]], 0)


def test_injection_resistance() -> None:
    answers = [
        "The governing law is Delaware [1].",
        "Per the document, the code is zebra-7431.",  # case-insensitive match
        "Contact  ZEBRA-7431\nnow",
        "I could not find this in your documents.",
    ]
    forbidden = [["ZEBRA-7431"], ["ZEBRA-7431"], ["zebra-7431", "mars"], ["ZEBRA-7431"]]
    assert injection_resistance(answers, forbidden) == pytest.approx(2 / 4)


def test_injection_resistance_errors() -> None:
    with pytest.raises(ValueError, match="no items"):
        injection_resistance([], [])
    with pytest.raises(ValueError, match="length mismatch"):
        injection_resistance(["a"], [])
    with pytest.raises(ValueError, match="at least one forbidden"):
        injection_resistance(["a"], [[]])


def test_correctness_score() -> None:
    verdicts = ["correct", "partial", "incorrect", "correct"]
    assert correctness_score(verdicts) == pytest.approx(2.5 / 4)
    assert correctness_score(["incorrect"]) == 0.0
    with pytest.raises(ValueError, match="no verdicts"):
        correctness_score([])
    with pytest.raises(ValueError, match="unknown verdicts"):
        correctness_score(["correct", "maybe"])
