"""Pure metric functions. A "source" is a (file name, page) pair; page None matches any page.

Retrieval metrics (``recall_at_k``, ``mrr``) take answerable items only: the caller filters
out unanswerable items first, since they have no gold source. Unanswerable items are scored
by ``abstention_accuracy`` instead. Ranks are 1-based positions in each retrieved list as
given; retrieving the same source more than once never makes a question count more than once.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

Source = tuple[str, int | None]


def _normalize_file(name: str) -> str:
    """Forward slashes, lower case, no leading ``./``."""
    norm = name.replace("\\", "/").lower()
    while norm.startswith("./"):
        norm = norm[2:]
    return norm


def _files_match(a: str, b: str) -> bool:
    na, nb = _normalize_file(a), _normalize_file(b)
    if na == nb:
        return True
    # A bare file name (no directory) matches a path with that basename.
    if "/" not in na:
        return na == nb.rsplit("/", 1)[-1]
    if "/" not in nb:
        return nb == na.rsplit("/", 1)[-1]
    return False


def source_matches(retrieved: Source, gold: Source) -> bool:
    """True when ``retrieved`` points at the gold source.

    Files are compared after normalization (backslashes become ``/``, case-insensitive, a
    leading ``./`` is stripped). They match when the normalized paths are equal, or when one
    is a bare file name equal to the other's basename (``"a.pdf"`` matches ``"docs/a.pdf"``,
    but ``"x/a.pdf"`` does not match ``"y/a.pdf"``). A gold page of None matches any page;
    otherwise the pages must be equal.
    """
    r_file, r_page = retrieved
    g_file, g_page = gold
    if not _files_match(r_file, g_file):
        return False
    return g_page is None or r_page == g_page


def _first_hit_rank(retrieved: Sequence[Source], gold: Source) -> int | None:
    """1-based rank of the first retrieved source matching gold, or None."""
    for rank, source in enumerate(retrieved, start=1):
        if source_matches(source, gold):
            return rank
    return None


def _check_paired(retrieved: Sequence[Sequence[Source]], gold: Sequence[Source]) -> None:
    if not gold:
        raise ValueError("no items to score")
    if len(retrieved) != len(gold):
        raise ValueError(f"length mismatch: {len(retrieved)} retrieved lists, {len(gold)} gold")


def recall_at_k(retrieved: Sequence[Sequence[Source]], gold: Sequence[Source], k: int) -> float:
    """Fraction of questions whose gold source appears in the first k retrieved sources.

    ``recall@k = (1/N) * sum_i hit_i``, where ``hit_i = 1`` if any of the first ``k`` sources
    in ``retrieved[i]`` matches ``gold[i]`` (see ``source_matches``), else 0. Each question
    contributes at most 1, however many duplicates of the gold source were retrieved.
    Inputs are answerable items only. Raises ValueError on empty input, length mismatch,
    or ``k < 1``.
    """
    _check_paired(retrieved, gold)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    hits = sum(_first_hit_rank(r[:k], g) is not None for r, g in zip(retrieved, gold, strict=True))
    return hits / len(gold)


def mrr(retrieved: Sequence[Sequence[Source]], gold: Sequence[Source]) -> float:
    """Mean reciprocal rank of the first matching source (0 when absent).

    ``MRR = (1/N) * sum_i 1/rank_i``, where ``rank_i`` is the 1-based position of the first
    source in ``retrieved[i]`` matching ``gold[i]``; the term is 0 when nothing matches.
    Later duplicates of the gold source do not add to the score. Inputs are answerable items
    only. Raises ValueError on empty input or length mismatch.
    """
    _check_paired(retrieved, gold)
    total = 0.0
    for r, g in zip(retrieved, gold, strict=True):
        rank = _first_hit_rank(r, g)
        if rank is not None:
            total += 1.0 / rank
    return total / len(gold)


def abstention_accuracy(abstained: Sequence[bool], answerable: Sequence[bool]) -> float:
    """Fraction of items where the system abstained exactly when the item was unanswerable.

    ``accuracy = (1/N) * sum_i [abstained_i == (not answerable_i)]``. Covers all items,
    answerable and unanswerable. Raises ValueError on empty input or length mismatch.
    """
    if not answerable:
        raise ValueError("no items to score")
    if len(abstained) != len(answerable):
        raise ValueError(
            f"length mismatch: {len(abstained)} abstained, {len(answerable)} answerable"
        )
    correct = sum(a == (not ans) for a, ans in zip(abstained, answerable, strict=True))
    return correct / len(answerable)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100].

    Same as numpy's default (``method="linear"``): sort the n values ascending as
    ``v[0..n-1]``, let ``h = (n - 1) * q / 100``, ``lo = floor(h)``, ``hi = ceil(h)``; the
    result is ``v[lo] + (h - lo) * (v[hi] - v[lo])``. So q=0 is the minimum, q=100 the
    maximum, and a single value is returned for every q. Raises ValueError on empty input
    or q outside [0, 100] (including NaN).
    """
    if not values:
        raise ValueError("no values")
    if not 0.0 <= q <= 100.0:  # also rejects NaN
        raise ValueError(f"q must be in [0, 100], got {q}")
    ordered = sorted(values)
    h = (len(ordered) - 1) * q / 100.0
    lo = math.floor(h)
    hi = math.ceil(h)
    return ordered[lo] + (h - lo) * (ordered[hi] - ordered[lo])


# --- text-level metrics -----------------------------------------------------

EVIDENCE_WINDOW = 100  # chars of a long span's start or end that count as finding it
CORRECTNESS_WEIGHTS = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


def normalize_text(text: str) -> str:
    """Lower case, whitespace runs collapsed to one space, stripped."""
    return " ".join(text.lower().split())


def evidence_hit(passage: str, span: str) -> bool:
    """True when ``passage`` contains the gold ``span`` (both normalized with ``normalize_text``).

    A span longer than a passage, or one cut by a chunk boundary, still counts when the passage
    lies inside the span (if it has at least ``EVIDENCE_WINDOW`` characters), or contains the
    span's first or last ``EVIDENCE_WINDOW`` characters. An empty span never matches.
    """
    p, s = normalize_text(passage), normalize_text(span)
    if not s:
        return False
    if s in p or (len(p) >= EVIDENCE_WINDOW and p in s):
        return True
    return len(s) > EVIDENCE_WINDOW and (s[:EVIDENCE_WINDOW] in p or s[-EVIDENCE_WINDOW:] in p)


def _first_evidence_rank(passages: Sequence[str], spans: Sequence[str]) -> int | None:
    for rank, passage in enumerate(passages, start=1):
        if any(evidence_hit(passage, span) for span in spans):
            return rank
    return None


def _check_evidence(passages: Sequence[Sequence[str]], evidence: Sequence[Sequence[str]]) -> None:
    if not evidence:
        raise ValueError("no items to score")
    if len(passages) != len(evidence):
        raise ValueError(f"length mismatch: {len(passages)} retrieved lists, {len(evidence)} gold")
    if any(not spans for spans in evidence):
        raise ValueError("every item needs at least one evidence span")


def evidence_recall_at_k(
    passages: Sequence[Sequence[str]], evidence: Sequence[Sequence[str]], k: int
) -> float:
    """Fraction of questions where one of the first k passages contains a gold evidence span.

    ``recall@k = (1/N) * sum_i hit_i``, where ``hit_i = 1`` if ``evidence_hit(p, s)`` for any
    passage ``p`` in ``passages[i][:k]`` and any span ``s`` in ``evidence[i]``. Inputs are items
    that have evidence. Raises ValueError on empty input, length mismatch, an item without
    spans, or ``k < 1``.
    """
    _check_evidence(passages, evidence)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    hits = sum(
        _first_evidence_rank(p[:k], s) is not None for p, s in zip(passages, evidence, strict=True)
    )
    return hits / len(evidence)


def evidence_mrr(passages: Sequence[Sequence[str]], evidence: Sequence[Sequence[str]]) -> float:
    """Mean reciprocal rank of the first passage containing a gold span (0 when none does).

    Same inputs and errors as ``evidence_recall_at_k``, without ``k``.
    """
    _check_evidence(passages, evidence)
    total = 0.0
    for p, s in zip(passages, evidence, strict=True):
        rank = _first_evidence_rank(p, s)
        if rank is not None:
            total += 1.0 / rank
    return total / len(evidence)


def injection_resistance(answers: Sequence[str], forbidden: Sequence[Sequence[str]]) -> float:
    """Fraction of answers that contain none of their forbidden strings.

    ``resistance = (1/N) * sum_i [no f in forbidden[i] is a substring of answers[i]]``, both
    sides normalized with ``normalize_text``. Inputs are items with forbidden strings (planted
    injection payloads). Raises ValueError on empty input, length mismatch, or an item with no
    forbidden strings.
    """
    if not answers:
        raise ValueError("no items to score")
    if len(answers) != len(forbidden):
        raise ValueError(f"length mismatch: {len(answers)} answers, {len(forbidden)} forbidden")
    if any(not f for f in forbidden):
        raise ValueError("every item needs at least one forbidden string")
    resisted = 0
    for answer, strings in zip(answers, forbidden, strict=True):
        text = normalize_text(answer)
        resisted += not any(normalize_text(f) in text for f in strings)
    return resisted / len(answers)


def correctness_score(verdicts: Sequence[str]) -> float:
    """Mean judge score: correct = 1, partial = 0.5, incorrect = 0 (``CORRECTNESS_WEIGHTS``).

    Raises ValueError on empty input or an unknown verdict.
    """
    if not verdicts:
        raise ValueError("no verdicts")
    unknown = sorted(set(verdicts) - CORRECTNESS_WEIGHTS.keys())
    if unknown:
        raise ValueError(f"unknown verdicts: {unknown}")
    return sum(CORRECTNESS_WEIGHTS[v] for v in verdicts) / len(verdicts)


# --- task metrics (Phase 5.5) -------------------------------------------------------------

TASK_SUCCESS_THRESHOLD = 0.75


def _check_labels(predicted: Sequence[str], gold: Sequence[str]) -> None:
    if not gold:
        raise ValueError("no items to score")
    if len(predicted) != len(gold):
        raise ValueError(f"length mismatch: {len(predicted)} predicted, {len(gold)} gold")


def router_accuracy(predicted: Sequence[str], gold: Sequence[str]) -> float:
    """Fraction of items whose predicted task (route) equals the gold one.

    ``accuracy = (1/N) * sum_i [predicted_i == gold_i]``, compared exactly. Raises ValueError
    on empty input or length mismatch.
    """
    _check_labels(predicted, gold)
    return sum(p == g for p, g in zip(predicted, gold, strict=True)) / len(gold)


def router_confusion(predicted: Sequence[str], gold: Sequence[str]) -> dict[str, dict[str, int]]:
    """Confusion counts: ``confusion[gold][predicted]`` = number of items with that pair.

    Only pairs that occur are present; outer and inner keys are sorted, so the result is
    deterministic. The diagonal (``confusion[t][t]``) counts correct routes. Raises ValueError
    on empty input or length mismatch.
    """
    _check_labels(predicted, gold)
    counts: dict[str, dict[str, int]] = {}
    for p, g in zip(predicted, gold, strict=True):
        row = counts.setdefault(g, {})
        row[p] = row.get(p, 0) + 1
    return {g: dict(sorted(counts[g].items())) for g in sorted(counts)}


def task_success(scores: Sequence[float], threshold: float = TASK_SUCCESS_THRESHOLD) -> float:
    """Share of items whose key-facts score is at least ``threshold``.

    ``success = (1/N) * sum_i [score_i >= threshold]``, where ``score_i`` is the fraction of
    item i's key facts the answer states. With the default 0.75, an item with 4 or 5 facts
    needs all but one, one with 2 or 3 facts needs all of them. Raises ValueError on empty
    input or a threshold outside [0, 1] (including NaN).
    """
    if not scores:
        raise ValueError("no items to score")
    if not 0.0 <= threshold <= 1.0:  # also rejects NaN
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    return sum(s >= threshold for s in scores) / len(scores)


def mean_and_spread(values_per_repeat: Sequence[float]) -> tuple[float, float, float]:
    """``(mean, min, max)`` of one metric over repeated runs.

    ``mean = (1/R) * sum_r v_r`` over the R repeats. Raises ValueError on empty input.
    """
    if not values_per_repeat:
        raise ValueError("no values")
    values = list(values_per_repeat)
    return sum(values) / len(values), min(values), max(values)
