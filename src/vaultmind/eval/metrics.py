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
