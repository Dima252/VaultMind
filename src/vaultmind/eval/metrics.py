"""Pure metric functions. A "source" is a (file name, page) pair; page None matches any page."""

from __future__ import annotations

from collections.abc import Sequence

Source = tuple[str, int | None]


def source_matches(retrieved: Source, gold: Source) -> bool:
    raise NotImplementedError


def recall_at_k(retrieved: Sequence[Sequence[Source]], gold: Sequence[Source], k: int) -> float:
    """Fraction of questions whose gold source appears in the first k retrieved sources."""
    raise NotImplementedError


def mrr(retrieved: Sequence[Sequence[Source]], gold: Sequence[Source]) -> float:
    """Mean reciprocal rank of the first matching source (0 when absent)."""
    raise NotImplementedError


def abstention_accuracy(abstained: Sequence[bool], answerable: Sequence[bool]) -> float:
    """Fraction of items where the system abstained exactly when the item was unanswerable."""
    raise NotImplementedError


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]."""
    raise NotImplementedError
