"""Rerank threshold sweep over a saved run, without calling any model again.

Score a config with `rerank_threshold: 0` (every top-k passage kept, each with its reranker
score), then replay the threshold: at threshold t, passages scoring below t are dropped, and
an item abstains when none are left or when the LLM abstained in the run. The LLM's own
abstentions are taken from the run with all passages in context, so confirm the chosen
threshold with a real run.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from vaultmind.eval.dataset import EvalItem
from vaultmind.eval.metrics import abstention_accuracy, evidence_recall_at_k
from vaultmind.eval.runner import ItemResult


@dataclass(frozen=True)
class SweepRow:
    threshold: float
    evidence_recall: float | None  # evidence recall@5 over the kept passages
    answered: float | None  # answerable items that still get an answer
    unanswerable_abstained: float | None  # unanswerable items that abstain
    abstention_accuracy: float


def _share(flags: Sequence[bool]) -> float | None:
    return sum(flags) / len(flags) if flags else None


def sweep(
    items: Sequence[EvalItem], results: Sequence[ItemResult], thresholds: Sequence[float]
) -> list[SweepRow]:
    """One row per threshold. `results` must come from a run of `items`, in order."""
    if len(items) != len(results):
        raise ValueError(f"length mismatch: {len(items)} items, {len(results)} results")
    for n, (item, result) in enumerate(zip(items, results, strict=True), start=1):
        if item.question != result.question:
            raise ValueError(f"results don't match the dataset at item {n}")

    answerable = [it.answerable for it in items]
    with_evidence = [i for i, it in enumerate(items) if it.evidence]
    rows: list[SweepRow] = []
    for t in thresholds:
        kept = [[s.text for s in r.retrieved if s.score >= t] for r in results]
        abstained = [r.abstained or not k for r, k in zip(results, kept, strict=True)]
        rows.append(
            SweepRow(
                threshold=t,
                evidence_recall=evidence_recall_at_k(
                    [kept[i] for i in with_evidence], [items[i].evidence for i in with_evidence], 5
                )
                if with_evidence
                else None,
                answered=_share([not a for a, ok in zip(abstained, answerable, strict=True) if ok]),
                unanswerable_abstained=_share(
                    [a for a, ok in zip(abstained, answerable, strict=True) if not ok]
                ),
                abstention_accuracy=abstention_accuracy(abstained, answerable),
            )
        )
    return rows


def format_sweep(rows: Sequence[SweepRow]) -> str:
    """ASCII table, one line per threshold."""

    def cell(value: float | None) -> str:
        return "-" if value is None else f"{value:.3f}"

    header = "threshold  evidence_recall@5  answered  unanswerable_abstained  abstention_accuracy"
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r.threshold:9.2f}  {cell(r.evidence_recall):>17}  {cell(r.answered):>8}  "
            f"{cell(r.unanswerable_abstained):>22}  {cell(r.abstention_accuracy):>19}"
        )
    return "\n".join(lines)
