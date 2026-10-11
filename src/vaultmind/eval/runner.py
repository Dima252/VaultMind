"""Eval runner: score one config on a dataset, save raw results, compare configs side by side."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from vaultmind.core.config import Config
from vaultmind.core.providers import LLM, Embedder
from vaultmind.eval.dataset import EvalItem
from vaultmind.store.index import ChunkIndex
from vaultmind.store.registry import Registry

# Every RunReport.metrics has exactly these keys, in this order (also the table's row order).
METRIC_NAMES: tuple[str, ...] = (
    "recall@1",
    "recall@5",
    "mrr",
    "evidence_recall@1",
    "evidence_recall@5",
    "evidence_mrr",
    "correctness",
    "abstention_accuracy",
    "injection_resistance",
    "retrieval_p50_ms",
    "retrieval_p95_ms",
    "generation_p50_ms",
    "generation_p95_ms",
    "total_p50_ms",
    "total_p95_ms",
)


@dataclass(frozen=True)
class RetrievedSource:
    file: str  # relative to paths.vault_dir with forward slashes; absolute if outside it
    page: int | None
    text: str  # the passage text shown to the LLM
    score: float


@dataclass(frozen=True)
class ItemResult:
    index: int  # 0-based position in the dataset
    question: str
    answerable: bool
    tags: list[str]
    retrieved: list[RetrievedSource]  # in rank order
    answer: str
    abstained: bool
    retrieval_ms: float
    generation_ms: float  # 0 when nothing was retrieved and no LLM call was made
    # Judge verdict (a Verdict value). With the judge on: "incorrect" for an abstention on an
    # answerable item (no LLM call), None for unanswerable items and judge errors.
    # With the judge off: always None.
    verdict: str | None
    judge_reason: str | None  # the judge's reason, or "judge error: ..."; None if not judged


@dataclass(frozen=True)
class RunReport:
    config_name: str
    dataset_name: str
    items: list[ItemResult]
    metrics: dict[str, float | None]  # keys = METRIC_NAMES; None when no item qualifies


def run_config(
    cfg: Config,
    items: Sequence[EvalItem],
    *,
    config_name: str,
    dataset_name: str,
    llm: LLM,
    embedder: Embedder,
    index: ChunkIndex,
    registry: Registry,
    judge_llm: LLM | None = None,
    on_item: Callable[[ItemResult], None] | None = None,
) -> RunReport:
    """Answer every item with the baseline pipeline and score the run.

    Retrieval and generation are timed separately. With ``judge_llm`` None, correctness is
    not measured: every verdict is None and the ``correctness`` metric is None.
    """
    raise NotImplementedError


def compute_metrics(
    items: Sequence[EvalItem], results: Sequence[ItemResult]
) -> dict[str, float | None]:
    """All METRIC_NAMES for paired ``items`` and ``results``."""
    raise NotImplementedError


def check_compatible(configs: Mapping[str, Config]) -> None:
    """Raise ValueError if two configs share a data_dir but differ in what the index depends on

    (embedder model or chunking settings), since they would silently score the wrong index.
    """
    raise NotImplementedError


def save_report(report: RunReport, out_dir: Path) -> Path:
    """Write ``<out_dir>/<config_name>__<dataset_name>.jsonl``: one ItemResult per line, then a
    final ``{"metrics": {...}}`` line. Returns the path."""
    raise NotImplementedError


def format_table(reports: Sequence[RunReport]) -> str:
    """ASCII table: one row per metric, one column per report (``config / dataset``)."""
    raise NotImplementedError
