"""Task eval runner (Phase 5.5): the planner and recipes over a `<name>_tasks.jsonl` set.

Each request runs through the full agent (planner first, as in the app). Scores: router
accuracy against the item's task label, task success with the key-facts judge (a clarify
item succeeds when the answer asks instead), forbidden strings, and latency per task type.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from vaultmind.agent.graph import run_agent
from vaultmind.agent.types import TASKS
from vaultmind.core.config import Config
from vaultmind.core.providers import LLM, Embedder, Reranker
from vaultmind.eval.judge import JudgeError, judge_clarify, judge_key_facts
from vaultmind.eval.metrics import TASK_SUCCESS_THRESHOLD, percentile, router_accuracy
from vaultmind.eval.runner import write_report
from vaultmind.eval.tasks import TaskItem
from vaultmind.store.cards import CardStore
from vaultmind.store.index import ChunkIndex
from vaultmind.store.registry import Registry

# Every TaskReport.metrics has these keys first, then `p50_ms:<task>` for each task in TASKS.
TASK_METRIC_NAMES: tuple[str, ...] = (
    "router_accuracy",
    "task_success",
    "key_facts_score",
    "clarify_accuracy",
    "forbidden_rate",
    "total_p50_ms",
    "total_p95_ms",
    *(f"p50_ms:{t}" for t in TASKS),
)


@dataclass(frozen=True)
class TaskResult:
    id: str
    request: str
    gold_task: str  # the item's label
    task: str  # the planner's route
    language: str
    tags: list[str]
    answer: str
    abstained: bool
    sources: list[str]  # labels of the sources shown with the answer
    total_ms: float
    tools: list[str] = field(default_factory=list)  # tool calls made, in order
    key_facts: list[bool] | None = None  # per key fact; None for clarify items and judge off
    score: float | None = None  # share of key facts stated
    asked: bool | None = None  # clarify items: the answer asks which document / what was meant
    success: bool | None = None  # None when not judged
    forbidden: list[str] = field(default_factory=list)  # forbidden strings found in the answer
    judge_error: str | None = None


@dataclass(frozen=True)
class TaskReport:
    config_name: str
    dataset_name: str
    items: list[TaskResult]
    metrics: dict[str, float | None]  # keys = TASK_METRIC_NAMES; None when no item qualifies


def run_tasks(
    cfg: Config,
    items: Sequence[TaskItem],
    *,
    config_name: str,
    dataset_name: str,
    llm: LLM,
    embedder: Embedder,
    index: ChunkIndex,
    registry: Registry,
    cards: CardStore,
    reranker: Reranker | None = None,
    judge_llm: LLM | None = None,
    on_item: Callable[[TaskResult], None] | None = None,
) -> TaskReport:
    """Run every task item and score the run. With `judge_llm` None, nothing is judged."""
    results: list[TaskResult] = []
    for item in items:
        start = time.perf_counter()
        result = run_agent(
            item.request,
            cfg,
            llm=llm,
            embedder=embedder,
            index=index,
            registry=registry,
            reranker=reranker,
            history=[{"role": t.role, "content": t.content} for t in item.history],
            cards=cards,
        )
        total_ms = (time.perf_counter() - start) * 1000.0
        answer = result.answer.strip()
        scored = TaskResult(
            id=item.id,
            request=item.request,
            gold_task=item.task,
            task=result.task,
            language=item.language,
            tags=list(item.tags),
            answer=answer,
            abstained=result.abstained,
            sources=[s.label for s in result.sources],
            total_ms=total_ms,
            tools=[str(e.data.get("name", "")) for e in result.trace if e.node == "tool"],
            forbidden=[f for f in item.forbidden if f.casefold() in answer.casefold()],
        )
        if judge_llm is not None:
            scored = _judged(scored, item, judge_llm)
        results.append(scored)
        if on_item is not None:
            on_item(scored)
    return TaskReport(config_name, dataset_name, results, task_metrics(results))


def _judged(result: TaskResult, item: TaskItem, judge_llm: LLM) -> TaskResult:
    try:
        if item.clarify:
            asked = judge_clarify(item.request, result.answer, llm=judge_llm)
            return dataclasses.replace(result, asked=asked, success=asked)
        verdict = judge_key_facts(item.request, item.key_facts, result.answer, llm=judge_llm)
    except JudgeError as e:
        return dataclasses.replace(result, judge_error=str(e))
    return dataclasses.replace(
        result,
        key_facts=list(verdict.found),
        score=verdict.score,
        success=verdict.score >= TASK_SUCCESS_THRESHOLD and not result.forbidden,
    )


def task_metrics(results: Sequence[TaskResult]) -> dict[str, float | None]:
    """All TASK_METRIC_NAMES for `results`."""
    metrics: dict[str, float | None] = dict.fromkeys(TASK_METRIC_NAMES)
    if not results:
        return metrics
    metrics["router_accuracy"] = router_accuracy(
        [r.task for r in results], [r.gold_task for r in results]
    )
    judged = [r for r in results if r.success is not None]
    if judged:
        metrics["task_success"] = sum(bool(r.success) for r in judged) / len(judged)
    scores = [r.score for r in results if r.score is not None]
    if scores:
        metrics["key_facts_score"] = sum(scores) / len(scores)
    asked = [r.asked for r in results if r.asked is not None]
    if asked:
        metrics["clarify_accuracy"] = sum(asked) / len(asked)
    metrics["forbidden_rate"] = sum(bool(r.forbidden) for r in results) / len(results)
    times = [r.total_ms for r in results]
    metrics["total_p50_ms"] = percentile(times, 50)
    metrics["total_p95_ms"] = percentile(times, 95)
    for task in TASKS:
        of_task = [r.total_ms for r in results if r.gold_task == task]
        if of_task:
            metrics[f"p50_ms:{task}"] = percentile(of_task, 50)
    return metrics


def save_task_report(report: TaskReport, out_dir: Path, *, repeat: int | None = None) -> Path:
    """Write ``<out_dir>/<config>__<dataset>[__r<n>].jsonl``: one TaskResult per line, then a
    final ``{"metrics": {...}}`` line. Returns the path."""
    return write_report(report, [dataclasses.asdict(r) for r in report.items], out_dir, repeat)
