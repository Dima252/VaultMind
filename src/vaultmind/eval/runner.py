"""Eval runner: score one config on a dataset, save raw results, compare configs side by side."""

from __future__ import annotations

import dataclasses
import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from vaultmind.agent.baseline import NOT_FOUND, Source, build_messages, is_abstention, retrieve
from vaultmind.agent.graph import run_agent
from vaultmind.agent.types import AgentResult
from vaultmind.core.config import Config
from vaultmind.core.providers import LLM, Embedder, Reranker
from vaultmind.eval.dataset import EvalItem
from vaultmind.eval.judge import JudgeError, judge
from vaultmind.eval.metrics import (
    abstention_accuracy,
    correctness_score,
    evidence_mrr,
    evidence_recall_at_k,
    injection_resistance,
    mean_and_spread,
    mrr,
    percentile,
    recall_at_k,
)
from vaultmind.store.cards import CardStore
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
    "rewrite_rate",
    "rewrite_recovered",
    "analyze_p50_ms",
    "plan_p50_ms",
    "lookup_route_rate",
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
    # Linear runs time the two stages directly. Agent runs (`agent.loop`) count the generate
    # and direct nodes as generation and everything else (analyze, rewrite, checks, graph
    # overhead) as retrieval, so retrieval_ms + generation_ms is the item's wall time.
    retrieval_ms: float
    generation_ms: float  # 0 when nothing was retrieved and no LLM call was made
    # Judge verdict (a Verdict value). With the judge on: "incorrect" for an abstention on an
    # answerable item (no LLM call), None for unanswerable items and judge errors.
    # With the judge off: always None.
    verdict: str | None
    judge_reason: str | None  # the judge's reason, or "judge error: ..."; None if not judged
    # Agent runs only (empty / 0 for linear runs and result files written before Phase 4):
    queries: list[str] = field(default_factory=list)  # every query searched, in order
    rewrites: int = 0  # rewrite rounds used
    node_ms: dict[str, float] = field(default_factory=dict)  # total time per graph node
    unsupported: list[str] = field(default_factory=list)  # groundedness check: flagged claims
    contradictions: list[str] = field(default_factory=list)  # "[a] claim | [b] claim"
    task: str = "lookup"  # the planner's route (always "lookup" without the router)


@dataclass(frozen=True)
class RunReport:
    config_name: str
    dataset_name: str
    items: list[ItemResult]
    metrics: dict[str, float | None]  # keys = METRIC_NAMES; None when no item qualifies


class Report(Protocol):
    """What saving and the table need from a report: `RunReport`, `task_runner.TaskReport`."""

    @property
    def config_name(self) -> str: ...
    @property
    def dataset_name(self) -> str: ...
    @property
    def metrics(self) -> Mapping[str, float | None]: ...


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _display_path(path: Path | None, vault_dir: Path) -> str:
    """Vault-relative path with forward slashes; absolute if outside the vault; "" if unknown."""
    if path is None:
        return ""
    resolved = path.resolve()
    try:
        return resolved.relative_to(vault_dir).as_posix()
    except ValueError:
        return resolved.as_posix()


def _to_retrieved(sources: Sequence[Source], vault_dir: Path) -> list[RetrievedSource]:
    return [
        RetrievedSource(_display_path(s.file_path, vault_dir), s.page, s.text, s.score)
        for s in sources
    ]


def _grade(
    item: EvalItem, answer: str, abstained: bool, judge_llm: LLM | None
) -> tuple[str | None, str | None]:
    """(verdict, reason) following the rules documented on ItemResult."""
    if judge_llm is None or not item.answerable:
        return None, None
    if abstained:
        return "incorrect", None
    try:
        result = judge(item.question, item.answer, answer, llm=judge_llm)
    except JudgeError as e:
        return None, f"judge error: {e}"
    return result.verdict.value, result.reason


def _linear_item(
    i: int,
    item: EvalItem,
    cfg: Config,
    llm: LLM,
    embedder: Embedder,
    index: ChunkIndex,
    registry: Registry,
    reranker: Reranker | None,
    vault_dir: Path,
    judge_llm: LLM | None,
) -> ItemResult:
    start = time.perf_counter()
    sources = retrieve(
        item.question, cfg, embedder=embedder, index=index, registry=registry, reranker=reranker
    )
    retrieval_ms = _elapsed_ms(start)

    if sources:
        start = time.perf_counter()
        answer = llm.generate(build_messages(item.question, sources)).strip()
        generation_ms = _elapsed_ms(start)
    else:
        answer, generation_ms = NOT_FOUND, 0.0

    abstained = is_abstention(answer)
    verdict, reason = _grade(item, answer, abstained, judge_llm)
    return ItemResult(
        index=i,
        question=item.question,
        answerable=item.answerable,
        tags=list(item.tags),
        retrieved=_to_retrieved(sources, vault_dir),
        answer=answer,
        abstained=abstained,
        retrieval_ms=retrieval_ms,
        generation_ms=generation_ms,
        verdict=verdict,
        judge_reason=reason,
    )


def _agent_item(
    i: int,
    item: EvalItem,
    result: AgentResult,
    total_ms: float,
    vault_dir: Path,
    judge_llm: LLM | None,
) -> ItemResult:
    answer = result.answer.strip()
    generation_ms = sum(result.node_ms(n) for n in ("generate", "direct", "respond", "human"))
    node_ms: dict[str, float] = {}
    for event in result.trace:
        node_ms[event.node] = node_ms.get(event.node, 0.0) + event.duration_ms
    verdict, reason = _grade(item, answer, result.abstained, judge_llm)
    # A not-found answer lists its closest matches as sources, but none passed grading. Score
    # retrieval on graded passages only, as the linear pipeline does.
    graded = [] if "not_found" in node_ms else result.sources
    return ItemResult(
        index=i,
        question=item.question,
        answerable=item.answerable,
        tags=list(item.tags),
        retrieved=_to_retrieved(graded, vault_dir),
        answer=answer,
        abstained=result.abstained,
        retrieval_ms=max(total_ms - generation_ms, 0.0),
        generation_ms=generation_ms,
        verdict=verdict,
        judge_reason=reason,
        queries=list(result.queries),
        rewrites=result.rewrites,
        node_ms=node_ms,
        unsupported=[c.claim for c in result.unsupported],
        contradictions=[
            f"[{c.source_a}] {c.claim_a} | [{c.source_b}] {c.claim_b}"
            for c in result.contradictions
        ],
        task=result.task,
    )


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
    reranker: Reranker | None = None,
    judge_llm: LLM | None = None,
    on_item: Callable[[ItemResult], None] | None = None,
    cards: CardStore | None = None,
) -> RunReport:
    """Answer every item and score the run.

    With ``agent.loop`` on, each item goes through the corrective agent (`agent.graph`) with
    its chat history; otherwise through the linear pipeline (`agent.baseline`), which ignores
    history; with ``cards`` and ``agent.router`` on, the planner routes each item first, as in
    the app. Timing is split as documented on ItemResult. With ``judge_llm`` None, correctness
    is not measured: every verdict is None and the ``correctness`` metric is None.
    """
    vault_dir = cfg.paths.vault_dir.resolve()
    results: list[ItemResult] = []
    for i, item in enumerate(items):
        if cfg.agent.loop:
            start = time.perf_counter()
            agent_result = run_agent(
                item.question,
                cfg,
                llm=llm,
                embedder=embedder,
                index=index,
                registry=registry,
                reranker=reranker,
                history=[{"role": t.role, "content": t.content} for t in item.history],
                cards=cards,
            )
            total_ms = _elapsed_ms(start)
            result = _agent_item(i, item, agent_result, total_ms, vault_dir, judge_llm)
        else:
            result = _linear_item(
                i, item, cfg, llm, embedder, index, registry, reranker, vault_dir, judge_llm
            )
        results.append(result)
        if on_item is not None:
            on_item(result)

    return RunReport(
        config_name=config_name,
        dataset_name=dataset_name,
        items=results,
        metrics=compute_metrics(items, results),
    )


def _p(values: Sequence[float], q: float) -> float | None:
    return percentile(values, q) if values else None


def compute_metrics(
    items: Sequence[EvalItem], results: Sequence[ItemResult]
) -> dict[str, float | None]:
    """All METRIC_NAMES for paired ``items`` and ``results``."""
    if len(items) != len(results):
        raise ValueError(f"length mismatch: {len(items)} items, {len(results)} results")
    pairs = list(zip(items, results, strict=True))
    metrics: dict[str, float | None] = dict.fromkeys(METRIC_NAMES)

    answerable = [(it, r) for it, r in pairs if it.answerable]
    if answerable:
        retrieved = [[(s.file, s.page) for s in r.retrieved] for _, r in answerable]
        gold = [(it.source_file or "", it.source_page) for it, _ in answerable]
        metrics["recall@1"] = recall_at_k(retrieved, gold, 1)
        metrics["recall@5"] = recall_at_k(retrieved, gold, 5)
        metrics["mrr"] = mrr(retrieved, gold)

    with_evidence = [(it, r) for it, r in pairs if it.evidence]
    if with_evidence:
        passages = [[s.text for s in r.retrieved] for _, r in with_evidence]
        evidence = [it.evidence for it, _ in with_evidence]
        metrics["evidence_recall@1"] = evidence_recall_at_k(passages, evidence, 1)
        metrics["evidence_recall@5"] = evidence_recall_at_k(passages, evidence, 5)
        metrics["evidence_mrr"] = evidence_mrr(passages, evidence)

    verdicts = [r.verdict for r in results if r.verdict is not None]
    if verdicts:
        metrics["correctness"] = correctness_score(verdicts)

    if pairs:
        metrics["abstention_accuracy"] = abstention_accuracy(
            [r.abstained for r in results], [it.answerable for it in items]
        )

    injected = [(it, r) for it, r in pairs if it.forbidden]
    if injected:
        metrics["injection_resistance"] = injection_resistance(
            [r.answer for _, r in injected], [it.forbidden for it, _ in injected]
        )

    latencies = {
        "retrieval": [r.retrieval_ms for r in results],
        "generation": [r.generation_ms for r in results],
        "total": [r.retrieval_ms + r.generation_ms for r in results],
    }
    for name, values in latencies.items():
        metrics[f"{name}_p50_ms"] = _p(values, 50)
        metrics[f"{name}_p95_ms"] = _p(values, 95)

    # Agent metrics: an agent run always records node timings; linear runs leave them empty.
    agent = [r for r in results if r.node_ms]
    if agent:
        rewritten = [r for r in agent if r.rewrites > 0]
        metrics["rewrite_rate"] = len(rewritten) / len(agent)
        if rewritten:
            metrics["rewrite_recovered"] = sum(not r.abstained for r in rewritten) / len(rewritten)
        metrics["analyze_p50_ms"] = _p(
            [r.node_ms["analyze"] for r in agent if "analyze" in r.node_ms], 50
        )
        planned = [r for r in agent if "plan" in r.node_ms]
        if planned:
            metrics["plan_p50_ms"] = _p([r.node_ms["plan"] for r in planned], 50)
            metrics["lookup_route_rate"] = sum(r.task == "lookup" for r in planned) / len(planned)
    return metrics


def check_compatible(configs: Mapping[str, Config]) -> None:
    """Raise ValueError if two configs share a data_dir but differ in what the index depends on

    (embedder model or chunking settings), since they would silently score the wrong index.
    """
    first_by_dir: dict[Path, tuple[str, Config]] = {}
    for name, cfg in configs.items():
        data_dir = cfg.paths.data_dir.resolve()
        if data_dir not in first_by_dir:
            first_by_dir[data_dir] = (name, cfg)
            continue
        other_name, other = first_by_dir[data_dir]
        if cfg.models.embedder != other.models.embedder:
            what = f"embedder ({other.models.embedder!r} vs {cfg.models.embedder!r})"
        elif cfg.chunking != other.chunking:
            what = "chunking settings"
        else:
            continue
        raise ValueError(
            f"configs {other_name!r} and {name!r} share data_dir {data_dir} but differ in "
            f"{what}; give each its own paths.data_dir"
        )


_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL"} | {
    f"{p}{n}" for p in ("COM", "LPT") for n in "123456789"
}


def _safe_name(name: str) -> str:
    """A file-name-safe version of ``name`` on Windows (and everywhere else)."""
    safe = _UNSAFE_CHARS.sub("_", name).strip(" .") or "_"
    if safe.split(".", 1)[0].upper() in _RESERVED_NAMES:
        safe = f"_{safe}"
    return safe


def save_report(report: RunReport, out_dir: Path, *, repeat: int | None = None) -> Path:
    """Write ``<out_dir>/<config_name>__<dataset_name>.jsonl``: one ItemResult per line, then a
    final ``{"metrics": {...}}`` line. Returns the path.

    With ``repeat`` (1-based, for ``eval run --repeat``), the name ends in ``__r<repeat>``."""
    return write_report(report, [dataclasses.asdict(r) for r in report.items], out_dir, repeat)


def write_report(
    report: Report, rows: Sequence[object], out_dir: Path, repeat: int | None = None
) -> Path:
    """`save_report` for any report: ``rows`` (its items as dicts), then its metrics line."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{_safe_name(report.config_name)}__{_safe_name(report.dataset_name)}"
    if repeat is not None:
        stem += f"__r{repeat}"
    path = out_dir / f"{stem}.jsonl"
    rows = [*rows, {"metrics": report.metrics}]
    content = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
    )
    path.write_text(content, encoding="utf-8", newline="\n")
    return path


def load_results(path: Path) -> list[ItemResult]:
    """The ItemResults of a file written by `save_report` (its metrics line is skipped).

    Files written before a field existed load with that field's default."""
    results: list[ItemResult] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if "metrics" in row:
            continue
        retrieved = [RetrievedSource(**s) for s in row.pop("retrieved")]
        results.append(ItemResult(**row, retrieved=retrieved))
    return results


def _format_value(name: str, value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.0f}" if name.endswith("_ms") else f"{value:.3f}"


def _ascii(text: str) -> str:
    return text.encode("ascii", errors="replace").decode("ascii")


def _format_repeats(name: str, values: Sequence[float | None]) -> str:
    """One value as is; several as ``mean (min-max)`` over the repeats that have a value."""
    present = [v for v in values if v is not None]
    if len(values) <= 1 or not present:
        return _format_value(name, present[0] if present else None)
    mean, low, high = mean_and_spread(present)
    return f"{_format_value(name, mean)} ({_format_value(name, low)}-{_format_value(name, high)})"


def format_table(reports: Sequence[RunReport]) -> str:
    """ASCII table: one row per metric, one column per report (``config / dataset``)."""
    return format_repeat_table([[r] for r in reports])


def format_repeat_table(
    runs: Sequence[Sequence[Report]], metric_names: Sequence[str] = METRIC_NAMES
) -> str:
    """Like `format_table`, with one column per (config, dataset) holding its repeated runs.

    A column with one run shows plain values. With several, each metric shows as
    ``mean (min-max)``, e.g. ``0.800 (0.767-0.833)``, over the runs where it isn't None (a
    metric can be None in one repeat, e.g. ``rewrite_recovered`` when nothing was rewritten);
    ``-`` when it is None in all of them.
    """
    if not runs or any(not group for group in runs):
        raise ValueError("no reports to format")
    header = ["metric"] + [_ascii(f"{g[0].config_name} / {g[0].dataset_name}") for g in runs]
    rows = [
        [name] + [_format_repeats(name, [r.metrics.get(name) for r in group]) for group in runs]
        for name in metric_names
    ]
    widths = [max(len(row[c]) for row in [header, *rows]) for c in range(len(header))]

    def line(cells: Sequence[str]) -> str:
        first = cells[0].ljust(widths[0])
        rest = [cell.rjust(w) for cell, w in zip(cells[1:], widths[1:], strict=True)]
        return "  ".join([first, *rest]).rstrip()

    rule = "  ".join("-" * w for w in widths)
    return "\n".join([line(header), rule, *(line(row) for row in rows)])
