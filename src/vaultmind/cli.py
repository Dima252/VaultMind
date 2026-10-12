"""Command-line entry point: `uv run vaultmind <command>`."""

from __future__ import annotations

import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
import yaml

from vaultmind.core.config import Config, load_config, resolve_config_path

if TYPE_CHECKING:
    from vaultmind.agent.baseline import Source
    from vaultmind.agent.types import AgentResult
    from vaultmind.core.providers import LLM, Embedder, Message, Reranker
    from vaultmind.store.index import ChunkIndex
    from vaultmind.store.registry import Registry
    from vaultmind.store.traces import StoredRun

app = typer.Typer(help="VaultMind: local, private agentic RAG over your documents.")
eval_app = typer.Typer(help="Evaluation datasets and runs.")
app.add_typer(eval_app, name="eval")

ConfigOption = Annotated[Path | None, typer.Option("--config", "-c", help="Path to a YAML config.")]


@app.callback()
def _main() -> None:
    # Legacy Windows code pages can't encode every character a document or model may produce.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")


@contextmanager
def _store(cfg: Config) -> Iterator[tuple[ChunkIndex, Registry]]:
    """Open the index and registry under `paths.data_dir`, with friendly errors."""
    from vaultmind.store.index import ChunkIndex, EmbedderMismatchError
    from vaultmind.store.registry import Registry

    try:
        index = ChunkIndex(cfg.paths.data_dir / "index", embedder_name=cfg.models.embedder)
    except EmbedderMismatchError as exc:
        typer.secho(f"[FAIL] {exc}", fg="red")
        raise typer.Exit(1) from exc
    registry = Registry(cfg.paths.data_dir / "vaultmind.sqlite")
    try:
        yield index, registry
    finally:
        registry.close()
        index.close()


@contextmanager
def _ollama_errors() -> Iterator[None]:
    from vaultmind.core.ollama import OllamaUnavailableError

    try:
        yield
    except OllamaUnavailableError as exc:
        typer.echo()
        typer.secho(f"[FAIL] {exc}", fg="red")
        raise typer.Exit(1) from exc


def _reranker(cfg: Config) -> Reranker | None:
    """The configured cross-encoder when `retrieval.rerank` is on, with friendly errors."""
    if not cfg.retrieval.rerank:
        return None
    from vaultmind.core.rerank import CrossEncoderReranker, RerankerUnavailableError

    try:
        return CrossEncoderReranker.from_config(cfg)
    except RerankerUnavailableError as exc:
        typer.secho(f"[FAIL] {exc} (or set retrieval.rerank: false)", fg="red")
        raise typer.Exit(1) from exc


@app.command()
def config(config_path: ConfigOption = None) -> None:
    """Print the effective configuration."""
    cfg = load_config(config_path)
    typer.echo(f"# source: {resolve_config_path(config_path) or 'built-in defaults'}")
    typer.echo(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))


@app.command()
def doctor(config_path: ConfigOption = None) -> None:
    """Check that Ollama is reachable and the configured models are pulled."""
    import ollama

    cfg = load_config(config_path)
    try:
        listed = ollama.Client(host=cfg.ollama.host).list()
    except Exception as exc:
        typer.secho(f"[FAIL] Ollama not reachable at {cfg.ollama.host}: {exc}", fg="red")
        raise typer.Exit(1) from exc

    typer.secho(f"[ OK ] Ollama reachable at {cfg.ollama.host}", fg="green")
    available = {m.model for m in listed.models}
    missing = [name for name in (cfg.models.llm, cfg.models.embedder) if name not in available]
    for name in (cfg.models.llm, cfg.models.embedder):
        if name in missing:
            typer.secho(f"[FAIL] {name} not pulled - run: ollama pull {name}", fg="red")
        else:
            typer.secho(f"[ OK ] {name}", fg="green")
    if cfg.retrieval.rerank and not _check_reranker(cfg.models.reranker):
        missing.append(cfg.models.reranker)
    if missing:
        raise typer.Exit(1)


def _check_reranker(name: str) -> bool:
    from importlib.util import find_spec

    from vaultmind.core.rerank import is_downloaded

    if find_spec("sentence_transformers") is None:
        typer.secho("[FAIL] reranker extra missing - run: uv sync --extra rerank", fg="red")
        return False
    if not is_downloaded(name):
        typer.secho(f"[FAIL] {name} not downloaded - run: vaultmind pull-reranker", fg="red")
        return False
    typer.secho(f"[ OK ] {name}", fg="green")
    return True


@app.command("pull-reranker")
def pull_reranker(config_path: ConfigOption = None) -> None:
    """Download the configured reranker model (the only step that uses the internet)."""
    from importlib.util import find_spec

    from vaultmind.core.rerank import download

    if find_spec("sentence_transformers") is None:
        typer.secho("[FAIL] Run first: uv sync --extra rerank", fg="red")
        raise typer.Exit(1)
    cfg = load_config(config_path)
    typer.echo(f"Downloading {cfg.models.reranker} from Hugging Face ...")
    path = download(cfg.models.reranker)
    typer.secho(f"[ OK ] {cfg.models.reranker} -> {path}", fg="green")


@app.command()
def ingest(
    path: Annotated[
        Path | None, typer.Argument(help="File or folder. Defaults to paths.vault_dir.")
    ] = None,
    config_path: ConfigOption = None,
) -> None:
    """Index new and changed documents; forget deleted ones."""
    from vaultmind.core.ollama import OllamaEmbedder
    from vaultmind.ingest.pipeline import ingest_path

    cfg = load_config(config_path)
    target = path or cfg.paths.vault_dir
    if not target.exists():
        typer.secho(f"[FAIL] {target} does not exist. Create it and add documents.", fg="red")
        raise typer.Exit(1)

    with _store(cfg) as (index, registry), _ollama_errors():
        summary = ingest_path(
            target,
            cfg,
            embedder=OllamaEmbedder.from_config(cfg),
            index=index,
            registry=registry,
            on_file=lambda f: typer.echo(f"  indexing {f.name}"),
        )
        chunks = index.count()

    for file, reason in summary.failed:
        typer.secho(f"[FAIL] {file.name}: {reason}", fg="red")
    typer.echo(
        f"added {len(summary.added)}, updated {len(summary.updated)}, "
        f"skipped {len(summary.skipped)}, failed {len(summary.failed)}, "
        f"removed {len(summary.removed)} | {chunks} chunks in the index"
    )


@app.command()
def ui(
    browser: Annotated[
        bool, typer.Option("--browser", help="Open in the default browser, not a native window.")
    ] = False,
    config_path: ConfigOption = None,
) -> None:
    """Open the desktop app (served on 127.0.0.1 only)."""
    from vaultmind.ui.app import main

    cfg = load_config(config_path)
    if browser:
        cfg = cfg.model_copy(update={"ui": cfg.ui.model_copy(update={"native": False})})
    main(cfg, resolve_config_path(config_path))


def stream_agent(
    question: str,
    cfg: Config,
    *,
    llm: LLM,
    embedder: Embedder,
    index: ChunkIndex,
    registry: Registry,
    reranker: Reranker | None,
    history: Sequence[Message] = (),
) -> Iterator[str | AgentResult]:
    """`agent.graph.stream_agent`, imported on first use to keep CLI startup fast."""
    from vaultmind.agent.graph import stream_agent as run

    return run(
        question,
        cfg,
        llm=llm,
        embedder=embedder,
        index=index,
        registry=registry,
        reranker=reranker,
        history=history,
    )


def _db_path(cfg: Config) -> Path:
    return cfg.paths.data_dir / "vaultmind.sqlite"


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Your question.")],
    show_trace: Annotated[
        bool, typer.Option("--trace", help="Print what each agent step did and how long it took.")
    ] = False,
    config_path: ConfigOption = None,
) -> None:
    """Answer a question from your indexed documents, with sources."""
    from vaultmind.agent.types import AgentResult
    from vaultmind.core.ollama import OllamaEmbedder, OllamaLLM

    cfg = load_config(config_path)
    if not cfg.agent.loop:
        _ask_linear(question, cfg)
        if show_trace:
            typer.echo()
            typer.echo("(no trace: agent.loop is off)")
        return

    result: AgentResult | None = None
    with _store(cfg) as (index, registry), _ollama_errors():
        if index.count() == 0:
            typer.secho("[FAIL] The index is empty. Run: vaultmind ingest", fg="red")
            raise typer.Exit(1)
        events = stream_agent(
            question,
            cfg,
            llm=OllamaLLM.from_config(cfg),
            embedder=OllamaEmbedder.from_config(cfg),
            index=index,
            registry=registry,
            reranker=_reranker(cfg),
        )
        for event in events:
            if isinstance(event, AgentResult):
                result = event
            else:
                typer.echo(event, nl=False)
        typer.echo()
    if result is None:
        typer.secho("[FAIL] The agent ended without a result.", fg="red")
        raise typer.Exit(1)

    run_id = _save_run(cfg, result)
    _print_agent_sources(result)
    if show_trace:
        from vaultmind.agent.trace_text import format_trace

        typer.echo()
        typer.echo("Trace:")
        for line in format_trace(result.trace):
            typer.echo(line)
    typer.echo()
    typer.echo(f"run {run_id}")


def _ask_linear(question: str, cfg: Config) -> None:
    """The linear pipeline (`agent.loop: false`), unchanged from Phase 3."""
    from vaultmind.agent.baseline import cited_sources, is_abstention, stream_answer
    from vaultmind.core.ollama import OllamaEmbedder, OllamaLLM

    with _store(cfg) as (index, registry), _ollama_errors():
        if index.count() == 0:
            typer.secho("[FAIL] The index is empty. Run: vaultmind ingest", fg="red")
            raise typer.Exit(1)
        sources, tokens = stream_answer(
            question,
            cfg,
            llm=OllamaLLM.from_config(cfg),
            embedder=OllamaEmbedder.from_config(cfg),
            index=index,
            registry=registry,
            reranker=_reranker(cfg),
        )
        parts: list[str] = []
        for token in tokens:
            parts.append(token)
            typer.echo(token, nl=False)
        typer.echo()

    text = "".join(parts)
    if is_abstention(text):
        heading, shown = "Closest matches:", sources
    else:
        heading, shown = "Sources:", cited_sources(text, sources) or sources
    if shown:
        typer.echo()
        typer.echo(heading)
        for s in shown:
            typer.echo(f"  [{s.n}] {s.label}")


def _save_run(cfg: Config, result: AgentResult) -> str:
    from vaultmind.store.traces import TraceStore

    store = TraceStore(_db_path(cfg))
    try:
        return store.save(result)
    finally:
        store.close()


def _print_agent_sources(result: AgentResult) -> None:
    """Sources (or closest matches) with their warnings, then conflicts and unsupported claims."""
    shown: list[Source]
    if result.abstained:
        heading, shown = "Closest matches:", result.sources
    else:
        heading, shown = "Sources:", result.cited or result.sources
    if shown:
        typer.echo()
        typer.echo(heading)
        for s in shown:
            typer.echo(f"  [{s.n}] {s.label}")
            for warning in s.warnings:
                typer.secho(f"      [!] {warning}", fg="yellow")
    if result.contradictions:
        typer.echo()
        typer.secho("[!] Sources disagree:", fg="yellow")
        for c in result.contradictions:
            typer.echo(f"  [{c.source_a}] {c.claim_a}")
            typer.echo(f"  [{c.source_b}] {c.claim_b}")
    if result.unsupported:
        typer.echo()
        typer.secho("[!] Not supported by the cited sources:", fg="yellow")
        for u in result.unsupported:
            cited = "".join(f"[{n}]" for n in u.cited) or "(no citation)"
            typer.echo(f"  {u.claim} {cited}: {u.reason}")


# --- traces ---------------------------------------------------------------------------


@app.command()
def trace(
    run_id: Annotated[
        str | None, typer.Argument(help="A run id printed by 'ask'. Omit to list recent runs.")
    ] = None,
    last: Annotated[int, typer.Option("--last", "-n", min=1, help="Runs to list.")] = 10,
    config_path: ConfigOption = None,
) -> None:
    """Show a stored agent run with its trace, or list the most recent runs."""
    from vaultmind.agent.trace_text import format_trace
    from vaultmind.store.traces import TraceStore

    cfg = load_config(config_path)
    store = TraceStore(_db_path(cfg))
    try:
        if run_id is None:
            _list_runs(store.recent(last))
            return
        run = store.get(run_id.strip())
    finally:
        store.close()

    if run is None:
        typer.secho(f"[FAIL] No run with id {run_id}. List them with: vaultmind trace", fg="red")
        raise typer.Exit(1)
    typer.echo(f"Run {run.id}  {_local_time(run)}  {_outcome(run)}")
    typer.echo(f"Question: {run.question}")
    typer.echo("Answer:")
    typer.echo(run.answer)
    typer.echo()
    typer.echo("Trace:")
    for line in format_trace(run.events):
        typer.echo(line)


def _list_runs(runs: Sequence[StoredRun]) -> None:
    from vaultmind.agent.trace_text import short

    if not runs:
        typer.echo('No runs stored yet. Ask something: vaultmind ask "..."')
    for run in runs:
        typer.echo(f"{run.id}  {_local_time(run)}  {_outcome(run):<9}  {short(run.question)}")


def _local_time(run: StoredRun) -> str:
    return run.created_at.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _outcome(run: StoredRun) -> str:
    return "abstained" if run.abstained else "answered"


@eval_app.command("import-cuad")
def import_cuad(
    zip_path: Annotated[
        Path, typer.Option("--zip", help="CUAD data.zip (contains CUADv1.json).")
    ] = Path("eval/corpora/cuad/data.zip"),
    out: Annotated[Path, typer.Option(help="Dataset JSONL to write.")] = Path(
        "eval/datasets/cuad.jsonl"
    ),
    contracts: Annotated[int, typer.Option(min=1, help="Contracts to copy into the vault.")] = 25,
    questions: Annotated[int, typer.Option(min=1, help="Questions to generate.")] = 60,
    unanswerable: Annotated[
        float, typer.Option(min=0.0, max=1.0, help="Share of unanswerable questions.")
    ] = 0.2,
    seed: Annotated[int, typer.Option(help="Sampling seed; same seed, same subset.")] = 0,
    max_chars: Annotated[int, typer.Option(help="Skip contracts longer than this.")] = 100_000,
    config_path: ConfigOption = None,
) -> None:
    """Copy a CUAD contract subset into <vault>/cuad/ and write its eval dataset."""
    from vaultmind.eval.cuad import SUBDIR
    from vaultmind.eval.cuad import import_cuad as run_import

    cfg = load_config(config_path)
    if not zip_path.exists():
        typer.secho(f"[FAIL] {zip_path} not found. Copy CUAD's data.zip there.", fg="red")
        raise typer.Exit(1)
    try:
        summary = run_import(
            zip_path,
            cfg.paths.vault_dir,
            out,
            n_contracts=contracts,
            n_questions=questions,
            unanswerable_share=unanswerable,
            seed=seed,
            max_chars=max_chars,
        )
    except ValueError as exc:
        typer.secho(f"[FAIL] {exc}", fg="red")
        raise typer.Exit(1) from exc

    n_unanswerable = sum(not item.answerable for item in summary.items)
    typer.echo(
        f"{len(summary.written)} contracts in {cfg.paths.vault_dir / SUBDIR} "
        f"({len(summary.removed)} stale removed)"
    )
    typer.echo(f"{len(summary.items)} questions ({n_unanswerable} unanswerable) written to {out}")
    typer.echo("Next: review the questions, then run: vaultmind ingest")


@eval_app.command("install-adversarial")
def install_adversarial(config_path: ConfigOption = None) -> None:
    """Copy the synthetic adversarial documents into <vault>/adversarial/."""
    from vaultmind.eval.adversarial import SUBDIR
    from vaultmind.eval.adversarial import install_adversarial as run_install

    cfg = load_config(config_path)
    try:
        written = run_install(cfg.paths.vault_dir)
    except FileNotFoundError as exc:
        typer.secho(f"[FAIL] {exc}", fg="red")
        raise typer.Exit(1) from exc
    typer.echo(f"{len(written)} documents in {cfg.paths.vault_dir / SUBDIR}")
    typer.echo("Next: vaultmind ingest")


def _dataset_path(name: str) -> Path:
    """A dataset name (``cuad``) or a path to a JSONL file."""
    return Path(name) if name.endswith(".jsonl") else Path("eval/datasets") / f"{name}.jsonl"


@eval_app.command("run")
def eval_run(
    config_paths: Annotated[
        list[Path] | None,
        typer.Option("--config", "-c", help="Config to score; repeat to compare several."),
    ] = None,
    datasets: Annotated[
        list[str] | None,
        typer.Option("--dataset", "-d", help="Dataset name or .jsonl path; repeatable."),
    ] = None,
    judge: Annotated[
        bool, typer.Option("--judge/--no-judge", help="Grade answers with the LLM judge.")
    ] = True,
    limit: Annotated[
        int | None, typer.Option(min=1, help="Only the first N items (quick runs).")
    ] = None,
    out_dir: Annotated[Path, typer.Option(help="Where raw results go.")] = Path("eval/results"),
) -> None:
    """Score configs on eval datasets and print a comparison table.

    The judge always uses the first config's LLM, so every config gets the same judge.
    """
    from datetime import datetime

    from vaultmind.core.ollama import OllamaEmbedder, OllamaLLM
    from vaultmind.eval.dataset import load_dataset
    from vaultmind.eval.runner import (
        ItemResult,
        RunReport,
        check_compatible,
        format_table,
        run_config,
        save_report,
    )

    paths: list[Path | None] = list(config_paths) if config_paths else [None]
    configs = {(path.stem if path else "default"): load_config(path) for path in paths}
    if len(configs) != len(paths):
        typer.secho("[FAIL] Config file names must be unique (they name the columns).", fg="red")
        raise typer.Exit(1)
    try:
        check_compatible(configs)
        loaded = {name: load_dataset(_dataset_path(name)) for name in (datasets or ["cuad"])}
    except (ValueError, OSError) as exc:
        typer.secho(f"[FAIL] {exc}", fg="red")
        raise typer.Exit(1) from exc

    run_dir = out_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    judge_llm = OllamaLLM.from_config(next(iter(configs.values()))) if judge else None
    reports: list[RunReport] = []
    rerankers: dict[str, Reranker | None] = {}  # loaded once per model, shared across configs
    for config_name, cfg in configs.items():
        with _store(cfg) as (index, registry), _ollama_errors():
            if index.count() == 0:
                typer.secho(
                    f"[FAIL] {config_name}: the index is empty. Run: vaultmind ingest", fg="red"
                )
                raise typer.Exit(1)
            llm, embedder = OllamaLLM.from_config(cfg), OllamaEmbedder.from_config(cfg)
            if cfg.retrieval.rerank and cfg.models.reranker not in rerankers:
                rerankers[cfg.models.reranker] = _reranker(cfg)
            reranker = rerankers.get(cfg.models.reranker) if cfg.retrieval.rerank else None
            for dataset_name, items in loaded.items():
                items = items[:limit] if limit else items
                typer.echo(f"{config_name} / {dataset_name}: {len(items)} items")

                def progress(
                    result: ItemResult, total: int = len(items), loop: bool = cfg.agent.loop
                ) -> None:
                    mark = "abstained" if result.abstained else (result.verdict or "answered")
                    extra = f" rewrites={result.rewrites}" if loop else ""
                    typer.echo(f"  [{result.index + 1}/{total}] {mark}{extra}")

                report = run_config(
                    cfg,
                    items,
                    config_name=config_name,
                    dataset_name=dataset_name,
                    llm=llm,
                    embedder=embedder,
                    index=index,
                    registry=registry,
                    reranker=reranker,
                    judge_llm=judge_llm,
                    on_item=progress,
                )
                save_report(report, run_dir)
                reports.append(report)

    typer.echo()
    typer.echo(format_table(reports))
    typer.echo()
    typer.echo(f"Raw results: {run_dir}")


@eval_app.command("sweep")
def eval_sweep(
    results: Annotated[
        Path, typer.Argument(help="A results file from 'eval run' with rerank_threshold: 0.")
    ],
    dataset: Annotated[
        str, typer.Option("--dataset", "-d", help="The dataset that run scored.")
    ] = "cuad",
    step: Annotated[float, typer.Option(min=0.01, max=1.0, help="Threshold step.")] = 0.05,
) -> None:
    """Replay rerank thresholds over a saved run: evidence recall vs. abstention."""
    from vaultmind.eval.dataset import load_dataset
    from vaultmind.eval.runner import load_results
    from vaultmind.eval.sweep import format_sweep, sweep

    try:
        items = load_dataset(_dataset_path(dataset))
        rows = sweep(
            items,
            load_results(results),
            [round(i * step, 4) for i in range(round(1 / step))],
        )
    except (ValueError, OSError) as exc:
        typer.secho(f"[FAIL] {exc}", fg="red")
        raise typer.Exit(1) from exc
    typer.echo(format_sweep(rows))


if __name__ == "__main__":
    app()
