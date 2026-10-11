"""Command-line entry point: `uv run vaultmind <command>`."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
import yaml

from vaultmind.core.config import Config, load_config, resolve_config_path

if TYPE_CHECKING:
    from vaultmind.store.index import ChunkIndex
    from vaultmind.store.registry import Registry

app = typer.Typer(help="VaultMind: local, private agentic RAG over your documents.")

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
    if missing:
        raise typer.Exit(1)


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
def ask(
    question: Annotated[str, typer.Argument(help="Your question.")],
    config_path: ConfigOption = None,
) -> None:
    """Answer a question from your indexed documents, with sources."""
    from vaultmind.agent.baseline import cited_sources, is_abstention, stream_answer
    from vaultmind.core.ollama import OllamaEmbedder, OllamaLLM

    cfg = load_config(config_path)
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


if __name__ == "__main__":
    app()
