"""Command-line entry point: `uv run vaultmind <command>`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import yaml

from vaultmind.core.config import load_config, resolve_config_path

app = typer.Typer(help="VaultMind: local, private agentic RAG over your documents.")

ConfigOption = Annotated[Path | None, typer.Option("--config", "-c", help="Path to a YAML config.")]


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


if __name__ == "__main__":
    app()
