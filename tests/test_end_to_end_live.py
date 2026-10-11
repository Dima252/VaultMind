"""End-to-end over a tiny corpus with the real models. Run with: uv run pytest -m ollama"""

from pathlib import Path

import pytest

from vaultmind.agent.baseline import Answer, answer
from vaultmind.core.config import load_config
from vaultmind.core.ollama import OllamaEmbedder, OllamaLLM
from vaultmind.ingest.pipeline import ingest_path
from vaultmind.store.index import ChunkIndex
from vaultmind.store.registry import Registry

pytestmark = pytest.mark.ollama


def test_ingest_then_answer_with_citation(tmp_path: Path) -> None:
    cfg = load_config()
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "zephyr.md").write_text(
        "# Project Zephyr\n\n## Budget\n\nThe Zephyr project budget for 2026 is 41,500 euros.\n\n"
        "## Team\n\nThe project lead is Mira Okafor.\n",
        encoding="utf-8",
    )
    (vault / "garden.txt").write_text(
        "Tomatoes need six hours of sun a day.\n\nWater basil in the morning.\n", encoding="utf-8"
    )
    embedder = OllamaEmbedder.from_config(cfg)
    index = ChunkIndex(tmp_path / "index", embedder_name=embedder.model_name)
    registry = Registry(tmp_path / "r.sqlite")
    try:
        summary = ingest_path(vault, cfg, embedder=embedder, index=index, registry=registry)
        assert len(summary.added) == 2

        llm = OllamaLLM.from_config(cfg)

        def ask(question: str) -> Answer:
            return answer(question, cfg, llm=llm, embedder=embedder, index=index, registry=registry)

        found = ask("What is the Zephyr budget?")
        assert "41,500" in found.text or "41500" in found.text
        assert any(s.file_path and s.file_path.name == "zephyr.md" for s in found.cited)

        assert ask("What is the capital of Peru?").abstained
    finally:
        registry.close()
