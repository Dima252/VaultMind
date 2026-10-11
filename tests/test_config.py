from pathlib import Path

import pytest
from pydantic import ValidationError

from vaultmind.core.config import DEFAULT_CONFIG, Config, load_config
from vaultmind.core.providers import LLM, Embedder, Reranker

from .fakes import FakeEmbedder, FakeLLM, FakeReranker


def test_default_yaml_matches_builtin_defaults() -> None:
    assert load_config(DEFAULT_CONFIG) == Config()


def test_partial_override(tmp_path: Path) -> None:
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("retrieval:\n  rerank: false\n  top_k: 3\n", encoding="utf-8")
    cfg = load_config(cfg_file)
    assert cfg.retrieval.rerank is False
    assert cfg.retrieval.top_k == 3
    assert cfg.agent.max_rewrites == 2


def test_unknown_key_rejected(tmp_path: Path) -> None:
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("retrieval:\n  rerankk: false\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(cfg_file)


def test_env_var_selects_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("agent:\n  max_rewrites: 0\n", encoding="utf-8")
    monkeypatch.setenv("VAULTMIND_CONFIG", str(cfg_file))
    assert load_config().agent.max_rewrites == 0


def test_fakes_satisfy_protocols() -> None:
    llm: LLM = FakeLLM()
    embedder: Embedder = FakeEmbedder()
    reranker: Reranker = FakeReranker()
    assert embedder.embed(["a"]) == embedder.embed(["a"])
    assert reranker.score("local rag", ["local rag app", "cooking"]) == [1.0, 0.0]
    assert llm.generate([{"role": "user", "content": "hi"}]) == ""
