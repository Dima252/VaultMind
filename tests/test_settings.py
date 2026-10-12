"""Settings persistence: only overrides go to config/local.yaml, written atomically."""

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from vaultmind.app.settings import needs_reindex, needs_restart, overrides, save_local
from vaultmind.core.config import Config, load_config


def _cfg(**sections: dict[str, Any]) -> Config:
    return Config.model_validate(sections)


def test_defaults_have_no_overrides() -> None:
    assert overrides(Config()) == {}


def test_overrides_keep_only_changed_leaves_nested() -> None:
    cfg = _cfg(
        models={"llm": "qwen3:8b", "num_ctx": 16384},  # num_ctx is the default
        retrieval={"top_k": 8, "rerank": False},
        agent={"rewrite": True},
    )
    assert overrides(cfg) == {
        "models": {"llm": "qwen3:8b"},
        "retrieval": {"top_k": 8, "rerank": False},
        "agent": {"rewrite": True},
    }


def test_path_values_serialize_as_strings_and_load_back(tmp_path: Path) -> None:
    vault = tmp_path / "My Docs" / "vault"
    cfg = _cfg(paths={"vault_dir": str(vault)})
    data = overrides(cfg)
    assert data == {"paths": {"vault_dir": str(vault)}}

    path = tmp_path / "local.yaml"
    save_local(cfg, path)
    loaded = load_config(path)
    assert loaded == cfg
    assert loaded.paths.vault_dir == vault


def test_save_then_load_round_trips(tmp_path: Path) -> None:
    cfg = _cfg(
        ollama={"keep_alive": "5m"},
        models={"llm": "qwen3:4b", "think": True},
        retrieval={"rerank_threshold": 0.25},
        ui={"port": 9000, "native": True},
    )
    path = tmp_path / "config" / "nested" / "local.yaml"  # parent folders are created

    save_local(cfg, path)

    assert load_config(path) == cfg
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Written by VaultMind's Settings page")
    assert "only the values that differ" in text
    assert text.index("ollama:") < text.index("models:") < text.index("ui:")  # config order


def test_unicode_is_written_as_is(tmp_path: Path) -> None:
    folder = "C:/Users/me/מסמכים"
    path = tmp_path / "local.yaml"
    save_local(_cfg(paths={"data_dir": folder}), path)
    assert "מסמכים" in path.read_text(encoding="utf-8")
    assert load_config(path).paths.data_dir == Path(folder)


def test_empty_overrides_still_load(tmp_path: Path) -> None:
    path = tmp_path / "local.yaml"
    save_local(Config(), path)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("#")
    assert yaml.safe_load(text) == {}
    assert text.rstrip().endswith("{}")
    assert load_config(path) == Config()


def test_write_is_atomic_and_leaves_no_temp_files(tmp_path: Path) -> None:
    path = tmp_path / "local.yaml"
    save_local(_cfg(retrieval={"top_k": 3}), path)
    save_local(_cfg(retrieval={"top_k": 4}), path)  # replaces the file
    assert [p.name for p in tmp_path.iterdir()] == ["local.yaml"]
    assert load_config(path).retrieval.top_k == 4


def test_invalid_config_is_rejected_and_the_old_file_kept(tmp_path: Path) -> None:
    path = tmp_path / "local.yaml"
    save_local(_cfg(retrieval={"top_k": 3}), path)
    before = path.read_text(encoding="utf-8")
    bad = Config()
    bad.retrieval.top_k = -1  # assignment isn't validated; saving must catch it

    with pytest.raises(ValidationError):
        save_local(bad, path)

    assert path.read_text(encoding="utf-8") == before
    assert [p.name for p in tmp_path.iterdir()] == ["local.yaml"]


def test_needs_restart() -> None:
    old = Config()
    new = _cfg(
        paths={"vault_dir": "./elsewhere"},
        models={"llm": "qwen3:8b"},
        retrieval={"top_k": 9},
        agent={"loop": False},
        ui={"port": 9000},
        ingest={"watch": False},
    )
    assert needs_restart(old, new) == ["paths.vault_dir", "models.llm", "ingest.watch", "ui.port"]
    assert needs_restart(old, old) == []
    assert needs_restart(old, _cfg(agent={"groundedness_check": True})) == []


def test_turning_the_reranker_on_needs_a_restart() -> None:
    off = _cfg(retrieval={"rerank": False})
    assert needs_restart(off, Config()) == ["retrieval.rerank"]  # the model must be loaded
    assert needs_restart(Config(), off) == []  # turning it off applies to the next question


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"models": {"embedder": "nomic-embed-text"}}, True),
        ({"chunking": {"child_tokens": 128}}, True),
        ({"chunking": {"overlap": 0.2}}, True),
        ({"models": {"llm": "qwen3:8b"}}, False),
        ({"retrieval": {"hybrid": False}}, False),
        ({}, False),
    ],
)
def test_needs_reindex(changes: dict[str, dict[str, Any]], expected: bool) -> None:
    assert needs_reindex(Config(), _cfg(**changes)) is expected
