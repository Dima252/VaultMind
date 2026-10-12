"""Settings page support: save only what differs from the defaults to `config/local.yaml`."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

from vaultmind.core.config import LOCAL_CONFIG, Config

_HEADER = (
    "# Written by VaultMind's Settings page. It holds only the values that differ from the\n"
    "# built-in defaults (config/default.yaml); everything else uses the default.\n"
)
# Read once when the app starts: the stores, the Ollama client, the loaded models, the window
# and the folder watcher. Retrieval and agent switches are read again for every question.
_RESTART_SECTIONS = frozenset({"paths", "ollama", "models", "ingest", "ui"})
_REINDEX_KEYS = ("models.embedder", "chunking.")


def overrides(cfg: Config) -> dict[str, Any]:
    """Nested dict of the values in `cfg` that differ from `Config()` (JSON/YAML-safe)."""
    return _diff(cfg.model_dump(mode="json"), Config().model_dump(mode="json"))


def save_local(cfg: Config, path: Path = LOCAL_CONFIG) -> None:
    """Write `overrides(cfg)` to `path` atomically, with a short header comment."""
    data = overrides(Config.model_validate(cfg.model_dump(mode="json")))  # raises if invalid
    text = _HEADER + yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def needs_restart(old: Config, new: Config) -> list[str]:
    """Dotted keys that changed and only take effect after a restart (`paths.*`, `ollama.*`,
    `models.*`, `ingest.*`, `ui.*`, and `retrieval.rerank` when it's switched on, since the
    reranker is loaded at startup). Retrieval and agent switches apply to the next question."""
    keys = [k for k in _changed(old, new) if k.split(".", 1)[0] in _RESTART_SECTIONS]
    if new.retrieval.rerank and not old.retrieval.rerank:
        keys.append("retrieval.rerank")
    return keys


def needs_reindex(old: Config, new: Config) -> bool:
    """True when the embedder or chunking changed: the index must be rebuilt."""
    return any(k == key or k.startswith(key) for k in _changed(old, new) for key in _REINDEX_KEYS)


def _diff(new: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in new.items():
        default = base.get(key)
        if isinstance(value, dict) and isinstance(default, dict):
            if nested := _diff(value, default):
                out[key] = nested
        elif value != default:
            out[key] = value
    return out


def _changed(old: Config, new: Config) -> list[str]:
    """Dotted leaf keys whose values differ, in config order."""
    return list(_leaves(_diff(new.model_dump(mode="json"), old.model_dump(mode="json"))))


def _leaves(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            out.update(_leaves(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out
