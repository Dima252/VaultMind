"""Settings page support: save only what differs from the defaults to `config/local.yaml`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vaultmind.core.config import LOCAL_CONFIG, Config


def overrides(cfg: Config) -> dict[str, Any]:
    """Nested dict of the values in `cfg` that differ from `Config()` (JSON/YAML-safe)."""
    raise NotImplementedError


def save_local(cfg: Config, path: Path = LOCAL_CONFIG) -> None:
    """Write `overrides(cfg)` to `path` atomically, with a short header comment."""
    raise NotImplementedError


def needs_restart(old: Config, new: Config) -> list[str]:
    """Dotted keys that changed and only take effect after a restart (`paths.*`, `ollama.*`,
    `models.*`, `ui.*`). Retrieval and agent switches apply to the next question."""
    raise NotImplementedError


def needs_reindex(old: Config, new: Config) -> bool:
    """True when the embedder or chunking changed: the index must be rebuilt."""
    raise NotImplementedError
