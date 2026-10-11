"""Synthetic adversarial documents: planted prompt injections and contradicting documents.

The fixtures are committed under ``eval/fixtures/adversarial/`` (fictional content, safe to
publish); their questions are in ``eval/datasets/adversarial.jsonl``.
"""

from __future__ import annotations

from pathlib import Path

FIXTURES_DIR = Path("eval/fixtures/adversarial")
SUBDIR = "adversarial"


def install_adversarial(vault_dir: Path, fixtures_dir: Path = FIXTURES_DIR) -> list[Path]:
    """Copy the fixtures into ``<vault_dir>/adversarial/`` and return the written paths.

    That folder belongs to this function: files there that aren't fixtures are removed.
    """
    raise NotImplementedError
