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
    A file is rewritten only when its bytes changed, so re-running doesn't trigger a re-ingest.
    Raises FileNotFoundError if ``fixtures_dir`` is missing or holds no ``*.md`` fixtures.
    """
    if not fixtures_dir.is_dir():
        raise FileNotFoundError(f"adversarial fixtures not found: {fixtures_dir}")
    sources = sorted(p for p in fixtures_dir.glob("*.md") if p.is_file())
    if not sources:
        raise FileNotFoundError(f"no adversarial fixtures (*.md) in {fixtures_dir}")

    dest_dir = vault_dir / SUBDIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for src in sources:
        dest = dest_dir / src.name
        data = src.read_bytes()
        if not dest.is_file() or dest.read_bytes() != data:
            dest.write_bytes(data)
        written.append(dest)

    keep = {p.name for p in written}
    for stale in dest_dir.iterdir():
        if stale.name not in keep and (stale.is_file() or stale.is_symlink()):
            stale.unlink()
    return sorted(written)
