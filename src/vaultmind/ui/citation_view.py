"""Citation viewer: the cited passage highlighted in its source (PDF page or text file)."""

from __future__ import annotations

from vaultmind.agent.baseline import Source


def open_citation(source: Source) -> None:
    """Open a dialog showing `source` in its file, highlighted (`app/highlight.py`)."""
    raise NotImplementedError
