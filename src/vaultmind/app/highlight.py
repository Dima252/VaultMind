"""Show a cited passage in its source: a PDF page image with the passage highlighted, or the
character span in a Markdown/text file. The user's files are never modified."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PageImage:
    png: bytes
    page: int  # 1-based, the page shown
    page_count: int
    highlighted: bool  # False: the passage wasn't found on the page; it's shown unmarked


def render_pdf_citation(pdf: Path, page: int, passage: str, *, zoom: float = 2.0) -> PageImage:
    """Render `page` of `pdf` with `passage` (a chunk's Markdown text) highlighted, in memory."""
    raise NotImplementedError


def highlighted_pdf_copy(pdf: Path, page: int, passage: str, out_dir: Path) -> Path:
    """A highlighted copy of `pdf` in `out_dir`, for opening in the system viewer."""
    raise NotImplementedError


def find_passage(document: str, passage: str) -> tuple[int, int] | None:
    """The [start, end) span of `passage` in a Markdown/text document, ignoring whitespace and
    Markdown markup differences. None when it can't be located."""
    raise NotImplementedError
