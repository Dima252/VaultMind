"""File -> ParsedDocument. PDF via pymupdf4llm; Markdown and plain text natively."""

from __future__ import annotations

from pathlib import Path

from vaultmind.core.models import ParsedDocument

SUPPORTED_EXTENSIONS = frozenset({".pdf", ".md", ".markdown", ".txt"})


class UnsupportedFileError(Exception):
    """The file extension is not in SUPPORTED_EXTENSIONS."""


class ParseError(Exception):
    """The file could not be parsed (corrupt, encrypted, no text layer, ...)."""


def parse(path: Path) -> ParsedDocument:
    raise NotImplementedError
