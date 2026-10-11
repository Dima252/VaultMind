"""File -> ParsedDocument. PDF via pymupdf4llm; Markdown and plain text natively."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vaultmind.core.models import Block, ParsedDocument

SUPPORTED_EXTENSIONS = frozenset({".pdf", ".md", ".markdown", ".txt"})


class UnsupportedFileError(Exception):
    """The file extension is not in SUPPORTED_EXTENSIONS."""


class ParseError(Exception):
    """The file could not be parsed (corrupt, encrypted, no text layer, ...)."""


def parse(path: Path) -> ParsedDocument:
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileError(f"unsupported file type {ext or '(none)'!r}: {path.name}")
    if ext == ".pdf":
        title, blocks = _parse_pdf(path)
    elif ext == ".txt":
        title, blocks = None, _text_blocks(_read_text(path))
    else:
        state = _HeadingState()
        blocks = _markdown_blocks(_read_text(path), None, state)
        title = state.first_h1
    if not blocks:
        raise ParseError(f"{path.name} contains no text")
    return ParsedDocument(path=path, title=title or path.stem, blocks=blocks)


# --- text decoding -------------------------------------------------------------------


def _read_text(path: Path) -> str:
    try:
        data = path.read_bytes()
    except OSError as e:
        raise ParseError(f"cannot read {path.name}: {e}") from e
    try:
        text = data.decode("utf-8-sig")  # strips a UTF-8 BOM if present
    except UnicodeDecodeError as e:
        raise ParseError(
            f"{path.name} is not valid UTF-8 (bad byte at offset {e.start}); "
            "other encodings are not supported yet"
        ) from e
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _block(text: str, page: int | None, section_path: tuple[str, ...]) -> Block | None:
    text = text.strip()
    return Block(text=text, page=page, section_path=section_path) if text else None


# --- plain text ----------------------------------------------------------------------

_BLANK_LINES = re.compile(r"\n[ \t]*\n\s*")


def _text_blocks(text: str) -> list[Block]:
    blocks = (_block(para, None, ()) for para in _BLANK_LINES.split(text))
    return [b for b in blocks if b is not None]


# --- Markdown ------------------------------------------------------------------------

_ATX_HEADING = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?[ \t]*$")
_CLOSING_HASHES = re.compile(r"(?:^|[ \t]+)#+$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_EMPHASIS = ("**", "__", "*", "_")


@dataclass
class _HeadingState:
    """Heading hierarchy carried across pages (and the document's first H1)."""

    stack: list[tuple[int, str]] = field(default_factory=list)
    first_h1: str | None = None

    @property
    def path(self) -> tuple[str, ...]:
        return tuple(title for _, title in self.stack)

    def push(self, level: int, title: str) -> None:
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        self.stack.append((level, title))
        if level == 1 and self.first_h1 is None:
            self.first_h1 = title


def _heading(line: str) -> tuple[int, str] | None:
    m = _ATX_HEADING.match(line)
    if not m:
        return None
    title = _CLOSING_HASHES.sub("", m.group(2) or "").strip()
    for marker in _EMPHASIS:  # pymupdf4llm often emits "## **Bold heading**"
        if len(title) > 2 * len(marker) and title.startswith(marker) and title.endswith(marker):
            title = title[len(marker) : -len(marker)].strip()
    return (len(m.group(1)), title) if title else None


def _markdown_blocks(text: str, page: int | None, state: _HeadingState) -> list[Block]:
    """One block per heading-delimited span; the heading line starts its block.

    `state` is updated in place so a caller can carry the hierarchy across pages.
    Fences are tracked per call: a code fence never spans a PDF page boundary.
    """
    blocks: list[Block] = []
    lines: list[str] = []
    path = state.path
    fence: str | None = None

    def flush() -> None:
        if (b := _block("\n".join(lines), page, path)) is not None:
            blocks.append(b)
        lines.clear()

    for line in text.split("\n"):
        if fence is not None:
            m = _FENCE.match(line)
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence):
                fence = None
        elif m := _FENCE.match(line):
            fence = m.group(1)
        elif heading := _heading(line):
            flush()
            state.push(*heading)
            path = state.path
        lines.append(line)
    flush()
    return blocks


# --- PDF -----------------------------------------------------------------------------


def _parse_pdf(path: Path) -> tuple[str | None, list[Block]]:
    import pymupdf
    import pymupdf4llm

    fitz: Any = pymupdf  # partially typed upstream
    try:
        doc: Any = fitz.open(path)
    except Exception as e:  # FileDataError, EmptyFileError, FileNotFoundError, ...
        raise ParseError(f"cannot open {path.name} as PDF: {e}") from e
    with doc:
        if doc.needs_pass:
            raise ParseError(f"{path.name} is password-protected")
        try:
            has_text = any(page.get_text().strip() for page in doc)
            pages: list[dict[str, Any]] = (
                pymupdf4llm.to_markdown(doc, page_chunks=True, use_ocr=False, show_progress=False)
                if has_text
                else []
            )
        except Exception as e:
            raise ParseError(f"cannot extract text from {path.name}: {e}") from e
        if not has_text:
            raise ParseError(
                f"{path.name} has no extractable text (likely scanned); OCR isn't supported yet"
            )
        meta_title = str((doc.metadata or {}).get("title") or "").strip()

    state = _HeadingState()
    blocks: list[Block] = []
    for i, chunk in enumerate(pages):
        page_no = int(chunk.get("metadata", {}).get("page_number") or i + 1)
        text = str(chunk.get("text") or "").replace("\r\n", "\n")
        blocks.extend(_markdown_blocks(text, page_no, state))
    return state.first_h1 or meta_title or None, blocks
