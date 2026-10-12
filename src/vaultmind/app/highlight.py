"""Show a cited passage in its source: a PDF page image with the passage highlighted, or the
character span in a Markdown/text file. The user's files are never modified.

The PDF is read into memory and annotated there; only `highlighted_pdf_copy` writes, and it
writes a new file. Finding the passage on the page:

1. strip the chunk's Markdown markup, cut it into sentence pieces (long ones into windows of
   ~8 words) and search each piece on the page;
2. when most pieces aren't found (e.g. right-to-left or mixed-direction text stored in another
   order), match single words of 3+ letters instead and highlight the lines of the densest
   contiguous run of them;
3. the cited page is tried first, then the pages before and after it (a chunk can straddle a
   page break); the first page with a match is shown. With none, the cited page unmarked.
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf as _pymupdf

pymupdf: Any = _pymupdf  # partially typed upstream


@dataclass(frozen=True)
class PageImage:
    png: bytes
    page: int  # 1-based, the page shown
    page_count: int
    highlighted: bool  # False: the passage wasn't found on the page; it's shown unmarked


def render_pdf_citation(pdf: Path, page: int, passage: str, *, zoom: float = 2.0) -> PageImage:
    """Render `page` of `pdf` with `passage` (a chunk's Markdown text) highlighted, in memory."""
    if zoom <= 0:
        raise ValueError(f"zoom must be positive, got {zoom}")
    doc = _open(pdf)
    try:
        shown, found = _annotate(doc, _clamp(page, doc.page_count), passage)
        pix = doc[shown].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), annots=True)
        return PageImage(pix.tobytes("png"), shown + 1, doc.page_count, found)
    finally:
        doc.close()


def highlighted_pdf_copy(pdf: Path, page: int, passage: str, out_dir: Path) -> Path:
    """A highlighted copy of `pdf` in `out_dir`, for opening in the system viewer."""
    out = out_dir / f"{pdf.stem}-highlight.pdf"
    doc = _open(pdf)
    try:
        shown, _ = _annotate(doc, _clamp(page, doc.page_count), passage)
        # Open the copy at the passage's page, in viewers that honour /OpenAction. A nicety:
        # the copy is still useful without it.
        with contextlib.suppress(Exception):
            target = f"[{doc[shown].xref} 0 R /XYZ null null null]"
            doc.xref_set_key(doc.pdf_catalog(), "OpenAction", target)
        out_dir.mkdir(parents=True, exist_ok=True)
        doc.save(out)
    finally:
        doc.close()
    return out


def find_passage(document: str, passage: str) -> tuple[int, int] | None:
    """The [start, end) span of `passage` in a Markdown/text document, ignoring whitespace and
    Markdown markup differences. None when it can't be located."""
    doc_text, index = _normalize(document)
    needle, _ = _normalize(passage)
    if not needle:
        return None
    if (pos := doc_text.find(needle)) >= 0:
        return index[pos], index[pos + len(needle) - 1] + 1
    # A difference in the middle (an edited line): anchor on both ends instead.
    if len(needle) < 2 * _ANCHOR:
        return None
    head, tail = needle[:_ANCHOR], needle[-_ANCHOR:]
    start = doc_text.find(head)
    while start >= 0:
        end = doc_text.find(tail, start + len(head))
        if end < 0:
            return None
        if end + len(tail) - start <= 1.5 * len(needle):
            return index[start], index[end + len(tail) - 1] + 1
        start = doc_text.find(head, start + 1)
    return None


# --- opening ---------------------------------------------------------------------------


def _open(pdf: Path) -> Any:
    """The PDF loaded from bytes, so nothing can be written back to (or lock) the user's file."""
    try:
        data = pdf.read_bytes()
    except OSError as e:
        raise ValueError(f"cannot read {pdf.name}: {e.strerror or e}") from e
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as e:  # MuPDF raises several types for broken files
        raise ValueError(f"{pdf.name} is not a readable PDF") from e
    if doc.needs_pass:
        doc.close()
        raise ValueError(f"{pdf.name} is password-protected, so it can't be shown")
    if doc.page_count == 0:
        doc.close()
        raise ValueError(f"{pdf.name} has no pages")
    return doc


def _clamp(page: int, page_count: int) -> int:
    """0-based index of the 1-based `page`, kept inside the document."""
    return min(max(page, 1), page_count) - 1


# --- annotating ------------------------------------------------------------------------

_WINDOW = 8  # words per searched piece
_MIN_WORDS = 4  # shorter pieces are too likely to match elsewhere
_MAX_PIECES = 64  # bounds the work for very long passages
_MIN_LETTERS = 3  # word-level matching ignores shorter words
_MIN_WORD_HITS = 4  # distinct words a word-level region needs (or all of a short passage's)
_MIN_WORD_SHARE = 0.3  # ... and at least this share of the distinct words that could match


def _annotate(doc: Any, index: int, passage: str) -> tuple[int, bool]:
    """Highlight `passage` near page `index`. Returns the page to show and whether it's marked."""
    pieces = _pieces(passage)
    words = _words(_strip_markdown(passage))
    for i in (index, index - 1, index + 1):  # the cited page first: it's what the label says
        if not 0 <= i < doc.page_count:
            continue
        page = doc[i]  # keep the page referenced: an annotation needs its page alive
        if quads := _page_quads(page, pieces, words):
            page.add_highlight_annot(quads).update()
            return i, True
    return index, False


def _page_quads(page: Any, pieces: list[str], words: set[str]) -> list[Any]:
    """Phrase hits for the pieces. When most pieces are missing (right-to-left or mixed-
    direction text stored in another order, unusual layouts), the word-level region instead,
    if there is one: it covers the passage's lines, phrase hits included."""
    hits = [h for piece in pieces if (h := page.search_for(piece, quads=True))]
    if 2 * len(hits) < len(pieces) and (region := _word_region(page, words)):
        return region
    return [q for h in hits for q in h]


def _pieces(passage: str) -> list[str]:
    """Searchable pieces of the passage: sentences, long ones cut into windows."""
    text = _strip_markdown(passage)
    pieces: list[str] = []
    for line in text.split("\n"):
        for sentence in _SENTENCE_END.split(line):
            pieces.extend(_windows(sentence.split()))
    if not pieces and text.strip():  # a short passage (a heading, a table cell): whole
        pieces = [" ".join(text.split())]
    return list(dict.fromkeys(pieces))[:_MAX_PIECES]


def _windows(words: list[str]) -> list[str]:
    if len(words) < _MIN_WORDS:
        return []
    if len(words) < _WINDOW + _MIN_WORDS:
        return [" ".join(words)]
    groups = [words[i : i + _WINDOW] for i in range(0, len(words), _WINDOW)]
    if len(groups[-1]) < _MIN_WORDS:
        groups[-2].extend(groups.pop())
    return [" ".join(g) for g in groups]


def _word_region(page: Any, words: set[str]) -> list[Any]:
    """One rect per line for the densest contiguous run of passage words on the page, in the
    page's stored word order. Empty when the run holds too few distinct passage words."""
    if not words:
        return []
    page_words = page.get_text("words")
    scores: list[int] = []  # matched words +1, other words -1, short words 0
    matched: list[str | None] = []
    on_page: set[str] = set()
    for *_, raw, _, _, _ in page_words:
        word = _word_key(raw)
        hit = None if word is None else word if word in words else word[::-1]  # visual order
        matched.append(hit if hit in words else None)
        scores.append(0 if word is None else 1 if hit in words else -1)
        if word is not None:
            on_page.add(word)
    start, end = _densest(scores)
    distinct = set(matched[start:end]) - {None}
    # A share of what could match: of the passage, or of the page when it holds only a part.
    possible = min(len(words), len(on_page))
    needed = min(len(words), max(_MIN_WORD_HITS, round(_MIN_WORD_SHARE * possible)))
    if len(distinct) < needed:
        return []
    lines: dict[tuple[int, int], Any] = {}  # (block, line) -> union of its words in the run
    marked: set[tuple[int, int]] = set()  # lines with a passage word (not just labels, numbers)
    for i in range(start, end):
        x0, y0, x1, y1, _, block, line, _ = page_words[i]
        rect = pymupdf.Rect(x0, y0, x1, y1)
        lines[block, line] = lines[block, line] | rect if (block, line) in lines else rect
        if matched[i] is not None:
            marked.add((block, line))
    return [rect for key, rect in lines.items() if key in marked]


def _densest(scores: list[int]) -> tuple[int, int]:
    """[start, end) of the maximum-sum run (Kadane); (0, 0) when nothing scores above 0."""
    best_sum, best, run_sum, run_start = 0, (0, 0), 0, 0
    for i, score in enumerate(scores):
        if run_sum <= 0:
            run_sum, run_start = 0, i
        run_sum += score
        if run_sum > best_sum:
            best_sum, best = run_sum, (run_start, i + 1)
    return best


def _words(text: str) -> set[str]:
    return {key for raw in text.split() if (key := _word_key(raw)) is not None}


def _word_key(raw: str) -> str | None:
    """A word without punctuation, lowercased; None when it has fewer than 3 letters."""
    word = _NON_WORD.sub("", raw).lower()
    return word if sum(ch.isalpha() for ch in word) >= _MIN_LETTERS else None


# --- Markdown --------------------------------------------------------------------------

_BR = re.compile(r"<br\s*/?>", re.IGNORECASE)
_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+")
_LIST_MARKER = re.compile(r"^\s*(?:[-*+•>]|\d{1,3}[.)])\s+")
_EMPHASIS = re.compile(r"\*\*|__|~~|`|(?<!\w)[*_]+(?=\S)|(?<=\S)[*_]+(?!\w)")
_RULE = re.compile(r"[-:=*_\s]*")  # table separator rows and horizontal rules
_SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+")
_NON_WORD = re.compile(r"[\W_]+")


def _strip_markdown(text: str) -> str:
    """The passage as plain text: one line per Markdown line or table cell, spaces collapsed."""
    text = _IMAGE.sub(r"\1", _BR.sub("\n", text))
    text = _LINK.sub(r"\1", text).replace("|", "\n")
    lines = []
    for line in text.split("\n"):
        line = _EMPHASIS.sub("", _LIST_MARKER.sub("", _HEADING.sub("", line)))
        if not _RULE.fullmatch(line):
            lines.append(" ".join(line.split()))
    return "\n".join(lines)


_ANCHOR = 40  # normalized characters matched at each end when the whole passage isn't found
_MARKUP = frozenset("*_#`~")  # dropped by find_passage; '|' and <br> count as whitespace


def _normalize(text: str) -> tuple[str, list[int]]:
    """Lowercased text without Markdown markup, whitespace collapsed to single spaces, and the
    original offset of every kept character."""
    out: list[str] = []
    index: list[int] = []
    space = False
    i = 0
    while i < len(text):
        if text[i] == "<" and (m := _BR.match(text, i)):
            space, i = True, m.end()
            continue
        ch = text[i]
        if ch.isspace() or ch == "|":
            space = True
        elif ch not in _MARKUP:
            if space and out:
                out.append(" ")
                index.append(i)
            space = False
            for low in ch.lower():
                out.append(low)
                index.append(i)
        i += 1
    return "".join(out), index
