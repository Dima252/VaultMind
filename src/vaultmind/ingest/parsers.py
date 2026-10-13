"""File -> ParsedDocument. PDF via pymupdf4llm; Markdown and plain text natively."""

from __future__ import annotations

import bisect
import re
import unicodedata
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


RTL_PAGE_SHARE = 0.2  # pages where at least this share of letters is RTL use _rtl_markdown
HIDDEN_JUNK_RATIO = 0.15  # hidden text layers with more suspicious tokens than this are rejected


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
            scans = [_scan_page(page) for page in doc]
        except Exception as e:
            raise ParseError(f"cannot extract text from {path.name}: {e}") from e
        _check_text_layer(path, scans)
        rtl = [i for i, scan in enumerate(scans) if scan.rtl_share >= RTL_PAGE_SHARE]
        ltr = sorted(set(range(len(scans))) - set(rtl))
        try:
            md_pages: dict[int, str] = _rtl_markdown(doc, rtl, scans)
            if ltr:
                chunks = pymupdf4llm.to_markdown(
                    doc, pages=ltr, page_chunks=True, use_ocr=False, show_progress=False
                )
                for i, chunk in zip(ltr, chunks, strict=False):
                    page_no = int(chunk.get("metadata", {}).get("page_number") or i + 1)
                    md_pages[page_no - 1] = str(chunk.get("text") or "").replace("\r\n", "\n")
        except Exception as e:
            raise ParseError(f"cannot extract text from {path.name}: {e}") from e
        meta_title = str((doc.metadata or {}).get("title") or "").strip()

    state = _HeadingState()
    blocks: list[Block] = []
    title: str | None = None
    for n, i in enumerate(sorted(md_pages)):
        blocks.extend(_markdown_blocks(md_pages[i], i + 1, state))
        if n == 0:
            title = state.first_h1  # a later H1 ("Learn more!") is no document title
    return title or meta_title or None, blocks


# --- PDF text layer ------------------------------------------------------------------


@dataclass(frozen=True)
class _Glyph:
    """One character (plus any combining marks) with its position on the page."""

    text: str
    x0: float
    x1: float
    baseline: float
    size: float
    line: int  # index of the MuPDF line it came from


@dataclass
class _PageScan:
    glyphs: list[_Glyph]  # visible, horizontal text only
    hidden: list[str]  # text of invisible spans (OCR / handwriting-recognition layers)
    visible: int  # visible non-space characters in any direction
    rtl_share: float  # RTL letters / all letters among visible glyphs


def _scan_page(page: Any) -> _PageScan:
    import pymupdf

    fitz: Any = pymupdf
    painted = fitz.mupdf.FZ_STEXT_FILLED | fitz.mupdf.FZ_STEXT_STROKED
    flags = fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_MEDIABOX_CLIP
    glyphs: list[_Glyph] = []
    hidden: list[str] = []
    visible = line_no = 0
    for block in page.get_text("rawdict", flags=flags)["blocks"]:
        for line in block.get("lines", ()):
            line_no += 1
            horizontal = abs(line["dir"][0] - 1) < 1e-3
            invisible = ""
            for span in line["spans"]:
                text = "".join(c["c"] for c in span["chars"])
                if span.get("alpha", 255) == 0 or not span.get("char_flags", painted) & painted:
                    invisible += text
                    continue
                visible += len("".join(text.split()))
                if horizontal:
                    glyphs.extend(_span_glyphs(span, line_no))
            if invisible:
                hidden.append(invisible)
    letters = [g.text[0] for g in glyphs if g.text[0].isalpha()]
    rtl = sum(1 for c in letters if _is_rtl(c))
    return _PageScan(glyphs, hidden, visible, rtl / len(letters) if letters else 0.0)


# Office documents write Symbol and Wingdings characters as private-use code points
# (U+F000 + the font's own code), which carry Greek letters, arrows and math operators.
_SYMBOL_FONT = {  # look-alike characters are the point of this table
    **{0x41 + i: ch for i, ch in enumerate("ΑΒΧΔΕΦΓΗΙϑΚΛΜΝΟΠΘΡΣΤΥςΩΞΨΖ")},
    **{0x61 + i: ch for i, ch in enumerate("αβχδεφγηιϕκλμνοπθρστυϖωξψζ")},
    **dict(zip(b"\"$'*-@^", "∀∃∋∗−≅⊥", strict=True)),  # noqa: RUF001
    **dict(zip(range(0xA3, 0xBD), "≤⁄∞ƒ♣♦♥♠↔←↑→↓°±″≥×∝∂•÷≠≡≈…", strict=True)),  # noqa: RUF001
    **dict(zip(range(0xC4, 0xE0), "⊗⊕∅∩∪⊃⊇⊄⊂⊆∈∉∠∇®©™∏√⋅¬∧∨⇔⇐⇑⇒⇓", strict=True)),  # noqa: RUF001
    0xE5: "∑",
}
_WINGDINGS_FONT = {0x9F: "•", 0xA7: "▪", 0xD8: "➢", 0xDF: "←", 0xE0: "→", 0xE1: "↑", 0xE2: "↓"}


def _private_use(ch: str, font: str) -> str:
    """The standard character for a Symbol/Wingdings private-use code point, else ""."""
    code = ord(ch) - 0xF000
    if "Symbol" in font and 0x20 <= code <= 0xFF:
        return _SYMBOL_FONT.get(code, chr(code) if code < 0x7F else "")
    if "Wingdings" in font:
        return _WINGDINGS_FONT.get(code, "")
    return ""


def _span_glyphs(span: dict[str, Any], line: int) -> list[_Glyph]:
    """Glyphs of one span. Combining marks stay with their base character; spaces (they are
    rebuilt from gaps) and invisible format characters are dropped; private-use characters
    of symbol fonts are translated, other private-use icons dropped."""
    out: list[_Glyph] = []
    size = span["size"]
    for c in span["chars"]:
        ch = c["c"]
        if unicodedata.category(ch) == "Co":
            ch = _private_use(ch, span.get("font", ""))
        if not ch:
            continue
        if unicodedata.combining(ch) and out:
            prev = out[-1]
            out[-1] = _Glyph(prev.text + ch, prev.x0, prev.x1, prev.baseline, prev.size, line)
        elif not (ch.isspace() or unicodedata.category(ch) == "Cf"):
            x0, _, x1, _ = c["bbox"]
            out.append(_Glyph(ch, x0, x1, c["origin"][1], size, line))
    return out


_SUSPICIOUS_TOKEN = re.compile(r"(?=.*[^\W\d_])(?=.*\d)|.{21,}")


def _check_text_layer(path: Path, scans: list[_PageScan]) -> None:
    """Reject PDFs without text, and those whose text is mostly a garbled hidden layer.

    Hidden text comes from OCR or handwriting recognition done by another app. A good layer
    is fine (pymupdf4llm handles it); a garbled one (words mixed with digits, run-together
    tokens) would only add junk chunks, so it is treated like a scan without text.
    """
    visible = sum(s.visible for s in scans)
    hidden_text = " ".join(h for s in scans for h in s.hidden)
    hidden = len("".join(hidden_text.split()))
    if not visible and not hidden:
        raise ParseError(
            f"{path.name} has no extractable text (likely scanned); OCR isn't supported yet"
        )
    if hidden <= visible:
        return
    tokens = [t.strip(".,:;!?()[]{}\"'") for t in hidden_text.split()]
    tokens = [t for t in tokens if len(t) >= 3]
    suspicious = sum(1 for t in tokens if _SUSPICIOUS_TOKEN.match(t))
    if not tokens or suspicious / len(tokens) > HIDDEN_JUNK_RATIO:
        raise ParseError(
            f"{path.name} has no usable text layer (its hidden OCR or handwriting-recognition "
            "text is garbled); OCR isn't supported yet"
        )


# --- right-to-left PDF pages ---------------------------------------------------------
#
# pymupdf4llm orders the pieces of a line left to right, which scrambles Hebrew and Arabic
# lines, and some PDF producers store RTL text in visual order anyway. Pages with enough RTL
# text are therefore rebuilt from glyph positions: glyphs are grouped into visual rows,
# sorted by x, and turned from visual into logical order with a simplified Unicode bidi
# algorithm. Headings come from font size, tables from PyMuPDF's table finder.

_RTL_RANGES = ((0x0590, 0x08FF), (0xFB1D, 0xFDFF), (0xFE70, 0xFEFF))
_MIRROR = str.maketrans("()[]{}<>«»", ")(][}{><»«")
_NUMBER_INNER = frozenset(".,:/-")  # kept inside a number: 2.5, 15.9.2026, 10:30
_NUMBER_AFFIX = frozenset("%‰°")
WORD_GAP = 0.15  # em: a wider gap between glyphs is a space
SEGMENT_GAP = 2.5  # em: a wider gap splits a row into separately ordered segments
HEADING_RATIO = 1.15  # rows at least this much larger than body text are headings
MAX_HEADING_CHARS = 150


def _is_rtl(ch: str) -> bool:
    o = ord(ch)
    return any(lo <= o <= hi for lo, hi in _RTL_RANGES)


def _strong(ch: str) -> str | None:
    """Bidi direction of a letter ("R" or "L"); None for digits, punctuation and symbols."""
    if not ch.isalpha():
        return None
    return "R" if _is_rtl(ch) else "L"


def _is_rtl_text(units: list[str]) -> bool:
    """Base direction: RTL when RTL letters are at least a quarter of the letters (this only
    runs on RTL pages, where a few English terms in a Hebrew line are common)."""
    kinds = [_strong(u[0]) for u in units]
    r, n = kinds.count("R"), kinds.count("R") + kinds.count("L")
    return r > 0 and 4 * r >= n


def _tokens(units: list[str]) -> tuple[list[str], list[str | None]]:
    """Visual-order units -> token texts and kinds ("R", "L", "D" for a number, None for a
    neutral). A number with its inner separators and affixes (``15.9.2026``, ``80%``) is one
    token, so its digits never get reversed."""
    texts: list[str] = []
    kinds: list[str | None] = []
    i, n = 0, len(units)
    while i < n:
        if not units[i][0].isdecimal():
            texts.append(units[i])
            kinds.append(_strong(units[i][0]))
            i += 1
            continue
        j = i + 1
        while j < n:
            if units[j][0].isdecimal():
                j += 1
            elif units[j] in _NUMBER_INNER and j + 1 < n and units[j + 1][0].isdecimal():
                j += 2
            else:
                break
        number = "".join(units[i:j])
        if texts and texts[-1] in _NUMBER_AFFIX and kinds[-1] is None:
            number = texts.pop() + number
            kinds.pop()
        if j < n and units[j] in _NUMBER_AFFIX:
            number += units[j]
            j += 1
        texts.append(number)
        kinds.append("D")
        i = j
    return texts, kinds


def _join_ltr_stretches(kinds: list[str | None]) -> None:
    """Between two RTL letters, everything from the first to the last LTR letter or number
    is one LTR run when it has an LTR letter: formulas (``x + y = 5``) and English phrases
    keep their visual order even with numbers and operators inside."""
    start = 0
    for end in [*(k for k, kind in enumerate(kinds) if kind == "R"), len(kinds)]:
        span = [k for k in range(start, end) if kinds[k] in ("L", "D")]
        if any(kinds[k] == "L" for k in span):
            for k in range(span[0], span[-1] + 1):
                kinds[k] = "L"
        start = end + 1


_OPENERS = {"(": ")", "[": "]", "{": "}"}
_CLOSERS = {v: k for k, v in _OPENERS.items()}


def _pair_brackets(texts: list[str], kinds: list[str | None]) -> dict[int, bool]:
    """Match bracket pairs in visual order; a pair whose content has a single direction
    takes that direction. Returns {index: is the left one of its pair}.

    Producers differ: most store the glyph that is shown, so a pair reads ``(...)`` left to
    right even in RTL text; others store the logical character and let the font mirror it,
    so the pair reads ``)...(``. Both are matched; the logical form is set from position.
    """
    sides: dict[int, bool] = {}
    stack: list[tuple[str, int]] = []
    for k, text in enumerate(texts):
        if text not in _OPENERS and text not in _CLOSERS:
            continue
        top = stack[-1][0] if stack else ""
        if _OPENERS.get(top) != text and _CLOSERS.get(top) != text:
            stack.append((text, k))
            continue
        _, open_k = stack.pop()
        sides[open_k], sides[k] = True, False
        inner = {kind for kind in kinds[open_k + 1 : k] if kind in ("R", "L")}
        if len(inner) == 1:
            kinds[open_k] = kinds[k] = inner.pop()
    return sides


def _bracket(texts: list[str], k: int, rtl_run: bool, left: bool | None) -> str:
    """The logical form of the bracket at visual index k. In an RTL run the visually right
    bracket of a pair opens it. An unpaired one opens when it sits at the right edge of a
    word in an RTL run (the left edge in an LTR run)."""
    text = texts[k]
    pair = (text, _OPENERS[text]) if text in _OPENERS else (_CLOSERS[text], text)
    if left is None:
        before = k > 0 and not texts[k - 1].isspace()
        after = k + 1 < len(texts) and not texts[k + 1].isspace()
        if before == after:  # no clue from the position
            return text.translate(_MIRROR) if rtl_run else text
        left = after
    return pair[0] if left != rtl_run else pair[1]


def _visual_to_logical(units: list[str], rtl: bool) -> str:
    """Reorder one visual line (units left to right) into logical order.

    A simplified inverse of the Unicode bidi algorithm: neutrals and numbers take the
    direction of their strong neighbours when both agree, else the base direction. Runs
    against the base direction are reversed; brackets get their logical form back.
    """
    texts, kinds = _tokens(units)
    if rtl and len(kinds) > 1 and kinds[-1] == "D" and texts[-2] in (".", ")"):
        kinds[-2] = kinds[-1] = "R"  # a list marker ("2." shown as ".2") starts an RTL line
    _join_ltr_stretches(kinds)
    sides = _pair_brackets(texts, kinds)
    base = "R" if rtl else "L"
    strong = [k if k in ("R", "L") else None for k in kinds]
    prev: list[str] = []
    last = base
    for kind in strong:
        prev.append(last)
        last = kind or last
    dirs: list[str] = [""] * len(kinds)
    nxt = base
    for k in range(len(kinds) - 1, -1, -1):
        dirs[k] = strong[k] or (prev[k] if prev[k] == nxt else base)
        nxt = strong[k] or nxt
    runs: list[tuple[str, list[str]]] = []
    for k, (text, d) in enumerate(zip(texts, dirs, strict=True)):
        if text in _OPENERS or text in _CLOSERS:
            text = _bracket(texts, k, d == "R", sides.get(k))
        elif d == "R":
            text = text.translate(_MIRROR)
        if runs and runs[-1][0] == d:
            runs[-1][1].append(text)
        else:
            runs.append((d, [text]))
    parts = ["".join(reversed(t)) if d == "R" else "".join(t) for d, t in runs]
    return "".join(reversed(parts) if rtl else parts)


def _segments(glyphs: list[_Glyph]) -> list[list[str]]:
    """Units (glyph texts and synthesized spaces) of a row, split where the gap is very wide."""
    ordered = sorted(glyphs, key=lambda g: (g.x0 + g.x1) / 2)
    segments: list[list[str]] = [[]]
    prev: _Glyph | None = None
    for g in ordered:
        if prev is not None:
            em = min(g.size, prev.size)
            gap = g.x0 - prev.x1
            if g.text == prev.text and abs(g.x0 - prev.x0) < 0.1 * em:
                continue  # the same glyph drawn twice (fake bold, shadows)
            if gap > SEGMENT_GAP * em:
                segments.append([])
            elif gap > WORD_GAP * em:
                segments[-1].append(" ")
        segments[-1].append(g.text)
        prev = g
    return segments


def _row_text(glyphs: list[_Glyph]) -> str:
    """Logical text of one visual row. A row without letters (``.1``, ``52``) is RTL (this
    only runs on RTL pages); a segment without letters takes the row's direction."""
    segments = _segments(glyphs)
    units = [u for s in segments for u in s]
    rtl = _is_rtl_text(units) if _has_letters(units) else True
    texts = [_visual_to_logical(s, _is_rtl_text(s) if _has_letters(s) else rtl) for s in segments]
    return "  ".join(reversed(texts) if rtl else texts).strip()


def _has_letters(units: list[str]) -> bool:
    return any(u[0].isalpha() for u in units)


ROW_BASELINE_TOLERANCE = 0.3  # em: MuPDF lines this close in baseline share a row


@dataclass
class _Row:
    glyphs: list[_Glyph]
    baseline: float
    top: float
    bottom: float
    spans: list[tuple[float, float]]  # x extents of the glyphs, sorted

    @property
    def size(self) -> float:
        """Most common font size in the row, rounded to a point."""
        sizes = [round(g.size) for g in self.glyphs]
        return float(max(set(sizes), key=sizes.count))

    def collides(self, line: list[_Glyph]) -> bool:
        """True when a glyph of `line` covers a glyph already in the row (another text box
        at the same height, e.g. a callout over the body text)."""
        for g in line:
            k = bisect.bisect_left(self.spans, (g.x1,))
            if not k:
                continue
            x0, x1 = self.spans[k - 1]
            if min(x1, g.x1) - max(x0, g.x0) > 0.25 * min(x1 - x0, g.x1 - g.x0):
                return True
        return False

    def add(self, line: list[_Glyph], top: float, bottom: float) -> None:
        self.glyphs.extend(line)
        self.top, self.bottom = min(top, self.top), max(bottom, self.bottom)
        self.spans = sorted(self.spans + [(g.x0, g.x1) for g in line])


def _rows(glyphs: list[_Glyph]) -> list[_Row]:
    """Group glyphs into visual rows: MuPDF lines with (nearly) the same baseline that don't
    overlap horizontally. RTL text often comes as many short lines per visual row."""
    lines: dict[int, list[_Glyph]] = {}
    for g in glyphs:
        lines.setdefault(g.line, []).append(g)
    bands = []
    for line in lines.values():
        size = max(g.size for g in line)
        main = sorted(g.baseline for g in line if g.size >= 0.9 * size)  # not sub/superscripts
        bands.append((main[len(main) // 2], size, line))
    bands.sort(key=lambda b: b[0])
    rows: list[_Row] = []
    for baseline, size, line in bands:
        top, bottom = baseline - 0.75 * size, baseline + 0.25 * size
        for row in reversed(rows[-4:]):
            tolerance = ROW_BASELINE_TOLERANCE * max(size, row.bottom - row.top)
            if abs(baseline - row.baseline) <= tolerance and not row.collides(line):
                row.add(line, top, bottom)
                break
        else:
            rows.append(_Row([], baseline, top, bottom, []))
            rows[-1].add(line, top, bottom)
    rows.sort(key=lambda r: r.baseline)
    return rows


@dataclass
class _Table:
    top: float
    markdown: str


def _tables(page: Any, glyphs: list[_Glyph]) -> tuple[list[_Table], list[_Glyph]]:
    """Markdown for the page's ruled tables (RTL tables keep their right-to-left column order)
    and the glyphs outside them."""
    try:
        found = page.find_tables().tables
    except Exception:  # the table finder is heuristic; fall back to plain rows
        return [], glyphs
    tables: list[_Table] = []
    rest = glyphs
    for table in found:
        rows = [row.cells for row in table.rows]
        if len(rows) < 2 or max(len(cells) for cells in rows) < 2:
            continue  # a box around text, not a table
        x0, y0, x1, y1 = table.bbox
        inside = [g for g in rest if x0 <= (g.x0 + g.x1) / 2 <= x1 and y0 <= g.baseline <= y1]
        cell_rows = [[_cell_text(inside, cell) for cell in cells] for cells in rows]
        if _is_rtl_text([c for cells in cell_rows for text in cells for c in text]):
            cell_rows = [cells[::-1] for cells in cell_rows]
        lines = ["| " + " | ".join(cells) + " |" for cells in cell_rows]
        lines.insert(1, "|" + "---|" * len(cell_rows[0]))
        tables.append(_Table(y0, "\n".join(lines)))
        ids = {id(g) for g in inside}
        rest = [g for g in rest if id(g) not in ids]
    return tables, rest


def _cell_text(glyphs: list[_Glyph], bbox: tuple[float, float, float, float] | None) -> str:
    if bbox is None:
        return ""
    x0, y0, x1, y1 = bbox
    mine = [g for g in glyphs if x0 <= (g.x0 + g.x1) / 2 <= x1 and y0 <= g.baseline <= y1]
    return " ".join(_row_text(r.glyphs) for r in _rows(mine)).replace("|", "\\|")


def _rtl_markdown(doc: Any, pages: list[int], scans: list[_PageScan]) -> dict[int, str]:
    """Markdown for the given (RTL) pages, with heading levels shared across the document."""
    layouts: dict[int, tuple[list[_Row], list[_Table]]] = {}
    for i in pages:
        tables, rest = _tables(doc[i], scans[i].glyphs)
        layouts[i] = (_rows(rest), tables)
    levels = _heading_levels([row for rows, _ in layouts.values() for row in rows])
    return {i: _page_markdown(rows, tables, levels) for i, (rows, tables) in layouts.items()}


def _heading_levels(rows: list[_Row]) -> dict[float, int]:
    """Font size -> heading level (1-3). Body text is the most common size by glyph count."""
    counts: dict[float, int] = {}
    for row in rows:
        counts[row.size] = counts.get(row.size, 0) + len(row.glyphs)
    if not counts:
        return {}
    body = max(counts, key=lambda s: counts[s])
    sizes = sorted({r.size for r in rows if r.size >= HEADING_RATIO * body}, reverse=True)
    return {size: min(level, 3) for level, size in enumerate(sizes, start=1)}


def _page_markdown(rows: list[_Row], tables: list[_Table], levels: dict[float, int]) -> str:
    items: list[tuple[float, _Row | _Table]] = [(r.top, r) for r in rows]
    items += [(t.top, t) for t in tables]
    items.sort(key=lambda item: item[0])
    paragraphs: list[str] = []
    prev: _Row | None = None
    prev_level = 0
    for _, item in items:
        if isinstance(item, _Table):
            paragraphs.append(item.markdown)
            prev = None
            continue
        text = _row_text(item.glyphs)
        if not text:
            continue
        level = levels.get(item.size, 0)
        if level and (len(text) > MAX_HEADING_CHARS or not any(c.isalpha() for c in text)):
            level = 0
        close = (
            prev is not None
            and level == prev_level
            and item.top - prev.bottom < 0.5 * prev.size
            and abs(item.size - prev.size) <= 0.15 * prev.size
        )
        if close:  # same paragraph, or a heading wrapped over two rows
            paragraphs[-1] += (" " if level else "\n") + text
        else:
            paragraphs.append("#" * level + " " + text if level else text)
        prev, prev_level = item, level
    return "\n\n".join(paragraphs) + "\n"
