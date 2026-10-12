"""Export an answer or a whole chat as Markdown or PDF, with its citations as file + page.

The PDF is rendered offline by PyMuPDF's `Story` from HTML converted from the small Markdown
subset this module writes (headings, paragraphs, lists, bold). Paragraphs whose first strong
character is right-to-left (Hebrew, Arabic) get `dir="rtl"`. The font is a system font with
Hebrew glyphs (Arial, DejaVu Sans, ...). If none is found, MuPDF's own fonts are used: Latin
text in its built-in serif, and Hebrew through MuPDF's bundled Noto Serif Hebrew fallback, which
stays legible but has no bold face. Nothing is downloaded either way.
"""

from __future__ import annotations

import html
import io
import re
import unicodedata
from collections.abc import Sequence
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import pymupdf as _pymupdf

from vaultmind.agent.baseline import cited_sources
from vaultmind.store.chats import Chat, ChatMessage

pymupdf: Any = _pymupdf  # partially typed upstream

# (regular, bold) font files with Latin and Hebrew glyphs, in order of preference.
_FONT_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ("C:/Windows/Fonts/tahoma.ttf", "C:/Windows/Fonts/tahomabd.ttf"),
    (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ),
    ("/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf"),
    (
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "",
    ),
    (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ),
    ("/usr/share/fonts/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"),
    (
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
    ),
    (
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ),
    (
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ),
)
_HEBREW_PROBE = 0x05D0  # alef

_CSS = """
body { font-size: 11pt; line-height: 1.35; }
h1 { font-size: 18pt; margin: 0 0 6pt 0; }
h2 { font-size: 14pt; margin: 16pt 0 6pt 0; }
h3 { font-size: 12pt; margin: 12pt 0 4pt 0; }
p { margin: 0 0 6pt 0; }
p.item { margin: 0 0 3pt 14pt; }
p.item-rtl { margin: 0 14pt 3pt 0; }
"""
_MARGIN = 50  # points
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_BULLET = re.compile(r"^[-*+]\s+(.*)$")
_NUMBERED = re.compile(r"^(\d+[.)])\s+(.*)$")
_BOLD = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
_SOURCES, _CLOSEST = "Sources", "Closest matches"
_SOURCE_LABELS = {f"**{_SOURCES}**", f"**{_CLOSEST}**"}


def answer_markdown(question: ChatMessage, answer: ChatMessage) -> str:
    """One exchange: the question, the answer, then the sources it cited (or closest matches)."""
    return "\n\n".join([_question_md(question), *_answer_md(answer)]) + "\n"


def chat_markdown(
    chat: Chat, messages: Sequence[ChatMessage], *, now: datetime | None = None
) -> str:
    """The whole chat: title, export time, and each exchange as in `answer_markdown`.

    `now` (local time, default: the current time) is for tests.
    """
    stamp = (now or datetime.now().astimezone()).strftime("%Y-%m-%d %H:%M")
    blocks = [f"# {_one_line(chat.title)}", f"Exported from VaultMind on {stamp}"]
    for message in messages:
        if message.role == "user":
            blocks.append(_question_md(message))
        else:
            blocks.extend(_answer_md(message))
    return "\n\n".join(blocks) + "\n"


def markdown_pdf(markdown: str) -> bytes:
    """Render exported Markdown as a PDF, offline (no downloaded fonts). Hebrew must be legible."""
    font = _font_files()
    css = _CSS
    archive: Any = None
    if font is not None:
        regular, bold = font
        archive = pymupdf.Archive()
        archive.add(regular.read_bytes(), "vm-regular" + regular.suffix.lower())
        faces = [f"@font-face {{ font-family: vm; src: url(vm-regular{regular.suffix.lower()}); }}"]
        if bold is not None:
            archive.add(bold.read_bytes(), "vm-bold" + bold.suffix.lower())
            faces.append(
                "@font-face { font-family: vm; font-weight: bold; "
                f"src: url(vm-bold{bold.suffix.lower()}); }}"
            )
        css = "\n".join(faces) + "\nbody { font-family: vm; }" + css

    buffer = io.BytesIO()
    story = pymupdf.Story(html=_to_html(markdown), user_css=css, archive=archive)
    writer = pymupdf.DocumentWriter(buffer)
    page = pymupdf.paper_rect("a4")
    where = pymupdf.Rect(_MARGIN, _MARGIN, page.width - _MARGIN, page.height - _MARGIN)
    more = True
    while more:
        device = writer.begin_page(page)
        more, _ = story.place(where)
        story.draw(device)
        writer.end_page()
    writer.close()

    # Story embeds whole font files (Arial is ~1 MB); keep only the glyphs used.
    doc = pymupdf.open(stream=buffer.getvalue(), filetype="pdf")
    try:
        title = next((m.group(2) for m in map(_HEADING.match, markdown.splitlines()) if m), "")
        doc.set_metadata({"title": title, "creator": "VaultMind"})
        doc.subset_fonts()
        data: bytes = doc.tobytes(garbage=3, deflate=True)
        return data
    finally:
        doc.close()


def _question_md(question: ChatMessage) -> str:
    return f"## {_one_line(question.content)}"


def _answer_md(answer: ChatMessage) -> list[str]:
    blocks = [answer.content.strip()] if answer.content.strip() else []
    if answer.abstained:
        heading, listed = _CLOSEST, list(answer.sources)
    else:
        heading, listed = _SOURCES, cited_sources(answer.content, answer.sources)
        listed = listed or list(answer.sources)
    if listed:
        blocks.append(f"**{heading}**")
        blocks.append("\n".join(f"- [{s.n}] {s.label}" for s in listed))
    for c in answer.contradictions:
        blocks.append(
            f"**Note:** sources [{c.source_a}] and [{c.source_b}] disagree: "
            f'"{_one_line(c.claim_a)}" vs. "{_one_line(c.claim_b)}"'
        )
    for u in answer.unsupported:
        cited = "".join(f"[{n}]" for n in u.cited) or "no source"
        reason = f"; {_one_line(u.reason)}" if u.reason.strip() else ""
        blocks.append(
            f'**Note:** possibly unsupported by its sources: "{_one_line(u.claim)}" '
            f"(cited {cited}{reason})"
        )
    return blocks


def _one_line(text: str) -> str:
    return " ".join(text.split())


@lru_cache(maxsize=1)
def _font_files() -> tuple[Path, Path | None] | None:
    """The first installed (regular, bold) pair whose regular face has Hebrew glyphs."""
    for regular_name, bold_name in _FONT_CANDIDATES:
        regular = Path(regular_name)
        if not regular.is_file():
            continue
        try:
            if not pymupdf.Font(fontfile=str(regular)).has_glyph(_HEBREW_PROBE):
                continue
        except Exception:  # unreadable or unsupported font file: try the next one
            continue
        bold = Path(bold_name) if bold_name else None
        return regular, bold if bold is not None and bold.is_file() else None
    return None


def _is_rtl(text: str) -> bool:
    """True when the first strong directional character is right-to-left (Hebrew, Arabic)."""
    for ch in text:
        kind = unicodedata.bidirectional(ch)
        if kind in ("R", "AL"):
            return True
        if kind == "L":
            return False
    return False


def _inline(text: str) -> str:
    escaped = html.escape(text, quote=False)
    return _BOLD.sub(lambda m: f"<b>{m.group(1) or m.group(2)}</b>", escaped)


def _block(tag: str, lines: Sequence[str], css_class: str = "", *, ltr: bool = False) -> str:
    """One HTML block; its direction comes from the Markdown text, before any markup."""
    rtl = not ltr and _is_rtl(" ".join(lines))
    attrs = ' dir="rtl"' if rtl else ""
    if css_class:
        attrs += f' class="{css_class}-rtl"' if rtl else f' class="{css_class}"'
    return f"<{tag}{attrs}>{'<br>'.join(_inline(ln) for ln in lines)}</{tag}>"


def _to_html(markdown: str) -> str:
    """Headings, paragraphs (single newlines kept as line breaks), lists and bold.

    List items are paragraphs with their own marker: MuPDF keeps list markers on the left even
    in a right-to-left list, so a drawn marker is what puts a Hebrew item's bullet on the right.
    Source lists stay left-to-right: a label is "[n] file, p. N, section" whatever the file
    name's script, and a Hebrew name still reads correctly as an embedded run.
    """
    out: list[str] = []
    paragraph: list[str] = []
    item: list[str] | None = None  # the open list item: marker + text lines
    source_list = False  # inside the list under a Sources / Closest matches label

    def flush() -> None:
        nonlocal item, source_list
        if paragraph:
            out.append(_block("p", paragraph))
            source_list = len(paragraph) == 1 and paragraph[0] in _SOURCE_LABELS
            paragraph.clear()
        if item is not None:
            out.append(_block("p", item, "item", ltr=source_list))
            item = None

    for raw in markdown.splitlines():
        line = raw.strip()
        heading = _HEADING.match(line)
        bullet = _BULLET.match(line)
        numbered = _NUMBERED.match(line)
        if not line:
            list_ends = item is not None
            flush()
            source_list = source_list and not list_ends
        elif heading:
            flush()
            source_list = False
            out.append(_block(f"h{min(len(heading.group(1)), 3)}", [heading.group(2)]))
        elif bullet or numbered:
            flush()
            if bullet:
                item = ["\u2022 " + bullet.group(1)]
            elif numbered:
                item = [f"{numbered.group(1)} {numbered.group(2)}"]
        elif item is not None and raw[:1].isspace():
            item.append(line)  # an indented continuation of the list item
        else:
            if item is not None:
                flush()
            paragraph.append(line)
    flush()
    return "\n".join(out)
