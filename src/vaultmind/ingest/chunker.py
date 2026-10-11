"""ParsedDocument -> heading-aware parent sections, each split into overlapping child chunks."""

from __future__ import annotations

import bisect
import math
import re
from dataclasses import dataclass

from vaultmind.core.config import ChunkingConfig
from vaultmind.core.models import Block, ChildChunk, ParentSection, ParsedDocument

CHARS_PER_TOKEN = 4
_PARAGRAPH_SEP = re.compile(r"\n[ \t]*\n\s*")
_SENTENCE_SEP = re.compile(r"(?<=[.!?])\s+|\s*\n\s*")
_PARENT_JOIN = "\n\n"


def estimate_tokens(text: str) -> int:
    """Approximate token count (no tokenizer download needed)."""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def chunk_document(
    doc: ParsedDocument, file_id: str, cfg: ChunkingConfig
) -> tuple[list[ParentSection], list[ChildChunk]]:
    parent_chars = cfg.parent_tokens * CHARS_PER_TOKEN
    child_chars = cfg.child_tokens * CHARS_PER_TOKEN
    overlap_chars = int(child_chars * cfg.overlap)

    parents: list[ParentSection] = []
    children: list[ChildChunk] = []
    for i, pieces in enumerate(_group_parents(_pieces(doc.blocks, parent_chars), parent_chars)):
        text = _PARENT_JOIN.join(p.text for p in pieces)
        pages = [p.page for p in pieces if p.page is not None]
        parent = ParentSection(
            id=f"{file_id}:p{i}",
            file_id=file_id,
            text=text,
            section_path=pieces[0].section_path,
            page_start=min(pages) if pages else None,
            page_end=max(pages) if pages else None,
        )
        parents.append(parent)

        offsets: list[int] = []  # start of each piece inside the parent text
        pos = 0
        for p in pieces:
            offsets.append(pos)
            pos += len(p.text) + len(_PARENT_JOIN)
        for j, (start, end) in enumerate(_windows(text, child_chars, overlap_chars)):
            origin = pieces[bisect.bisect_right(offsets, start) - 1]
            chunk_text = text[start:end]
            children.append(
                ChildChunk(
                    id=f"{parent.id}:c{j}",
                    parent_id=parent.id,
                    file_id=file_id,
                    text=chunk_text,
                    embed_text=_context_header(doc.title, origin.section_path)
                    + "\n\n"
                    + chunk_text,
                    page=origin.page,
                    section_path=origin.section_path,
                )
            )
    return parents, children


def _context_header(title: str, section_path: tuple[str, ...]) -> str:
    """Build "Title > Section > Sub", not repeating the title when it is also the first heading."""
    if section_path and section_path[0] == title:
        section_path = section_path[1:]
    return " > ".join((title, *section_path))


# --- parents -------------------------------------------------------------------------


def _pieces(blocks: list[Block], max_chars: int) -> list[Block]:
    """Blocks with surrounding whitespace removed; any block over `max_chars` is split."""
    out: list[Block] = []
    for block in blocks:
        text = block.text.strip()
        if not text:
            continue
        if len(text) <= max_chars:
            out.append(Block(text=text, page=block.page, section_path=block.section_path))
            continue
        for start, end in _windows(text, max_chars, 0):
            out.append(Block(text[start:end], block.page, block.section_path))
    return out


def _group_parents(pieces: list[Block], max_chars: int) -> list[list[Block]]:
    """Greedy packing up to `max_chars`, preferring a new parent at a heading change."""
    groups: list[list[Block]] = []
    current: list[Block] = []
    size = 0
    for piece in pieces:
        if current:
            too_big = size + len(_PARENT_JOIN) + len(piece.text) > max_chars
            new_section = piece.section_path != current[-1].section_path
            if too_big or (new_section and size >= max_chars / 2):
                groups.append(current)
                current, size = [], 0
        size += len(piece.text) + (len(_PARENT_JOIN) if current else 0)
        current.append(piece)
    if current:
        groups.append(current)
    return groups


# --- windows -------------------------------------------------------------------------


@dataclass(frozen=True)
class _Unit:
    """An unbreakable span of text (a sentence or a piece of an over-long one)."""

    start: int
    end: int
    paragraph_end: bool


def _windows(text: str, max_chars: int, overlap_chars: int) -> list[tuple[int, int]]:
    """Split `text` into spans of at most `max_chars`, consecutive spans sharing about
    `overlap_chars`. Spans end at paragraph boundaries when that keeps them at least
    half full, otherwise at sentence boundaries, otherwise at whitespace or anywhere.
    """
    units = _units(text, max_chars)
    windows: list[tuple[int, int]] = []
    i = fresh = 0  # fresh: first unit not covered by the previous window
    while i < len(units):
        start = units[i].start
        j = i
        while j + 1 < len(units) and units[j + 1].end - start <= max_chars:
            j += 1
        if j + 1 < len(units) and not units[j].paragraph_end:
            for k in range(j - 1, max(i, fresh) - 1, -1):
                if units[k].end - start < max_chars / 2:
                    break
                if units[k].paragraph_end:
                    j = k
                    break
        windows.append((start, units[j].end))
        if j + 1 == len(units):
            break
        i, fresh = _next_start(units, i, j, max_chars, overlap_chars), j + 1
    return windows


def _next_start(units: list[_Unit], i: int, j: int, max_chars: int, overlap_chars: int) -> int:
    """First unit of the window after units[i..j], backed off to overlap the previous one."""
    nxt = j + 1
    if overlap_chars <= 0:
        return nxt
    end = units[j].end
    k = nxt
    while k - 1 > i and end - units[k - 1].start <= overlap_chars:
        k -= 1
    if k == nxt and j > i and units[j].end - units[j].start <= max_chars / 2:
        k = j  # the last unit alone is longer than the overlap: share it anyway
    while k < nxt and units[nxt].end - units[k].start > max_chars:
        k += 1  # the next window must still reach new text
    return k


def _units(text: str, max_chars: int) -> list[_Unit]:
    sentence_limit = max(1, max_chars // 2)
    piece_size = max(1, max_chars // 4)
    units: list[_Unit] = []
    for p_start, p_end in _spans(text, _PARAGRAPH_SEP, 0, len(text)):
        sentences = _spans(text, _SENTENCE_SEP, p_start, p_end)
        for s_start, s_end in sentences:
            if s_end - s_start <= sentence_limit:
                units.append(_Unit(s_start, s_end, paragraph_end=False))
            else:
                units.extend(_Unit(a, b, False) for a, b in _hard(text, s_start, s_end, piece_size))
        last = units[-1]
        units[-1] = _Unit(last.start, last.end, paragraph_end=True)
    return units


def _spans(text: str, sep: re.Pattern[str], start: int, end: int) -> list[tuple[int, int]]:
    """Non-blank spans of text[start:end] between matches of `sep`, trimmed of whitespace."""
    bounds: list[tuple[int, int]] = []
    pos = start
    for m in sep.finditer(text, start, end):
        bounds.append((pos, m.start()))
        pos = m.end()
    bounds.append((pos, end))
    return [t for a, b in bounds if (t := _trim(text, a, b))[0] < t[1]]


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _hard(text: str, start: int, end: int, size: int) -> list[tuple[int, int]]:
    """Split a span into pieces of at most `size`, at the last whitespace when possible."""
    out: list[tuple[int, int]] = []
    while start < end:
        cut = min(start + size, end)
        if cut < end:
            space = max(text.rfind(" ", start, cut + 1), text.rfind("\t", start, cut + 1))
            if space > start:
                cut = space
        a, b = _trim(text, start, cut)
        if a < b:
            out.append((a, b))
        start = cut
        while start < end and text[start].isspace():
            start += 1
    return out
