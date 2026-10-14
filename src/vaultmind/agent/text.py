"""Small text helpers the agent modules share: model JSON output and one-line text."""

from __future__ import annotations

import json
import re
from typing import Any

_BOUNDARY = re.compile(r"[.!?](?=\s)|\n")
# A period after these doesn't end a sentence: titles and initials never do, the others
# don't unless a capital letter follows ("Ostrel LLC. It runs" ends; "Inc. and" doesn't).
_TITLES = frozenset(("dr", "mr", "mrs", "ms", "prof", "st", "no", "vs"))
_ABBREVIATIONS = frozenset(("inc", "ltd", "llc", "plc", "co", "corp", "e.g", "i.e"))


def parse_json(text: str) -> dict[str, Any]:
    """The JSON object in `text` (from its first "{" to its last "}"); ValueError when none."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object")
    parsed = json.loads(text[start : end + 1])  # JSONDecodeError is a ValueError
    if not isinstance(parsed, dict):
        raise ValueError("not a JSON object")
    return parsed


def clean(value: object) -> str:
    """`value` with its whitespace collapsed when it is a string, else ""."""
    return " ".join(value.split()) if isinstance(value, str) else ""


def split_sentences(text: str) -> list[str]:
    """Sentences (or lines) of `text`, each on one line. A period after "Inc.", "e.g." or an
    initial doesn't end one, even before a line break: card summaries were written with a
    break after "Inc." ("between Corvenna Cloud Services Inc.\\n(Provider) and ...")."""
    sentences: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        if match.group() == "\n":
            before = text[start : match.start()].rstrip()
            if before.endswith(".") and not _ends_sentence(text, start + len(before) - 1):
                continue
            end = match.start()
        elif match.group() == "." and not _ends_sentence(text, match.start()):
            continue
        else:
            end = match.end()
        if piece := " ".join(text[start:end].split()):
            sentences.append(piece)
        start = end
    if tail := " ".join(text[start:].split()):
        sentences.append(tail)
    return sentences


def _ends_sentence(text: str, period: int) -> bool:
    words = text[:period].split()
    word = words[-1].lower() if words else ""
    if word in _TITLES or len(word) == 1:
        return False
    if word in _ABBREVIATIONS:
        return text[period + 1 :].lstrip()[:1].isupper()
    return True


def one_line(text: str, limit: int) -> str:
    """`text` on one line, cut to `limit` characters with "..." when longer."""
    line = " ".join(text.split())
    return line if len(line) <= limit else line[: limit - 3].rstrip() + "..."
