"""Small text helpers the agent modules share: model JSON output, one-line text, token estimates."""

from __future__ import annotations

import json
import math
import re
from typing import Any

# Tokens per character of the answer model's tokenizer, by character class. Fitted on the 12
# personal documents against Ollama's prompt_eval_count for qwen3:14b (2026-10-14) so that it
# never undercounts (it overcounts by 1.0-1.35x). chars/4 undercounted mixed Hebrew PDFs by
# half: Hebrew letters split finely, and qwen splits every digit into its own token.
_TOKENS_PER_CHAR = (
    (re.compile("[֐-׿]"), 0.55),  # Hebrew
    (re.compile("[A-Za-z]"), 0.2),
    (re.compile("[0-9]"), 1.0),
    (re.compile(r"\s"), 1 / 3),
)
_OTHER_TOKENS_PER_CHAR = 1.0  # punctuation, symbols, other scripts

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


def estimate_tokens(text: str) -> int:
    """At least as many tokens as `text` takes in a prompt, give or take a third."""
    total = 0.0
    counted = 0
    for pattern, weight in _TOKENS_PER_CHAR:
        n = len(pattern.findall(text))
        total += n * weight
        counted += n
    return math.ceil(round(total + (len(text) - counted) * _OTHER_TOKENS_PER_CHAR, 6))


def one_line(text: str, limit: int) -> str:
    """`text` on one line, cut to `limit` characters with "..." when longer."""
    line = " ".join(text.split())
    return line if len(line) <= limit else line[: limit - 3].rstrip() + "..."
