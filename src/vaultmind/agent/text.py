"""Small text helpers the agent modules share: model JSON output and one-line text."""

from __future__ import annotations

import json
from typing import Any


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


def one_line(text: str, limit: int) -> str:
    """`text` on one line, cut to `limit` characters with "..." when longer."""
    line = " ".join(text.split())
    return line if len(line) <= limit else line[: limit - 3].rstrip() + "..."
