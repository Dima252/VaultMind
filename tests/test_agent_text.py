"""Token estimates (agent/text.py)."""

from __future__ import annotations

from vaultmind.agent.text import estimate_tokens


def test_estimate_tokens_by_character_class() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("a" * 500) == 100
    assert estimate_tokens("ש" * 100) == 55
    assert estimate_tokens("2026") == 4  # qwen splits every digit
    assert estimate_tokens("   ") == 1
    assert estimate_tokens("%%") == 2
    assert estimate_tokens("hello שלום 12") == 6  # 1 + 2.2 + 2 + 0.67, rounded up
