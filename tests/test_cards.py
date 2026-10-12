"""CardStore: document cards in the app database, keyed by content, model and prompt version."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from vaultmind.core.models import DocCard, KeyDate, SectionSummary
from vaultmind.store.cards import CardStore
from vaultmind.store.registry import Registry

T0 = datetime(2026, 10, 12, 9, 0, tzinfo=UTC)


def _card(sha: str = "a" * 64, **changes: object) -> DocCard:
    card = DocCard(
        sha256=sha,
        model="qwen3:14b",
        prompt_version=1,
        title="מבוא לאלגוריתמים",
        doc_type="syllabus",
        language="he",
        summary="שורה ראשונה.\nSecond line.",
        outline=("מבוא", "Grading"),
        entities=("Bellman-Ford",),
        dates=(KeyDate("15.3.2026", "תרגיל 2"),),
        sections=(
            SectionSummary("f:p0", ("מבוא",), 1, 2, "What the course covers."),
            SectionSummary("f:p1", (), None, None, "No pages."),
        ),
        created_at=T0,
    )
    return replace(card, **changes)  # type: ignore[arg-type]


def test_round_trip(tmp_path: Path) -> None:
    store = CardStore(tmp_path / "db.sqlite")
    card = _card()
    store.put(card)
    assert store.get(card.sha256, model="qwen3:14b", prompt_version=1) == card
    assert store.get(card.sha256, model="qwen3:14b", prompt_version=2) is None
    assert store.get(card.sha256, model="other", prompt_version=1) is None
    assert store.latest("b" * 64) is None


def test_put_replaces_same_key(tmp_path: Path) -> None:
    store = CardStore(tmp_path / "db.sqlite")
    store.put(_card())
    store.put(_card(summary="new"))
    assert store.latest("a" * 64) == _card(summary="new")


def test_latest_prefers_newest_of_any_version(tmp_path: Path) -> None:
    store = CardStore(tmp_path / "db.sqlite")
    store.put(_card(prompt_version=2, created_at=T0 + timedelta(hours=1), summary="v2"))
    store.put(_card(prompt_version=1, summary="v1"))
    latest = store.latest("a" * 64)
    assert latest is not None and latest.summary == "v2"


def test_prune_keeps_listed_content(tmp_path: Path) -> None:
    store = CardStore(tmp_path / "db.sqlite")
    store.put(_card("a" * 64))
    store.put(_card("a" * 64, prompt_version=2))
    store.put(_card("b" * 64))
    assert store.prune({"b" * 64}) == 2
    assert store.latest("a" * 64) is None
    assert store.latest("b" * 64) is not None


def test_shares_the_database_and_closes(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    registry = Registry(db)
    store = CardStore(db)
    store.put(_card())
    assert registry.all() == []
    store.close()
    store.close()
    with pytest.raises(RuntimeError):
        store.latest("a" * 64)
    registry.close()
