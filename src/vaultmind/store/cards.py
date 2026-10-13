"""Document cards in the app database (`doc_cards` table, registry migration 4).

A card is keyed by (sha256, model, prompt_version), so it's written once per file content and
rewritten only when the model or the card prompt changes. Cards are not tied to file ids:
a renamed or moved file keeps its card, and `prune` drops cards no file has any more.
"""

from __future__ import annotations

import json
from collections.abc import Collection
from dataclasses import asdict
from datetime import datetime
from typing import Any

from vaultmind.core.models import DocCard, KeyDate, SectionSummary
from vaultmind.store.registry import SqliteStore


class CardStore(SqliteStore):
    def put(self, card: DocCard) -> None:
        """Store a card, replacing one with the same key."""
        with self._lock:
            self._db().execute(
                """
                INSERT OR REPLACE INTO doc_cards (sha256, model, prompt_version, created_at, card)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    card.sha256,
                    card.model,
                    card.prompt_version,
                    card.created_at.isoformat(),
                    json.dumps(_to_json(card), ensure_ascii=False),
                ),
            )

    def get(self, sha256: str, *, model: str, prompt_version: int) -> DocCard | None:
        """The card for exactly this content, model and prompt version."""
        with self._lock:
            row = (
                self._db()
                .execute(
                    "SELECT card FROM doc_cards"
                    " WHERE sha256 = ? AND model = ? AND prompt_version = ?",
                    (sha256, model, prompt_version),
                )
                .fetchone()
            )
        return _from_json(json.loads(row["card"])) if row else None

    def latest(self, sha256: str) -> DocCard | None:
        """The newest card for this content from any model or prompt version. Readers use
        this: a card from an older prompt beats no card while a new one is written."""
        with self._lock:
            row = (
                self._db()
                .execute(
                    "SELECT card FROM doc_cards WHERE sha256 = ? ORDER BY created_at DESC LIMIT 1",
                    (sha256,),
                )
                .fetchone()
            )
        return _from_json(json.loads(row["card"])) if row else None

    def prune(self, keep: Collection[str]) -> int:
        """Delete cards whose sha256 is not in `keep`; returns how many rows went."""
        keep_set = set(keep)
        with self._lock:
            db = self._db()
            shas = [r[0] for r in db.execute("SELECT DISTINCT sha256 FROM doc_cards")]
            gone = [s for s in shas if s not in keep_set]
            removed = 0
            for sha in gone:
                removed += db.execute("DELETE FROM doc_cards WHERE sha256 = ?", (sha,)).rowcount
        return removed


def _to_json(card: DocCard) -> dict[str, Any]:
    data = asdict(card)
    data["created_at"] = card.created_at.isoformat()
    return data


def _from_json(data: dict[str, Any]) -> DocCard:
    return DocCard(
        sha256=data["sha256"],
        model=data["model"],
        prompt_version=data["prompt_version"],
        title=data["title"],
        doc_type=data["doc_type"],
        language=data["language"],
        summary=data["summary"],
        outline=tuple(data["outline"]),
        entities=tuple(data["entities"]),
        dates=tuple(KeyDate(**d) for d in data["dates"]),
        sections=tuple(
            SectionSummary(
                parent_id=s["parent_id"],
                section_path=tuple(s["section_path"]),
                page_start=s["page_start"],
                page_end=s["page_end"],
                summary=s["summary"],
            )
            for s in data["sections"]
        ),
        created_at=datetime.fromisoformat(data["created_at"]),
    )
