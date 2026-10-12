"""Chat history in the app database (`chats`, `messages` tables).

The schema lives in `store/registry.py` (migration 3); opening a ChatStore applies `migrate`.
An assistant message keeps everything needed to show it again without re-running the agent:
its sources (numbered as the answer cites them), abstention, contradictions, unsupported claims,
and the id of its stored run (`store/traces.py`) for the trace.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from vaultmind.agent.baseline import Source
from vaultmind.agent.types import AgentResult, Contradiction, UnsupportedClaim
from vaultmind.core.providers import Message
from vaultmind.store.registry import migrate

Role = Literal["user", "assistant"]
DEFAULT_TITLE = "New chat"

_TITLE_MAX = 60
_CHAT_COLUMNS = "id, title, created_at, updated_at"
_MESSAGE_COLUMNS = "chat_id, seq, role, content, created_at, run_id, meta"


@dataclass(frozen=True)
class Chat:
    id: str
    title: str
    created_at: datetime  # UTC
    updated_at: datetime  # UTC; the last message added


@dataclass(frozen=True)
class ChatMessage:
    chat_id: str
    seq: int  # 0-based position in the chat
    role: Role
    content: str
    created_at: datetime  # UTC
    run_id: str | None = None  # assistant: the stored agent run, for the trace
    sources: tuple[Source, ...] = ()  # assistant: AgentResult.sources, numbering kept
    abstained: bool = False
    contradictions: tuple[Contradiction, ...] = ()
    unsupported: tuple[UnsupportedClaim, ...] = ()


class ChatStore:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = sqlite3.connect(
            db_path, check_same_thread=False, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            migrate(self._conn)
        except BaseException:
            self.close()
            raise

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("chat store is closed")
        return self._conn

    def create(self, title: str = DEFAULT_TITLE) -> Chat:
        now = datetime.now(UTC)
        chat = Chat(id=uuid.uuid4().hex, title=title, created_at=now, updated_at=now)
        with self._lock:
            self._db().execute(
                f"INSERT INTO chats ({_CHAT_COLUMNS}) VALUES (?, ?, ?, ?)",
                (chat.id, chat.title, now.isoformat(), now.isoformat()),
            )
        return chat

    def get(self, chat_id: str) -> Chat | None:
        with self._lock:
            row = (
                self._db()
                .execute(f"SELECT {_CHAT_COLUMNS} FROM chats WHERE id = ?", (chat_id,))
                .fetchone()
            )
        return _to_chat(row) if row else None

    def recent(self, limit: int = 50) -> list[Chat]:
        """Most recently updated first."""
        with self._lock:
            rows = (
                self._db()
                .execute(
                    f"SELECT {_CHAT_COLUMNS} FROM chats "
                    "ORDER BY updated_at DESC, rowid DESC LIMIT ?",
                    (limit,),
                )
                .fetchall()
            )
        return [_to_chat(r) for r in rows]

    def rename(self, chat_id: str, title: str) -> None:
        """Raises KeyError for an unknown chat."""
        with self._lock:
            cursor = self._db().execute("UPDATE chats SET title = ? WHERE id = ?", (title, chat_id))
        if cursor.rowcount == 0:
            raise KeyError(chat_id)

    def delete(self, chat_id: str) -> None:
        """Deletes the chat and its messages. Stored runs are kept (`vaultmind trace`)."""
        with self._lock:
            self._db().execute("DELETE FROM chats WHERE id = ?", (chat_id,))

    def messages(self, chat_id: str) -> list[ChatMessage]:
        """In order. Empty for an unknown chat."""
        with self._lock:
            rows = (
                self._db()
                .execute(
                    f"SELECT {_MESSAGE_COLUMNS} FROM messages WHERE chat_id = ? ORDER BY seq",
                    (chat_id,),
                )
                .fetchall()
            )
        return [_to_message(r) for r in rows]

    def add_exchange(
        self, chat_id: str, result: AgentResult, run_id: str | None
    ) -> tuple[ChatMessage, ChatMessage]:
        """Store the question and its answer in one transaction; returns (user, assistant).

        Bumps `updated_at`. The first exchange of a chat still titled DEFAULT_TITLE renames it
        after the question (one line, cut to about 60 characters). KeyError for an unknown chat.
        """
        now = datetime.now(UTC)
        user = ChatMessage(chat_id, 0, "user", result.question, now)
        assistant = ChatMessage(
            chat_id,
            1,
            "assistant",
            result.answer,
            now,
            run_id=run_id,
            sources=tuple(result.sources),
            abstained=result.abstained,
            contradictions=tuple(result.contradictions),
            unsupported=tuple(result.unsupported),
        )
        meta = json.dumps(_meta(assistant), ensure_ascii=False, separators=(",", ":"))
        stamp = now.isoformat()
        with self._lock:
            db = self._db()
            db.execute("BEGIN IMMEDIATE")
            try:
                row = db.execute("SELECT title FROM chats WHERE id = ?", (chat_id,)).fetchone()
                if row is None:
                    raise KeyError(chat_id)
                last = db.execute(
                    "SELECT MAX(seq) FROM messages WHERE chat_id = ?", (chat_id,)
                ).fetchone()[0]
                seq = 0 if last is None else int(last) + 1
                db.execute(
                    f"INSERT INTO messages ({_MESSAGE_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (chat_id, seq, "user", result.question, stamp, None, "{}"),
                )
                db.execute(
                    f"INSERT INTO messages ({_MESSAGE_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (chat_id, seq + 1, "assistant", result.answer, stamp, run_id, meta),
                )
                title = row["title"]
                if seq == 0 and title == DEFAULT_TITLE:
                    title = _title_from(result.question) or title
                db.execute(
                    "UPDATE chats SET updated_at = ?, title = ? WHERE id = ?",
                    (stamp, title, chat_id),
                )
            except BaseException:
                db.execute("ROLLBACK")
                raise
            db.execute("COMMIT")
        return replace(user, seq=seq), replace(assistant, seq=seq + 1)

    def history(self, chat_id: str) -> list[Message]:
        """The chat as `stream_agent` history (user/assistant turns, oldest first)."""
        with self._lock:
            rows = (
                self._db()
                .execute(
                    "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY seq",
                    (chat_id,),
                )
                .fetchall()
            )
        return [{"role": cast(Role, r["role"]), "content": r["content"]} for r in rows]

    def close(self) -> None:
        """Idempotent."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


def _title_from(question: str) -> str:
    """The first non-blank line, whitespace collapsed, cut near _TITLE_MAX at a word break."""
    line = next((" ".join(ln.split()) for ln in question.splitlines() if ln.strip()), "")
    if len(line) <= _TITLE_MAX:
        return line
    cut = line[:_TITLE_MAX]
    space = cut.rfind(" ")
    if space >= _TITLE_MAX // 2:
        cut = cut[:space]
    return cut.rstrip(" ,;:.-") + "..."


def _meta(message: ChatMessage) -> dict[str, Any]:
    return {
        "sources": [_source_json(s) for s in message.sources],
        "abstained": message.abstained,
        "contradictions": [
            {
                "claim_a": c.claim_a,
                "source_a": c.source_a,
                "claim_b": c.claim_b,
                "source_b": c.source_b,
            }
            for c in message.contradictions
        ],
        "unsupported": [
            {"claim": u.claim, "cited": list(u.cited), "reason": u.reason}
            for u in message.unsupported
        ],
    }


def _source_json(s: Source) -> dict[str, Any]:
    return {
        "n": int(s.n),
        "file_path": str(s.file_path) if s.file_path is not None else None,
        "page": int(s.page) if s.page is not None else None,
        "section_path": list(s.section_path),
        "text": s.text,
        "score": float(s.score),
        "chunk_id": s.chunk_id,
        "warnings": list(s.warnings),
    }


def _source_from(d: dict[str, Any]) -> Source:
    file_path = d.get("file_path")
    return Source(
        n=int(d["n"]),
        file_path=Path(file_path) if file_path is not None else None,
        page=d.get("page"),
        section_path=tuple(d.get("section_path", ())),
        text=d.get("text", ""),
        score=float(d.get("score", 0.0)),
        chunk_id=d.get("chunk_id"),
        warnings=tuple(d.get("warnings", ())),
    )


def _to_chat(row: sqlite3.Row) -> Chat:
    return Chat(
        id=row["id"],
        title=row["title"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def _to_message(row: sqlite3.Row) -> ChatMessage:
    meta: dict[str, Any] = json.loads(row["meta"] or "{}")
    return ChatMessage(
        chat_id=row["chat_id"],
        seq=int(row["seq"]),
        role=cast(Role, row["role"]),
        content=row["content"],
        created_at=datetime.fromisoformat(row["created_at"]),
        run_id=row["run_id"],
        sources=tuple(_source_from(s) for s in meta.get("sources", ())),
        abstained=bool(meta.get("abstained", False)),
        contradictions=tuple(
            Contradiction(c["claim_a"], int(c["source_a"]), c["claim_b"], int(c["source_b"]))
            for c in meta.get("contradictions", ())
        ),
        unsupported=tuple(
            UnsupportedClaim(u["claim"], tuple(int(n) for n in u.get("cited", ())), u["reason"])
            for u in meta.get("unsupported", ())
        ),
    )
