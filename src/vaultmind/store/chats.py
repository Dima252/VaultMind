"""Chat history in the app database (`chats`, `messages` tables).

The schema lives in `store/registry.py` (migration 3); opening a ChatStore applies `migrate`.
An assistant message keeps everything needed to show it again without re-running the agent:
its sources (numbered as the answer cites them), abstention, contradictions, unsupported claims,
and the id of its stored run (`store/traces.py`) for the trace.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from vaultmind.agent.baseline import Source
from vaultmind.agent.types import AgentResult, Contradiction, UnsupportedClaim
from vaultmind.core.providers import Message

Role = Literal["user", "assistant"]
DEFAULT_TITLE = "New chat"


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
        raise NotImplementedError

    def create(self, title: str = DEFAULT_TITLE) -> Chat:
        raise NotImplementedError

    def get(self, chat_id: str) -> Chat | None:
        raise NotImplementedError

    def recent(self, limit: int = 50) -> list[Chat]:
        """Most recently updated first."""
        raise NotImplementedError

    def rename(self, chat_id: str, title: str) -> None:
        """Raises KeyError for an unknown chat."""
        raise NotImplementedError

    def delete(self, chat_id: str) -> None:
        """Deletes the chat and its messages. Stored runs are kept (`vaultmind trace`)."""
        raise NotImplementedError

    def messages(self, chat_id: str) -> list[ChatMessage]:
        """In order. Empty for an unknown chat."""
        raise NotImplementedError

    def add_exchange(
        self, chat_id: str, result: AgentResult, run_id: str | None
    ) -> tuple[ChatMessage, ChatMessage]:
        """Store the question and its answer in one transaction; returns (user, assistant).

        Bumps `updated_at`. The first exchange of a chat still titled DEFAULT_TITLE renames it
        after the question (one line, cut to about 60 characters). KeyError for an unknown chat.
        """
        raise NotImplementedError

    def history(self, chat_id: str) -> list[Message]:
        """The chat as `stream_agent` history (user/assistant turns, oldest first)."""
        raise NotImplementedError

    def close(self) -> None:
        """Idempotent."""
        raise NotImplementedError
