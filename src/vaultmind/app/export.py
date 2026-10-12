"""Export an answer or a whole chat as Markdown or PDF, with its citations as file + page."""

from __future__ import annotations

from collections.abc import Sequence

from vaultmind.store.chats import Chat, ChatMessage


def answer_markdown(question: ChatMessage, answer: ChatMessage) -> str:
    """One exchange: the question, the answer, then the sources it cited (or closest matches)."""
    raise NotImplementedError


def chat_markdown(chat: Chat, messages: Sequence[ChatMessage]) -> str:
    """The whole chat: title, export time, and each exchange as in `answer_markdown`."""
    raise NotImplementedError


def markdown_pdf(markdown: str) -> bytes:
    """Render exported Markdown as a PDF, offline (no downloaded fonts). Hebrew must be legible."""
    raise NotImplementedError
