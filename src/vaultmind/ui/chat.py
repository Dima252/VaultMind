"""Chat page: chat list, streamed answers with citation chips, trace and export per answer.

`Engine.ask` is a blocking generator. A worker thread drives it and hands every item back to the
event loop (`call_soon_threadsafe`), so the page never blocks. The thread always runs the
generator to the end, so an answer is stored even when the page is left mid-answer.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass

from nicegui import app, run, ui

from vaultmind.agent.baseline import Source, cited_sources
from vaultmind.app.engine import Engine
from vaultmind.app.export import answer_markdown, chat_markdown, markdown_pdf
from vaultmind.store.chats import DEFAULT_TITLE, Chat, ChatMessage
from vaultmind.ui.citation_view import open_citation
from vaultmind.ui.state import state
from vaultmind.ui.trace_view import trace_view

_OPEN_CHAT = "vaultmind.chat_id"  # app.storage.client: the open chat ("" = a new, unsaved one)
_ANSWERING = "vaultmind.answering"  # app.storage.client: an answer is being generated
_UNSAFE_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def page() -> None:
    """Build the chat page. Uses `state().engine` (ready when this is called)."""
    engine = state().engine
    if engine is None:
        raise RuntimeError("the chat page needs a loaded engine")
    _ChatPage(engine).build()


@dataclass(frozen=True)
class _Failed:
    error: Exception


_END = object()  # the generator finished


class _ChatPage:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.chat_id: str | None = self._initial_chat()  # None: a new chat, saved on first ask
        self.hint: ui.label | None = None  # the empty-chat hint, while shown

    def _initial_chat(self) -> str | None:
        stored = app.storage.client.get(_OPEN_CHAT)
        if stored == "":
            return None
        if isinstance(stored, str) and self.engine.chats.get(stored) is not None:
            return stored
        recent = self.engine.chats.recent(1)
        return recent[0].id if recent else None

    # --- layout ------------------------------------------------------------------------

    def build(self) -> None:
        with (
            ui.row()
            .classes("w-full no-wrap gap-0 items-stretch")
            .style("height: calc(100vh - 6rem)")
        ):
            with ui.column().classes("w-44 sm:w-56 lg:w-64 shrink-0 h-full gap-2 pr-3 border-r"):
                ui.button("New chat", icon="add", on_click=self.new_chat).props(
                    "outline no-caps"
                ).classes("w-full").mark("new-chat")
                with ui.scroll_area().classes("flex-grow w-full"):
                    self.chat_list = ui.list().props("dense").classes("w-full")
            with ui.column().classes("flex-grow min-w-0 h-full gap-2 pl-3"):
                with ui.row().classes("w-full items-center no-wrap"):
                    self.title = ui.label().classes("text-lg font-semibold truncate min-w-0")
                    self.title.props('dir="auto"')
                    ui.space()
                    with ui.button(icon="download").props("flat dense") as export:
                        export.tooltip("Export this chat")
                        with ui.menu():
                            ui.menu_item("Chat as Markdown", self.export_chat_md).mark(
                                "export-chat-md"
                            )
                            ui.menu_item("Chat as PDF", self.export_chat_pdf).mark(
                                "export-chat-pdf"
                            )
                    self.export_button = export
                self.scroll = ui.scroll_area().classes("flex-grow w-full vm-messages")  # app.CSS
                with self.scroll:
                    self.messages = ui.column().classes("w-full gap-4 pr-3").mark("messages")
                with ui.row().classes("w-full items-end no-wrap gap-2"):
                    self.input = (
                        ui.textarea(placeholder="Ask about your documents")
                        .props("autogrow outlined dense rows=1")
                        .classes("flex-grow")
                        .mark("question")
                    )
                    self.input.props('dir="auto" input-style="max-height: 12rem"')
                    # Enter sends; Shift+Enter (not "exact") falls through to a newline.
                    self.input.on("keydown.enter.exact.prevent", self.send)
                    self.spinner = ui.spinner(size="md")
                    self.send_button = ui.button(icon="send", on_click=self.send).props("round")
                    self.send_button.mark("send")
        self._show_chat()
        self._set_busy(bool(app.storage.client.get(_ANSWERING)))
        if app.storage.client.get(_ANSWERING):  # left mid-answer and came back: wait for it
            self._wait_timer = ui.timer(0.5, self._check_answered)

    def _check_answered(self) -> None:
        if not app.storage.client.get(_ANSWERING):
            self._wait_timer.cancel()
            self._set_busy(False)
            self._show_chat()

    def _set_busy(self, busy: bool) -> None:
        self.input.set_enabled(not busy)
        self.send_button.set_enabled(not busy)
        self.spinner.set_visibility(busy)

    def _select(self, chat_id: str | None) -> None:
        self.chat_id = chat_id
        app.storage.client[_OPEN_CHAT] = chat_id or ""
        self._show_chat()

    def _show_chat(self) -> None:
        """Render the open chat: drawer, header and messages."""
        app.storage.client[_OPEN_CHAT] = self.chat_id or ""
        chat = self.engine.chats.get(self.chat_id) if self.chat_id else None
        messages = self.engine.chats.messages(chat.id) if chat else []
        self._refresh_list()
        self.title.set_text(chat.title if chat else DEFAULT_TITLE)
        self.export_button.set_enabled(bool(messages))
        self.messages.clear()
        with self.messages:
            self.hint = None if messages else _empty_hint()
            question: ChatMessage | None = None
            for message in messages:
                if message.role == "user":
                    question = message
                    _question_bubble(message.content)
                else:
                    with ui.column().classes("w-full gap-2"):
                        self._answer(question, message)
        self.scroll.scroll_to(percent=1)

    def _refresh_list(self) -> None:
        self.chat_list.clear()
        with self.chat_list:
            for chat in self.engine.chats.recent():
                self._chat_item(chat)

    def _chat_item(self, chat: Chat) -> None:
        item = ui.item(on_click=lambda c=chat: self._select(c.id)).mark(f"chat-{chat.id}")
        if chat.id == self.chat_id:
            item.classes("bg-primary/10 rounded").props("active")
        with item:
            with ui.item_section():
                ui.item_label(chat.title).classes("truncate").props('dir="auto"')
            with (
                ui.item_section().props("side"),
                ui.button(icon="more_vert").props("flat round dense size=sm"),
                ui.menu(),
            ):
                ui.menu_item("Rename", lambda c=chat: self.rename(c)).mark(f"rename-{chat.id}")
                ui.menu_item("Delete", lambda c=chat: self.delete(c)).mark(f"delete-{chat.id}")

    # --- chat actions --------------------------------------------------------------------

    def new_chat(self) -> None:
        self._select(None)
        self.input.run_method("focus")

    async def rename(self, chat: Chat) -> None:
        with ui.dialog() as dialog, ui.card().classes("w-96"):
            ui.label("Rename chat").classes("text-base font-semibold")
            name = ui.input(value=chat.title).props('autofocus dir="auto"').classes("w-full")
            name.on("keydown.enter", lambda: dialog.submit(name.value))
            with ui.row().classes("w-full justify-end"):
                ui.button("Cancel", on_click=lambda: dialog.submit(None)).props("flat")
                ui.button("Save", on_click=lambda: dialog.submit(name.value)).mark("rename-save")
        title = await dialog
        dialog.delete()
        title = " ".join(str(title or "").split())
        if not title or title == chat.title:
            return
        with contextlib.suppress(KeyError):  # deleted meanwhile
            self.engine.chats.rename(chat.id, title)
        self._show_chat()

    async def delete(self, chat: Chat) -> None:
        with ui.dialog() as dialog, ui.card():
            ui.label(f'Delete "{chat.title}"?').classes("text-base").props('dir="auto"')
            ui.label("Its messages are removed for good.").classes("text-sm")
            with ui.row().classes("w-full justify-end"):
                ui.button("Cancel", on_click=lambda: dialog.submit(False)).props("flat")
                ui.button("Delete", color="negative", on_click=lambda: dialog.submit(True)).mark(
                    "delete-confirm"
                )
        confirmed = await dialog
        dialog.delete()
        if not confirmed:
            return
        self.engine.chats.delete(chat.id)
        if chat.id == self.chat_id:
            recent = self.engine.chats.recent(1)
            self._select(recent[0].id if recent else None)
        else:
            self._refresh_list()

    # --- asking --------------------------------------------------------------------------

    async def send(self) -> None:
        question = (self.input.value or "").strip()
        if not question or app.storage.client.get(_ANSWERING):
            return
        if self.chat_id is None:
            self.chat_id = self.engine.chats.create().id
            app.storage.client[_OPEN_CHAT] = self.chat_id
        chat_id = self.chat_id
        app.storage.client[_ANSWERING] = True
        self._set_busy(True)
        self.input.value = ""
        with self.messages:
            if self.hint is not None and not self.hint.is_deleted:
                self.hint.delete()
            bubble = _question_bubble(question)
            pending = ui.column().classes("w-full gap-2")
            with pending:
                answer_md = ui.markdown("").props('dir="auto"').classes("w-full")
                ui.spinner("dots", size="lg").mark("answer-spinner")
        self.scroll.scroll_to(percent=1)

        def show(text: str) -> None:
            if not answer_md.is_deleted:
                answer_md.set_content(text)
                self.scroll.scroll_to(percent=1)

        try:
            answer = await _stream_answer(self.engine, chat_id, question, show)
        except Exception as e:
            ui.notify(f"Couldn't answer: {e or type(e).__name__}", type="negative")
            bubble.delete()
            pending.delete()
            if not self.input.is_deleted:
                self.input.value = question
            return
        finally:
            app.storage.client[_ANSWERING] = False
            if not self.input.is_deleted:
                self._set_busy(False)
        if self.messages.is_deleted:  # the page was left mid-answer
            return
        if self.chat_id != chat_id:  # another chat was opened meanwhile
            self._refresh_list()
            return
        asked = ChatMessage(chat_id, answer.seq - 1, "user", question, answer.created_at)
        pending.clear()
        with pending:
            self._answer(asked, answer)
        self._refresh_list()
        chat = self.engine.chats.get(chat_id)
        self.title.set_text(chat.title if chat else DEFAULT_TITLE)
        self.export_button.set_enabled(True)
        self.scroll.scroll_to(percent=1)

    # --- one answer ----------------------------------------------------------------------

    def _answer(self, question: ChatMessage | None, answer: ChatMessage) -> None:
        """The answer with its citation chips, notes, trace and export actions."""
        ui.markdown(answer.content).props('dir="auto"').classes("w-full")
        _citation_chips(answer)
        _notes(answer)
        with ui.row().classes("w-full items-center gap-1"):
            if answer.run_id:
                _trace_expansion(self.engine, answer.run_id)
            ui.space()
            if question is not None:
                q = question
                ui.button(
                    icon="description",
                    on_click=lambda: _download_md(answer_markdown(q, answer), _first_line(q)),
                ).props("flat dense size=sm").tooltip("Export answer as Markdown").mark(
                    "export-answer-md"
                )
                ui.button(
                    icon="picture_as_pdf",
                    on_click=lambda: _download_pdf(answer_markdown(q, answer), _first_line(q)),
                ).props("flat dense size=sm").tooltip("Export answer as PDF").mark(
                    "export-answer-pdf"
                )

    # --- export --------------------------------------------------------------------------

    def _chat_export(self) -> tuple[Chat, str] | None:
        chat = self.engine.chats.get(self.chat_id) if self.chat_id else None
        if chat is None:
            return None
        return chat, chat_markdown(chat, self.engine.chats.messages(chat.id))

    def export_chat_md(self) -> None:
        if exported := self._chat_export():
            chat, markdown = exported
            _download_md(markdown, chat.title)

    async def export_chat_pdf(self) -> None:
        if exported := self._chat_export():
            chat, markdown = exported
            await _download_pdf(markdown, chat.title)


# --- streaming -------------------------------------------------------------------------


async def _stream_answer(
    engine: Engine, chat_id: str, question: str, show: Callable[[str], None]
) -> ChatMessage:
    """Run `engine.ask` on a worker thread; call `show` with the answer so far as tokens
    arrive; return the stored assistant message. Re-raises the engine's error."""
    loop = asyncio.get_running_loop()
    inbox: asyncio.Queue[object] = asyncio.Queue()

    def put(item: object) -> None:
        with contextlib.suppress(RuntimeError):  # the loop is gone (app shutting down)
            loop.call_soon_threadsafe(inbox.put_nowait, item)

    def pump() -> None:
        try:
            for item in engine.ask(chat_id, question):
                put(item)
        except Exception as e:
            put(_Failed(e))
        put(_END)

    threading.Thread(target=pump, name="vaultmind-ask", daemon=True).start()
    text = ""
    while True:
        items = [await inbox.get()]
        while not inbox.empty():  # take everything that arrived meanwhile: one update per batch
            items.append(inbox.get_nowait())
        for item in items:
            if isinstance(item, str):
                text += item
            elif isinstance(item, ChatMessage):
                return item
            elif isinstance(item, _Failed):
                raise item.error
            elif item is _END:
                raise RuntimeError("the answer ended without being stored")
        show(text)


# --- pieces ----------------------------------------------------------------------------


def _empty_hint() -> ui.label:
    return (
        ui.label("Ask a question about your documents. Answers cite the passages they use.")
        .classes("text-grey self-center mt-8")
        .mark("empty-hint")
    )


def _question_bubble(text: str) -> ui.row:
    with ui.row().classes("w-full justify-end") as row:
        # Tailwind has no "primary" color here, so the bubble's tint is inline.
        ui.label(text).classes(
            "whitespace-pre-wrap break-words rounded-lg px-3 py-2 max-w-[80%]"
        ).style("background: rgba(88, 152, 212, 0.22)").props('dir="auto"')
    return row


CHIP_NAME_CHARS = 48  # longer file names are shortened in the middle; the tooltip has it all


def _chip_text(source: Source) -> str:
    name = source.file_path.name if source.file_path else "unknown file"
    if len(name) > CHIP_NAME_CHARS:
        half = (CHIP_NAME_CHARS - 1) // 2
        name = f"{name[:half]}…{name[-half:]}"
    page = f", p. {source.page}" if source.page is not None else ""
    # Isolated left-to-right (LRI ... PDI): a Hebrew name keeps ".pdf" at its end and can't
    # reorder "[n]" and "p. N"; Hebrew words inside it still read right-to-left.
    return f"[{source.n}] ⁦{name}⁩{page}"


def _citation_chips(answer: ChatMessage) -> None:
    if not answer.sources:
        return
    if answer.abstained:
        heading, shown = "Closest matches", list(answer.sources)
    else:
        heading = "Sources"
        shown = cited_sources(answer.content, answer.sources) or list(answer.sources)
    with ui.row().classes("w-full items-center gap-1"):
        ui.label(heading).classes("text-xs text-grey mr-1")
        for source in shown:
            warned = bool(source.warnings)
            chip = ui.chip(
                _chip_text(source),
                icon="warning" if warned else "description",
                color="warning" if warned else ("grey-5" if answer.abstained else "primary"),
                on_click=lambda s=source: open_citation(s),
            ).props('outline dense dir="ltr"')
            chip.classes("max-w-full")
            chip.mark("citation")
            with chip:
                ui.tooltip("\n".join([source.label, *source.warnings])).classes(
                    "whitespace-pre-line"
                )


def _notes(answer: ChatMessage) -> None:
    for c in answer.contradictions:
        _note(
            "Sources disagree",
            f'[{c.source_a}] "{_one_line(c.claim_a)}" vs. [{c.source_b}] "{_one_line(c.claim_b)}"',
        )
    for u in answer.unsupported:
        cited = "".join(f"[{n}]" for n in u.cited) or "no source"
        reason = f"; {_one_line(u.reason)}" if u.reason.strip() else ""
        _note(
            "Possibly unsupported by its sources",
            f'"{_one_line(u.claim)}" (cited {cited}{reason})',
        )


def _note(title: str, text: str) -> None:
    with ui.row().classes("w-full no-wrap items-start gap-2 border-l-4 border-amber-500 pl-2 py-1"):
        ui.icon("warning", color="amber-8").classes("mt-0.5")
        with ui.column().classes("gap-0 min-w-0"):
            ui.label(title).classes("text-sm font-semibold")
            ui.label(text).classes("text-sm break-words").props('dir="auto"')


def _trace_expansion(engine: Engine, run_id: str) -> None:
    loaded = False

    async def load(opened: bool) -> None:
        nonlocal loaded
        if not opened or loaded:
            return
        loaded = True
        run_ = await run.io_bound(engine.traces.get, run_id)
        if body.is_deleted:
            return
        body.clear()
        with body:
            if run_ is None:
                ui.label("The trace for this answer wasn't found.").classes("text-sm text-grey")
            else:
                trace_view(run_.events)

    expansion = ui.expansion(
        "Trace", icon="timeline", on_value_change=lambda e: load(bool(e.value))
    )
    with expansion.props("dense").classes("text-sm").mark("trace"):
        body = ui.column().classes("w-full")
        with body:
            ui.spinner()


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _first_line(message: ChatMessage) -> str:
    return _one_line(message.content)[:60] or "answer"


def _filename(title: str, ext: str) -> str:
    name = _UNSAFE_FILENAME.sub(" ", title)
    name = " ".join(name.split()).strip(". ")[:80] or "VaultMind"
    return f"{name}.{ext}"


def _download_md(markdown: str, title: str = "answer") -> None:
    ui.download(markdown.encode("utf-8"), _filename(title, "md"), "text/markdown")


async def _download_pdf(markdown: str, title: str) -> None:
    try:
        pdf = await run.io_bound(markdown_pdf, markdown)
    except Exception as e:
        ui.notify(f"Couldn't create the PDF: {e or type(e).__name__}", type="negative")
        return
    if pdf is not None:  # None: the app is shutting down
        ui.download(pdf, _filename(title, "pdf"), "application/pdf")
