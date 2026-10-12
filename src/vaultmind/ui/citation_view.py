"""Citation viewer: the cited passage highlighted in its source (PDF page or text file).

Rendering and file reads run off the event loop. Problems (a moved file, a broken PDF) show
as a readable message in the dialog, never as a stack trace.
"""

from __future__ import annotations

import base64
import html
import os
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path

from nicegui import background_tasks, run, ui

from vaultmind.agent.baseline import Source
from vaultmind.app.highlight import (
    PageImage,
    find_passage,
    highlighted_pdf_copy,
    render_pdf_citation,
)
from vaultmind.ui.state import state

_TEXT_SUFFIXES = frozenset({".md", ".markdown", ".txt"})
_MAX_SHOWN = 200_000  # characters of a text file shown around the passage
NOT_FOUND_ON_PAGE = "Passage not found on the page; showing the page."
NOT_FOUND_IN_FILE = "Passage not found in the file (it may have changed since indexing)."


def open_citation(source: Source) -> None:
    """Open a dialog showing `source` in its file, highlighted (`app/highlight.py`)."""
    with (
        ui.dialog() as dialog,
        ui.card().classes("w-[90vw] gap-2").style("max-width: 1100px"),
    ):
        with ui.row().classes("w-full items-center no-wrap"):
            ui.label(source.label).classes("text-base font-semibold min-w-0 break-words").props(
                'dir="auto"'
            )
            ui.space()
            ui.button(icon="close", on_click=dialog.close).props("flat round dense")
        body = ui.column().classes("w-full gap-2")
        with body:
            ui.spinner(size="lg")
    dialog.on("hide", dialog.delete)
    dialog.open()
    background_tasks.create(_fill(body, source), name="citation view")


async def _fill(body: ui.column, source: Source) -> None:
    path = source.file_path
    if path is None:
        _replace(body, lambda: _passage_only(source, "This source has no file."))
    elif not path.is_file():
        note = f"The file is no longer at {path}."
        _replace(body, lambda: _passage_only(source, note))
    elif path.suffix.lower() == ".pdf":
        await _PdfView(body, path, source).show(source.page or 1)
    elif path.suffix.lower() in _TEXT_SUFFIXES:
        await _show_text(body, path, source)
    else:
        note = f"{path.suffix} files can't be shown here."
        _replace(body, lambda: _passage_only(source, note))


def _replace(body: ui.column, build: Callable[[], None]) -> None:
    """Swap the dialog body for new content (a no-op once the dialog is gone)."""
    if body.is_deleted:
        return
    body.clear()
    with body:
        build()


def _passage_only(source: Source, note: str) -> None:
    ui.label(note).classes("text-warning")
    ui.label(source.text).classes("whitespace-pre-wrap text-sm").props('dir="auto"')


# --- PDF ---------------------------------------------------------------------------------


class _PdfView:
    """One page at a time; the passage is highlighted on the page the citation points at."""

    def __init__(self, body: ui.column, path: Path, source: Source) -> None:
        self.body = body
        self.path = path
        self.source = source
        self.cited: int | None = None  # the page the passage was shown on (after the first render)

    async def show(self, page: int) -> None:
        # Other pages render without the passage: highlighting also searches the neighbouring
        # pages and would jump back to the cited one.
        passage = self.source.text if self.cited in (None, page) else ""
        try:
            image = await run.io_bound(render_pdf_citation, self.path, page, passage)
        except ValueError as e:  # unreadable, encrypted, empty: messages fit for the user
            message = str(e)
            _replace(self.body, lambda: _passage_only(self.source, message))
            return
        except Exception as e:
            message = f"Couldn't render {self.path.name}: {e or type(e).__name__}"
            _replace(self.body, lambda: _passage_only(self.source, message))
            return
        if image is None:  # the app is shutting down
            return
        if self.cited is None:
            self.cited = image.page
        _replace(self.body, lambda: self._build(image))

    def _build(self, image: PageImage) -> None:
        if image.page == self.cited and not image.highlighted:
            ui.label(NOT_FOUND_ON_PAGE).classes("text-warning")
        with ui.row().classes("w-full items-center gap-2"):
            back = ui.button(icon="chevron_left", on_click=lambda: self.show(image.page - 1))
            back.props("flat dense").tooltip("Previous page")
            back.set_enabled(image.page > 1)
            ui.label(f"Page {image.page} of {image.page_count}")
            forward = ui.button(icon="chevron_right", on_click=lambda: self.show(image.page + 1))
            forward.props("flat dense").tooltip("Next page")
            forward.set_enabled(image.page < image.page_count)
            ui.space()
            ui.button("Open in viewer", icon="open_in_new", on_click=self.open_external).props(
                "flat dense"
            )
        data = base64.b64encode(image.png).decode("ascii")
        with ui.scroll_area().classes("w-full h-[70vh] border rounded"):
            ui.image(f"data:image/png;base64,{data}").classes("w-full")

    async def open_external(self) -> None:
        """A highlighted copy in the app's temp folder, opened with the system's PDF viewer."""
        page = self.cited or self.source.page or 1
        try:
            copy = await run.io_bound(_highlighted_copy, self.path, page, self.source.text)
            if copy is not None:  # None: the app is shutting down
                _open_with_system(copy)
        except Exception as e:
            ui.notify(f"Couldn't open {self.path.name}: {e or type(e).__name__}", type="negative")


def _highlighted_copy(pdf: Path, page: int, passage: str) -> Path:
    tmp = state().cfg.paths.data_dir / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    # A folder per copy: a viewer may still hold the previous copy open (and locked).
    out_dir = Path(tempfile.mkdtemp(prefix="cite-", dir=tmp))
    return highlighted_pdf_copy(pdf, page, passage, out_dir)


def _open_with_system(path: Path) -> None:
    """Open a file with the OS default app, without waiting for it."""
    if sys.platform == "win32":
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


# --- Markdown / text ---------------------------------------------------------------------


async def _show_text(body: ui.column, path: Path, source: Source) -> None:
    try:
        found = await run.io_bound(_read_and_find, path, source.text)
    except OSError as e:
        message = f"Couldn't read {path.name}: {e.strerror or e}"
        _replace(body, lambda: _passage_only(source, message))
        return
    if found is None:  # the app is shutting down
        return
    text, span = found
    if span is None:
        _replace(body, lambda: _passage_only(source, NOT_FOUND_IN_FILE))
        return
    mark_id = f"vm-mark-{uuid.uuid4().hex[:8]}"
    _replace(body, lambda: _text_with_mark(path, text, span, mark_id))
    if not body.is_deleted:
        body.client.run_javascript(
            f"setTimeout(() => document.getElementById('{mark_id}')"
            "?.scrollIntoView({block: 'center'}), 150)"
        )


def _read_and_find(path: Path, passage: str) -> tuple[str, tuple[int, int] | None]:
    text = path.read_text("utf-8-sig", errors="replace")
    return text, find_passage(text, passage)


def marked_html(text: str, span: tuple[int, int], mark_id: str) -> str:
    """`text` escaped for HTML, with the [start, end) span wrapped in <mark id=mark_id>.

    A very large file is cut to a window around the passage."""
    start, end = span
    lo, hi = 0, len(text)
    if len(text) > _MAX_SHOWN:
        lo = max(0, start - _MAX_SHOWN // 2)
        hi = min(len(text), max(end, lo + _MAX_SHOWN))
    return (
        ("...\n" if lo else "")
        + html.escape(text[lo:start])
        + f'<mark id="{mark_id}">{html.escape(text[start:end])}</mark>'
        + html.escape(text[end:hi])
        + ("\n..." if hi < len(text) else "")
    )


def _text_with_mark(path: Path, text: str, span: tuple[int, int], mark_id: str) -> None:
    with ui.row().classes("w-full items-center"):
        ui.space()
        ui.button("Open file", icon="open_in_new", on_click=lambda: _open_file(path)).props(
            "flat dense"
        )
    with ui.scroll_area().classes("w-full h-[70vh] border rounded"):
        # Escaped by marked_html; unicode-bidi: plaintext gives each line its own direction.
        ui.html(
            '<pre style="white-space: pre-wrap; unicode-bidi: plaintext; font-family: inherit; '
            f'margin: 0">{marked_html(text, span, mark_id)}</pre>',
            sanitize=False,
        ).classes("w-full p-2 text-sm")


def _open_file(path: Path) -> None:
    try:
        _open_with_system(path)
    except OSError as e:
        ui.notify(f"Couldn't open {path.name}: {e.strerror or e}", type="negative")
