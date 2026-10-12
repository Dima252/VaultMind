"""The desktop window: NiceGUI pages calling the engine in-process (there is no HTTP API).

Served on 127.0.0.1 only. NiceGUI binds 0.0.0.0 by default outside native mode, so the host
is always passed explicitly.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from nicegui import app, run, ui

from vaultmind.core.config import Config
from vaultmind.ui import badge, chat, library, settings_page
from vaultmind.ui.state import init_state, state

HOST = "127.0.0.1"
WINDOW_SIZE = (1280, 860)


def main(cfg: Config, config_source: Path | None = None) -> None:
    """Run the app until its window is closed."""
    init_state(cfg, config_source)
    app.native.settings["ALLOW_DOWNLOADS"] = True  # exports use ui.download in the native window
    app.on_startup(_load_engine)
    app.on_shutdown(lambda: state().close())
    ui.run(
        root,
        host=HOST,
        port=cfg.ui.port,
        title="VaultMind",
        native=cfg.ui.native,
        window_size=WINDOW_SIZE if cfg.ui.native else None,
        dark=None,
        reload=False,
        show_welcome_message=False,
        on_air=None,
        fastapi_docs=False,
    )


async def _load_engine() -> None:
    await run.io_bound(state().load)


def root() -> None:
    """Every page: the header, then the routed content."""
    with ui.header().classes("items-center gap-4 px-4 py-2"):
        ui.label("VaultMind").classes("text-lg font-semibold")
        ui.link("Chat", "/").classes("text-white")
        ui.link("Library", "/library").classes("text-white")
        ui.link("Settings", "/settings").classes("text-white")
        ui.space()
        library.queue_indicator()
        badge.privacy_badge()
    ui.sub_pages(
        {
            "/": _when_ready(chat.page),
            "/library": _when_ready(library.page),
            "/settings": settings_page.page,  # usable even when the engine failed to load
        }
    )


def _when_ready(build: Callable[[], None]) -> Callable[[], None]:
    """Show the loading or error panel until the engine is ready, then the page."""

    def page() -> None:
        shown = state().phase

        @ui.refreshable
        def body() -> None:
            s = state()
            if s.phase == "ready":
                build()
            elif s.phase == "loading":
                with ui.row().classes("items-center gap-3 p-8"):
                    ui.spinner(size="lg")
                    ui.label().bind_text_from(s, "message")
            else:
                with ui.column().classes("p-8 gap-3"):
                    ui.label("VaultMind couldn't start.").classes("text-lg text-negative")
                    ui.label(s.message).classes("whitespace-pre-wrap")
                    ui.button("Retry", on_click=_retry)

        def check() -> None:
            nonlocal shown
            if state().phase != shown:
                shown = state().phase
                body.refresh()

        body()
        ui.timer(0.5, check)

    return page


async def _retry() -> None:
    state().phase = "loading"
    await run.io_bound(state().load)
