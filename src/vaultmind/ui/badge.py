"""Header privacy badge: "localhost only", or red with the remote addresses it saw."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from nicegui import run, ui

from vaultmind.app import privacy
from vaultmind.app.privacy import PrivacyStatus
from vaultmind.ui.state import state

_EXPLANATION = (
    "VaultMind checks the network connections of its own process and of every process it "
    "started (for example the browser processes behind the app window). Connections to this "
    "computer (localhost) are fine; anything else is listed here and turns the badge red. "
    "Ollama runs as a separate program, so its connections are not covered by this check."
)


def privacy_badge() -> None:
    """Build the badge; it re-checks every `ui.privacy_check_s` seconds (`app/privacy.py`)."""
    latest: PrivacyStatus | None = None
    failure: str | None = None

    chip = ui.chip("Checking connections", icon="shield", color="grey").props("dense")
    chip.on_click(lambda: _details(latest, failure))
    with chip:
        tooltip = ui.tooltip("Checking this app's network connections")

    async def refresh() -> None:
        nonlocal latest, failure
        try:
            status = await off_loop(privacy.check)
        except Exception as exc:
            failure = str(exc) or type(exc).__name__
            chip.text, tooltip.text = "Privacy check failed", failure
            chip.set_background_color("grey")
            return
        if status is None:  # the app is shutting down
            return
        latest, failure = status, None
        chip.text = _label(status)
        chip.set_background_color("positive" if status.local_only else "negative")
        chip.icon = "verified_user" if status.local_only else "gpp_bad"
        tooltip.text = _summary(status)

    ui.timer(state().cfg.ui.privacy_check_s, refresh)


async def off_loop[**P, R](fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R | None:
    """`run.io_bound`, but a cancelled caller stays cancelled. io_bound swallows the
    cancellation and returns None, which keeps a `ui.timer` loop running forever when its
    page is torn down mid-call (it hung the test run after a failing test)."""
    result = await run.io_bound(fn, *args, **kwargs)
    task = asyncio.current_task()
    if result is None and task is not None and task.cancelling():
        raise asyncio.CancelledError
    return result


def _label(status: PrivacyStatus) -> str:
    if status.local_only:
        return "localhost only"
    n = len(status.remote)
    return f"{n} remote connection{'s' if n != 1 else ''}"


def _summary(status: PrivacyStatus) -> str:
    checked = f"Last checked {_local_time(status)}"
    if status.local_only:
        return f"No connections outside this computer. {checked}"
    return "Remote: " + ", ".join(status.remote) + f". {checked}"


def _local_time(status: PrivacyStatus) -> str:
    return status.checked_at.astimezone().strftime("%H:%M:%S")


def _details(status: PrivacyStatus | None, failure: str | None) -> None:
    with ui.dialog() as dialog, ui.card().classes("max-w-xl"):
        ui.label("Privacy check").classes("text-lg font-semibold")
        ui.label(_EXPLANATION)
        if failure is not None:
            ui.label(f"The last check failed: {failure}").classes("text-negative")
        elif status is None:
            ui.label("The first check hasn't finished yet.")
        elif status.local_only:
            ui.label("No connections outside this computer.").classes("text-positive")
        else:
            ui.label("Connections outside this computer:").classes("text-negative")
            for address in status.remote:
                ui.label(address).classes("font-mono")
        if status is not None:
            ui.label(f"Last checked at {_local_time(status)}.").classes("text-sm text-grey-7")
        ui.button("Close", on_click=dialog.close)
    dialog.on("hide", dialog.delete)  # built per click; don't pile up closed dialogs
    dialog.open()
