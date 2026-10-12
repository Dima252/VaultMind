"""The app shell: header, routing, and the loading/error panel before the engine is ready."""

from collections.abc import Iterator

import pytest
from nicegui.testing.user_simulation import user_simulation

from vaultmind.core.config import Config
from vaultmind.ui import state as ui_state
from vaultmind.ui.app import root


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def app_state() -> Iterator[ui_state.AppState]:
    yield ui_state.init_state(Config())
    ui_state._state = None


@pytest.mark.anyio
async def test_loading_panel_until_ready(app_state: ui_state.AppState) -> None:
    app_state.message = "Loading models"
    async with user_simulation(root=root) as user:
        await user.open("/")
        await user.should_see("VaultMind")
        await user.should_see("Loading models")


@pytest.mark.anyio
async def test_failed_engine_shows_error_and_retry(app_state: ui_state.AppState) -> None:
    app_state.phase, app_state.message = "failed", "Ollama is not running"
    async with user_simulation(root=root) as user:
        await user.open("/library")
        await user.should_see("Ollama is not running")
        await user.should_see("Retry")


@pytest.mark.anyio
async def test_settings_open_without_engine(app_state: ui_state.AppState) -> None:
    app_state.phase = "failed"
    async with user_simulation(root=root) as user:
        await user.open("/settings")
        await user.should_see("Settings")
