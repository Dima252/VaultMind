"""Library page, queue indicator, privacy badge and settings page, driven with NiceGUI's
simulated user against real stores and the fake embedder (no models)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from nicegui import ui
from nicegui.elements.upload_files import SmallFileUpload
from nicegui.testing.user import User
from nicegui.testing.user_simulation import user_simulation

from vaultmind.app import privacy
from vaultmind.app.engine import Engine
from vaultmind.app.privacy import PrivacyStatus
from vaultmind.core.config import Config
from vaultmind.core.models import FileRecord, file_id_for
from vaultmind.ingest.pipeline import ingest_path
from vaultmind.ingest.queue import IngestQueue, JobEvent
from vaultmind.store.chats import ChatStore
from vaultmind.store.index import ChunkIndex
from vaultmind.store.registry import Registry
from vaultmind.store.traces import TraceStore
from vaultmind.ui import library, settings_page
from vaultmind.ui import state as ui_state
from vaultmind.ui.app import root

from .fakes import FakeEmbedder

LOCAL_ONLY = PrivacyStatus(local_only=True, remote=(), checked_at=datetime.now(UTC))


class RecordingQueue(IngestQueue):
    """A real queue that records what the page asked for. Not started unless a test does."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.calls: list[tuple[str, Path]] = []

    def submit(self, path: Path) -> None:
        self.calls.append(("submit", path.resolve()))
        super().submit(path)

    def forget(self, path: Path) -> None:
        self.calls.append(("forget", path.resolve()))
        super().forget(path)

    def scan(self) -> None:
        self.calls.append(("scan", self._cfg.paths.vault_dir))
        super().scan()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def quiet(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Fast polling, a local privacy check, no Ollama, and Save writes under tmp_path."""
    monkeypatch.setattr(library, "REFRESH_S", 0.05)
    monkeypatch.setattr(library, "INDICATOR_S", 0.05)
    monkeypatch.setattr(privacy, "check", lambda pid=None: LOCAL_ONLY)
    monkeypatch.setattr(settings_page, "LOCAL_PATH", tmp_path / "config" / "local.yaml")

    def ollama_down(host: str) -> list[str]:
        raise ConnectionError("Ollama is not running")

    monkeypatch.setattr(settings_page, "list_ollama_models", ollama_down)


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config.model_validate(
        {
            "paths": {"vault_dir": str(tmp_path / "vault"), "data_dir": str(tmp_path / "data")},
            "retrieval": {"rerank": False},
            "ui": {"privacy_check_s": 0.05},
        }
    )


@pytest.fixture
def engine(cfg: Config) -> Iterator[Engine]:
    """An engine over a vault with one good file, one in a subfolder and one broken PDF."""
    vault = cfg.paths.vault_dir
    (vault / "contracts").mkdir(parents=True)
    (vault / "fees.md").write_text("# Fees\n\nPayment is due within sixty days.\n", "utf-8")
    (vault / "contracts" / "lease.txt").write_text("The lease runs for two years.\n", "utf-8")
    (vault / "broken.pdf").write_bytes(b"not a pdf")
    embedder = FakeEmbedder()
    db = cfg.paths.data_dir / "vaultmind.sqlite"
    index = ChunkIndex(cfg.paths.data_dir / "index", embedder_name=embedder.model_name)
    registry = Registry(db)
    ingest_path(vault, cfg, embedder=embedder, index=index, registry=registry)
    eng = Engine(
        cfg,
        llm=None,  # type: ignore[arg-type]  # the library and settings pages never ask
        embedder=embedder,
        reranker=None,
        index=index,
        registry=registry,
        traces=TraceStore(db),
        chats=ChatStore(db),
    )
    yield eng
    eng.close()


@pytest.fixture
def app_state(cfg: Config, engine: Engine) -> Iterator[ui_state.AppState]:
    s = ui_state.init_state(cfg)
    engine.queue = RecordingQueue(
        cfg,
        embedder=engine.embedder,
        index=engine.index,
        registry=engine.registry,
        on_event=s.on_job,
    )
    s.engine, s.phase = engine, "ready"
    yield s
    ui_state._state = None


def _queue(s: ui_state.AppState) -> RecordingQueue:
    assert s.engine is not None and isinstance(s.engine.queue, RecordingQueue)
    return s.engine.queue


def _table(user: User) -> ui.table:
    return next(iter(user.find(ui.table).elements))


def _status(user: User) -> dict[str, str]:
    return {row["name"]: row["status"] for row in _table(user).rows}


async def _wait_for(check: Callable[[], bool], tries: int = 60) -> None:
    """Let the page's timers run until `check()` holds."""
    for _ in range(tries):
        if check():
            return
        await asyncio.sleep(0.05)
    assert check()


# -- library page --


@pytest.mark.anyio
async def test_library_lists_files_with_status(app_state: ui_state.AppState) -> None:
    async with user_simulation(root=root) as user:
        await user.open("/library")
        await user.should_see("3 files, 2 ready, 1 failed")
        await user.should_see("chunks in the index")
        rows = {row["name"]: row for row in _table(user).rows}
        assert rows["fees.md"]["status"] == "ready" and rows["fees.md"]["folder"] == ""
        assert rows["lease.txt"]["folder"] == "contracts"
        assert rows["broken.pdf"]["status"] == "failed" and rows["broken.pdf"]["error"]
        assert rows["fees.md"]["indexed"] and all(r["in_vault"] for r in rows.values())


@pytest.mark.anyio
async def test_filters_narrow_the_table(app_state: ui_state.AppState) -> None:
    async with user_simulation(root=root) as user:
        await user.open("/library")
        user.find(ui.select).elements.pop().value = "failed"
        assert list(_status(user)) == ["broken.pdf"]
        user.find(ui.select).elements.pop().value = "all"
        user.find(ui.input).type("lease")
        assert list(_status(user)) == ["lease.txt"]


@pytest.mark.anyio
async def test_live_indexing_event_overrides_ready(app_state: ui_state.AppState) -> None:
    fees = (app_state.cfg.paths.vault_dir / "fees.md").resolve()
    async with user_simulation(root=root) as user:
        await user.open("/library")
        assert _status(user)["fees.md"] == "ready"
        app_state.on_job(JobEvent(fees, "indexing"))
        await _wait_for(lambda: _status(user)["fees.md"] == "indexing")
        await user.should_see("1 in progress")
        app_state.on_job(JobEvent(fees, "ready"))
        await _wait_for(lambda: _status(user)["fees.md"] == "ready")


@pytest.mark.anyio
async def test_upload_saves_into_vault_and_submits(app_state: ui_state.AppState) -> None:
    vault = app_state.cfg.paths.vault_dir
    async with user_simulation(root=root) as user:
        await user.open("/library")
        upload = next(iter(user.find(ui.upload).elements))
        await upload.handle_uploads([SmallFileUpload("notes.md", "text/markdown", b"# A\n\none")])
        await user.should_see("Added notes.md")
        # Same name, other content: kept side by side. Same content again: reused.
        await upload.handle_uploads([SmallFileUpload("notes.md", "text/markdown", b"# B\n\ntwo")])
        await user.should_see("Added notes (2).md")
        await upload.handle_uploads([SmallFileUpload("notes.md", "text/markdown", b"# A\n\none")])
        await upload.handle_uploads([SmallFileUpload("evil.exe", "application/x", b"MZ")])
        await user.should_see("not a supported file type")
        # The queued files show up in the table right away.
        await _wait_for(lambda: _status(user).get("notes (2).md") == "queued")

    assert (vault / "notes.md").read_bytes() == b"# A\n\none"
    assert (vault / "notes (2).md").read_bytes() == b"# B\n\ntwo"
    assert not (vault / "evil.exe").exists()
    assert not list(vault.glob(".*.tmp"))
    submitted = [p.name for kind, p in _queue(app_state).calls if kind == "submit"]
    assert submitted == ["notes.md", "notes (2).md", "notes.md"]


@pytest.mark.anyio
async def test_uploaded_file_gets_indexed(app_state: ui_state.AppState) -> None:
    queue = _queue(app_state)
    queue.start()
    async with user_simulation(root=root) as user:
        await user.open("/library")
        upload = next(iter(user.find(ui.upload).elements))
        await upload.handle_uploads([SmallFileUpload("new.txt", "text/plain", b"Fresh text.\n")])
        await _wait_for(lambda: _status(user).get("new.txt") == "ready")
        await user.should_see("4 files, 3 ready")
    queue.stop()


@pytest.mark.anyio
async def test_remove_asks_first_and_deletes_only_after_confirming(
    app_state: ui_state.AppState,
) -> None:
    fees = (app_state.cfg.paths.vault_dir / "fees.md").resolve()
    async with user_simulation(root=root) as user:
        await user.open("/library")
        row = next(r for r in _table(user).rows if r["name"] == "fees.md")

        user.find(ui.table).trigger("remove", row)
        await user.should_see("deleted from the vault folder")
        user.find("Cancel").click()
        await user.should_not_see("deleted from the vault folder")
        assert fees.exists()

        user.find(ui.table).trigger("remove", row)
        await user.should_see("deleted from the vault folder")
        assert fees.exists()
        user.find(marker="confirm").click()
        await user.should_see("Deleted fees.md")
    assert not fees.exists()
    assert ("forget", fees) in _queue(app_state).calls


@pytest.mark.anyio
async def test_remove_refuses_files_outside_the_vault(
    app_state: ui_state.AppState, tmp_path: Path
) -> None:
    outside = tmp_path / "elsewhere.md"
    outside.write_text("keep me", "utf-8")
    sneaky = app_state.cfg.paths.vault_dir / ".." / "elsewhere.md"
    async with user_simulation(root=root) as user:
        await user.open("/library")
        user.find(ui.table).trigger("remove", {"path": str(sneaky)})
        await user.should_see("Only files inside the vault folder")
    assert outside.exists()


@pytest.mark.anyio
async def test_reindex_marks_the_file_queued_and_submits(app_state: ui_state.AppState) -> None:
    assert app_state.engine is not None
    fees = (app_state.cfg.paths.vault_dir / "fees.md").resolve()
    async with user_simulation(root=root) as user:
        await user.open("/library")
        row = next(r for r in _table(user).rows if r["name"] == "fees.md")
        user.find(ui.table).trigger("reindex", row)
        await user.should_see("Re-indexing fees.md")
        record = app_state.engine.registry.get(file_id_for(fees))
        assert record is not None and record.status == "queued"
        assert ("submit", fees) in _queue(app_state).calls

        user.find("Re-scan vault").click()
        assert any(kind == "scan" for kind, _ in _queue(app_state).calls)


@pytest.mark.anyio
async def test_open_uses_the_os(
    app_state: ui_state.AppState, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[Path] = []
    monkeypatch.setattr(library, "open_with_os", opened.append)
    async with user_simulation(root=root) as user:
        await user.open("/library")
        row = next(r for r in _table(user).rows if r["name"] == "lease.txt")
        user.find(ui.table).trigger("open", row)
        await _wait_for(lambda: bool(opened))
    assert opened[0].name == "lease.txt"


def test_file_rows_merge_registry_and_events(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    ready = FileRecord(id="a", path=vault / "a.md", sha256="x", status="ready")
    failed = FileRecord(id="b", path=vault / "b.md", sha256="x", status="failed", error="bad")
    jobs = {
        vault / "a.md": JobEvent(vault / "a.md", "queued"),
        vault / "b.md": JobEvent(vault / "b.md", "ready"),  # registry is as new: failed stays
        vault / "c.md": JobEvent(vault / "c.md", "failed", "locked"),  # no registry row
        vault / "d.md": JobEvent(vault / "d.md", "removed"),
    }
    rows = {r["name"]: r for r in library.file_rows([ready, failed], jobs, vault)}
    assert {n: r["status"] for n, r in rows.items()} == {
        "a.md": "queued",
        "b.md": "failed",
        "c.md": "failed",
    }
    assert rows["c.md"]["error"] == "locked" and rows["a.md"]["error"] == ""


def test_place_upload_names(tmp_path: Path) -> None:
    def tmp(data: bytes) -> Path:
        path = tmp_path / ".upload.tmp"
        path.write_bytes(data)
        return path

    vault = tmp_path / "vault"
    vault.mkdir()
    assert library.place_upload(tmp(b"1"), "a.md", vault).name == "a.md"
    assert library.place_upload(tmp(b"2"), "a.md", vault).name == "a (2).md"
    assert library.place_upload(tmp(b"3"), "a.md", vault).name == "a (3).md"
    assert library.place_upload(tmp(b"2"), "a.md", vault).name == "a (2).md"
    assert sorted(p.name for p in vault.iterdir()) == ["a (2).md", "a (3).md", "a.md"]


# -- header: queue indicator and privacy badge --


@pytest.mark.anyio
async def test_queue_indicator_shows_while_jobs_are_pending(app_state: ui_state.AppState) -> None:
    async with user_simulation(root=root) as user:
        await user.open("/settings")
        await user.should_not_see(marker="queue-indicator")
        _queue(app_state).submit(app_state.cfg.paths.vault_dir / "fees.md")
        await user.should_see("Indexing 1 file")
        user.find(marker="queue-indicator").click()
        await user.should_see("chunks in the index")  # navigated to the library


@pytest.mark.anyio
async def test_queue_indicator_hidden_without_engine(cfg: Config) -> None:
    ui_state.init_state(cfg)
    try:
        async with user_simulation(root=root) as user:
            await user.open("/settings")
            await user.should_not_see(marker="queue-indicator")
    finally:
        ui_state._state = None


@pytest.mark.anyio
async def test_privacy_badge_local(app_state: ui_state.AppState) -> None:
    async with user_simulation(root=root) as user:
        await user.open("/settings")
        await user.should_see("localhost only")


@pytest.mark.anyio
async def test_privacy_badge_remote(
    app_state: ui_state.AppState, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = PrivacyStatus(False, ("93.184.216.34:443 (python.exe)",), datetime.now(UTC))
    monkeypatch.setattr(privacy, "check", lambda pid=None: remote)
    async with user_simulation(root=root) as user:
        await user.open("/settings")
        await user.should_see("1 remote connection")
        user.find(ui.chip).click()
        await user.should_see("93.184.216.34:443 (python.exe)")
        await user.should_see("Ollama runs as a separate program")


# -- settings page --


def _set(user: User, key: str, value: Any) -> None:
    for element in user.find(marker=key).elements:
        element.value = value  # type: ignore[attr-defined]


def _saved(app_state: ui_state.AppState) -> dict[str, Any]:
    data = yaml.safe_load(settings_page.LOCAL_PATH.read_text("utf-8"))
    assert isinstance(data, dict)
    return data


@pytest.mark.anyio
async def test_retrieval_change_saves_and_applies(app_state: ui_state.AppState) -> None:
    assert app_state.engine is not None
    async with user_simulation(root=root) as user:
        await user.open("/settings")
        _set(user, "retrieval.top_k", 7.0)  # ui.number gives floats
        _set(user, "agent.groundedness_check", True)
        user.find(kind=ui.button, content="Save").click()
        await user.should_see("Applied")

    vault, data = app_state.cfg.paths.vault_dir, app_state.cfg.paths.data_dir
    assert _saved(app_state) == {
        "paths": {"vault_dir": str(vault), "data_dir": str(data)},
        "retrieval": {"rerank": False, "top_k": 7},
        "agent": {"groundedness_check": True},
        "ui": {"privacy_check_s": 0.05},
    }
    assert app_state.engine.cfg.retrieval.top_k == 7
    assert app_state.cfg.agent.groundedness_check is True


@pytest.mark.anyio
async def test_model_change_asks_for_restart(app_state: ui_state.AppState) -> None:
    assert app_state.engine is not None
    async with user_simulation(root=root) as user:
        await user.open("/settings")
        _set(user, "models.llm", "qwen3:8b")
        user.find(kind=ui.button, content="Save").click()
        await user.should_see("Restart VaultMind to apply: models.llm")
    assert _saved(app_state)["models"] == {"llm": "qwen3:8b"}
    assert app_state.engine.cfg.models.llm == "qwen3:14b"


@pytest.mark.anyio
async def test_embedder_change_warns_about_reindex(app_state: ui_state.AppState) -> None:
    async with user_simulation(root=root) as user:
        await user.open("/settings")
        _set(user, "models.embedder", "nomic-embed-text")
        user.find(kind=ui.button, content="Save").click()
        await user.should_see("the index must be rebuilt")


@pytest.mark.anyio
async def test_invalid_input_shows_error_and_writes_nothing(app_state: ui_state.AppState) -> None:
    async with user_simulation(root=root) as user:
        await user.open("/settings")
        _set(user, "models.num_ctx", 0)
        user.find(kind=ui.button, content="Save").click()
        await user.should_see("greater than 0")
        await user.should_see("nothing was saved")
        _set(user, "models.num_ctx", 16384)
        _set(user, "paths.vault_dir", str(app_state.cfg.paths.vault_dir / "fees.md"))
        user.find(kind=ui.button, content="Save").click()
        await user.should_see("not a folder")
    assert not settings_page.LOCAL_PATH.exists()


@pytest.mark.anyio
async def test_reset_fills_defaults_without_saving(app_state: ui_state.AppState) -> None:
    async with user_simulation(root=root) as user:
        await user.open("/settings")
        user.find("Reset to defaults").click()
        await user.should_see("Nothing is saved until you press Save")
        user.find(marker="confirm").click()
        await user.should_see("Defaults filled in")
        (rerank,) = user.find(marker="retrieval.rerank").elements
        (vault,) = user.find(marker="paths.vault_dir").elements
        assert rerank.value is True  # type: ignore[attr-defined]
        assert vault.value == "vault"  # type: ignore[attr-defined]
    assert not settings_page.LOCAL_PATH.exists()


@pytest.mark.anyio
async def test_explicit_config_file_is_flagged(
    app_state: ui_state.AppState, tmp_path: Path
) -> None:
    app_state.config_source = tmp_path / "custom.yaml"
    async with user_simulation(root=root) as user:
        await user.open("/settings")
        user.find(kind=ui.button, content="Save").click()
        await user.should_see("won't be used until you launch it without --config")


@pytest.mark.anyio
async def test_browser_flag_is_not_saved(app_state: ui_state.AppState) -> None:
    # `vaultmind ui --browser` turns native off for the run; the config file still says on.
    app_state.cfg = app_state.cfg.model_copy(
        update={"ui": app_state.cfg.ui.model_copy(update={"native": False})}
    )
    async with user_simulation(root=root) as user:
        await user.open("/settings")
        user.find(kind=ui.button, content="Save").click()
        await user.should_see("Applied")
    assert "native" not in _saved(app_state).get("ui", {})


@pytest.mark.anyio
async def test_failed_engine_takes_fixed_settings_on_retry(cfg: Config) -> None:
    s = ui_state.init_state(cfg)
    s.phase = "failed"
    try:
        async with user_simulation(root=root) as user:
            await user.open("/settings")
            await user.should_see("Ollama isn't reachable")
            _set(user, "models.llm", "qwen3:8b")
            user.find(kind=ui.button, content="Save").click()
            await user.should_see("Press Retry")
        assert s.cfg.models.llm == "qwen3:8b"
    finally:
        ui_state._state = None


@pytest.mark.anyio
async def test_model_names_from_ollama(
    app_state: ui_state.AppState, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings_page, "list_ollama_models", lambda host: ["a:1b", "b:2b"])
    async with user_simulation(root=root) as user:
        await user.open("/settings")
        await user.should_see("Pulled in Ollama: a:1b, b:2b")
