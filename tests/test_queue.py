"""IngestQueue: background indexing with real stores and the fake embedder."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from vaultmind.core.config import Config, PathsConfig
from vaultmind.core.models import FileRecord, file_id_for
from vaultmind.ingest.queue import IngestQueue, JobEvent, JobState
from vaultmind.store.index import ChunkIndex
from vaultmind.store.registry import Registry

from .fakes import FakeEmbedder

LONG_TEXT = "# Alpha\n\n" + "\n\n".join(f"Paragraph {i} " + "word " * 200 for i in range(6))


class OllamaDown(RuntimeError):
    """Stands in for `OllamaUnavailableError` (no Ollama import in unit tests)."""


class FlakyEmbedder(FakeEmbedder):
    """Raises while `down` is set, like an embedder whose Ollama is unreachable."""

    def __init__(self) -> None:
        super().__init__()
        self.down = False

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if self.down:
            raise OllamaDown("Cannot reach Ollama: is Ollama running at http://127.0.0.1:11434?")
        return super().embed(texts)


class GatedEmbedder(FakeEmbedder):
    """Blocks inside `embed` until released, so a test can act while a file is indexing."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.entered.set()
        assert self.release.wait(10)
        return super().embed(texts)


class Recorder:
    def __init__(self) -> None:
        self.events: list[JobEvent] = []
        self._lock = threading.Lock()

    def __call__(self, event: JobEvent) -> None:
        with self._lock:
            self.events.append(event)

    def states(self, name: str) -> list[JobState]:
        with self._lock:
            return [e.state for e in self.events if e.path.name == name]

    def of(self, state: JobState) -> list[str]:
        with self._lock:
            return sorted(e.path.name for e in self.events if e.state == state)

    def clear(self) -> None:
        with self._lock:
            self.events.clear()


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    root = tmp_path / "vault"
    (root / "sub").mkdir(parents=True)
    (root / "a.md").write_text("# Alpha\n\nAlpha is the first letter.\n", encoding="utf-8")
    (root / "sub" / "b.txt").write_text("Bravo is the second.\n", encoding="utf-8")
    return root


@pytest.fixture
def cfg(tmp_path: Path, vault: Path) -> Config:
    return Config(paths=PathsConfig(vault_dir=vault, data_dir=tmp_path / "data"))


@pytest.fixture
def stores(cfg: Config) -> Iterator[tuple[ChunkIndex, Registry]]:
    index = ChunkIndex(cfg.paths.data_dir / "index", embedder_name=FakeEmbedder.model_name)
    registry = Registry(cfg.paths.data_dir / "vaultmind.sqlite")
    yield index, registry
    registry.close()
    index.close()


@pytest.fixture
def rec() -> Recorder:
    return Recorder()


def make_queue(
    cfg: Config,
    stores: tuple[ChunkIndex, Registry],
    rec: Recorder,
    embedder: FakeEmbedder | None = None,
) -> IngestQueue:
    index, registry = stores
    return IngestQueue(
        cfg, embedder=embedder or FakeEmbedder(), index=index, registry=registry, on_event=rec
    )


@pytest.fixture
def queue(cfg: Config, stores: tuple[ChunkIndex, Registry], rec: Recorder) -> Iterator[IngestQueue]:
    q = make_queue(cfg, stores, rec)
    q.start()
    yield q
    q.stop()


def stop_while_running(q: IngestQueue, embedder: GatedEmbedder) -> None:
    """Call `stop` while a file is inside `embed`, then let that file finish."""
    stopper = threading.Thread(target=q.stop)
    stopper.start()
    deadline = time.monotonic() + 10
    while not q._stopped and time.monotonic() < deadline:
        time.sleep(0.01)
    embedder.release.set()
    stopper.join(10)
    assert not stopper.is_alive()


def statuses(registry: Registry) -> dict[str, str]:
    return {r.path.name: r.status for r in registry.all()}


def test_index_folder_emits_events_and_ready_records(
    queue: IngestQueue, vault: Path, stores: tuple[ChunkIndex, Registry], rec: Recorder
) -> None:
    queue.submit(vault)
    assert queue.wait_idle(10)
    index, registry = stores
    assert statuses(registry) == {"a.md": "ready", "b.txt": "ready"}
    assert index.count() == 2
    assert rec.states("a.md") == ["indexing", "ready"]
    assert rec.states("b.txt") == ["indexing", "ready"]
    assert rec.of("queued") == []  # a folder has no per-file queued event
    assert queue.pending() == 0


def test_single_file_queued_then_indexed(queue: IngestQueue, vault: Path, rec: Recorder) -> None:
    queue.submit(vault / "a.md")
    assert queue.wait_idle(10)
    assert rec.states("a.md") == ["queued", "indexing", "ready"]


def test_unchanged_resubmit_does_not_reindex(
    queue: IngestQueue, vault: Path, rec: Recorder
) -> None:
    queue.submit(vault)
    assert queue.wait_idle(10)
    rec.clear()
    queue.submit(vault)
    queue.submit(vault / "a.md")
    assert queue.wait_idle(10)
    assert rec.of("indexing") == []
    assert rec.states("a.md") == ["queued", "ready"]  # the queued state is closed


def test_modified_file_is_reindexed(
    queue: IngestQueue, vault: Path, stores: tuple[ChunkIndex, Registry], rec: Recorder
) -> None:
    queue.submit(vault)
    assert queue.wait_idle(10)
    rec.clear()
    (vault / "a.md").write_text(LONG_TEXT, encoding="utf-8")
    queue.submit(vault / "a.md")
    assert queue.wait_idle(10)
    index, _ = stores
    assert rec.states("a.md") == ["queued", "indexing", "ready"]
    assert index.count() > 2


def test_forget_removes_file(
    queue: IngestQueue, vault: Path, stores: tuple[ChunkIndex, Registry], rec: Recorder
) -> None:
    queue.submit(vault)
    assert queue.wait_idle(10)
    (vault / "a.md").unlink()
    queue.forget(vault / "a.md")
    queue.forget(vault / "never-seen.md")  # harmless
    assert queue.wait_idle(10)
    index, registry = stores
    assert statuses(registry) == {"b.txt": "ready"}
    assert index.count() == 1
    assert rec.of("removed") == ["a.md"]


def test_forget_folder_removes_files_under_it(
    queue: IngestQueue, vault: Path, stores: tuple[ChunkIndex, Registry], rec: Recorder
) -> None:
    queue.submit(vault)
    assert queue.wait_idle(10)
    (vault / "sub" / "b.txt").unlink()
    (vault / "sub").rmdir()
    queue.forget(vault / "sub")
    assert queue.wait_idle(10)
    _, registry = stores
    assert statuses(registry) == {"a.md": "ready"}
    assert rec.of("removed") == ["b.txt"]


def test_scan_indexes_vault_and_forgets_records_outside_it(
    queue: IngestQueue,
    cfg: Config,
    tmp_path: Path,
    stores: tuple[ChunkIndex, Registry],
    rec: Recorder,
) -> None:
    _, registry = stores
    elsewhere = tmp_path / "old-vault" / "gone.md"
    registry.upsert(
        FileRecord(id=file_id_for(elsewhere), path=elsewhere, sha256="x", status="ready")
    )
    queue.scan()
    assert queue.wait_idle(10)
    assert statuses(registry) == {"a.md": "ready", "b.txt": "ready"}
    assert rec.of("removed") == ["gone.md"]


def test_scan_creates_missing_vault(
    tmp_path: Path, stores: tuple[ChunkIndex, Registry], rec: Recorder
) -> None:
    cfg = Config(paths=PathsConfig(vault_dir=tmp_path / "new-vault", data_dir=tmp_path / "data"))
    q = make_queue(cfg, stores, rec)
    q.start()
    q.scan()
    assert q.wait_idle(10)
    q.stop()
    assert (tmp_path / "new-vault").is_dir()


def test_duplicate_waiting_submits_collapse(
    cfg: Config, stores: tuple[ChunkIndex, Registry], vault: Path, rec: Recorder
) -> None:
    q = make_queue(cfg, stores, rec)  # not started: everything stays waiting
    for _ in range(3):
        q.submit(vault / "a.md")
        q.submit(vault)
    assert q.pending() == 2
    assert rec.of("queued") == ["a.md"]
    q.start()
    assert q.wait_idle(10)
    q.stop()
    assert rec.states("a.md") == ["queued", "indexing", "ready"]


def test_submit_while_running_runs_again(
    cfg: Config, stores: tuple[ChunkIndex, Registry], vault: Path, rec: Recorder
) -> None:
    embedder = GatedEmbedder()
    q = make_queue(cfg, stores, rec, embedder)
    q.start()
    q.submit(vault / "a.md")
    assert embedder.entered.wait(10)
    (vault / "a.md").write_text(LONG_TEXT, encoding="utf-8")  # changed mid-indexing
    q.submit(vault / "a.md")
    assert q.pending() == 2
    embedder.release.set()
    assert q.wait_idle(10)
    q.stop()
    assert rec.states("a.md") == [
        "queued",
        "indexing",
        "queued",
        "ready",
        "indexing",
        "ready",
    ]


def test_embedder_failure_emits_failed_and_worker_continues(
    cfg: Config, stores: tuple[ChunkIndex, Registry], vault: Path, rec: Recorder
) -> None:
    embedder = FlakyEmbedder()
    embedder.down = True
    q = make_queue(cfg, stores, rec, embedder)
    q.start()
    q.submit(vault / "a.md")
    assert q.wait_idle(10)
    failed = [e for e in rec.events if e.state == "failed"]
    assert [e.path.name for e in failed] == ["a.md"]
    assert failed[0].error is not None and "Cannot reach Ollama" in failed[0].error
    _, registry = stores
    assert statuses(registry) == {"a.md": "failed"}

    embedder.down = False
    q.submit(vault / "sub" / "b.txt")
    q.scan()  # retries the failed file
    assert q.wait_idle(10)
    q.stop()
    assert statuses(registry) == {"a.md": "ready", "b.txt": "ready"}


def test_folder_failure_reports_the_file_being_indexed(
    cfg: Config, stores: tuple[ChunkIndex, Registry], vault: Path, rec: Recorder
) -> None:
    embedder = FlakyEmbedder()
    embedder.down = True
    q = make_queue(cfg, stores, rec, embedder)
    q.start()
    q.submit(vault)
    assert q.wait_idle(10)
    q.stop()
    assert rec.of("failed") == ["a.md"]  # the run stops at the first file; scan retries later


def test_parse_failure_is_reported_with_reason(
    queue: IngestQueue, vault: Path, rec: Recorder
) -> None:
    (vault / "broken.pdf").write_bytes(b"not a pdf")
    queue.submit(vault)
    assert queue.wait_idle(10)
    failed = [e for e in rec.events if e.state == "failed"]
    assert [e.path.name for e in failed] == ["broken.pdf"]
    assert failed[0].error
    assert rec.of("ready") == ["a.md", "b.txt"]


def test_raising_callback_does_not_kill_worker(
    cfg: Config, stores: tuple[ChunkIndex, Registry], vault: Path
) -> None:
    calls: list[JobEvent] = []

    def bad(event: JobEvent) -> None:
        calls.append(event)
        raise ValueError("ui bug")

    index, registry = stores
    q = IngestQueue(cfg, embedder=FakeEmbedder(), index=index, registry=registry, on_event=bad)
    q.start()
    q.submit(vault / "a.md")
    q.submit(vault / "sub" / "b.txt")
    assert q.wait_idle(10)
    q.stop()
    assert statuses(registry) == {"a.md": "ready", "b.txt": "ready"}
    assert len(calls) == 6


def test_stop_is_idempotent_and_drops_waiting_jobs(
    cfg: Config, stores: tuple[ChunkIndex, Registry], vault: Path, rec: Recorder
) -> None:
    embedder = GatedEmbedder()
    q = make_queue(cfg, stores, rec, embedder)
    q.start()
    q.submit(vault / "a.md")
    assert embedder.entered.wait(10)
    q.submit(vault / "sub" / "b.txt")
    assert q.pending() == 2

    stop_while_running(q, embedder)  # the running file finishes
    q.stop()
    q.submit(vault)  # ignored after stop
    assert q.pending() == 0
    assert q.wait_idle(0)
    _, registry = stores
    assert statuses(registry) == {"a.md": "ready"}
    assert rec.states("b.txt") == ["queued"]


def test_stop_ends_a_folder_job_between_files(
    cfg: Config, stores: tuple[ChunkIndex, Registry], vault: Path, rec: Recorder
) -> None:
    embedder = GatedEmbedder()
    q = make_queue(cfg, stores, rec, embedder)
    q.start()
    q.submit(vault)
    assert embedder.entered.wait(10)
    stop_while_running(q, embedder)
    _, registry = stores
    assert statuses(registry) == {"a.md": "ready"}  # b.txt waits for the next scan
    assert rec.states("a.md") == ["indexing", "ready"]


def test_stop_before_start(cfg: Config, stores: tuple[ChunkIndex, Registry], rec: Recorder) -> None:
    q = make_queue(cfg, stores, rec)
    q.stop()
    q.start()  # no worker after stop
    q.stop()
    assert q.pending() == 0
