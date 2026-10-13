"""VaultWatcher: real file-system events into a recording fake queue."""

from __future__ import annotations

import shutil
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import cast

import pytest
from watchdog.events import DirCreatedEvent, FileCreatedEvent, FileMovedEvent

from vaultmind.core.config import Config, PathsConfig
from vaultmind.ingest.queue import IngestQueue
from vaultmind.ingest.watcher import VaultWatcher
from vaultmind.store.index import ChunkIndex
from vaultmind.store.registry import Registry

from .fakes import FakeEmbedder

DEBOUNCE = 0.2


class FakeQueue:
    """Records what the watcher asks for."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls: list[tuple[str, Path]] = []

    def submit(self, path: Path) -> None:
        with self._lock:
            self.calls.append(("submit", path))

    def forget(self, path: Path) -> None:
        with self._lock:
            self.calls.append(("forget", path))

    def names(self, kind: str) -> list[str]:
        with self._lock:
            return [p.name for k, p in self.calls if k == kind]

    def clear(self) -> None:
        with self._lock:
            self.calls.clear()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    path = tmp_path / "vault"
    path.mkdir()
    return path.resolve()


@pytest.fixture
def fake() -> FakeQueue:
    return FakeQueue()


@pytest.fixture
def watcher(root: Path, fake: FakeQueue) -> Iterator[VaultWatcher]:
    w = VaultWatcher(root, cast(IngestQueue, fake), debounce_s=DEBOUNCE)
    w.start()
    time.sleep(0.1)  # let the observer attach before the test touches files
    yield w
    w.stop()


def eventually(check: Callable[[], bool], timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(0.02)
    return check()


def settle() -> None:
    """Long enough for any stray event to pass its debounce."""
    time.sleep(DEBOUNCE * 3)


def test_create_submits_once_after_the_burst(
    watcher: VaultWatcher, root: Path, fake: FakeQueue
) -> None:
    path = root / "a.md"
    with path.open("w", encoding="utf-8") as f:
        for i in range(5):
            f.write(f"line {i}\n")
            f.flush()
            time.sleep(0.03)
    assert eventually(lambda: fake.names("submit") == ["a.md"])
    settle()
    assert fake.names("submit") == ["a.md"]


def test_rapid_modifications_collapse(watcher: VaultWatcher, root: Path, fake: FakeQueue) -> None:
    path = root / "notes.txt"
    path.write_text("v0", encoding="utf-8")
    assert eventually(lambda: fake.names("submit") == ["notes.txt"])
    fake.clear()
    for i in range(5):
        path.write_text(f"version {i} " * (i + 1), encoding="utf-8")
        time.sleep(0.03)
    assert eventually(lambda: fake.names("submit") == ["notes.txt"])
    settle()
    assert fake.names("submit") == ["notes.txt"]


def test_unchanged_file_events_do_not_resubmit(
    watcher: VaultWatcher, root: Path, fake: FakeQueue
) -> None:
    path = root / "a.md"
    path.write_text("hello", encoding="utf-8")
    assert eventually(lambda: fake.names("submit") == ["a.md"])
    path.read_text(encoding="utf-8")  # reads and attribute changes count as "modified" on Windows
    path.chmod(0o444)
    path.chmod(0o666)
    settle()
    assert fake.names("submit") == ["a.md"]


def test_existing_files_read_after_start_are_not_resubmitted(root: Path, fake: FakeQueue) -> None:
    path = root / "old.md"
    path.write_text("already indexed by the startup scan", encoding="utf-8")
    w = VaultWatcher(root, cast(IngestQueue, fake), debounce_s=DEBOUNCE)
    w.start()
    try:
        time.sleep(0.1)
        path.read_text(encoding="utf-8")  # the scan hashing it: a "modified" event on Windows
        path.chmod(0o444)
        path.chmod(0o666)
        settle()
        assert fake.names("submit") == []
        path.write_text("edited", encoding="utf-8")
        assert eventually(lambda: fake.names("submit") == ["old.md"])
    finally:
        w.stop()


def test_delete_forgets(watcher: VaultWatcher, root: Path, fake: FakeQueue) -> None:
    path = root / "a.md"
    path.write_text("hello", encoding="utf-8")
    assert eventually(lambda: fake.names("submit") == ["a.md"])
    path.unlink()
    assert eventually(lambda: "a.md" in fake.names("forget"))


def test_delete_cancels_a_pending_submit(root: Path, fake: FakeQueue, tmp_path: Path) -> None:
    w = VaultWatcher(root, cast(IngestQueue, fake), debounce_s=0.6)
    w.start()
    time.sleep(0.1)
    try:
        path = root / "a.md"
        path.write_text("hello", encoding="utf-8")
        time.sleep(0.2)
        path.unlink()
        assert eventually(lambda: "a.md" in fake.names("forget"))
        time.sleep(0.7)
        assert fake.names("submit") == []
    finally:
        w.stop()


def test_rename_forgets_source_and_submits_destination(
    watcher: VaultWatcher, root: Path, fake: FakeQueue
) -> None:
    src = root / "draft.md"
    src.write_text("hello", encoding="utf-8")
    assert eventually(lambda: fake.names("submit") == ["draft.md"])
    fake.clear()
    src.rename(root / "final.md")
    assert eventually(lambda: "draft.md" in fake.names("forget"))
    assert eventually(lambda: fake.names("submit") == ["final.md"])


def test_ignored_files(watcher: VaultWatcher, root: Path, fake: FakeQueue) -> None:
    (root / ".hidden").mkdir()
    (root / ".hidden" / "secret.md").write_text("x", encoding="utf-8")
    (root / ".dotfile.md").write_text("x", encoding="utf-8")
    (root / "~$report.md").write_text("x", encoding="utf-8")
    (root / "download.pdf.crdownload").write_text("x", encoding="utf-8")
    (root / "big.part").write_text("x", encoding="utf-8")
    (root / "edit.tmp").write_text("x", encoding="utf-8")
    (root / ".a.md.swp").write_text("x", encoding="utf-8")
    (root / "notes.docx").write_text("x", encoding="utf-8")
    (root / "image.PNG").write_text("x", encoding="utf-8")
    (root / "UPPER.PDF").write_text("x", encoding="utf-8")  # supported, any case
    assert eventually(lambda: fake.names("submit") == ["UPPER.PDF"])
    settle()
    assert fake.names("submit") == ["UPPER.PDF"]


def test_folder_delete_forgets_it(watcher: VaultWatcher, root: Path, fake: FakeQueue) -> None:
    folder = root / "project"
    folder.mkdir()
    (folder / "a.md").write_text("a", encoding="utf-8")
    (folder / "b.txt").write_text("b", encoding="utf-8")
    assert eventually(lambda: {"a.md", "b.txt"} <= set(fake.names("submit")))
    shutil.rmtree(folder)
    # The queue expands a folder forget to the registered files under it; per-file
    # events may or may not arrive depending on the platform.
    assert eventually(lambda: "project" in fake.names("forget"))


def test_folder_moved_in_is_submitted(
    watcher: VaultWatcher, root: Path, fake: FakeQueue, tmp_path: Path
) -> None:
    outside = tmp_path / "incoming"
    outside.mkdir()
    (outside / "c.md").write_text("c", encoding="utf-8")
    shutil.move(str(outside), str(root / "incoming"))
    assert eventually(lambda: "incoming" in fake.names("submit") or "c.md" in fake.names("submit"))


def test_stop_is_idempotent_and_cancels_timers(root: Path, fake: FakeQueue) -> None:
    w = VaultWatcher(root, cast(IngestQueue, fake), debounce_s=0.3)
    w.start()
    time.sleep(0.1)
    (root / "a.md").write_text("hello", encoding="utf-8")
    time.sleep(0.1)
    w.stop()
    w.stop()
    time.sleep(0.4)
    assert fake.names("submit") == []
    w.start()  # no restart after stop
    (root / "b.md").write_text("hello", encoding="utf-8")
    time.sleep(0.4)
    assert fake.names("submit") == []


def test_with_real_queue(root: Path, tmp_path: Path) -> None:
    """Drop a file in, it gets indexed; delete it, it's gone from the registry."""
    cfg = Config(paths=PathsConfig(vault_dir=root, data_dir=tmp_path / "data"))
    index = ChunkIndex(cfg.paths.data_dir / "index", embedder_name=FakeEmbedder.model_name)
    registry = Registry(cfg.paths.data_dir / "vaultmind.sqlite")
    queue = IngestQueue(cfg, embedder=FakeEmbedder(), index=index, registry=registry)
    queue.start()
    w = VaultWatcher(root, queue, debounce_s=DEBOUNCE)
    w.start()
    time.sleep(0.1)
    try:
        (root / "a.md").write_text("# Alpha\n\nAlpha is first.\n", encoding="utf-8")
        assert eventually(lambda: [r.status for r in registry.all()] == ["ready"])
        (root / "a.md").unlink()
        assert eventually(lambda: registry.all() == [] and queue.pending() == 0)
    finally:
        w.stop()
        queue.stop()
        registry.close()
        index.close()


def test_replayed_creation_of_root_and_known_files_is_ignored(root: Path, fake: FakeQueue) -> None:
    """macOS FSEvents replays creations from just before the watch started."""
    path = root / "old.md"
    path.write_text("already indexed by the startup scan", encoding="utf-8")
    w = VaultWatcher(root, cast(IngestQueue, fake), debounce_s=DEBOUNCE)
    w._remember_existing()
    w.handle(DirCreatedEvent(str(root)))
    w.handle(FileCreatedEvent(str(path)))
    settle()
    assert fake.names("submit") == []
    w.handle(FileMovedEvent(str(root / "draft.md"), str(path)))  # moved over it: always new
    assert eventually(lambda: fake.names("submit") == ["old.md"])
    w.stop()
