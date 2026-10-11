from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from vaultmind.core.config import Config
from vaultmind.core.models import file_id_for
from vaultmind.ingest.pipeline import IngestSummary, ingest_path, iter_files
from vaultmind.store.index import ChunkIndex
from vaultmind.store.registry import Registry

from .fakes import FakeEmbedder

CFG = Config()


@pytest.fixture
def store(tmp_path: Path) -> Iterator[tuple[ChunkIndex, Registry]]:
    index = ChunkIndex(tmp_path / "data" / "index", embedder_name=FakeEmbedder.model_name)
    registry = Registry(tmp_path / "data" / "vaultmind.sqlite")
    yield index, registry
    registry.close()
    index.close()


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    root = tmp_path / "vault"
    (root / "sub").mkdir(parents=True)
    (root / "a.md").write_text("# Alpha\n\nAlpha is the first letter.\n", encoding="utf-8")
    (root / "sub" / "b.txt").write_text("Bravo is the second.\n", encoding="utf-8")
    (root / "notes.docx").write_bytes(b"not supported")
    return root


def run(vault: Path, store: tuple[ChunkIndex, Registry]) -> IngestSummary:
    index, registry = store
    return ingest_path(vault, CFG, embedder=FakeEmbedder(), index=index, registry=registry)


def names(paths: Sequence[Path]) -> list[str]:
    return sorted(p.name for p in paths)


def test_iter_files_filters_extensions(vault: Path) -> None:
    assert names(iter_files(vault)) == ["a.md", "b.txt"]
    assert iter_files(vault / "notes.docx") == []


def test_first_run_indexes_everything(vault: Path, store: tuple[ChunkIndex, Registry]) -> None:
    summary = run(vault, store)
    index, registry = store
    assert names(summary.added) == ["a.md", "b.txt"]
    assert index.count() == 2
    records = registry.all()
    assert {r.status for r in records} == {"ready"}
    assert {r.embedder for r in records} == {FakeEmbedder.model_name}


def test_unchanged_files_are_skipped(vault: Path, store: tuple[ChunkIndex, Registry]) -> None:
    run(vault, store)
    summary = run(vault, store)
    assert names(summary.skipped) == ["a.md", "b.txt"]
    assert not summary.added and not summary.updated


def test_modified_file_is_reindexed(vault: Path, store: tuple[ChunkIndex, Registry]) -> None:
    run(vault, store)
    long_text = "# Alpha\n\n" + "\n\n".join(f"Paragraph {i} " + "word " * 200 for i in range(6))
    (vault / "a.md").write_text(long_text, encoding="utf-8")
    summary = run(vault, store)
    index, _ = store
    assert names(summary.updated) == ["a.md"]
    assert names(summary.skipped) == ["b.txt"]
    assert index.count() > 2

    (vault / "a.md").write_text("# Alpha\n\nShort again.\n", encoding="utf-8")
    run(vault, store)
    assert index.count() == 2  # old chunks of the longer version are gone


def test_deleted_file_is_removed(vault: Path, store: tuple[ChunkIndex, Registry]) -> None:
    run(vault, store)
    (vault / "sub" / "b.txt").unlink()
    summary = run(vault, store)
    index, registry = store
    assert names(summary.removed) == ["b.txt"]
    assert index.count() == 1
    assert [r.path.name for r in registry.all()] == ["a.md"]


def test_ingesting_one_file_does_not_remove_others(
    vault: Path, store: tuple[ChunkIndex, Registry]
) -> None:
    run(vault, store)
    index, registry = store
    summary = ingest_path(
        vault / "a.md", CFG, embedder=FakeEmbedder(), index=index, registry=registry
    )
    assert names(summary.skipped) == ["a.md"]
    assert not summary.removed
    assert len(registry.all()) == 2


def test_parse_failure_is_recorded_and_retried(
    vault: Path, store: tuple[ChunkIndex, Registry]
) -> None:
    bad = vault / "bad.txt"
    bad.write_bytes(b"\xff\xfe broken")
    summary = run(vault, store)
    _, registry = store
    assert [(p.name, "UTF-8" in reason) for p, reason in summary.failed] == [("bad.txt", True)]
    record = registry.get(file_id_for(bad))
    assert record is not None and record.status == "failed" and record.error

    retry = run(vault, store)  # failed files are not skipped: they are tried again
    assert [p.name for p, _ in retry.failed] == ["bad.txt"]
    assert names(retry.skipped) == ["a.md", "b.txt"]


def test_embedder_error_marks_file_failed_and_stops(
    vault: Path, store: tuple[ChunkIndex, Registry]
) -> None:
    class BrokenEmbedder(FakeEmbedder):
        def embed(self, texts: Sequence[str]) -> list[list[float]]:
            raise RuntimeError("ollama down")

    index, registry = store
    with pytest.raises(RuntimeError, match="ollama down"):
        ingest_path(vault, CFG, embedder=BrokenEmbedder(), index=index, registry=registry)
    statuses = {r.path.name: r.status for r in registry.all()}
    assert statuses == {"a.md": "failed"}
