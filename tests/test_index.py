"""ChunkIndex: LanceDB storage of child chunks, vectors and parent sections."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from vaultmind.core.models import ChildChunk, ParentSection
from vaultmind.store.index import ChunkIndex, EmbedderMismatchError

from .fakes import FakeEmbedder

EMBEDDER = FakeEmbedder()
FILE_A = "aaaaaaaaaaaaaaaa"
FILE_B = "bbbbbbbbbbbbbbbb"


def _chunks(
    file_id: str, n: int = 3, *, page: int | None = 1
) -> tuple[list[ParentSection], list[ChildChunk]]:
    parent = ParentSection(
        id=f"{file_id}:p0",
        file_id=file_id,
        text=f"parent of {file_id}",
        section_path=("Title", "Intro"),
        page_start=page,
        page_end=page,
    )
    children = [
        ChildChunk(
            id=f"{parent.id}:c{i}",
            parent_id=parent.id,
            file_id=file_id,
            text=f"chunk {i} of {file_id}",
            embed_text=f"Title > Intro\n\nchunk {i} of {file_id}",
            page=page,
            section_path=("Title", "Intro"),
        )
        for i in range(n)
    ]
    return [parent], children


def _add(index: ChunkIndex, file_id: str, n: int = 3, *, page: int | None = 1) -> list[ChildChunk]:
    parents, children = _chunks(file_id, n, page=page)
    index.add(parents, children, EMBEDDER.embed([c.embed_text for c in children]))
    return children


def _index(path: Path) -> ChunkIndex:
    return ChunkIndex(path / "index", embedder_name=EMBEDDER.model_name)


def test_search_returns_identical_vector_first(tmp_path: Path) -> None:
    index = _index(tmp_path)
    children = _add(index, FILE_A, 5)
    target = children[3]

    hits = index.vector_search(EMBEDDER.embed([target.embed_text])[0], k=5)

    assert hits[0].chunk == target
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 <= s <= 1.0 for s in scores)
    assert len(hits) == 5


def test_search_respects_k(tmp_path: Path) -> None:
    index = _index(tmp_path)
    _add(index, FILE_A, 5)
    assert len(index.vector_search(EMBEDDER.embed(["q"])[0], k=2)) == 2
    assert index.vector_search(EMBEDDER.embed(["q"])[0], k=0) == []


def test_search_empty_index(tmp_path: Path) -> None:
    index = _index(tmp_path)
    assert index.vector_search(EMBEDDER.embed(["anything"])[0], k=5) == []
    assert index.count() == 0


def test_search_after_deleting_everything(tmp_path: Path) -> None:
    index = _index(tmp_path)
    _add(index, FILE_A)
    index.delete_file(FILE_A)
    assert index.vector_search(EMBEDDER.embed(["anything"])[0], k=5) == []


def test_page_none_and_section_path_round_trip(tmp_path: Path) -> None:
    index = _index(tmp_path)
    parents, children = _chunks(FILE_A, 1, page=None)
    child = replace(children[0], section_path=())
    index.add(parents, [child], EMBEDDER.embed([child.embed_text]))

    hit = index.vector_search(EMBEDDER.embed([child.embed_text])[0], k=1)[0]
    assert hit.chunk == child
    assert hit.chunk.page is None
    assert hit.chunk.section_path == ()

    parent = index.get_parents([parents[0].id])[parents[0].id]
    assert parent == parents[0]
    assert parent.section_path == ("Title", "Intro")
    assert parent.page_start is None


def test_reopen_keeps_data(tmp_path: Path) -> None:
    children = _add(_index(tmp_path), FILE_A)
    reopened = _index(tmp_path)
    assert reopened.count() == len(children)
    hits = reopened.vector_search(EMBEDDER.embed([children[0].embed_text])[0], k=1)
    assert hits[0].chunk == children[0]


def test_different_embedder_raises(tmp_path: Path) -> None:
    _add(_index(tmp_path), FILE_A)
    with pytest.raises(EmbedderMismatchError, match="Re-index"):
        ChunkIndex(tmp_path / "index", embedder_name="other-embedder")


def test_embedder_recorded_even_before_first_add(tmp_path: Path) -> None:
    _index(tmp_path)
    with pytest.raises(EmbedderMismatchError):
        ChunkIndex(tmp_path / "index", embedder_name="other-embedder")


def test_dimension_mismatch_raises(tmp_path: Path) -> None:
    index = _index(tmp_path)
    _add(index, FILE_A)
    parents, children = _chunks(FILE_B, 1)
    wrong = FakeEmbedder(dim=8).embed([children[0].embed_text])
    with pytest.raises(ValueError, match="dimension"):
        index.add(parents, children, wrong)
    with pytest.raises(ValueError, match="dimension"):
        index.vector_search(wrong[0], k=1)


def test_dimension_fixed_across_reopen(tmp_path: Path) -> None:
    _add(_index(tmp_path), FILE_A)
    parents, children = _chunks(FILE_B, 1)
    with pytest.raises(ValueError, match="dimension"):
        _index(tmp_path).add(parents, children, FakeEmbedder(dim=8).embed(["x"]))


def test_length_mismatch_raises(tmp_path: Path) -> None:
    index = _index(tmp_path)
    parents, children = _chunks(FILE_A, 2)
    with pytest.raises(ValueError, match="vectors"):
        index.add(parents, children, EMBEDDER.embed(["only one"]))


def test_readding_same_ids_replaces(tmp_path: Path) -> None:
    index = _index(tmp_path)
    _add(index, FILE_A, 3)
    parents, children = _chunks(FILE_A, 3)
    edited = [replace(c, text="edited") for c in children]
    index.add(parents, edited, EMBEDDER.embed([c.embed_text for c in edited]))

    assert index.count() == 3
    hits = index.vector_search(EMBEDDER.embed([edited[0].embed_text])[0], k=10)
    assert len(hits) == 3
    assert {h.chunk.text for h in hits} == {"edited"}
    assert len(index.get_parents([parents[0].id])) == 1


def test_delete_file_removes_only_that_file(tmp_path: Path) -> None:
    index = _index(tmp_path)
    _add(index, FILE_A, 2)
    b_children = _add(index, FILE_B, 3)

    index.delete_file(FILE_A)

    assert index.count() == 3
    hits = index.vector_search(EMBEDDER.embed(["query"])[0], k=10)
    assert {h.chunk.file_id for h in hits} == {FILE_B}
    assert index.get_parents([f"{FILE_A}:p0"]) == {}
    assert set(index.get_parents([b_children[0].parent_id])) == {b_children[0].parent_id}


def test_delete_file_rejects_unsafe_id(tmp_path: Path) -> None:
    index = _index(tmp_path)
    _add(index, FILE_A)
    with pytest.raises(ValueError):
        index.delete_file("x' OR '1'='1")
    assert index.count() == 3


def test_get_parents_returns_only_found(tmp_path: Path) -> None:
    index = _index(tmp_path)
    _add(index, FILE_A)
    _add(index, FILE_B)

    found = index.get_parents([f"{FILE_A}:p0", f"{FILE_B}:p0", "cccccccccccccccc:p0"])

    assert set(found) == {f"{FILE_A}:p0", f"{FILE_B}:p0"}
    assert found[f"{FILE_A}:p0"].file_id == FILE_A
    assert index.get_parents([]) == {}


def _keyword_chunks(index: ChunkIndex) -> list[ChildChunk]:
    parents, children = _chunks(FILE_A, 3)
    texts = ["governing law of Delaware", "payment due in sixty days", "termination for cause"]
    children = [
        replace(c, text=t, embed_text=f"Supply Agreement > Terms\n\n{t}")
        for c, t in zip(children, texts, strict=True)
    ]
    index.add(parents, children, EMBEDDER.embed([c.embed_text for c in children]))
    return children


def test_fts_search_ranks_keyword_match_first(tmp_path: Path) -> None:
    index = _index(tmp_path)
    children = _keyword_chunks(index)
    hits = index.fts_search("Which laws govern the agreement?", k=3)  # stemmed: laws -> law
    assert hits[0].chunk == children[0]
    assert hits[0].score > 0
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_fts_search_matches_title_prefix(tmp_path: Path) -> None:
    index = _index(tmp_path)
    _keyword_chunks(index)
    assert len(index.fts_search("supply", k=10)) == 3  # every chunk carries the title


def test_fts_search_sees_rows_added_and_deleted_after_refresh(tmp_path: Path) -> None:
    index = _index(tmp_path)
    _add(index, FILE_A, 2)
    index.refresh_fts()
    _add(index, FILE_B, 2)
    assert {h.chunk.file_id for h in index.fts_search(FILE_B, k=10)} == {FILE_B}
    index.delete_file(FILE_A)
    assert index.fts_search(FILE_A, k=10) == []


def test_fts_search_edge_cases(tmp_path: Path) -> None:
    index = _index(tmp_path)
    assert index.fts_search("anything", k=5) == []  # no children table yet
    index.refresh_fts()  # no-op on an empty index
    _add(index, FILE_A, 2)
    assert index.fts_search("   ", k=5) == []
    assert index.fts_search("chunk", k=0) == []
    # Not parsed as query syntax: punctuation is dropped and the words are matched.
    assert len(index.fts_search('"chunk" AND (x) -y: *?', k=5)) == 2


def test_parents_for_file_in_document_order(tmp_path: Path) -> None:
    index = _index(tmp_path)
    _add(index, FILE_B)
    parents = [
        ParentSection(f"{FILE_A}:p{i}", FILE_A, f"section {i}", ("T", f"S{i}"), i + 1, i + 1)
        for i in (10, 2, 0, 1)
    ]
    child = ChildChunk(f"{FILE_A}:p0:c0", f"{FILE_A}:p0", FILE_A, "c", "c", 1, ())
    index.add(parents, [child], EMBEDDER.embed(["c"]))
    assert [p.id.rsplit(":", 1)[1] for p in index.parents_for_file(FILE_A)] == [
        "p0",
        "p1",
        "p2",
        "p10",
    ]
    assert index.parents_for_file("cccccccccccccccc") == []
    with pytest.raises(ValueError):
        index.parents_for_file("x' OR '1'='1")
