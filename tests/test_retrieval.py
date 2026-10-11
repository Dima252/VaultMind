"""Retrieval stages: vector, hybrid (RRF), rerank + threshold."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from vaultmind.agent.baseline import retrieve
from vaultmind.agent.retrieval import RRF_K, rerank, rrf_fuse, search
from vaultmind.core.config import Config, RetrievalConfig
from vaultmind.core.models import ChildChunk, FileRecord, ParentSection, SearchHit
from vaultmind.store.index import ChunkIndex
from vaultmind.store.registry import Registry

from .fakes import FakeEmbedder, FakeReranker

FILE_ID = "0123456789abcdef"
PARENT = ParentSection(f"{FILE_ID}:p0", FILE_ID, "Whole contract section.", ("Terms",), 1, 1)
TEXTS = [
    "This Agreement is governed by the laws of Delaware.",
    "Payment is due within sixty days of invoice.",
    "Either party may terminate for convenience on notice.",
    "Notices must be sent by registered mail.",
]
CHILDREN = [
    ChildChunk(
        f"{PARENT.id}:c{i}", PARENT.id, FILE_ID, t, f"Contract > Terms\n\n{t}", 1, ("Terms",)
    )
    for i, t in enumerate(TEXTS)
]


def _cfg(**retrieval: object) -> Config:
    return Config(retrieval=RetrievalConfig.model_validate(retrieval))


def _hit(chunk_id: str, score: float = 0.0) -> SearchHit:
    chunk = ChildChunk(chunk_id, "p", FILE_ID, chunk_id, chunk_id, None, ())
    return SearchHit(chunk=chunk, score=score)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[tuple[ChunkIndex, Registry]]:
    embedder = FakeEmbedder()
    index = ChunkIndex(tmp_path / "index", embedder_name=embedder.model_name)
    registry = Registry(tmp_path / "r.sqlite")
    index.add([PARENT], CHILDREN, embedder.embed([c.embed_text for c in CHILDREN]))
    registry.upsert(FileRecord(FILE_ID, tmp_path / "contract.txt", "sha", "ready"))
    yield index, registry
    registry.close()
    index.close()


def test_rrf_fuse_sums_reciprocal_ranks() -> None:
    fused = rrf_fuse([[_hit("a"), _hit("b")], [_hit("b"), _hit("c")]])
    assert [h.chunk.id for h in fused] == ["b", "a", "c"]
    assert fused[0].score == pytest.approx(1 / (RRF_K + 2) + 1 / (RRF_K + 1))
    assert fused[1].score == pytest.approx(1 / (RRF_K + 1))
    assert rrf_fuse([]) == []


def test_rerank_rescores_and_sorts() -> None:
    hits = [_hit("apple pie"), _hit("law of delaware"), _hit("delaware")]
    reranked = rerank("law of delaware", hits, FakeReranker())
    assert [h.chunk.id for h in reranked] == ["law of delaware", "delaware", "apple pie"]
    assert reranked[0].score == pytest.approx(1.0)


def test_vector_only_is_plain_top_k(store: tuple[ChunkIndex, Registry]) -> None:
    index, _ = store
    cfg = _cfg(hybrid=False, rerank=False, top_k=2)
    hits = search(CHILDREN[2].embed_text, cfg, embedder=FakeEmbedder(), index=index)
    assert len(hits) == 2
    assert hits[0].chunk == CHILDREN[2]
    assert hits[0].score == pytest.approx(1.0)  # cosine similarity


def test_hybrid_finds_keyword_match_vectors_miss(store: tuple[ChunkIndex, Registry]) -> None:
    index, _ = store
    # The fake embedder is a hash, so vector search ranks a paraphrase arbitrarily;
    # BM25 still puts the keyword match on top after fusion (it gets a vector rank too).
    query = "which laws govern"
    cfg = _cfg(hybrid=True, rerank=False, top_k=4, candidates=4)
    hits = search(query, cfg, embedder=FakeEmbedder(), index=index)
    assert CHILDREN[0] in [h.chunk for h in hits]
    vector_only = search(
        query, _cfg(hybrid=False, rerank=False, top_k=4), embedder=FakeEmbedder(), index=index
    )
    fts_rank = [h.chunk for h in index.fts_search(query, k=4)].index(CHILDREN[0])
    assert fts_rank == 0
    assert len(hits) == len(vector_only) == 4


def test_rerank_threshold_drops_weak_hits(store: tuple[ChunkIndex, Registry]) -> None:
    index, _ = store
    cfg = _cfg(hybrid=True, rerank=True, rerank_threshold=0.5, top_k=5)
    query = "terminate for convenience"
    hits = search(query, cfg, embedder=FakeEmbedder(), index=index, reranker=FakeReranker())
    assert [h.chunk for h in hits] == [CHILDREN[2]]
    assert hits[0].score == pytest.approx(1.0)


def test_rerank_threshold_can_reject_everything(store: tuple[ChunkIndex, Registry]) -> None:
    index, _ = store
    cfg = _cfg(rerank=True, rerank_threshold=0.5)
    hits = search(
        "capital of peru", cfg, embedder=FakeEmbedder(), index=index, reranker=FakeReranker()
    )
    assert hits == []


def test_rerank_without_reranker_raises(store: tuple[ChunkIndex, Registry]) -> None:
    index, _ = store
    with pytest.raises(ValueError, match="no reranker"):
        search("q", _cfg(rerank=True), embedder=FakeEmbedder(), index=index)


def test_small_to_big_off_sends_child_text(store: tuple[ChunkIndex, Registry]) -> None:
    index, registry = store
    question = CHILDREN[1].embed_text
    kw = {"embedder": FakeEmbedder(), "index": index, "registry": registry}
    big = retrieve(question, _cfg(hybrid=False, rerank=False, small_to_big=True), **kw)  # type: ignore[arg-type]
    small = retrieve(
        question,
        _cfg(hybrid=False, rerank=False, small_to_big=False),
        **kw,  # type: ignore[arg-type]
    )
    assert [s.text for s in big] == [PARENT.text]  # four children collapse into one parent
    assert [s.n for s in small] == [1, 2, 3, 4]
    assert small[0].text == CHILDREN[1].text
    assert small[0].label == "contract.txt, p. 1, Terms"
