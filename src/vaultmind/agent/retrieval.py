"""Child-chunk retrieval: vector or hybrid (vector + BM25, fused with RRF), then reranking.

Each stage is a `retrieval.*` config switch so the eval can measure what it contributes.
"""

from __future__ import annotations

from collections.abc import Sequence

from vaultmind.core.config import Config
from vaultmind.core.models import SearchHit
from vaultmind.core.providers import Embedder, Reranker
from vaultmind.store.index import ChunkIndex

RRF_K = 60  # the usual constant from Cormack et al. (2009); damps the weight of top ranks


def rrf_fuse(rankings: Sequence[Sequence[SearchHit]], k: int = RRF_K) -> list[SearchHit]:
    """Reciprocal rank fusion: score = sum of 1 / (k + rank) over the rankings a chunk is in.

    Returns each chunk once, best first, with its fused score. Ties keep first-seen order.
    """
    scores: dict[str, float] = {}
    chunks: dict[str, SearchHit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            chunk_id = hit.chunk.id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            chunks.setdefault(chunk_id, hit)
    order = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [SearchHit(chunk=chunks[cid].chunk, score=scores[cid]) for cid in order]


def rerank(query: str, hits: Sequence[SearchHit], reranker: Reranker) -> list[SearchHit]:
    """Re-score hits with the cross-encoder (on title + section + text), best first."""
    scores = reranker.score(query, [h.chunk.embed_text for h in hits])
    rescored = [SearchHit(chunk=h.chunk, score=s) for h, s in zip(hits, scores, strict=True)]
    return sorted(rescored, key=lambda h: h.score, reverse=True)


def search(
    query: str,
    cfg: Config,
    *,
    embedder: Embedder,
    index: ChunkIndex,
    reranker: Reranker | None = None,
) -> list[SearchHit]:
    """Up to `top_k` child hits, best first.

    With reranking on, `candidates` hits are reranked and those scoring below
    `rerank_threshold` are dropped, so the result can be empty. Without reranking or hybrid
    search, this is plain top-k vector search.
    """
    r = cfg.retrieval
    if r.rerank and reranker is None:
        raise ValueError("retrieval.rerank is on but no reranker was given")
    n = r.candidates if (r.hybrid or r.rerank) else r.top_k

    [vector] = embedder.embed([query])
    hits = index.vector_search(vector, k=n)
    if r.hybrid:
        hits = rrf_fuse([hits, index.fts_search(query, k=n)])[:n]
    if r.rerank and reranker is not None:
        hits = [h for h in rerank(query, hits, reranker) if h.score >= r.rerank_threshold]
    return hits[: r.top_k]
