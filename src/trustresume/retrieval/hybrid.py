"""Hybrid retrieval: vector (Chroma) + keyword (SQLite FTS5), fused by rank.

Pure vector search misses exact-term matches an embedding treats as merely
"nearby" (a job description naming "Kubernetes" specifically shouldn't lose
to a chunk about "Docker" just because they're semantically close); pure
keyword search misses paraphrases with no shared vocabulary. Combining both
is the standard fix in production RAG systems.

Fusion is by **Reciprocal Rank Fusion (RRF)**, not by combining raw scores —
Chroma's cosine similarity and SQLite's BM25 rank live on incomparable
scales (a similarity in [0, 1] vs. an unbounded, sign-flipped log-odds score),
so averaging or weighting them directly would be combining apples and
oranges. RRF sidesteps that by fusing on rank position alone: each result's
score is ``1 / (k + rank)`` summed across whichever list(s) it appears in, so
a chunk found by both searches — even at different ranks — naturally outranks
one found by only one. ``k`` (default 60, the value from the original RRF
paper) dampens the influence of a rank-1 hit relative to lower ranks.

Milestone M2 (storage + retrieval); hybrid retrieval added post-M2.
"""

from __future__ import annotations

import sqlite3

from trustresume.models import DocumentType, EvidenceChunk, EvidenceSet
from trustresume.storage import ChunkRepository

from .vector_store import ChromaVectorStore

DEFAULT_RRF_K = 60


class HybridRetriever:
    """Combines :class:`ChromaVectorStore` and :class:`ChunkRepository`'s
    keyword search into one ranked :class:`EvidenceSet`, via RRF.

    Matches ``ChromaVectorStore``'s ``search(user_id, query, limit)`` shape so
    it's a drop-in for callers (``EvidenceRetrievalAgent``,
    ``TrustResumeApp.search_evidence``) that only need ranked evidence back,
    regardless of which retrieval strategy produced it.
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        chunk_repository: ChunkRepository,
        *,
        rrf_k: int = DEFAULT_RRF_K,
        candidates_per_source: int = 20,
    ) -> None:
        self._vectors = vector_store
        self._chunks = chunk_repository
        self._rrf_k = rrf_k
        # Each source is asked for more than the final `limit` so fusion has
        # real overlap to work with — if both searches were capped at the
        # final limit, a chunk ranked #6 in each (but never top-5) would be
        # invisible to both, even though summing its two rank-6 RRF scores
        # might beat a chunk that only one source ranked #1.
        self._candidates_per_source = candidates_per_source

    def search(self, *, user_id: str, query: str, limit: int = 5) -> EvidenceSet:
        """Hybrid search scoped to ``user_id``: vector + keyword, fused by RRF."""
        vector_hits = self._vectors.search(
            user_id=user_id, query=query, limit=self._candidates_per_source
        ).chunks
        keyword_rows = self._chunks.search_keywords(
            user_id=user_id, query=query, limit=self._candidates_per_source
        )
        keyword_chunks = [_row_to_chunk(row, user_id=user_id) for row in keyword_rows]

        fused = _reciprocal_rank_fusion([vector_hits, keyword_chunks], k=self._rrf_k)
        return EvidenceSet(user_id=user_id, query=query, chunks=fused[:limit])


def _row_to_chunk(row: sqlite3.Row, *, user_id: str) -> EvidenceChunk:
    """A ``chunks`` table row (with a BM25 ``rank`` column) -> ``EvidenceChunk``.

    ``score`` is left ``None`` — BM25's raw rank isn't on the same scale as
    Chroma's cosine similarity, and RRF (below) only ever needs each source's
    *rank position*, never the raw score, so there's nothing meaningful to
    put there.
    """
    return EvidenceChunk(
        chunk_id=row["chunk_id"],
        user_id=user_id,
        document_id=row["document_id"],
        document_type=DocumentType(row["document_type"]),
        source_document=row["source_document"],
        text=row["text"],
    )


def _reciprocal_rank_fusion(
    ranked_lists: list[list[EvidenceChunk]], *, k: int
) -> list[EvidenceChunk]:
    """Fuse multiple rank-ordered chunk lists into one, by RRF score (descending).

    A chunk appearing in more than one list keeps the first-seen chunk object
    (so its richer Chroma metadata — e.g. a similarity ``score`` — wins over
    the keyword-search version when both found it) but accumulates RRF score
    from every list it appears in.
    """
    rrf_scores: dict[str, float] = {}
    chunk_by_id: dict[str, EvidenceChunk] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            chunk_by_id.setdefault(chunk.chunk_id, chunk)

    ordered_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
    return [chunk_by_id[cid] for cid in ordered_ids]
