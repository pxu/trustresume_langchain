"""Retrieval layer: Chroma (vector) + SQLite FTS5 (keyword), fused by RRF.

``embedder`` wraps the local embedding model behind LangChain's ``Embeddings``
interface; ``vector_store`` owns upsert/search/delete against Chroma, always
scoped by ``user_id``; ``hybrid`` fuses vector search with the SQLite keyword
search in ``storage.ChunkRepository.search_keywords`` (ADR-0001's hybrid
store, extended post-M2 to also search the SQLite half, not just mirror it);
``query`` builds the search-query string from a ``JobDescription``, shared by
``EvidenceRetrievalAgent`` and offline evaluation tooling.

Milestone M2 (storage + retrieval); hybrid retrieval added post-M2.
"""

from __future__ import annotations

from .embedder import DEFAULT_MODEL, FastEmbedEmbeddings
from .hybrid import DEFAULT_RRF_K, HybridRetriever
from .query import build_query, per_skill_queries, query_terms
from .vector_store import COLLECTION_NAME, ChromaVectorStore

__all__ = [
    "FastEmbedEmbeddings",
    "DEFAULT_MODEL",
    "ChromaVectorStore",
    "COLLECTION_NAME",
    "HybridRetriever",
    "DEFAULT_RRF_K",
    "build_query",
    "query_terms",
    "per_skill_queries",
]
