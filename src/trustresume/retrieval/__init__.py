"""Semantic retrieval layer (Chroma): embedded chunk storage + search.

The Chroma half of the hybrid store (ADR-0001). ``embedder`` wraps the local
embedding model behind LangChain's ``Embeddings`` interface; ``vector_store``
owns upsert/search/delete against Chroma, always scoped by ``user_id``;
``query`` builds the search-query string from a ``JobDescription``, shared by
``EvidenceRetrievalAgent`` and offline evaluation tooling.

Milestone M2 (storage + retrieval).
"""

from __future__ import annotations

from .embedder import DEFAULT_MODEL, FastEmbedEmbeddings
from .query import build_query, per_skill_queries, query_terms
from .vector_store import COLLECTION_NAME, ChromaVectorStore

__all__ = [
    "FastEmbedEmbeddings",
    "DEFAULT_MODEL",
    "ChromaVectorStore",
    "COLLECTION_NAME",
    "build_query",
    "query_terms",
    "per_skill_queries",
]
