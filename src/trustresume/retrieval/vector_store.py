"""Chroma-backed vector store for candidate evidence chunks.

The vector half of the hybrid store (ADR-0001): a chunk lives here as an
embedded, searchable document, and as a metadata row in ``storage/`` (SQLite),
joined by the shared ``chunk_id``. Every write and read is scoped by
``user_id`` — ``search``'s server-side metadata filter is the isolation
boundary (ADR-0001).

Unlike Qdrant, Chroma accepts arbitrary string ids directly, so ``chunk_id``
is used as the Chroma document id itself — no uuid5 indirection needed.

Milestone M2 (storage + retrieval).
"""

from __future__ import annotations

from typing import Any

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from trustresume.models import DocumentType, EvidenceChunk, EvidenceSet

COLLECTION_NAME = "trustresume_chunks"


class ChromaVectorStore:
    """Thin wrapper around ``langchain_chroma.Chroma`` matching the shape the
    rest of the codebase depends on: ``upsert_chunks``, ``search``,
    ``delete_chunks``.
    """

    def __init__(
        self,
        client: Any,  # chromadb.ClientAPI — chromadb ships no type stubs
        embedder: Embeddings,
        *,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self._store = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embedder,
            # Cosine distance so the score conversion below (1 - distance) is
            # a similarity, matching the original Qdrant COSINE-distance store.
            collection_metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: list[EvidenceChunk]) -> None:
        """Embed and upsert chunks, keyed by their own ``chunk_id``.

        ``add_texts`` upserts in place when given an id that already exists
        (verified empirically), so re-ingesting the same chunk overwrites
        rather than duplicates — the same idempotency the original's uuid5
        point-id scheme provided for Qdrant.
        """
        if not chunks:
            return
        self._store.add_texts(
            texts=[chunk.text for chunk in chunks],
            ids=[chunk.chunk_id for chunk in chunks],
            metadatas=[
                {
                    "chunk_id": chunk.chunk_id,
                    "user_id": chunk.user_id,
                    "document_id": chunk.document_id,
                    "document_type": chunk.document_type.value,
                    "source_document": chunk.source_document or "",
                }
                for chunk in chunks
            ],
        )

    def search(
        self, *, user_id: str, query: str, limit: int = 5, document_ids: list[str] | None = None
    ) -> EvidenceSet:
        """Semantic search scoped to ``user_id``.

        The ``filter`` kwarg is a server-side metadata filter — the sole
        mechanism enforcing per-user isolation (ADR-0001). Every call passes
        it; there is no filter-less search path.

        ``document_ids``, when given, additionally restricts results to that
        set (job-scoped retrieval — see
        ``DocumentRepository.list_eligible_document_ids``). Chroma's filter
        grammar rejects a flat multi-key dict (verified empirically: raises
        ``ValueError``, "Expected where to have exactly one operator") — two
        conditions must be combined via an explicit ``$and``. An empty
        ``document_ids`` list is short-circuited to an empty result without
        calling Chroma at all, since an empty ``$in`` list is itself invalid
        there (also verified empirically) rather than trivially matching
        nothing.
        """
        if document_ids is not None:
            if not document_ids:
                return EvidenceSet(user_id=user_id, query=query, chunks=[])
            where: dict[str, object] = {
                "$and": [{"user_id": user_id}, {"document_id": {"$in": document_ids}}]
            }
        else:
            where = {"user_id": user_id}
        # langchain-chroma's own stub types `filter` as dict[str, str] | None,
        # narrower than the nested $and/$in dicts Chroma's filter grammar
        # actually accepts and requires here (verified empirically).
        hits = self._store.similarity_search_with_score(
            query,
            k=limit,
            filter=where,  # type: ignore[arg-type]
        )

        chunks = [
            EvidenceChunk(
                chunk_id=str(doc.metadata.get("chunk_id", doc.id)),
                user_id=user_id,
                document_id=str(doc.metadata.get("document_id", "")),
                document_type=DocumentType(
                    doc.metadata.get("document_type", DocumentType.OTHER.value)
                ),
                source_document=(doc.metadata.get("source_document") or None),
                text=doc.page_content,
                # Chroma's cosine distance varies inversely with similarity;
                # convert so a higher EvidenceChunk.score still means "more
                # relevant", matching the original Qdrant COSINE similarity.
                score=1.0 - distance,
            )
            for doc, distance in hits
        ]
        return EvidenceSet(user_id=user_id, query=query, chunks=chunks)

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        self._store.delete(ids=chunk_ids)
