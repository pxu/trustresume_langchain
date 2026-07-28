"""IngestionService — the pipeline that lands a document in both stores.

Ties the pure steps (parse → clean → chunk) to the two stores from ADR-0001,
and is the one place responsible for keeping them in sync: for every chunk it
writes the SQLite metadata row **and** the Chroma vector, and on document
delete it removes both. Everything is scoped by ``user_id``.

Independent write path, not a pipeline step: it runs on its own trigger
(document upload/delete), not on a ``generate()`` call — see "Ingestion is a
separate write path" in ``architecture/high-level-design.md``. Because of
that, it's also the thing that invalidates the cached Candidate Profile: after
any mutation for a user, it flags that user's profile stale so
``CandidateProfileService`` recomputes it next time it's needed, instead of
serving stale skills/certifications.

The service takes its collaborators (repositories, vector store) by injection
rather than constructing them, so the orchestrator/UI wire one connection +
one Chroma client through the whole app, and tests pass in-memory versions.

Milestone M3 (ingestion).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable

from trustresume.models import DocumentType, EvidenceChunk
from trustresume.retrieval import ChromaVectorStore
from trustresume.storage import CandidateProfileRepository, ChunkRepository, DocumentRepository

from .chunker import chunk_text, clean_text
from .parser import parse_document


class IngestionService:
    """Ingest documents for a user into SQLite + Chroma, kept in sync."""

    def __init__(
        self,
        *,
        documents: DocumentRepository,
        chunks: ChunkRepository,
        vector_store: ChromaVectorStore,
        candidate_profiles: CandidateProfileRepository,
        id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        self._documents = documents
        self._chunks = chunks
        self._vectors = vector_store
        self._candidate_profiles = candidate_profiles
        self._new_id = id_factory

    def ingest_text(
        self,
        *,
        user_id: str,
        filename: str,
        text: str,
        document_type: DocumentType = DocumentType.OTHER,
    ) -> str:
        """Ingest already-extracted text; return the new document id.

        Order matters: the SQLite metadata rows are written first, then the
        Chroma vectors. If embedding/upsert fails, the SQLite rows are rolled
        back so the two stores don't drift out of sync.
        """
        cleaned = clean_text(text)
        pieces = chunk_text(cleaned)

        document_id = self._documents.create(
            user_id=user_id,
            filename=filename,
            document_type=document_type,
        )

        evidence_chunks = [
            EvidenceChunk(
                chunk_id=self._new_id(),
                user_id=user_id,
                document_id=document_id,
                document_type=document_type,
                source_document=filename,
                text=piece,
            )
            for piece in pieces
        ]

        for index, chunk in enumerate(evidence_chunks):
            self._chunks.add(chunk, chunk_index=index)

        try:
            self._vectors.upsert_chunks(evidence_chunks)
        except Exception:
            # Keep the stores consistent: undo the SQLite chunk rows we just
            # wrote if the vector upsert failed. The document row is left as an
            # empty document, which ``list_for_user`` still reports honestly.
            self._chunks.delete_for_document(user_id=user_id, document_id=document_id)
            raise

        self._candidate_profiles.mark_stale(user_id)
        return document_id

    def ingest_file(
        self,
        *,
        user_id: str,
        path: str,
        document_type: DocumentType = DocumentType.OTHER,
        filename: str | None = None,
    ) -> str:
        """Parse a file from disk and ingest it; return the new document id."""
        from pathlib import Path

        text = parse_document(path)
        return self.ingest_text(
            user_id=user_id,
            filename=filename or Path(path).name,
            text=text,
            document_type=document_type,
        )

    def delete_document(self, *, user_id: str, document_id: str) -> None:
        """Remove a document's chunks from both stores, keeping them in sync."""
        chunk_ids = self._chunks.delete_for_document(user_id=user_id, document_id=document_id)
        self._vectors.delete_chunks(chunk_ids)
        self._candidate_profiles.mark_stale(user_id)
