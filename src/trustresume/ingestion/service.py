"""IngestionService — the pipeline that lands a document in both stores.

Ties the pure steps (parse → clean → chunk) to the two stores from ADR-0001,
and is the one place responsible for keeping them in sync: for every chunk it
writes the SQLite metadata row **and** the Chroma vector, and on document
delete it removes the chunk rows, the Chroma vectors, *and* the ``documents``
row itself — leaving the document row behind would keep its content hash
"occupying" that content forever (see ``ingest_text``'s dedup check),
silently no-opping any future re-ingest of the same content. Everything is
scoped by ``user_id``.

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

import hashlib
import logging
import sqlite3
import uuid
from collections.abc import Callable

from trustresume.models import DocumentType, EvidenceChunk
from trustresume.retrieval import ChromaVectorStore
from trustresume.storage import CandidateProfileRepository, ChunkRepository, DocumentRepository

from .chunker import chunk_text, clean_text
from .parser import parse_bytes, parse_document

logger = logging.getLogger(__name__)


def content_hash(cleaned_text: str) -> str:
    """A stable identity for "this document's content", for duplicate detection.

    Hashes the *cleaned* text (post ``clean_text``), not the raw upload bytes
    — two uploads that differ only in whitespace/line-ending/encoding (the
    same resume re-exported from a different tool) still hash identically,
    since ``clean_text`` already normalizes exactly that kind of noise.

    Public (not ``_content_hash``) because ``orchestration.
    candidate_profile_service`` reuses it for its own diagnostic doc-hash —
    one hashing convention, not two independently-written ``sha256(...)``
    call sites that happen to agree today.
    """
    return hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest()


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
        """Ingest already-extracted text; return the document id.

        Deduplicates by content: if this user already has a document whose
        *cleaned* text hashes identically (a re-upload of the same resume,
        possibly re-exported/re-encoded), this is a no-op that returns the
        existing document id — nothing is re-chunked, re-embedded, or
        re-written in either store. Without this, uploading "the same"
        resume twice would double every chunk in both SQLite and Chroma,
        double-counting it in every future retrieval.

        Order matters for a genuinely new document: the SQLite metadata rows
        are written first, then the Chroma vectors. If embedding/upsert
        fails, the SQLite rows are rolled back so the two stores don't drift
        out of sync.
        """
        cleaned = clean_text(text)
        this_hash = content_hash(cleaned)

        existing = self._documents.find_by_content_hash(user_id=user_id, content_hash=this_hash)
        if existing is not None:
            logger.info(
                "duplicate document ingest skipped",
                extra={
                    "user_id": user_id,
                    "document_id": existing["id"],
                    "doc_filename": filename,
                },
            )
            return str(existing["id"])

        pieces = chunk_text(cleaned)

        try:
            document_id = self._documents.create(
                user_id=user_id,
                filename=filename,
                document_type=document_type,
                content_hash=this_hash,
            )
        except sqlite3.IntegrityError as exc:
            # The find_by_content_hash check above and this insert aren't one
            # atomic transaction, so a concurrent ingest of the same content
            # (two requests racing) can slip between them; the DB's unique
            # index (idx_documents_user_content) is the real guarantee here.
            # Re-read rather than raise, so the caller still gets a valid,
            # idempotent document id instead of a transient race surfacing as
            # an error.
            existing = self._documents.find_by_content_hash(user_id=user_id, content_hash=this_hash)
            if existing is None:
                # The unique-index violation we just caught guarantees a row
                # with this (user_id, content_hash) exists — if this ever
                # fires, something more surprising than a simple race is
                # going on (e.g. a concurrent delete of that exact row
                # between the IntegrityError and this re-read), so raise
                # loudly rather than silently return an invalid id.
                raise RuntimeError(
                    f"IntegrityError on content_hash={this_hash!r} for user_id={user_id!r}, "
                    "but no matching document row was found on re-read"
                ) from exc
            return str(existing["id"])

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
            # Keep the stores consistent: undo everything we just wrote for
            # this document, including the `documents` row itself — leaving
            # it behind would keep its content_hash live in
            # idx_documents_user_content, permanently blocking any future
            # retry of the same content (find_by_content_hash would keep
            # matching this now-orphaned, chunk-less row forever).
            logger.exception(
                "chroma upsert failed, rolling back sqlite chunk and document rows",
                extra={
                    "user_id": user_id,
                    "document_id": document_id,
                    "chunks": len(evidence_chunks),
                },
            )
            self._chunks.delete_for_document(user_id=user_id, document_id=document_id)
            self._documents.delete(user_id=user_id, document_id=document_id)
            raise

        logger.info(
            "document ingested",
            extra={
                "user_id": user_id,
                "document_id": document_id,
                # Not "filename" — that's a reserved LogRecord attribute name
                # (the caller's source file) and passing it via `extra`
                # raises KeyError inside stdlib logging.
                "doc_filename": filename,
                "document_type": document_type.value,
                "chunks": len(evidence_chunks),
            },
        )
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

    def ingest_bytes(
        self,
        *,
        user_id: str,
        filename: str,
        data: bytes,
        document_type: DocumentType = DocumentType.OTHER,
    ) -> str:
        """Parse an in-memory upload (e.g. a web multipart file) and ingest it.

        The no-temp-file counterpart to :meth:`ingest_file`, for callers that
        already have the upload's bytes in hand (FastAPI's ``UploadFile``,
        Streamlit's ``file_uploader``) and shouldn't have to round-trip
        through disk just to reuse the file-based parser.
        """
        text = parse_bytes(filename, data)
        return self.ingest_text(
            user_id=user_id,
            filename=filename,
            text=text,
            document_type=document_type,
        )

    def delete_document(self, *, user_id: str, document_id: str) -> None:
        """Remove a document's chunks from both stores and the document row itself.

        Deleting the ``documents`` row (not just its chunks) matters for
        content-hash dedup: leaving it behind would keep its ``content_hash``
        occupying the ``idx_documents_user_content`` unique index, so a
        future re-upload of the same content would match it via
        ``find_by_content_hash`` and be silently treated as "already
        ingested" — producing a document with zero chunks instead of
        actually re-ingesting.

        ``mark_stale`` runs in a ``finally`` so a mid-method failure (Chroma
        blip deleting vectors, etc.) can't leave the cached Candidate Profile
        pointing at documents that no longer exist in SQLite — the same
        keep-the-cache-honest discipline ``ingest_text`` applies to its own
        Chroma failure path.
        """
        try:
            chunk_ids = self._chunks.delete_for_document(user_id=user_id, document_id=document_id)
            self._vectors.delete_chunks(chunk_ids)
            self._documents.delete(user_id=user_id, document_id=document_id)
        finally:
            self._candidate_profiles.mark_stale(user_id)
        logger.info(
            "document deleted",
            extra={"user_id": user_id, "document_id": document_id, "chunks": len(chunk_ids)},
        )
