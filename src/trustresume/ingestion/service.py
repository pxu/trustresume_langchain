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
from trustresume.storage import (
    CandidateProfileRepository,
    ChunkRepository,
    DocumentRepository,
    JobDocumentRepository,
)

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
        job_documents: JobDocumentRepository | None = None,
        id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        self._documents = documents
        self._chunks = chunks
        self._vectors = vector_store
        self._candidate_profiles = candidate_profiles
        self._job_documents = job_documents
        self._new_id = id_factory

    def ingest_text(
        self,
        *,
        user_id: str,
        filename: str,
        text: str,
        document_type: DocumentType = DocumentType.OTHER,
        job_id: str | None = None,
    ) -> str:
        """Ingest already-extracted text; return the document id.

        Identity/dedup is checked in two stages:

        1. **Filename-first**: if this user already has a document with this
           exact filename, the upload is treated as an update to that same
           logical document, not a new one — regardless of which job (if
           any) it's uploaded under. If the content is unchanged (same
           cleaned-text hash), this is a no-op. If it differs, the document
           is re-chunked and re-embedded *in place* (same ``document_id``,
           so its job links and history survive) — see :meth:`_reindex_in_place`.
        2. **Content-hash fallback**: for a filename never seen before, the
           existing cross-filename dedup (same *cleaned* text, different
           filename) still applies — a re-upload of "the same" document
           under a new name is still recognized as a duplicate rather than
           creating a second copy.

        If ``job_id`` is given, the resolved document is linked to that job
        (idempotently) regardless of which of the above paths resolved it —
        a document can be linked to more than one job.
        """
        if job_id is not None and self._job_documents is None:
            raise ValueError(
                "job_id was given but this IngestionService has no "
                "job_documents collaborator injected to record the link"
            )

        cleaned = clean_text(text)
        this_hash = content_hash(cleaned)

        by_filename = self._documents.find_by_filename(user_id=user_id, filename=filename)
        if by_filename is not None:
            document_id = str(by_filename["id"])
            if by_filename["content_hash"] == this_hash:
                logger.info(
                    "filename match, content unchanged, ingest skipped",
                    extra={
                        "user_id": user_id,
                        "document_id": document_id,
                        "doc_filename": filename,
                    },
                )
            else:
                self._reindex_in_place(
                    user_id=user_id,
                    document_id=document_id,
                    filename=filename,
                    document_type=document_type,
                    cleaned=cleaned,
                    this_hash=this_hash,
                )
            self._link_job(job_id, document_id)
            return document_id

        document_id = self._ingest_new_document(
            user_id=user_id,
            filename=filename,
            document_type=document_type,
            cleaned=cleaned,
            this_hash=this_hash,
        )
        self._link_job(job_id, document_id)
        return document_id

    def _link_job(self, job_id: str | None, document_id: str) -> None:
        """Associate ``document_id`` with ``job_id``, if one was given."""
        if job_id is not None:
            assert self._job_documents is not None  # checked in ingest_text
            self._job_documents.link(job_id=job_id, document_id=document_id)

    def _write_chunk_rows(
        self,
        *,
        user_id: str,
        document_id: str,
        filename: str,
        document_type: DocumentType,
        pieces: list[str],
    ) -> list[EvidenceChunk]:
        """Build ``EvidenceChunk``s for freshly-chunked text and insert their SQLite rows.

        Shared by the genuinely-new-document path and the re-index-in-place
        path — both need "chunk text -> ``EvidenceChunk`` objects -> SQLite
        rows"; they differ only in how they react to a subsequent Chroma
        upsert failure, which stays in each caller.
        """
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
        return evidence_chunks

    def _ingest_new_document(
        self,
        *,
        user_id: str,
        filename: str,
        document_type: DocumentType,
        cleaned: str,
        this_hash: str,
    ) -> str:
        """The genuinely-new-document path: content-hash dedup, then create + write.

        Reached only when :meth:`ingest_text` found no existing document
        with this filename — falls back to the pre-existing content-hash
        dedup so a re-upload of identical content under a never-before-seen
        filename is still recognized as a duplicate.
        """
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

        evidence_chunks = self._write_chunk_rows(
            user_id=user_id,
            document_id=document_id,
            filename=filename,
            document_type=document_type,
            pieces=pieces,
        )

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

    def _reindex_in_place(
        self,
        *,
        user_id: str,
        document_id: str,
        filename: str,
        document_type: DocumentType,
        cleaned: str,
        this_hash: str,
    ) -> None:
        """Re-chunk and re-embed a same-named document whose content changed.

        Preserves ``document_id`` (and therefore its job links and any other
        history keyed on it) rather than treating the upload as a new
        document. Unlike :meth:`_ingest_new_document`'s rollback-on-failure,
        a Chroma failure here can't cleanly restore the *old* chunks — they
        were already deleted before the new ones were written, and there is
        no cross-store transaction spanning SQLite and Chroma anywhere in
        this codebase to make that atomic. This is an accepted, logged gap:
        on failure the document's `content_hash` is left pointing at the old
        content while its chunks are incomplete/absent, mirroring the same
        "no cross-store transaction" pragmatism the create path already
        accepts, just now also reachable on update.
        """
        old_chunk_ids = self._chunks.delete_for_document(user_id=user_id, document_id=document_id)
        self._vectors.delete_chunks(old_chunk_ids)

        pieces = chunk_text(cleaned)
        evidence_chunks = self._write_chunk_rows(
            user_id=user_id,
            document_id=document_id,
            filename=filename,
            document_type=document_type,
            pieces=pieces,
        )

        try:
            self._vectors.upsert_chunks(evidence_chunks)
        except Exception:
            logger.exception(
                "chroma upsert failed while re-indexing an updated document "
                "in place; its old chunks were already removed",
                extra={
                    "user_id": user_id,
                    "document_id": document_id,
                    "chunks": len(evidence_chunks),
                },
            )
            raise

        self._documents.update_content_hash(
            user_id=user_id, document_id=document_id, content_hash=this_hash, filename=filename
        )
        logger.info(
            "document re-indexed in place",
            extra={
                "user_id": user_id,
                "document_id": document_id,
                "doc_filename": filename,
                "document_type": document_type.value,
                "chunks": len(evidence_chunks),
            },
        )
        self._candidate_profiles.mark_stale(user_id)

    def ingest_file(
        self,
        *,
        user_id: str,
        path: str,
        document_type: DocumentType = DocumentType.OTHER,
        filename: str | None = None,
        job_id: str | None = None,
    ) -> str:
        """Parse a file from disk and ingest it; return the new document id."""
        from pathlib import Path

        text = parse_document(path)
        return self.ingest_text(
            user_id=user_id,
            filename=filename or Path(path).name,
            text=text,
            document_type=document_type,
            job_id=job_id,
        )

    def ingest_bytes(
        self,
        *,
        user_id: str,
        filename: str,
        data: bytes,
        document_type: DocumentType = DocumentType.OTHER,
        job_id: str | None = None,
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
            job_id=job_id,
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
