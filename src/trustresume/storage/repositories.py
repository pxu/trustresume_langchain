"""SQLite repositories for TrustResume's structured data.

One repository per aggregate (users, documents, chunks, generated resumes,
evaluations). The design rule that makes ADR-0001 real: **every read and write
that touches user data takes a ``user_id``**, and every query filters on it.
There is no "get by id" that skips the owner check — a caller cannot
accidentally read across users, because the method signatures don't allow it.

Repositories hold a ``sqlite3.Connection`` and never open their own; the caller
owns the connection lifecycle (so a single request can share one connection and
one transaction across repositories). ``id``/timestamp generation is injected
via ``clock``/``id_factory`` so tests are deterministic.

Milestone M2 (storage + retrieval).
"""

from __future__ import annotations

import re
import sqlite3
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from trustresume.models import (
    ATSReport,
    CandidateProfile,
    DocumentType,
    EvidenceChunk,
    ResumeDraft,
    TrustReport,
)


def _new_id() -> str:
    """Default id factory: a uuid4 hex string."""
    return uuid.uuid4().hex


def _now_iso() -> str:
    """Default clock: current time as an ISO-8601 UTC string."""
    return datetime.now(UTC).isoformat()


class _BaseRepository:
    """Shared plumbing: a connection plus injectable id/clock for determinism."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        id_factory: Callable[[], str] = _new_id,
        clock: Callable[[], str] = _now_iso,
    ) -> None:
        self._conn = conn
        self._new_id = id_factory
        self._now = clock


class UserRepository(_BaseRepository):
    """CRUD for ``users``."""

    def create(self, name: str, *, user_id: str | None = None) -> str:
        """Insert a user and return its id (generated when not supplied)."""
        uid = user_id or self._new_id()
        self._conn.execute(
            "INSERT INTO users (id, name, created_at) VALUES (?, ?, ?)",
            (uid, name, self._now()),
        )
        self._conn.commit()
        return uid

    def exists(self, user_id: str) -> bool:
        """Whether a user with this id exists."""
        row = self._conn.execute("SELECT 1 FROM users WHERE id = ?", (user_id,)).fetchone()
        return row is not None


class DocumentRepository(_BaseRepository):
    """Metadata for uploaded source documents."""

    def create(
        self,
        *,
        user_id: str,
        filename: str,
        document_type: DocumentType,
        content_hash: str,
        document_id: str | None = None,
    ) -> str:
        """Insert a document row and return its id.

        ``content_hash`` backs the ``idx_documents_user_content`` unique index
        (same user + same hash -> ``sqlite3.IntegrityError``) — the DB-level
        half of duplicate-ingestion detection; ``IngestionService`` checks
        first via :meth:`find_by_content_hash` so a duplicate upload returns
        the existing document id instead of hitting this constraint at all.
        """
        doc_id = document_id or self._new_id()
        self._conn.execute(
            "INSERT INTO documents "
            "(id, user_id, filename, document_type, content_hash, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (doc_id, user_id, filename, document_type.value, content_hash, self._now()),
        )
        self._conn.commit()
        return doc_id

    def find_by_content_hash(self, *, user_id: str, content_hash: str) -> sqlite3.Row | None:
        """The existing document row for this user + content hash, if any.

        The read half of duplicate-ingestion detection: ``IngestionService``
        calls this before writing anything, so re-uploading the same document
        is a cheap no-op read rather than a wasted embed/chunk/write cycle.
        """
        row: sqlite3.Row | None = self._conn.execute(
            "SELECT * FROM documents WHERE user_id = ? AND content_hash = ?",
            (user_id, content_hash),
        ).fetchone()
        return row

    def delete(self, *, user_id: str, document_id: str) -> None:
        """Delete one document row.

        Without this, ``IngestionService.delete_document`` had no way to
        remove the ``documents`` row itself (only the chunk rows) — the row's
        ``content_hash`` stayed live in ``idx_documents_user_content``
        forever, so re-uploading identical content after a delete was
        silently treated as "already ingested" and produced a document with
        zero chunks. Called by ``IngestionService`` on both an explicit
        delete and a failed post-create Chroma upsert (same reason: an
        orphaned ``documents`` row with no chunks must not survive).
        """
        self._conn.execute(
            "DELETE FROM documents WHERE user_id = ? AND id = ?", (user_id, document_id)
        )
        self._conn.commit()

    def list_for_user(self, user_id: str) -> list[sqlite3.Row]:
        """All documents owned by ``user_id``, newest first."""
        return self._conn.execute(
            "SELECT * FROM documents WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()


class ChunkRepository(_BaseRepository):
    """The SQLite metadata mirror of the Qdrant vectors.

    A chunk's row here and its Qdrant point are written/deleted together by the
    ingestion pipeline; this repository owns only the SQLite half.
    """

    def add(self, chunk: EvidenceChunk, *, chunk_index: int) -> None:
        """Insert one chunk's metadata row.

        Takes an :class:`EvidenceChunk` so the ingestion pipeline can build the
        model once and store it in both stores from the same object.
        """
        self._conn.execute(
            "INSERT INTO chunks (chunk_id, user_id, document_id, chunk_index, "
            "document_type, source_document, text, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                chunk.chunk_id,
                chunk.user_id,
                chunk.document_id,
                chunk_index,
                chunk.document_type.value,
                chunk.source_document,
                chunk.text,
                self._now(),
            ),
        )
        self._conn.commit()

    def list_for_user(self, user_id: str) -> list[sqlite3.Row]:
        """All chunk rows owned by ``user_id``."""
        return self._conn.execute(
            "SELECT * FROM chunks WHERE user_id = ? ORDER BY document_id, chunk_index",
            (user_id,),
        ).fetchall()

    def delete_for_document(self, *, user_id: str, document_id: str) -> list[str]:
        """Delete all chunk rows for a document and return their ids.

        The returned ids let the caller delete the matching Qdrant points, so
        the two stores stay in sync (ADR-0001).
        """
        rows = self._conn.execute(
            "SELECT chunk_id FROM chunks WHERE user_id = ? AND document_id = ?",
            (user_id, document_id),
        ).fetchall()
        self._conn.execute(
            "DELETE FROM chunks WHERE user_id = ? AND document_id = ?",
            (user_id, document_id),
        )
        self._conn.commit()
        return [r["chunk_id"] for r in rows]

    def search_keywords(self, *, user_id: str, query: str, limit: int = 5) -> list[sqlite3.Row]:
        """Full-text (BM25) search over this user's chunks via the ``chunks_fts`` index.

        The keyword half of hybrid retrieval (``retrieval.hybrid.HybridRetriever``)
        — a term the embedding model treats as semantically fungible with a
        synonym (e.g. "Lambda" vs. "serverless") but that a job description
        names exactly still gets matched. Returns rows ordered by BM25 rank
        (SQLite's ``bm25()``: lower is more relevant, so ``ORDER BY`` ascending);
        an empty/non-alphanumeric query returns no rows rather than raising.
        """
        fts_query = _to_fts5_query(query)
        if not fts_query:
            return []
        return self._conn.execute(
            "SELECT chunks.*, bm25(chunks_fts) AS rank FROM chunks_fts "
            "JOIN chunks ON chunks.rowid = chunks_fts.rowid "
            "WHERE chunks_fts MATCH ? AND chunks.user_id = ? "
            "ORDER BY rank, chunks.chunk_id LIMIT ?",
            (fts_query, user_id, limit),
        ).fetchall()


_FTS5_TOKEN = re.compile(r"[A-Za-z0-9]+")


def _to_fts5_query(query: str) -> str:
    """Turn free text into a safe FTS5 ``MATCH`` expression.

    FTS5's query syntax treats punctuation specially (``-``, ``"``, ``*``,
    ``:``, ``/``, ...), so passing a raw job-description string straight to
    ``MATCH`` raises a syntax error the moment it contains e.g. "AI/ML" or a
    parenthesis. Tokenizing to bare alphanumeric words and quoting each as its
    own phrase, OR-joined, sidesteps that entirely — every token becomes a
    literal-text match, and OR means any matching token surfaces the chunk
    (BM25 ranks chunks matching more/rarer tokens higher).
    """
    tokens = _FTS5_TOKEN.findall(query)
    return " OR ".join(f'"{t}"' for t in tokens)


class CandidateProfileCacheEntry:
    """A cached :class:`CandidateProfile` row plus its staleness bookkeeping."""

    __slots__ = ("profile", "doc_hash", "stale", "updated_at")

    def __init__(
        self, *, profile: CandidateProfile, doc_hash: str, stale: bool, updated_at: str
    ) -> None:
        self.profile = profile
        self.doc_hash = doc_hash
        self.stale = stale
        self.updated_at = updated_at


class CandidateProfileRepository(_BaseRepository):
    """One cached, job-independent Candidate Profile per user.

    Unlike the other repositories this is a single-row-per-user cache, not an
    append-only log: ``upsert`` replaces the prior row entirely.
    """

    def get(self, user_id: str) -> CandidateProfileCacheEntry | None:
        """The cached profile for this user, or ``None`` if never computed."""
        row = self._conn.execute(
            "SELECT profile_json, doc_hash, stale, updated_at FROM candidate_profiles "
            "WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if row is None:
            return None
        return CandidateProfileCacheEntry(
            profile=CandidateProfile.model_validate_json(row["profile_json"]),
            doc_hash=row["doc_hash"],
            stale=bool(row["stale"]),
            updated_at=row["updated_at"],
        )

    def upsert(self, *, user_id: str, profile: CandidateProfile, doc_hash: str) -> None:
        """Replace the cached profile for a user and clear its stale flag."""
        self._conn.execute(
            "INSERT INTO candidate_profiles (user_id, profile_json, doc_hash, stale, updated_at) "
            "VALUES (?, ?, ?, 0, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET "
            "profile_json = excluded.profile_json, doc_hash = excluded.doc_hash, "
            "stale = 0, updated_at = excluded.updated_at",
            (user_id, profile.model_dump_json(), doc_hash, self._now()),
        )
        self._conn.commit()

    def mark_stale(self, user_id: str) -> None:
        """Flag the cached profile as needing recomputation.

        A no-op when there's no row yet — a user's first ``get_or_refresh``
        computes fresh regardless of this flag (see
        ``CandidateProfileService``).
        """
        self._conn.execute(
            "UPDATE candidate_profiles SET stale = 1 WHERE user_id = ?",
            (user_id,),
        )
        self._conn.commit()


class ResumeRepository(_BaseRepository):
    """Exported resume drafts and their scores at export time."""

    def create(
        self,
        *,
        user_id: str,
        draft: ResumeDraft,
        job_title: str | None,
        trust_score: float,
        ats_score: float,
        passed: bool,
    ) -> str:
        """Persist an exported draft; return its id."""
        resume_id = self._new_id()
        self._conn.execute(
            "INSERT INTO generated_resumes (id, user_id, job_title, iteration, summary, "
            "content_json, trust_score, ats_score, passed, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                resume_id,
                user_id,
                job_title,
                draft.iteration,
                draft.summary,
                draft.model_dump_json(),
                trust_score,
                ats_score,
                int(passed),
                self._now(),
            ),
        )
        self._conn.commit()
        return resume_id

    def get(self, *, user_id: str, resume_id: str) -> ResumeDraft | None:
        """Rehydrate a stored draft, or ``None`` if not found for this user."""
        row = self._conn.execute(
            "SELECT content_json FROM generated_resumes WHERE id = ? AND user_id = ?",
            (resume_id, user_id),
        ).fetchone()
        if row is None:
            return None
        return ResumeDraft.model_validate_json(row["content_json"])


class EvaluationRepository(_BaseRepository):
    """Trust + ATS report pairs attached to an exported resume."""

    def create(
        self,
        *,
        user_id: str,
        resume_id: str,
        trust: TrustReport,
        ats: ATSReport,
    ) -> str:
        """Persist a report pair; return its id."""
        eval_id = self._new_id()
        self._conn.execute(
            "INSERT INTO evaluations (id, resume_id, user_id, iteration, trust_score, "
            "ats_score, trust_report_json, ats_report_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                eval_id,
                resume_id,
                user_id,
                trust.iteration,
                trust.score,
                ats.score,
                trust.model_dump_json(),
                ats.model_dump_json(),
                self._now(),
            ),
        )
        self._conn.commit()
        return eval_id

    def list_for_resume(self, *, user_id: str, resume_id: str) -> list[dict[str, object]]:
        """Return stored report pairs for a resume as rehydrated models.

        Each item is ``{"trust": TrustReport, "ats": ATSReport}``.
        """
        rows = self._conn.execute(
            "SELECT trust_report_json, ats_report_json FROM evaluations "
            "WHERE user_id = ? AND resume_id = ? ORDER BY iteration",
            (user_id, resume_id),
        ).fetchall()
        return [
            {
                "trust": TrustReport.model_validate_json(r["trust_report_json"]),
                "ats": ATSReport.model_validate_json(r["ats_report_json"]),
            }
            for r in rows
        ]
