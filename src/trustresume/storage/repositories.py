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
    JobDescription,
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

    def exists(self, *, user_id: str, document_id: str) -> bool:
        """Whether ``document_id`` exists and is owned by ``user_id``.

        A single indexed row check for the ownership gate on delete —
        cheaper than fetching every column of every document a user has via
        :meth:`list_for_user` just to scan for one id.
        """
        row = self._conn.execute(
            "SELECT 1 FROM documents WHERE user_id = ? AND id = ?", (user_id, document_id)
        ).fetchone()
        return row is not None

    def list_for_user(self, user_id: str) -> list[sqlite3.Row]:
        """All documents owned by ``user_id``, newest first."""
        return self._conn.execute(
            "SELECT * FROM documents WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()

    def find_by_filename(self, *, user_id: str, filename: str) -> sqlite3.Row | None:
        """The existing document row for this user + filename, if any.

        The primary half of the filename-first identity check
        (``IngestionService.ingest_text``): a same-named re-upload is treated
        as an update to the same logical document, not a new one — this is
        what lets that path find the row to update in place.
        """
        row: sqlite3.Row | None = self._conn.execute(
            "SELECT * FROM documents WHERE user_id = ? AND filename = ?",
            (user_id, filename),
        ).fetchone()
        return row

    def update_content_hash(
        self, *, user_id: str, document_id: str, content_hash: str, filename: str
    ) -> None:
        """Update an existing document row's content identity in place.

        Called when a same-named re-upload's content has actually changed
        (``IngestionService.ingest_text``'s in-place-update path) — the
        ``document_id`` (and therefore its job links, per ``job_documents``)
        is preserved; only what the row says about its current content
        changes. ``filename`` is set explicitly rather than assumed
        unchanged, even though the caller matched on it, so this stays
        correct if that assumption ever changes.
        """
        self._conn.execute(
            "UPDATE documents SET content_hash = ?, filename = ? WHERE user_id = ? AND id = ?",
            (content_hash, filename, user_id, document_id),
        )
        self._conn.commit()

    def list_eligible_document_ids(self, *, user_id: str, job_id: str | None) -> list[str]:
        """Document ids eligible for retrieval in a given job's context.

        The generic pool (documents with **no** job association at all,
        across every job) unioned with documents explicitly linked to
        ``job_id`` when one is given. A document linked to a *different* job
        only is not part of this job's eligible set — "generic" means
        unlinked to any job, not merely unlinked to this one.

        ``job_id=None`` returns just the generic pool (the pre-existing,
        job-agnostic retrieval behavior, unchanged).
        """
        query = (
            "SELECT d.id FROM documents d WHERE d.user_id = ? AND ("
            "NOT EXISTS (SELECT 1 FROM job_documents jd WHERE jd.document_id = d.id)"
        )
        params: list[str] = [user_id]
        if job_id is not None:
            query += " OR d.id IN (SELECT document_id FROM job_documents WHERE job_id = ?)"
            params.append(job_id)
        query += ")"
        rows = self._conn.execute(query, params).fetchall()
        return [r["id"] for r in rows]


class JobRepository(_BaseRepository):
    """CRUD for ``jobs`` — a persisted, extracted job posting.

    Stores the full ``JobDescription`` as JSON (round-trippable via
    ``model_validate_json``) plus flattened ``title``/``company``/``summary``
    columns for cheap listing without deserializing every row — the same
    flattening pattern ``generated_resumes.job_title`` already uses.
    """

    def create(
        self,
        *,
        user_id: str,
        raw_posting: str,
        job: JobDescription,
        summary: str | None,
        job_id: str | None = None,
    ) -> str:
        """Insert a job row and return its id."""
        jid = job_id or self._new_id()
        now = self._now()
        self._conn.execute(
            "INSERT INTO jobs (id, user_id, title, company, summary, raw_posting, "
            "job_description_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                jid,
                user_id,
                job.title,
                job.company,
                summary,
                raw_posting,
                job.model_dump_json(),
                now,
                now,
            ),
        )
        self._conn.commit()
        return jid

    def get(self, *, user_id: str, job_id: str) -> sqlite3.Row | None:
        """The job row for this user + id, or ``None`` if not found/not owned."""
        row: sqlite3.Row | None = self._conn.execute(
            "SELECT * FROM jobs WHERE user_id = ? AND id = ?",
            (user_id, job_id),
        ).fetchone()
        return row

    def list_for_user(self, user_id: str) -> list[sqlite3.Row]:
        """All jobs owned by ``user_id``, newest first."""
        return self._conn.execute(
            "SELECT * FROM jobs WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()

    def update(
        self,
        *,
        user_id: str,
        job_id: str,
        raw_posting: str,
        job: JobDescription,
        summary: str | None,
    ) -> bool:
        """Replace a job's posting text and its extraction; return whether it existed.

        A full replace (new posting -> full re-extraction), not a partial
        field patch — matches "update re-extracts" rather than exposing a
        general-purpose PATCH.
        """
        cursor = self._conn.execute(
            "UPDATE jobs SET title = ?, company = ?, summary = ?, raw_posting = ?, "
            "job_description_json = ?, updated_at = ? WHERE user_id = ? AND id = ?",
            (
                job.title,
                job.company,
                summary,
                raw_posting,
                job.model_dump_json(),
                self._now(),
                user_id,
                job_id,
            ),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def delete(self, *, user_id: str, job_id: str) -> None:
        """Delete one job row.

        ``job_documents`` link rows cascade automatically (FK ON DELETE
        CASCADE); ``generated_resumes.job_id`` is set NULL on its rows
        rather than cascaded, so past generations for this job remain —
        see ``storage/schema.py``.
        """
        self._conn.execute("DELETE FROM jobs WHERE user_id = ? AND id = ?", (user_id, job_id))
        self._conn.commit()

    def exists(self, *, user_id: str, job_id: str) -> bool:
        """Whether ``job_id`` exists and is owned by ``user_id``."""
        row = self._conn.execute(
            "SELECT 1 FROM jobs WHERE user_id = ? AND id = ?", (user_id, job_id)
        ).fetchone()
        return row is not None


class JobDocumentRepository(_BaseRepository):
    """The many-to-many link between ``jobs`` and ``documents``.

    Kept as its own repository (not folded into ``JobRepository`` or
    ``DocumentRepository``) since it owns a genuinely distinct aggregate —
    the link itself, not either side of it — matching this module's
    one-repository-per-aggregate design.
    """

    def link(self, *, job_id: str, document_id: str) -> None:
        """Associate a document with a job. Idempotent — re-linking is a no-op.

        ``INSERT OR IGNORE`` on the composite primary key means a caller
        never needs to check existence first; re-uploading the same
        document under the same job repeatedly is always safe.
        """
        self._conn.execute(
            "INSERT OR IGNORE INTO job_documents (job_id, document_id, created_at) "
            "VALUES (?, ?, ?)",
            (job_id, document_id, self._now()),
        )
        self._conn.commit()

    def unlink(self, *, job_id: str, document_id: str) -> None:
        """Remove a document's association with a job (the document itself is untouched)."""
        self._conn.execute(
            "DELETE FROM job_documents WHERE job_id = ? AND document_id = ?",
            (job_id, document_id),
        )
        self._conn.commit()

    def list_document_ids_for_job(self, job_id: str) -> list[str]:
        """Every document id explicitly linked to ``job_id``."""
        rows = self._conn.execute(
            "SELECT document_id FROM job_documents WHERE job_id = ?", (job_id,)
        ).fetchall()
        return [r["document_id"] for r in rows]

    def list_job_ids_for_document(self, document_id: str) -> list[str]:
        """Every job id ``document_id`` is linked to."""
        rows = self._conn.execute(
            "SELECT job_id FROM job_documents WHERE document_id = ?", (document_id,)
        ).fetchall()
        return [r["job_id"] for r in rows]


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

    def search_keywords(
        self,
        *,
        user_id: str,
        query: str,
        limit: int = 5,
        document_ids: list[str] | None = None,
    ) -> list[sqlite3.Row]:
        """Full-text (BM25) search over this user's chunks via the ``chunks_fts`` index.

        The keyword half of hybrid retrieval (``retrieval.hybrid.HybridRetriever``)
        — a term the embedding model treats as semantically fungible with a
        synonym (e.g. "Lambda" vs. "serverless") but that a job description
        names exactly still gets matched. Returns rows ordered by BM25 rank
        (SQLite's ``bm25()``: lower is more relevant, so ``ORDER BY`` ascending);
        an empty/non-alphanumeric query returns no rows rather than raising.

        ``document_ids``, when given, additionally restricts results to that
        set (job-scoped retrieval — see ``DocumentRepository.list_eligible_document_ids``).
        An empty list short-circuits to no rows without querying at all,
        rather than relying on SQL's ``IN ()`` (valid but wasteful/confusing
        here, and best avoided for the same reason the FTS5 empty-query case
        above already short-circuits).
        """
        if document_ids is not None and not document_ids:
            return []
        fts_query = _to_fts5_query(query)
        if not fts_query:
            return []
        sql = (
            "SELECT chunks.*, bm25(chunks_fts) AS rank FROM chunks_fts "
            "JOIN chunks ON chunks.rowid = chunks_fts.rowid "
            "WHERE chunks_fts MATCH ? AND chunks.user_id = ? "
        )
        params: list[str | int] = [fts_query, user_id]
        if document_ids is not None:
            placeholders = ",".join("?" * len(document_ids))
            sql += f"AND chunks.document_id IN ({placeholders}) "
            params.extend(document_ids)
        sql += "ORDER BY rank, chunks.chunk_id LIMIT ?"
        params.append(limit)
        return self._conn.execute(sql, params).fetchall()


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
        job_id: str | None = None,
        pdf_bytes: bytes | None = None,
        markdown_text: str | None = None,
        rejection_reason: str | None = None,
        improvement_suggestions: str | None = None,
    ) -> str:
        """Persist an exported draft; return its id.

        ``pdf_bytes``/``markdown_text`` are the rendered export forms of
        ``draft`` (see ``trustresume.export``), computed once at persist
        time. ``rejection_reason``/``improvement_suggestions`` are only ever
        set for a draft that didn't pass the quality gate — both ``None``
        for a passing draft.
        """
        resume_id = self._new_id()
        self._conn.execute(
            "INSERT INTO generated_resumes (id, user_id, job_id, job_title, iteration, "
            "summary, content_json, trust_score, ats_score, passed, pdf_bytes, "
            "markdown_text, rejection_reason, improvement_suggestions, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                resume_id,
                user_id,
                job_id,
                job_title,
                draft.iteration,
                draft.summary,
                draft.model_dump_json(),
                trust_score,
                ats_score,
                int(passed),
                pdf_bytes,
                markdown_text,
                rejection_reason,
                improvement_suggestions,
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

    def get_row(self, *, user_id: str, resume_id: str) -> sqlite3.Row | None:
        """The raw stored row, or ``None`` if not found for this user.

        A row-returning sibling to :meth:`get` — needed so a caller (the API
        layer) can serve ``pdf_bytes``/``markdown_text``/``rejection_reason``
        directly without a second, model-only round trip that would drop
        the binary/export fields ``get`` never rehydrates.
        """
        row: sqlite3.Row | None = self._conn.execute(
            "SELECT * FROM generated_resumes WHERE id = ? AND user_id = ?",
            (resume_id, user_id),
        ).fetchone()
        return row

    def list_for_job(self, *, user_id: str, job_id: str) -> list[sqlite3.Row]:
        """All resumes generated for a job, newest first."""
        return self._conn.execute(
            "SELECT * FROM generated_resumes WHERE user_id = ? AND job_id = ? "
            "ORDER BY created_at DESC",
            (user_id, job_id),
        ).fetchall()


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
