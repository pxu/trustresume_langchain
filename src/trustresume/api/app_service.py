"""TrustResumeApp — the application facade the API (and tests) drive.

This is where every milestone's pieces are wired into one object: the two
stores (M2), ingestion (M3), the agents (M4) behind the orchestrator (M5) —
including the cached Candidate Profile service — and the extracted trust/eval
logic (M6). The FastAPI server (M7) and any other
front end are thin layers over this — all real logic lives here so it can be
tested without HTTP or a browser.

Construction is explicit and injectable: a shared SQLite connection and one
Chroma client thread through everything, and the LLM model + embedder are
supplied by the caller (real ones in ``build_default_app``; fakes in tests).

Milestone M7 (FastAPI backend).
"""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from trustresume.agents import (
    ATSEvaluationAgent,
    CandidateProfileAgent,
    EvidenceRetrievalAgent,
    JobDescriptionAgent,
    ResumeWriterAgent,
    TrustHarnessAgent,
)
from trustresume.export import render_markdown, render_pdf
from trustresume.ingestion import IngestionService
from trustresume.models import DocumentType, EvidenceSet, JobDescription, WorkflowState
from trustresume.orchestration import CandidateProfileService, Orchestrator, build_feedback
from trustresume.orchestration.rejection import build_rejection_reason
from trustresume.retrieval import ChromaVectorStore, HybridRetriever
from trustresume.retrieval.vector_store import COLLECTION_NAME
from trustresume.storage import (
    CandidateProfileRepository,
    ChunkRepository,
    DocumentRepository,
    EvaluationRepository,
    JobDocumentRepository,
    JobRepository,
    ResumeRepository,
    UserRepository,
    connect,
    init_db,
)

from .model_factory import LLMConfig, build_model

_SUMMARY_FALLBACK_CHARS = 120


def _job_summary(job: JobDescription) -> str | None:
    """A short, human-readable label for a job — not part of ``JobDescription``
    itself, synthesized once at job-creation/update time so listing jobs
    doesn't need to re-derive it from the full extraction every time.

    ``"{title} at {company}"`` when both are known; whichever one is known
    alone; a truncated prefix of the raw posting when neither extracted
    (the LLM found no title/company to name) — still gives a caller
    something to show rather than a blank label.
    """
    if job.title and job.company:
        return f"{job.title} at {job.company}"
    if job.title:
        return job.title
    if job.company:
        return job.company
    text = job.raw_text.strip()
    if not text:
        return None
    return text[:_SUMMARY_FALLBACK_CHARS]


class TrustResumeApp:
    """Full application: ingest documents, generate a resume, persist the result."""

    def __init__(
        self,
        *,
        connection: sqlite3.Connection,
        chroma_client: Any,  # chromadb.ClientAPI — chromadb ships no type stubs
        embedder: Embeddings,
        model: BaseChatModel,
        chroma_collection_name: str = COLLECTION_NAME,
    ) -> None:
        init_db(connection)
        self._connection = connection
        self._chroma_client = chroma_client

        # Storage (M2)
        self._users = UserRepository(connection)
        self._documents = DocumentRepository(connection)
        self._chunks = ChunkRepository(connection)
        self._candidate_profiles = CandidateProfileRepository(connection)
        self._resumes = ResumeRepository(connection)
        self._evaluations = EvaluationRepository(connection)
        self._jobs = JobRepository(connection)
        self._job_documents = JobDocumentRepository(connection)

        # Retrieval (M2). ``chroma_collection_name`` defaults to the shared
        # production name; tests override it to a unique value per app
        # instance — chromadb.EphemeralClient() caches its underlying storage
        # by collection name across client instances within one process, so
        # same-named collections leak state across "fresh" apps otherwise.
        self._vectors = ChromaVectorStore(
            chroma_client, embedder, collection_name=chroma_collection_name
        )
        # Hybrid retrieval: fuses the Chroma search above with a keyword
        # (BM25) search over the same chunks' SQLite rows (chunks_fts) — see
        # retrieval/hybrid.py. Both agents' retrieval (below) and the
        # standalone search_evidence() go through this, not ChromaVectorStore
        # directly, so every retrieval path in the app benefits equally.
        self._retriever = HybridRetriever(self._vectors, self._chunks)

        # Ingestion (M3)
        self._ingestion = IngestionService(
            documents=self._documents,
            chunks=self._chunks,
            vector_store=self._vectors,
            candidate_profiles=self._candidate_profiles,
            job_documents=self._job_documents,
        )

        # Agents (M4) + orchestrator (M5). One JobDescriptionAgent instance is
        # shared between create_job/update_job (which extract at job-creation
        # time) and the orchestrator (whose _analyze_job node is a no-op
        # whenever a persisted job is passed in — see Orchestrator._analyze_job)
        # — a single source of truth, not two agents that happen to agree.
        self._job_agent = JobDescriptionAgent(model)
        candidate_profile_service = CandidateProfileService(
            agent=CandidateProfileAgent(model),
            chunks=self._chunks,
            profiles=self._candidate_profiles,
        )
        self._orchestrator = Orchestrator(
            job_description_agent=self._job_agent,
            candidate_profile_service=candidate_profile_service,
            retrieval_agent=EvidenceRetrievalAgent(self._retriever),
            resume_agent=ResumeWriterAgent(model),
            trust_agent=TrustHarnessAgent(model),
            evaluation_agent=ATSEvaluationAgent(),
        )

    # --- user + document management ---------------------------------------

    def ensure_user(self, name: str, *, user_id: str) -> str:
        """Create the user if absent; return the id. Idempotent.

        The existence check and the insert aren't one atomic transaction, so
        two concurrent first-calls for the same new ``user_id`` can both see
        ``exists() == False`` and both try to create it; the DB's primary-key
        constraint is the real guarantee, and the loser just re-reads rather
        than surfacing an ``IntegrityError`` — the same race/handling
        ``DocumentRepository.create`` already applies for content-hash
        collisions.
        """
        if not self._users.exists(user_id):
            with contextlib.suppress(sqlite3.IntegrityError):
                self._users.create(name, user_id=user_id)
        return user_id

    def add_document(
        self,
        *,
        user_id: str,
        filename: str,
        text: str,
        document_type: DocumentType = DocumentType.OTHER,
    ) -> str:
        """Ingest an uploaded document's text; return its id."""
        return self._ingestion.ingest_text(
            user_id=user_id,
            filename=filename,
            text=text,
            document_type=document_type,
        )

    def add_document_bytes(
        self,
        *,
        user_id: str,
        filename: str,
        data: bytes,
        document_type: DocumentType = DocumentType.OTHER,
    ) -> str:
        """Ingest an uploaded file's raw bytes (e.g. a FastAPI ``UploadFile``); return its id."""
        return self._ingestion.ingest_bytes(
            user_id=user_id,
            filename=filename,
            data=data,
            document_type=document_type,
        )

    def list_documents(self, user_id: str) -> list[dict[str, object]]:
        """Document metadata for the user, as plain dicts for the API layer."""
        return [dict(row) for row in self._documents.list_for_user(user_id)]

    def delete_document(self, *, user_id: str, document_id: str) -> bool:
        """Remove a document (and its chunks/vectors) if the user owns it.

        Returns whether a document was actually found and removed, so the API
        layer can 404 rather than silently no-op on an unknown/foreign id.
        """
        if not self._documents.exists(user_id=user_id, document_id=document_id):
            return False
        self._ingestion.delete_document(user_id=user_id, document_id=document_id)
        return True

    # --- job management ------------------------------------------------------

    def create_job(self, *, user_id: str, job_posting: str) -> sqlite3.Row:
        """Extract and persist a job posting; return the stored row.

        Extraction happens now, not lazily on first generation — a caller
        listing/inspecting a job sees its real title/company/summary
        immediately, and a later ``generate_for_job`` skips re-extraction
        entirely (``Orchestrator._analyze_job`` is a no-op when a persisted
        ``JobDescription`` is supplied).
        """
        job = asyncio.run(self._job_agent.run(job_posting))
        summary = _job_summary(job)
        job_id = self._jobs.create(
            user_id=user_id, raw_posting=job_posting, job=job, summary=summary
        )
        row = self._jobs.get(user_id=user_id, job_id=job_id)
        assert row is not None  # just inserted under the same connection
        return row

    def get_job(self, *, user_id: str, job_id: str) -> sqlite3.Row | None:
        """The stored job row for this user + id, or ``None`` if not found/owned."""
        return self._jobs.get(user_id=user_id, job_id=job_id)

    def list_jobs(self, user_id: str) -> list[sqlite3.Row]:
        """Every job owned by this user, newest first."""
        return self._jobs.list_for_user(user_id)

    def update_job(self, *, user_id: str, job_id: str, job_posting: str) -> sqlite3.Row | None:
        """Replace a job's posting text and re-extract; return the updated row.

        ``None`` if the job doesn't exist or isn't owned by this user —
        checked *before* running extraction, so a call against an unowned
        or nonexistent job never pays for an LLM call it can't use. A full
        replace-and-re-extract (matching "update re-extracts"), not a
        partial field patch.
        """
        if not self._jobs.exists(user_id=user_id, job_id=job_id):
            return None
        job = asyncio.run(self._job_agent.run(job_posting))
        summary = _job_summary(job)
        self._jobs.update(
            user_id=user_id, job_id=job_id, raw_posting=job_posting, job=job, summary=summary
        )
        return self._jobs.get(user_id=user_id, job_id=job_id)

    def delete_job(self, *, user_id: str, job_id: str) -> bool:
        """Delete a job if the user owns it; return whether one was found.

        Past resumes generated for this job are not deleted — their
        ``job_id`` is set NULL (``storage/schema.py``'s ``ON DELETE SET
        NULL``), and their flattened ``job_title`` keeps them meaningful.
        """
        if not self._jobs.exists(user_id=user_id, job_id=job_id):
            return False
        self._jobs.delete(user_id=user_id, job_id=job_id)
        return True

    def link_document_to_job(self, *, user_id: str, job_id: str, document_id: str) -> bool:
        """Associate an existing document with a job; return whether both were owned.

        Ownership of *both* sides is checked before linking — a caller
        can't link a job or document they don't own, even if they happen to
        know its id.
        """
        if not self._jobs.exists(user_id=user_id, job_id=job_id):
            return False
        if not self._documents.exists(user_id=user_id, document_id=document_id):
            return False
        self._job_documents.link(job_id=job_id, document_id=document_id)
        return True

    def upload_document_for_job(
        self,
        *,
        user_id: str,
        job_id: str,
        filename: str,
        data: bytes,
        document_type: DocumentType = DocumentType.OTHER,
    ) -> str | None:
        """Ingest an upload and link it to a job in one step.

        ``None`` if ``job_id`` isn't owned by this user — checked before
        ingestion runs, so an upload against someone else's (or a
        nonexistent) job never touches either store.
        """
        if not self._jobs.exists(user_id=user_id, job_id=job_id):
            return None
        return self._ingestion.ingest_bytes(
            user_id=user_id,
            filename=filename,
            data=data,
            document_type=document_type,
            job_id=job_id,
        )

    def list_documents_for_job(
        self, *, user_id: str, job_id: str
    ) -> list[dict[str, object]] | None:
        """Documents eligible for this job (generic pool + job-linked), as plain dicts.

        ``None`` if ``job_id`` isn't owned by this user. Mirrors
        ``list_documents``'s dict-of-columns shape for the API layer, scoped
        by :meth:`DocumentRepository.list_eligible_document_ids` rather than
        every document the user owns.
        """
        if not self._jobs.exists(user_id=user_id, job_id=job_id):
            return None
        eligible_ids = set(
            self._documents.list_eligible_document_ids(user_id=user_id, job_id=job_id)
        )
        return [
            dict(row) for row in self._documents.list_for_user(user_id) if row["id"] in eligible_ids
        ]

    # --- retrieval (ad-hoc, outside a generation run) -----------------------

    def search_evidence(self, *, user_id: str, query: str, limit: int = 5) -> EvidenceSet:
        """Hybrid (vector + keyword) search over a user's own ingested evidence.

        The same hybrid search a generation's Evidence Retrieval step runs
        internally, exposed standalone so a caller (the UI's "search" tab) can
        inspect retrieval quality directly instead of only seeing its effect
        buried inside a full ``generate()`` run. Still user-scoped (ADR-0001).
        """
        return self._retriever.search(user_id=user_id, query=query, limit=limit)

    # --- generation --------------------------------------------------------

    def generate(self, *, user_id: str, job_posting: str) -> WorkflowState:
        """Run the full pipeline and persist the final draft + its scores.

        Returns the whole :class:`WorkflowState` so the caller can surface the
        real Trust/ATS scores and the pass/fail — including for a capped-out
        draft (ADR-0005).
        """
        state = asyncio.run(self._orchestrator.run(user_id=user_id, job_posting=job_posting))
        self._persist(state)
        return state

    def generate_for_job(self, *, user_id: str, job_id: str) -> WorkflowState | None:
        """Generate against a persisted job; return the result, or ``None`` if unowned.

        Re-uses the job's already-extracted ``JobDescription`` (no re-run of
        the Job Description agent — see ``Orchestrator._analyze_job``) and
        scopes retrieval to that job's eligible documents (generic pool +
        job-linked, resolved once here via
        ``DocumentRepository.list_eligible_document_ids`` rather than inside
        the orchestrator, which has no ``DocumentRepository`` dependency of
        its own).
        """
        row = self._jobs.get(user_id=user_id, job_id=job_id)
        if row is None:
            return None
        job = JobDescription.model_validate_json(row["job_description_json"])
        document_ids = self._documents.list_eligible_document_ids(user_id=user_id, job_id=job_id)
        state = asyncio.run(
            self._orchestrator.run(
                user_id=user_id, job=job, job_id=job_id, document_ids=document_ids
            )
        )
        self._persist(state)
        return state

    def get_resume(self, *, user_id: str, resume_id: str) -> sqlite3.Row | None:
        """The stored resume row for this user + id, or ``None`` if not found/owned."""
        return self._resumes.get_row(user_id=user_id, resume_id=resume_id)

    def list_resumes_for_job(self, *, user_id: str, job_id: str) -> list[sqlite3.Row] | None:
        """Every resume generated for this job, newest first, or ``None`` if unowned."""
        if not self._jobs.exists(user_id=user_id, job_id=job_id):
            return None
        return self._resumes.list_for_job(user_id=user_id, job_id=job_id)

    def _persist(self, state: WorkflowState) -> None:
        """Save the final draft, its exports, and its evaluation to SQLite.

        Every persisted resume gets rendered PDF/Markdown bytes unconditionally
        — including a run through the legacy raw-``job_posting`` ``generate()``
        path, which previously had no export data to persist at all. A draft
        that didn't pass the quality gate additionally gets a rejection reason
        (:func:`build_rejection_reason`) and improvement suggestions (the same
        ``build_feedback`` output the orchestrator would have used for one more
        rewrite, had the iteration cap not already been hit) — both ``None``
        for a passing draft. Sets ``state.resume_id`` on success so the caller
        can deep-link to the export routes without a second lookup.
        """
        draft = state.current_draft
        trust = state.current_trust
        ats = state.current_ats
        if draft is None or trust is None or ats is None:
            return

        rejection_reason: str | None = None
        improvement_suggestions: str | None = None
        if not state.passed:
            rejection_reason = build_rejection_reason(state.gate, trust, ats)
            improvement_suggestions = build_feedback(trust, ats)

        resume_id = self._resumes.create(
            user_id=state.user_id,
            draft=draft,
            job_title=state.job.title if state.job else None,
            trust_score=trust.score,
            ats_score=ats.score,
            passed=state.passed,
            job_id=state.job_id,
            pdf_bytes=render_pdf(draft),
            markdown_text=render_markdown(draft),
            rejection_reason=rejection_reason,
            improvement_suggestions=improvement_suggestions,
        )
        self._evaluations.create(user_id=state.user_id, resume_id=resume_id, trust=trust, ats=ats)
        state.resume_id = resume_id

    # --- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        """Release the SQLite connection and Chroma client this app owns.

        Both are handed to ``__init__`` by the caller (``build_default_app``
        for the real app, a fixture for tests), so this doesn't destroy
        anything the caller didn't already know it was giving up — it just
        makes sure a long-running process (the FastAPI server, a one-shot
        script) doesn't leak the open file handle/connection past its own
        lifetime.
        """
        self._connection.close()
        self._chroma_client.close()

    def __enter__(self) -> TrustResumeApp:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def build_default_app(
    *,
    db_path: str = "trustresume.db",
    chroma_path: str = "chroma_data",
    llm_config: LLMConfig | None = None,
) -> TrustResumeApp:
    """Assemble a production app: file-backed stores + fastembed + an LLM.

    Kept out of ``TrustResumeApp.__init__`` so tests can build the app with
    in-memory stores and fakes without triggering model downloads or provider
    calls.

    ``llm_config`` selects the provider backing the LLM agents (Bedrock by
    default; also OpenAI, Google/Gemini, or an offline ``test`` model). See
    :class:`trustresume.api.model_factory.LLMConfig`. When ``None``, config is
    read from the environment.
    """
    import chromadb

    from trustresume.retrieval import FastEmbedEmbeddings

    config = llm_config or LLMConfig.from_env()
    return TrustResumeApp(
        connection=connect(db_path),
        chroma_client=chromadb.PersistentClient(path=chroma_path),
        embedder=FastEmbedEmbeddings(),
        model=build_model(config),
    )
