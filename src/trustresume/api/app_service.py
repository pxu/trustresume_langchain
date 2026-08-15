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
import logging
import sqlite3
import uuid
from pathlib import Path
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
from trustresume.export import render_markdown, render_pdf, write_run_artifacts
from trustresume.ingestion import IngestionService
from trustresume.models import (
    ATSReport,
    DocumentType,
    EvidenceSet,
    JobDescription,
    QualityGate,
    TrustReport,
    WorkflowState,
)
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

logger = logging.getLogger(__name__)

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
        extraction_model: BaseChatModel | None = None,
        writer_model: BaseChatModel | None = None,
        verifier_model: BaseChatModel | None = None,
        chroma_collection_name: str = COLLECTION_NAME,
        output_dir: Path | None = None,
        checkpoint_path: str | None = None,
    ) -> None:
        init_db(connection)
        self._connection = connection
        self._chroma_client = chroma_client
        # ``None`` (the default) writes nothing to disk. Deliberately *not*
        # defaulted to a real path: every test constructs this class directly,
        # and a default would have the offline suite scattering directories.
        # ``build_default_app`` is the one place that sets it.
        self._output_dir = output_dir

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

        # Agents (M4) + orchestrator (M5). Each LLM-backed agent takes the
        # model for its *role* (extraction / writer / verifier — see
        # model_factory.AGENT_ROLES), falling back to the single shared
        # ``model`` when the caller didn't tier them. One model everywhere is
        # still the default; tiering is opt-in config, not a required setup
        # step, and tests keep passing one fake.
        extraction = extraction_model or model
        writer = writer_model or model
        verifier = verifier_model or model

        # One JobDescriptionAgent instance is shared between
        # create_job/update_job (which extract at job-creation time) and the
        # orchestrator (whose _analyze_job node is a no-op whenever a
        # persisted job is passed in — see Orchestrator._analyze_job) — a
        # single source of truth, not two agents that happen to agree.
        self._job_agent = JobDescriptionAgent(extraction)
        candidate_profile_service = CandidateProfileService(
            agent=CandidateProfileAgent(extraction),
            chunks=self._chunks,
            profiles=self._candidate_profiles,
        )
        # Durable execution (ADR-0015), opt-in. ``None`` disables checkpointing
        # entirely — the default for tests and the standard install. When set,
        # a crashed generation can be resumed via :meth:`resume_run`.
        self._durable = checkpoint_path is not None
        self._orchestrator = Orchestrator(
            job_description_agent=self._job_agent,
            candidate_profile_service=candidate_profile_service,
            retrieval_agent=EvidenceRetrievalAgent(self._retriever),
            resume_agent=ResumeWriterAgent(writer),
            trust_agent=TrustHarnessAgent(verifier),
            evaluation_agent=ATSEvaluationAgent(),
            checkpoint_path=checkpoint_path,
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

    def generate(
        self, *, user_id: str, job_posting: str, gate: QualityGate | None = None
    ) -> WorkflowState:
        """Run the full pipeline and persist the final draft + its scores.

        Returns the whole :class:`WorkflowState` so the caller can surface the
        real Trust/ATS scores and the pass/fail — including for a capped-out
        draft (ADR-0005). ``gate`` overrides the orchestrator's own default
        (``QualityGate()``, cap 3) when given.
        """
        run_id = self._new_run_id()
        state = asyncio.run(
            self._orchestrator.run(
                user_id=user_id, job_posting=job_posting, run_id=run_id, gate=gate
            )
        )
        self._persist(state)
        return state

    def generate_for_job(
        self, *, user_id: str, job_id: str, gate: QualityGate | None = None
    ) -> WorkflowState | None:
        """Generate against a persisted job; return the result, or ``None`` if unowned.

        Re-uses the job's already-extracted ``JobDescription`` (no re-run of
        the Job Description agent — see ``Orchestrator._analyze_job``) and
        scopes retrieval to that job's eligible documents (generic pool +
        job-linked, resolved once here via
        ``DocumentRepository.list_eligible_document_ids`` rather than inside
        the orchestrator, which has no ``DocumentRepository`` dependency of
        its own). ``gate`` overrides the orchestrator's own default
        (``QualityGate()``, cap 3) when given.
        """
        row = self._jobs.get(user_id=user_id, job_id=job_id)
        if row is None:
            return None
        job = JobDescription.model_validate_json(row["job_description_json"])
        document_ids = self._documents.list_eligible_document_ids(user_id=user_id, job_id=job_id)
        run_id = self._new_run_id()
        state = asyncio.run(
            self._orchestrator.run(
                user_id=user_id,
                job=job,
                job_id=job_id,
                document_ids=document_ids,
                run_id=run_id,
                gate=gate,
            )
        )
        self._persist(state)
        return state

    def resume_run(self, *, user_id: str, run_id: str) -> WorkflowState | None:
        """Resume a checkpointed generation from its last completed node (ADR-0015).

        Returns ``None`` when durable execution is disabled, when no checkpoint
        exists for ``run_id``, or when the checkpointed run belongs to a
        different user — all three collapse to "not found" so a caller can't
        probe another user's run ids (ADR-0001's isolation still holds). On
        success the resumed draft is persisted exactly as a fresh generation
        would be.
        """
        if not self._durable:
            return None
        state = asyncio.run(self._orchestrator.resume(run_id=run_id))
        if state is None or state.user_id != user_id:
            return None
        self._persist(state)
        return state

    @staticmethod
    def _new_run_id() -> str:
        """A fresh LangGraph ``thread_id`` for one generation.

        Minted before the run (not derived from ``resume_id``, which only
        exists post-run in ``_persist``) so a crash-resume has a stable key to
        address the run by. Harmless when durable execution is off — the
        orchestrator ignores ``run_id`` with no checkpointer to write to.
        """
        return uuid.uuid4().hex

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

        "Final" is the best-scoring draft (``state.final_*``), not the last
        one generated — the orchestrator's quality loop always runs every
        draft up to the gate's iteration cap (no early exit on a pass), so
        ``final_*`` picks the best of them: a passing draft always beats a
        failing one, and among drafts on the same side of that line, higher
        ATS wins (see :attr:`WorkflowState.final_index`).

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
        draft = state.final_draft
        trust = state.final_trust
        ats = state.final_ats
        if draft is None or trust is None or ats is None:
            return

        rejection_reason: str | None = None
        improvement_suggestions: str | None = None
        if not state.final_passed:
            rejection_reason = build_rejection_reason(state.gate, trust, ats)
            improvement_suggestions = build_feedback(trust, ats)

        # Rendered once and reused for both the database row and the artifact
        # directory, so the file on disk is byte-identical to what the API
        # serves rather than a second, independently-rendered copy.
        pdf_bytes = render_pdf(draft)
        markdown_text = render_markdown(draft)

        resume_id = self._resumes.create(
            user_id=state.user_id,
            draft=draft,
            job_title=state.job.title if state.job else None,
            trust_score=trust.score,
            ats_score=ats.score,
            passed=state.final_passed,
            job_id=state.job_id,
            pdf_bytes=pdf_bytes,
            markdown_text=markdown_text,
            rejection_reason=rejection_reason,
            improvement_suggestions=improvement_suggestions,
            usage=state.usage,
        )
        self._evaluations.create(user_id=state.user_id, resume_id=resume_id, trust=trust, ats=ats)
        state.resume_id = resume_id

        if self._output_dir is not None:
            self._write_artifacts(
                state=state,
                resume_id=resume_id,
                trust=trust,
                ats=ats,
                markdown=markdown_text,
                pdf=pdf_bytes,
                rejection_reason=rejection_reason,
                improvement_suggestions=improvement_suggestions,
            )

    def _write_artifacts(
        self,
        *,
        state: WorkflowState,
        resume_id: str,
        trust: TrustReport,
        ats: ATSReport,
        markdown: str,
        pdf: bytes,
        rejection_reason: str | None,
        improvement_suggestions: str | None,
    ) -> None:
        """Mirror the run to a browsable directory — best effort, never fatal.

        The database write above already succeeded and is the source of truth;
        a full disk or a read-only mount must not turn a completed generation
        into an error the user sees. Logged at warning level so the failure is
        visible without being disruptive.
        """
        assert self._output_dir is not None  # guarded by the caller
        try:
            run_dir = write_run_artifacts(
                self._output_dir,
                state=state,
                resume_id=resume_id,
                trust=trust,
                ats=ats,
                markdown=markdown,
                pdf=pdf,
                rejection_reason=rejection_reason,
                improvement_suggestions=improvement_suggestions,
            )
        except OSError as exc:
            logger.warning(
                "could not write run artifacts",
                extra={"output_dir": str(self._output_dir), "error": str(exc)},
            )
            return
        logger.info("run artifacts written", extra={"run_dir": str(run_dir)})

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
    output_dir: str | None = "output",
    checkpoint_path: str | None = None,
) -> TrustResumeApp:
    """Assemble a production app: file-backed stores + fastembed + an LLM.

    Kept out of ``TrustResumeApp.__init__`` so tests can build the app with
    in-memory stores and fakes without triggering model downloads or provider
    calls.

    ``llm_config`` selects the provider backing the LLM agents (Bedrock by
    default; also OpenAI, Google/Gemini, or an offline ``test`` model). See
    :class:`trustresume.api.model_factory.LLMConfig`. When ``None``, config is
    read from the environment.

    ``output_dir`` mirrors every generation to a browsable directory tree
    (``trustresume.export.artifacts``). Pass ``None`` to disable it — the
    database still holds everything either way.

    ``checkpoint_path`` enables durable execution (ADR-0015): a SQLite file
    LangGraph writes per-node checkpoints to, so a crashed generation can be
    resumed via ``TrustResumeApp.resume_run``. ``None`` (the default) leaves
    checkpointing off; the ``durable`` extra is only needed when it is set.
    """
    import chromadb

    from trustresume.retrieval import FastEmbedEmbeddings

    config = llm_config or LLMConfig.from_env()
    extraction_model = build_model(config, role="extraction")
    return TrustResumeApp(
        connection=connect(db_path),
        chroma_client=chromadb.PersistentClient(path=chroma_path),
        embedder=FastEmbedEmbeddings(),
        # One model object per role rather than one shared instance: with no
        # role overrides configured these are three identically-configured
        # clients (cheap — a client, not a loaded model), and with overrides
        # they are genuinely different models. Building them here keeps
        # TrustResumeApp itself free of provider/config knowledge.
        #
        # ``model`` is the fallback for any role left unset; since all three
        # are set here it is never consulted, so it reuses the extraction
        # client rather than building a fourth. That fourth build wasn't free:
        # under Bedrock each one opens a boto3 Session and resolves the
        # profile's credentials, which is slow with SSO.
        model=extraction_model,
        extraction_model=extraction_model,
        writer_model=build_model(config, role="writer"),
        verifier_model=build_model(config, role="verifier"),
        output_dir=Path(output_dir) if output_dir else None,
        checkpoint_path=checkpoint_path,
    )
