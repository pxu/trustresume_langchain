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
from trustresume.ingestion import IngestionService
from trustresume.models import DocumentType, WorkflowState
from trustresume.orchestration import CandidateProfileService, Orchestrator
from trustresume.retrieval import ChromaVectorStore
from trustresume.storage import (
    CandidateProfileRepository,
    ChunkRepository,
    DocumentRepository,
    EvaluationRepository,
    ResumeRepository,
    UserRepository,
    connect,
    init_db,
)

from .model_factory import LLMConfig, build_model


class TrustResumeApp:
    """Full application: ingest documents, generate a resume, persist the result."""

    def __init__(
        self,
        *,
        connection: sqlite3.Connection,
        chroma_client: Any,  # chromadb.ClientAPI — chromadb ships no type stubs
        embedder: Embeddings,
        model: BaseChatModel,
    ) -> None:
        init_db(connection)

        # Storage (M2)
        self._users = UserRepository(connection)
        self._documents = DocumentRepository(connection)
        self._chunks = ChunkRepository(connection)
        self._candidate_profiles = CandidateProfileRepository(connection)
        self._resumes = ResumeRepository(connection)
        self._evaluations = EvaluationRepository(connection)

        # Retrieval (M2)
        self._vectors = ChromaVectorStore(chroma_client, embedder)

        # Ingestion (M3)
        self._ingestion = IngestionService(
            documents=self._documents,
            chunks=self._chunks,
            vector_store=self._vectors,
            candidate_profiles=self._candidate_profiles,
        )

        # Agents (M4) + orchestrator (M5)
        candidate_profile_service = CandidateProfileService(
            agent=CandidateProfileAgent(model),
            chunks=self._chunks,
            profiles=self._candidate_profiles,
        )
        self._orchestrator = Orchestrator(
            job_description_agent=JobDescriptionAgent(model),
            candidate_profile_service=candidate_profile_service,
            retrieval_agent=EvidenceRetrievalAgent(self._vectors),
            resume_agent=ResumeWriterAgent(model),
            trust_agent=TrustHarnessAgent(model),
            evaluation_agent=ATSEvaluationAgent(),
        )

    # --- user + document management ---------------------------------------

    def ensure_user(self, name: str, *, user_id: str) -> str:
        """Create the user if absent; return the id. Idempotent."""
        if not self._users.exists(user_id):
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

    def list_documents(self, user_id: str) -> list[dict[str, object]]:
        """Document metadata for the user, as plain dicts for the API layer."""
        return [dict(row) for row in self._documents.list_for_user(user_id)]

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

    def _persist(self, state: WorkflowState) -> None:
        """Save the final draft and its evaluation to SQLite."""
        draft = state.current_draft
        trust = state.current_trust
        ats = state.current_ats
        if draft is None or trust is None or ats is None:
            return
        resume_id = self._resumes.create(
            user_id=state.user_id,
            draft=draft,
            job_title=state.job.title if state.job else None,
            trust_score=trust.score,
            ats_score=ats.score,
            passed=state.passed,
        )
        self._evaluations.create(user_id=state.user_id, resume_id=resume_id, trust=trust, ats=ats)


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
