"""FastAPI backend.

Thin HTTP layer over :class:`trustresume.api.app_service.TrustResumeApp` — it
translates requests into facade calls and facade results into
``schemas`` responses, and owns no business logic. ``create_app`` takes the
facade by injection so tests drive it through ``TestClient`` with an in-memory,
fake-backed app; ``build_served_app`` is the ``--factory`` uvicorn serves.

Run: ``uvicorn trustresume.api.server:build_served_app --factory --reload``
(the ``--factory`` form defers building the real stores/LLM model until
uvicorn starts, so importing this module stays side-effect-free for tests).

Milestone M7 (API).
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware

from trustresume.ingestion import UnsupportedDocumentError
from trustresume.models import DocumentType, JobDescription, ResumeDraft

from .app_service import NoEvidenceError, TrustResumeApp
from .schemas import (
    AddDocumentRequest,
    CreateJobRequest,
    DocumentSummary,
    GenerateForJobRequest,
    GenerateRequest,
    GenerateResponse,
    JobDetail,
    JobSummary,
    ResumeDetail,
    ResumeSummary,
    SearchRequest,
    SearchResponse,
    UsageView,
)

logger = logging.getLogger(__name__)

# Fallback identity for a caller that sends no ``X-User-Id`` header — keeps
# the walk-up-and-try-it flow (and every existing client) working unchanged.
# Every store call is user-scoped (ADR-0001); ``resolve_user`` below is the
# single place that decides *which* user a request belongs to.
DEMO_USER_ID = "demo-user"
USER_ID_HEADER = "X-User-Id"
# Conservative on purpose: these ids become SQLite primary keys and Chroma
# metadata filter values, and they show up in logs. Queries are parameterized
# (so this isn't an injection defense) — it's about keeping identifiers
# bounded and readable rather than accepting arbitrary header bytes.
_USER_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,64}")

# A résumé/job-posting upload has no legitimate reason to approach this;
# caps memory use per upload instead of buffering an unbounded body.
_MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def _job_summary_response(row: Any) -> JobSummary:
    return JobSummary(
        id=str(row["id"]),
        title=row["title"],
        company=row["company"],
        summary=row["summary"],
        created_at=str(row["created_at"]),
    )


def _job_detail_response(row: Any) -> JobDetail:
    job = JobDescription.model_validate_json(row["job_description_json"])
    return JobDetail(
        id=str(row["id"]),
        title=row["title"],
        company=row["company"],
        summary=row["summary"],
        created_at=str(row["created_at"]),
        raw_posting=str(row["raw_posting"]),
        seniority=job.seniority.value,
        required_skills=job.required_skills,
        preferred_skills=job.preferred_skills,
        responsibilities=job.responsibilities,
        keywords=job.keywords,
    )


def _resume_summary_response(row: Any) -> ResumeSummary:
    return ResumeSummary(
        id=str(row["id"]),
        job_id=row["job_id"],
        job_title=row["job_title"],
        iteration=int(row["iteration"]),
        trust_score=float(row["trust_score"]),
        ats_score=float(row["ats_score"]),
        passed=bool(row["passed"]),
        created_at=str(row["created_at"]),
    )


def _resume_detail_response(row: Any) -> ResumeDetail:
    draft = ResumeDraft.model_validate_json(row["content_json"])
    return ResumeDetail(
        id=str(row["id"]),
        job_id=row["job_id"],
        job_title=row["job_title"],
        iteration=int(row["iteration"]),
        trust_score=float(row["trust_score"]),
        ats_score=float(row["ats_score"]),
        passed=bool(row["passed"]),
        created_at=str(row["created_at"]),
        draft=draft,
        rejection_reason=row["rejection_reason"],
        improvement_suggestions=row["improvement_suggestions"],
        usage=_usage_view(row),
    )


def _usage_view(row: Any) -> UsageView | None:
    """Rebuild the usage view from a resume row's flattened columns.

    ``None`` when the run wasn't measured (a row written before telemetry
    existed, or by a caller that passed no usage) — distinguished from a
    measured-but-free run by checking for NULL rather than for zero.
    """
    if row["llm_calls"] is None:
        return None
    input_tokens = int(row["input_tokens"] or 0)
    output_tokens = int(row["output_tokens"] or 0)
    return UsageView(
        llm_calls=int(row["llm_calls"]),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        duration_ms=round(float(row["duration_ms"] or 0.0), 1),
        cost_usd=row["cost_usd"],
    )


def resolve_user(
    request: Request,
    x_user_id: Annotated[str | None, Header(alias=USER_ID_HEADER)] = None,
) -> str:
    """The caller's user id, from ``X-User-Id``, defaulting to the demo user.

    This is *identity, not authentication* — the header is trusted as sent,
    which suits a portfolio/demo deployment and is not a login system. What it
    buys is that ADR-0001's per-user isolation becomes exercisable end to end:
    two callers with different ids genuinely cannot see each other's
    documents, jobs, or resumes, and there's a test that proves it. Making it
    real authentication means replacing this one function with a
    token-verifying dependency — every store call underneath is already
    user-scoped.

    Unknown ids are created on first use, so a client picks its own id with no
    signup step. Ids are validated rather than trusted blindly: a stray or
    hostile header shouldn't be able to create rows keyed by megabyte-long or
    control-character-laden strings.

    Defined at module level, and reaching the facade through
    ``request.app.state``, because ``from __future__ import annotations``
    makes every annotation a string: a dependency nested inside ``create_app``
    would leave ``CurrentUser`` unresolvable from module globals, and FastAPI
    silently degrades that to "unknown query parameter" — every route then
    422s. Caught only by running the app, not by mypy or ruff.
    """
    if x_user_id is None or not x_user_id.strip():
        return DEMO_USER_ID
    user_id = x_user_id.strip()
    if not _USER_ID_PATTERN.fullmatch(user_id):
        raise HTTPException(
            status_code=400,
            detail=f"invalid {USER_ID_HEADER}: expected 1-64 characters matching [A-Za-z0-9._-]",
        )
    facade: TrustResumeApp = request.app.state.facade
    facade.ensure_user(user_id, user_id=user_id)
    return user_id


#: The resolved caller identity, injected into every user-scoped route.
CurrentUser = Annotated[str, Depends(resolve_user)]


def create_app(app_facade: TrustResumeApp) -> FastAPI:
    """Build the FastAPI application around an injected facade."""

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Release the facade's SQLite connection/Chroma client on server
        # stop. Tests build a fresh in-memory facade per test via
        # TestClient (which drives this same lifespan) and don't otherwise
        # call close(); ``build_served_app`` builds one real, file-backed
        # facade for uvicorn's whole process lifetime — this is what stops
        # it outliving the server.
        yield
        app_facade.close()

    api = FastAPI(title="TrustResume API", version="0.1.0", lifespan=lifespan)

    # A local frontend, if one is added later, would run on a different
    # origin; allow it to call us.
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:3000"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ``resolve_user`` reads the facade off app state rather than closing over
    # ``app_facade`` — see the note on that function for why it can't be a
    # closure here.
    api.state.facade = app_facade
    app_facade.ensure_user("Demo User", user_id=DEMO_USER_ID)

    @api.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        """Point visitors at the interactive docs."""
        return {
            "service": "TrustResume API",
            "docs": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json",
            "health": "/api/health",
            "ping": "/api/ping (live LLM probe for the configured provider; needs the poc extra)",
        }

    @api.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.get("/api/ping")
    def ping() -> dict[str, str]:
        """Live connectivity probe for whichever LLM provider is configured.

        Invokes the standalone smoke test (``poc/llm_smoke_test.py``) — a real
        round-trip through the provider resolved by
        :meth:`~trustresume.api.model_factory.LLMConfig.from_env` (Bedrock,
        OpenAI, Google, or ``test``) — and relays its answer. Useful for
        confirming from a client (Postman/curl) that the configured provider,
        credentials, and LangChain are wired up end to end.

        Requires the optional ``providers`` extra (for non-Bedrock providers)
        *and* valid credentials for the configured provider; neither is
        needed to run the rest of the API, so failures are reported as HTTP
        errors rather than crashing anything:

        - ``501`` if the required provider SDK isn't installed.
        - ``502`` if the LLM call itself fails (creds/region/model).
        """
        from .model_factory import LLMConfig

        try:
            from trustresume.poc.llm_smoke_test import run_smoke_test
        except ImportError as exc:
            raise HTTPException(
                status_code=501,
                detail=(
                    "LLM smoke test unavailable — install the optional extra with "
                    f'`pip install -e ".[providers]"`. ({exc})'
                ),
            ) from exc

        config = LLMConfig.from_env()
        try:
            answer = run_smoke_test(config=config)
        except Exception as exc:  # provider/creds/region/model failures
            raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}") from exc

        return {"status": "ok", "backend": config.provider, "response": answer}

    @api.get("/api/documents", response_model=list[DocumentSummary])
    def list_documents(user_id: CurrentUser) -> list[DocumentSummary]:
        return [
            DocumentSummary(
                id=str(d["id"]),
                filename=str(d["filename"]),
                document_type=str(d["document_type"]),
            )
            for d in app_facade.list_documents(user_id)
        ]

    @api.post("/api/documents", response_model=DocumentSummary, status_code=201)
    def add_document(req: AddDocumentRequest, user_id: CurrentUser) -> DocumentSummary:
        doc_id = app_facade.add_document(
            user_id=user_id,
            filename=req.filename,
            text=req.text,
            document_type=req.document_type,
        )
        return DocumentSummary(
            id=doc_id, filename=req.filename, document_type=req.document_type.value
        )

    @api.post("/api/documents/upload", response_model=DocumentSummary, status_code=201)
    async def upload_document(
        user_id: CurrentUser,
        file: UploadFile = File(...),  # noqa: B008 — FastAPI's DI idiom for upload params
        document_type: DocumentType = Form(DocumentType.OTHER),  # noqa: B008
    ) -> DocumentSummary:
        """Ingest a raw file upload (.txt/.md/.docx) — no client-side parsing needed.

        Complements ``POST /api/documents`` (which takes already-extracted
        text): this route reads the upload's bytes and parses them
        server-side, so a plain file-picker UI (the Streamlit frontend) can
        hand over the file exactly as the user selected it.
        """
        data = await file.read(_MAX_UPLOAD_BYTES + 1)
        if not data:
            raise HTTPException(status_code=422, detail="uploaded file is empty")
        if len(data) > _MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"uploaded file exceeds the {_MAX_UPLOAD_BYTES} byte limit",
            )
        try:
            doc_id = app_facade.add_document_bytes(
                user_id=user_id,
                filename=file.filename or "upload",
                data=data,
                document_type=document_type,
            )
        except UnsupportedDocumentError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return DocumentSummary(
            id=doc_id, filename=file.filename or "upload", document_type=document_type.value
        )

    @api.delete("/api/documents/{document_id}", status_code=204)
    def delete_document(document_id: str, user_id: CurrentUser) -> None:
        """Remove one of the demo user's documents from both stores.

        404s if ``document_id`` doesn't exist or belongs to a different user
        — ``TrustResumeApp.delete_document`` checks ownership before deleting.
        """
        found = app_facade.delete_document(user_id=user_id, document_id=document_id)
        if not found:
            raise HTTPException(status_code=404, detail="document not found")

    @api.post("/api/jobs", response_model=JobSummary, status_code=201)
    def create_job(req: CreateJobRequest, user_id: CurrentUser) -> JobSummary:
        """Persist a job posting, extracting its structured fields now.

        Complements the legacy ``POST /api/generate`` (a raw posting string,
        never persisted): a created job gets an id that can be listed,
        inspected, re-used for scoped document uploads, and generated
        against repeatedly without re-extracting each time.
        """
        row = app_facade.create_job(user_id=user_id, job_posting=req.job_posting)
        return _job_summary_response(row)

    @api.get("/api/jobs", response_model=list[JobSummary])
    def list_jobs(user_id: CurrentUser) -> list[JobSummary]:
        return [_job_summary_response(row) for row in app_facade.list_jobs(user_id)]

    @api.get("/api/jobs/{job_id}", response_model=JobDetail)
    def get_job(job_id: str, user_id: CurrentUser) -> JobDetail:
        row = app_facade.get_job(user_id=user_id, job_id=job_id)
        if row is None:
            raise HTTPException(status_code=404, detail="job not found")
        return _job_detail_response(row)

    @api.put("/api/jobs/{job_id}", response_model=JobDetail)
    def update_job(job_id: str, req: CreateJobRequest, user_id: CurrentUser) -> JobDetail:
        """Replace a job's posting text and re-extract it."""
        row = app_facade.update_job(user_id=user_id, job_id=job_id, job_posting=req.job_posting)
        if row is None:
            raise HTTPException(status_code=404, detail="job not found")
        return _job_detail_response(row)

    @api.delete("/api/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str, user_id: CurrentUser) -> None:
        """Delete a job. Past resumes generated for it are kept (job_id -> NULL)."""
        found = app_facade.delete_job(user_id=user_id, job_id=job_id)
        if not found:
            raise HTTPException(status_code=404, detail="job not found")

    @api.post(
        "/api/jobs/{job_id}/documents/upload", response_model=DocumentSummary, status_code=201
    )
    async def upload_document_for_job(
        job_id: str,
        user_id: CurrentUser,
        file: UploadFile = File(...),  # noqa: B008
        document_type: DocumentType = Form(DocumentType.OTHER),  # noqa: B008
    ) -> DocumentSummary:
        """Upload a document and link it to a job in one step.

        Same size/parse-error handling as ``POST /api/documents/upload``;
        additionally 404s if ``job_id`` doesn't exist or isn't owned by the
        demo user, checked before any parsing/ingestion happens.
        """
        data = await file.read(_MAX_UPLOAD_BYTES + 1)
        if not data:
            raise HTTPException(status_code=422, detail="uploaded file is empty")
        if len(data) > _MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"uploaded file exceeds the {_MAX_UPLOAD_BYTES} byte limit",
            )
        try:
            doc_id = app_facade.upload_document_for_job(
                user_id=user_id,
                job_id=job_id,
                filename=file.filename or "upload",
                data=data,
                document_type=document_type,
            )
        except UnsupportedDocumentError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if doc_id is None:
            raise HTTPException(status_code=404, detail="job not found")
        return DocumentSummary(
            id=doc_id, filename=file.filename or "upload", document_type=document_type.value
        )

    @api.get("/api/jobs/{job_id}/documents", response_model=list[DocumentSummary])
    def list_documents_for_job(job_id: str, user_id: CurrentUser) -> list[DocumentSummary]:
        """Documents eligible for this job: the demo user's generic (unlinked)
        pool, plus any documents explicitly linked to this job."""
        docs = app_facade.list_documents_for_job(user_id=user_id, job_id=job_id)
        if docs is None:
            raise HTTPException(status_code=404, detail="job not found")
        return [
            DocumentSummary(
                id=str(d["id"]), filename=str(d["filename"]), document_type=str(d["document_type"])
            )
            for d in docs
        ]

    @api.post("/api/search", response_model=SearchResponse)
    def search(req: SearchRequest, user_id: CurrentUser) -> SearchResponse:
        """Ad-hoc semantic search over the demo user's own ingested evidence.

        Exposes retrieval directly (the same Chroma search a generation runs
        internally) so a caller can inspect what the RAG pipeline would
        retrieve for a query, without running a full — and much more
        expensive — ``/api/generate``.
        """
        evidence = app_facade.search_evidence(user_id=user_id, query=req.query, limit=req.limit)
        return SearchResponse.from_evidence(evidence)

    @api.post("/api/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest, user_id: CurrentUser) -> GenerateResponse:
        logger.info("generate requested", extra={"user_id": user_id})
        try:
            state = app_facade.generate(
                user_id=user_id, job_posting=req.job_posting, gate=req.to_gate()
            )
        except NoEvidenceError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            return GenerateResponse.from_state(state)
        except ValueError as exc:  # no scored draft produced
            logger.exception("generate produced no scored draft", extra={"user_id": user_id})
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @api.post("/api/jobs/{job_id}/generate", response_model=GenerateResponse)
    def generate_for_job(
        job_id: str, user_id: CurrentUser, req: GenerateForJobRequest | None = None
    ) -> GenerateResponse:
        """Generate against a persisted job, scoped to its eligible documents.

        Re-uses the job's already-extracted ``JobDescription`` — no
        re-extraction — and retrieves only from the generic pool plus
        whatever's linked to this job, not every document the demo user owns.
        ``req`` is optional — existing callers that post no body at all are
        unaffected.
        """
        logger.info("generate-for-job requested", extra={"user_id": user_id, "job_id": job_id})
        try:
            state = app_facade.generate_for_job(
                user_id=user_id, job_id=job_id, gate=req.to_gate() if req else None
            )
        except NoEvidenceError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if state is None:
            raise HTTPException(status_code=404, detail="job not found")
        try:
            return GenerateResponse.from_state(state)
        except ValueError as exc:  # no scored draft produced
            logger.exception(
                "generate-for-job produced no scored draft",
                extra={"user_id": user_id, "job_id": job_id},
            )
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @api.post("/api/runs/{run_id}/resume", response_model=GenerateResponse)
    def resume_run(run_id: str, user_id: CurrentUser) -> GenerateResponse:
        """Resume a crashed generation from its last completed node (ADR-0015).

        404 when durable execution is disabled, when no checkpoint exists for
        ``run_id``, or when the run belongs to another user — all indistinguishable
        by design, so a caller can't probe other users' run ids (ADR-0001).
        """
        logger.info("resume-run requested", extra={"user_id": user_id, "run_id": run_id})
        state = app_facade.resume_run(user_id=user_id, run_id=run_id)
        if state is None:
            raise HTTPException(status_code=404, detail="run not found")
        try:
            return GenerateResponse.from_state(state)
        except ValueError as exc:  # no scored draft produced
            logger.exception(
                "resume-run produced no scored draft",
                extra={"user_id": user_id, "run_id": run_id},
            )
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @api.get("/api/jobs/{job_id}/resumes", response_model=list[ResumeSummary])
    def list_resumes_for_job(job_id: str, user_id: CurrentUser) -> list[ResumeSummary]:
        rows = app_facade.list_resumes_for_job(user_id=user_id, job_id=job_id)
        if rows is None:
            raise HTTPException(status_code=404, detail="job not found")
        return [_resume_summary_response(row) for row in rows]

    @api.get("/api/resumes/{resume_id}", response_model=ResumeDetail)
    def get_resume(resume_id: str, user_id: CurrentUser) -> ResumeDetail:
        row = app_facade.get_resume(user_id=user_id, resume_id=resume_id)
        if row is None:
            raise HTTPException(status_code=404, detail="resume not found")
        return _resume_detail_response(row)

    @api.get("/api/resumes/{resume_id}/pdf")
    def download_resume_pdf(resume_id: str, user_id: CurrentUser) -> Response:
        row = app_facade.get_resume(user_id=user_id, resume_id=resume_id)
        if row is None or row["pdf_bytes"] is None:
            raise HTTPException(status_code=404, detail="resume not found")
        return Response(content=bytes(row["pdf_bytes"]), media_type="application/pdf")

    @api.get("/api/resumes/{resume_id}/markdown")
    def download_resume_markdown(resume_id: str, user_id: CurrentUser) -> Response:
        row = app_facade.get_resume(user_id=user_id, resume_id=resume_id)
        if row is None or row["markdown_text"] is None:
            raise HTTPException(status_code=404, detail="resume not found")
        return Response(content=str(row["markdown_text"]), media_type="text/markdown")

    return api


def build_served_app() -> FastAPI:
    """Factory for uvicorn: the production app, configured from the environment.

    Used via ``--factory`` so importing this module has no side effects — tests
    import ``create_app`` and inject their own in-memory facade instead.

    Environment variables (all optional):

    - ``TRUSTRESUME_LLM_PROVIDER`` — ``bedrock`` (default), ``openai``,
      ``google``, or ``test`` (offline, no keys). Legacy ``TRUSTRESUME_LLM``
      is still honored.
    - ``TRUSTRESUME_LLM_MODEL`` — model id/name (provider default if unset).
    - ``TRUSTRESUME_AWS_PROFILE`` / ``TRUSTRESUME_AWS_REGION`` — Bedrock only.
    - ``TRUSTRESUME_DB_PATH`` — SQLite file path (default ``trustresume.db``).
    - ``TRUSTRESUME_CHROMA_PATH`` — Chroma storage dir (default ``chroma_data``).
    - ``TRUSTRESUME_CHECKPOINT_PATH`` — SQLite file for durable execution
      (ADR-0015); unset/empty disables checkpointing (the default). Requires
      the ``durable`` extra when set.
    - Provider API keys are read from their conventional env vars
      (``OPENAI_API_KEY``, ``GOOGLE_API_KEY``) by the provider SDK.

    To run the API without any provider credentials (e.g. to explore it from
    Postman)::

        TRUSTRESUME_LLM_PROVIDER=test uvicorn \\
            trustresume.api.server:build_served_app --factory --port 8000

    To switch to OpenAI::

        TRUSTRESUME_LLM_PROVIDER=openai OPENAI_API_KEY=sk-... uvicorn \\
            trustresume.api.server:build_served_app --factory --port 8000
    """
    import os

    from trustresume.logging_config import configure_logging

    from .app_service import build_default_app
    from .model_factory import LLMConfig

    configure_logging()
    return create_app(
        build_default_app(
            db_path=os.getenv("TRUSTRESUME_DB_PATH", "trustresume.db"),
            chroma_path=os.getenv("TRUSTRESUME_CHROMA_PATH", "chroma_data"),
            llm_config=LLMConfig.from_env(),
            # Empty string disables the on-disk mirror without needing a
            # separate boolean flag.
            output_dir=os.getenv("TRUSTRESUME_OUTPUT_DIR", "output") or None,
            # Unset/empty leaves durable execution off (same or-None idiom).
            checkpoint_path=os.getenv("TRUSTRESUME_CHECKPOINT_PATH") or None,
        )
    )
