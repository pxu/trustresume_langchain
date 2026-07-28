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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .app_service import TrustResumeApp
from .schemas import (
    AddDocumentRequest,
    DocumentSummary,
    GenerateRequest,
    GenerateResponse,
)

# Single-user demo id (mirrors the MVP scope); every store call is still
# user-scoped (ADR-0001), so real auth slots in by replacing this.
DEMO_USER_ID = "demo-user"


def create_app(app_facade: TrustResumeApp) -> FastAPI:
    """Build the FastAPI application around an injected facade."""
    api = FastAPI(title="TrustResume API", version="0.1.0")

    # A local frontend, if one is added later, would run on a different
    # origin; allow it to call us.
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:3000"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
    def list_documents() -> list[DocumentSummary]:
        return [
            DocumentSummary(
                id=str(d["id"]),
                filename=str(d["filename"]),
                document_type=str(d["document_type"]),
            )
            for d in app_facade.list_documents(DEMO_USER_ID)
        ]

    @api.post("/api/documents", response_model=DocumentSummary, status_code=201)
    def add_document(req: AddDocumentRequest) -> DocumentSummary:
        doc_id = app_facade.add_document(
            user_id=DEMO_USER_ID,
            filename=req.filename,
            text=req.text,
            document_type=req.document_type,
        )
        return DocumentSummary(
            id=doc_id, filename=req.filename, document_type=req.document_type.value
        )

    @api.post("/api/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest) -> GenerateResponse:
        state = app_facade.generate(user_id=DEMO_USER_ID, job_posting=req.job_posting)
        try:
            return GenerateResponse.from_state(state)
        except ValueError as exc:  # no scored draft produced
            raise HTTPException(status_code=500, detail=str(exc)) from exc

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

    from .app_service import build_default_app
    from .model_factory import LLMConfig

    return create_app(
        build_default_app(
            db_path=os.getenv("TRUSTRESUME_DB_PATH", "trustresume.db"),
            chroma_path=os.getenv("TRUSTRESUME_CHROMA_PATH", "chroma_data"),
            llm_config=LLMConfig.from_env(),
        )
    )
