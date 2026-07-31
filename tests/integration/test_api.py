"""Integration tests for the FastAPI backend (M7).

Drives the real HTTP layer via ``TestClient`` over a fully in-memory facade
(fake embedder + a scripted chat model), so the endpoints, request/response
schemas, and the facade wiring are all exercised without a browser, network,
or AWS.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import chromadb
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from tests.fakes import FakeEmbeddings, FakeToolCallingChatModel
from trustresume.api import TrustResumeApp, create_app
from trustresume.storage import connect

from .test_app_service import (
    _CANDIDATE_PROFILE_CALL,
    _FULL_GENERATION,
    _JOB_DESCRIPTION_CALL,
    _RESUME_DRAFT_CALL,
    _TRUST_CALL,
)


@pytest.fixture
def client() -> TestClient:
    facade = TrustResumeApp(
        connection=connect(":memory:"),
        chroma_client=chromadb.EphemeralClient(),
        embedder=FakeEmbeddings(),
        model=FakeToolCallingChatModel(messages=iter(_FULL_GENERATION)),
        # Unique collection name per test — see the note on
        # TrustResumeApp.__init__'s chroma_collection_name parameter.
        chroma_collection_name=f"test-{uuid.uuid4().hex}",
    )
    return TestClient(create_app(facade))


def test_lifespan_closesFacadeOnShutdown() -> None:
    """FastAPI only runs lifespan startup/shutdown when the TestClient is used
    as a context manager — the ``client`` fixture above doesn't, so this needs
    its own facade + client to actually exercise ``create_app``'s shutdown
    hook (``TrustResumeApp.close()``).
    """
    import sqlite3

    facade = TrustResumeApp(
        connection=connect(":memory:"),
        chroma_client=chromadb.EphemeralClient(),
        embedder=FakeEmbeddings(),
        model=FakeToolCallingChatModel(messages=iter(_FULL_GENERATION)),
        chroma_collection_name=f"test-{uuid.uuid4().hex}",
    )

    with TestClient(create_app(facade)) as test_client:
        resp = test_client.get("/api/health")
        assert resp.status_code == 200

    with pytest.raises(sqlite3.ProgrammingError):
        facade._connection.execute("SELECT 1")


def test_health(client: TestClient) -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_root_pointsAtDocs(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["docs"] == "/docs"
    assert body["openapi"] == "/openapi.json"


def test_openapi_schemaServed(client: TestClient) -> None:
    # FastAPI auto-serves this; it's what Postman imports.
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    paths = resp.json()["paths"]
    assert "/api/generate" in paths
    assert "/api/documents" in paths


def test_documents_addListRoundTrip(client: TestClient) -> None:
    assert client.get("/api/documents").json() == []

    resp = client.post(
        "/api/documents",
        json={"filename": "resume.txt", "text": "Python and AWS work.", "document_type": "RESUME"},
    )
    assert resp.status_code == 201
    assert resp.json()["filename"] == "resume.txt"

    listed = client.get("/api/documents").json()
    assert len(listed) == 1
    assert listed[0]["document_type"] == "RESUME"


def test_addDocument_validationError(client: TestClient) -> None:
    # Empty text violates the schema's min_length.
    resp = client.post("/api/documents", json={"filename": "x", "text": ""})
    assert resp.status_code == 422


def test_addDocument_whitespaceOnlyText_rejectedLikeEmpty(client: TestClient) -> None:
    # A whitespace-only body isn't "empty" by min_length's raw check, but it's
    # exactly as useless — must be rejected too, not create a zero-chunk document.
    resp = client.post("/api/documents", json={"filename": "x", "text": "   \n\t  "})
    assert resp.status_code == 422


def test_uploadDocument_parsesFileServerSide(client: TestClient) -> None:
    resp = client.post(
        "/api/documents/upload",
        files={"file": ("resume.txt", b"Built Python services on AWS.", "text/plain")},
        data={"document_type": "RESUME"},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["filename"] == "resume.txt"
    assert body["document_type"] == "RESUME"

    listed = client.get("/api/documents").json()
    assert len(listed) == 1


def test_uploadDocument_unsupportedType_returns422(client: TestClient) -> None:
    resp = client.post(
        "/api/documents/upload",
        files={"file": ("image.png", b"\x89PNG", "image/png")},
    )
    assert resp.status_code == 422


def test_uploadDocument_emptyFile_returns422(client: TestClient) -> None:
    resp = client.post(
        "/api/documents/upload",
        files={"file": ("resume.txt", b"", "text/plain")},
    )
    assert resp.status_code == 422


def test_uploadDocument_overSizeLimit_returns413(client: TestClient) -> None:
    from trustresume.api.server import _MAX_UPLOAD_BYTES

    oversized = b"x" * (_MAX_UPLOAD_BYTES + 1)
    resp = client.post(
        "/api/documents/upload",
        files={"file": ("resume.txt", oversized, "text/plain")},
    )
    assert resp.status_code == 413


def test_deleteDocument_removesIt(client: TestClient) -> None:
    resp = client.post(
        "/api/documents",
        json={"filename": "resume.txt", "text": "Python and AWS work."},
    )
    doc_id = resp.json()["id"]

    del_resp = client.delete(f"/api/documents/{doc_id}")
    assert del_resp.status_code == 204
    assert client.get("/api/documents").json() == []


def test_deleteDocument_unknownId_returns404(client: TestClient) -> None:
    resp = client.delete("/api/documents/does-not-exist")
    assert resp.status_code == 404


def test_search_returnsRankedEvidence(client: TestClient) -> None:
    client.post(
        "/api/documents",
        json={"filename": "r.txt", "text": "Built Python services on AWS."},
    )
    resp = client.post("/api/search", json={"query": "Python AWS", "limit": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "Python AWS"
    assert len(body["chunks"]) == 1
    assert "Python" in body["chunks"][0]["text"]


def test_search_requiresNonEmptyQuery(client: TestClient) -> None:
    resp = client.post("/api/search", json={"query": ""})
    assert resp.status_code == 422


def test_search_whitespaceOnlyQuery_rejectedLikeEmpty(client: TestClient) -> None:
    resp = client.post("/api/search", json={"query": "   "})
    assert resp.status_code == 422


def test_generate_returnsScoresAndDraft(client: TestClient) -> None:
    client.post(
        "/api/documents",
        json={
            "filename": "r.txt",
            "text": "Built Python services on AWS.",
            "document_type": "RESUME",
        },
    )
    resp = client.post("/api/generate", json={"job_posting": "Senior Python Engineer"})
    assert resp.status_code == 200

    body = resp.json()
    assert "draft" in body
    assert 0 <= body["trust_score"] <= 100
    assert 0 <= body["ats_score"] <= 100
    assert isinstance(body["passed"], bool)
    assert isinstance(body["hallucinations"], list)


def test_generate_requiresJobPosting(client: TestClient) -> None:
    resp = client.post("/api/generate", json={"job_posting": ""})
    assert resp.status_code == 422


def test_generate_whitespaceOnlyJobPosting_rejectedLikeEmpty(client: TestClient) -> None:
    resp = client.post("/api/generate", json={"job_posting": "  \n  "})
    assert resp.status_code == 422


def test_generate_noScoredDraftProduced_returns500(monkeypatch: pytest.MonkeyPatch) -> None:
    """A workflow that produces no scored draft is a facade-layer edge case
    (not reachable through the real orchestrator/gate in a normal run) — stub
    ``generate`` on a fresh facade to return an empty state, so the route's
    ``ValueError`` -> HTTP 500 branch is exercised directly.
    """
    from trustresume.models import WorkflowState

    facade = TrustResumeApp(
        connection=connect(":memory:"),
        chroma_client=chromadb.EphemeralClient(),
        embedder=FakeEmbeddings(),
        model=FakeToolCallingChatModel(messages=iter(_FULL_GENERATION)),
        chroma_collection_name=f"test-{uuid.uuid4().hex}",
    )
    monkeypatch.setattr(
        facade, "generate", lambda *, user_id, job_posting: WorkflowState(user_id=user_id)
    )
    test_client = TestClient(create_app(facade))

    resp = test_client.post("/api/generate", json={"job_posting": "Some job"})
    assert resp.status_code == 500
    assert "no scored draft" in resp.json()["detail"]


def test_ping_relaysLlmResult(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # Stub the smoke test so the endpoint's success path runs without the
    # providers extra/creds.
    import sys
    import types

    # Pin the provider via env (highest precedence in LLMConfig.load) so the
    # assertion doesn't depend on ambient config/llm*.json overlays.
    monkeypatch.setenv("TRUSTRESUME_LLM_PROVIDER", "bedrock")

    fake_module = types.ModuleType("trustresume.poc.llm_smoke_test")
    fake_module.run_smoke_test = lambda **_: "Paris ~18C, Tokyo ~22C."  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "trustresume.poc.llm_smoke_test", fake_module)

    resp = client.get("/api/ping")
    assert resp.status_code == 200
    body = resp.json()
    assert body["backend"] == "bedrock"
    assert "Tokyo" in body["response"]


def test_ping_smokeTestModuleMissing_returns501(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Setting a sys.modules entry to None is the documented way to force an
    # ImportError on lookup — simulates the `providers` extra not being
    # installed, without actually uninstalling anything.
    import sys

    monkeypatch.setitem(sys.modules, "trustresume.poc.llm_smoke_test", None)

    resp = client.get("/api/ping")
    assert resp.status_code == 501
    assert "providers" in resp.json()["detail"]


def test_ping_llmFailure_returns502(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    def _boom(**_: object) -> str:
        raise RuntimeError("no credentials")

    fake_module = types.ModuleType("trustresume.poc.llm_smoke_test")
    fake_module.run_smoke_test = _boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "trustresume.poc.llm_smoke_test", fake_module)

    resp = client.get("/api/ping")
    assert resp.status_code == 502
    assert "LLM call failed" in resp.json()["detail"]


def test_buildServedApp_wiresRealAppFromEnv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``build_served_app`` — the ``--factory`` uvicorn actually serves — end to
    end with the offline ``test`` provider and file-backed stores under
    ``tmp_path``, so this exercises the real wiring (env parsing,
    ``build_default_app``, ``configure_logging``) that every other test in
    this module bypasses by injecting its own in-memory facade.
    """
    from trustresume.api.server import build_served_app

    monkeypatch.setenv("TRUSTRESUME_LLM_PROVIDER", "test")
    monkeypatch.setenv("TRUSTRESUME_DB_PATH", str(tmp_path / "served.db"))
    monkeypatch.setenv("TRUSTRESUME_CHROMA_PATH", str(tmp_path / "served_chroma"))

    app = build_served_app()
    client = TestClient(app)

    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# --- job CRUD, job-scoped documents/generation, resume export routes ------


def _client_with_scripted_calls(*call_groups: list[AIMessage]) -> TestClient:
    messages = [msg for group in call_groups for msg in group]
    facade = TrustResumeApp(
        connection=connect(":memory:"),
        chroma_client=chromadb.EphemeralClient(),
        embedder=FakeEmbeddings(),
        model=FakeToolCallingChatModel(messages=iter(messages)),
        chroma_collection_name=f"test-{uuid.uuid4().hex}",
    )
    return TestClient(create_app(facade))


def test_createJob_listJobs_getJob() -> None:
    client = _client_with_scripted_calls([_JOB_DESCRIPTION_CALL])

    create_resp = client.post("/api/jobs", json={"job_posting": "Senior Python Engineer role"})
    assert create_resp.status_code == 201
    job = create_resp.json()
    assert job["title"] == "Senior Python Engineer"

    list_resp = client.get("/api/jobs")
    assert list_resp.status_code == 200
    assert len(list_resp.json()) == 1

    get_resp = client.get(f"/api/jobs/{job['id']}")
    assert get_resp.status_code == 200
    assert get_resp.json()["keywords"] == ["python", "aws"]

    assert client.get("/api/jobs/nonexistent").status_code == 404


def test_updateJob_putReExtracts_missingReturns404() -> None:
    client = _client_with_scripted_calls([_JOB_DESCRIPTION_CALL, _JOB_DESCRIPTION_CALL])
    job = client.post("/api/jobs", json={"job_posting": "Senior Python Engineer role"}).json()

    resp = client.put(f"/api/jobs/{job['id']}", json={"job_posting": "Staff Python Engineer role"})
    assert resp.status_code == 200
    assert resp.json()["title"] == "Senior Python Engineer"  # scripted extraction is fixed

    assert client.put("/api/jobs/nonexistent", json={"job_posting": "x"}).status_code == 404


def test_deleteJob_removesIt_missingReturns404() -> None:
    client = _client_with_scripted_calls([_JOB_DESCRIPTION_CALL])
    job = client.post("/api/jobs", json={"job_posting": "role"}).json()

    assert client.delete(f"/api/jobs/{job['id']}").status_code == 204
    assert client.get(f"/api/jobs/{job['id']}").status_code == 404
    assert client.delete(f"/api/jobs/{job['id']}").status_code == 404


def test_uploadDocumentForJob_linksItAndListDocumentsForJobIncludesIt() -> None:
    client = _client_with_scripted_calls([_JOB_DESCRIPTION_CALL])
    job = client.post("/api/jobs", json={"job_posting": "role"}).json()

    resp = client.post(
        f"/api/jobs/{job['id']}/documents/upload",
        files={"file": ("resume.txt", b"Built Python services on AWS.", "text/plain")},
        data={"document_type": "RESUME"},
    )
    assert resp.status_code == 201

    docs_resp = client.get(f"/api/jobs/{job['id']}/documents")
    assert docs_resp.status_code == 200
    assert len(docs_resp.json()) == 1

    assert (
        client.post(
            "/api/jobs/nonexistent/documents/upload",
            files={"file": ("x.txt", b"content", "text/plain")},
        ).status_code
        == 404
    )
    assert client.get("/api/jobs/nonexistent/documents").status_code == 404


def test_generateForJob_persistsAndExposesResumeId_missingJobReturns404() -> None:
    client = _client_with_scripted_calls(
        [_JOB_DESCRIPTION_CALL], [_CANDIDATE_PROFILE_CALL, _RESUME_DRAFT_CALL, _TRUST_CALL]
    )
    job = client.post("/api/jobs", json={"job_posting": "Senior Python Engineer role"}).json()
    client.post(
        f"/api/jobs/{job['id']}/documents/upload",
        files={"file": ("resume.txt", b"Built Python services on AWS.", "text/plain")},
    )

    resp = client.post(f"/api/jobs/{job['id']}/generate")
    assert resp.status_code == 200
    body = resp.json()
    assert body["passed"] is True
    assert body["resume_id"] is not None

    assert client.post("/api/jobs/nonexistent/generate").status_code == 404


def test_listResumesForJob_getResume_downloadPdfAndMarkdown() -> None:
    client = _client_with_scripted_calls(_FULL_GENERATION)
    job = client.post("/api/jobs", json={"job_posting": "Senior Python Engineer role"}).json()
    client.post(
        f"/api/jobs/{job['id']}/documents/upload",
        files={"file": ("resume.txt", b"Built Python services on AWS.", "text/plain")},
    )
    resume_id = client.post(f"/api/jobs/{job['id']}/generate").json()["resume_id"]

    resumes_resp = client.get(f"/api/jobs/{job['id']}/resumes")
    assert resumes_resp.status_code == 200
    assert len(resumes_resp.json()) == 1
    assert client.get("/api/jobs/nonexistent/resumes").status_code == 404

    detail_resp = client.get(f"/api/resumes/{resume_id}")
    assert detail_resp.status_code == 200
    assert "draft" in detail_resp.json()
    assert client.get("/api/resumes/nonexistent").status_code == 404

    pdf_resp = client.get(f"/api/resumes/{resume_id}/pdf")
    assert pdf_resp.status_code == 200
    assert pdf_resp.headers["content-type"] == "application/pdf"
    assert pdf_resp.content[:5] == b"%PDF-"
    assert client.get("/api/resumes/nonexistent/pdf").status_code == 404

    md_resp = client.get(f"/api/resumes/{resume_id}/markdown")
    assert md_resp.status_code == 200
    assert md_resp.headers["content-type"].startswith("text/markdown")
    assert "Skills" in md_resp.text
    assert client.get("/api/resumes/nonexistent/markdown").status_code == 404


def test_deleteJob_pastResumeStillDownloadable() -> None:
    """A resume generated for a since-deleted job must remain fully servable
    (get/pdf/markdown) — job deletion only nulls job_id on the resume row, it
    never cascades to it.
    """
    client = _client_with_scripted_calls(_FULL_GENERATION)
    job = client.post("/api/jobs", json={"job_posting": "role"}).json()
    client.post(
        f"/api/jobs/{job['id']}/documents/upload",
        files={"file": ("resume.txt", b"Built Python services on AWS.", "text/plain")},
    )
    resume_id = client.post(f"/api/jobs/{job['id']}/generate").json()["resume_id"]

    assert client.delete(f"/api/jobs/{job['id']}").status_code == 204

    assert client.get(f"/api/resumes/{resume_id}").json()["job_id"] is None
    assert client.get(f"/api/resumes/{resume_id}/pdf").status_code == 200


def test_uploadDocumentForJob_emptyFile_returns422() -> None:
    client = _client_with_scripted_calls([_JOB_DESCRIPTION_CALL])
    job = client.post("/api/jobs", json={"job_posting": "role"}).json()

    resp = client.post(
        f"/api/jobs/{job['id']}/documents/upload",
        files={"file": ("resume.txt", b"", "text/plain")},
    )
    assert resp.status_code == 422


def test_uploadDocumentForJob_overSizeLimit_returns413() -> None:
    from trustresume.api.server import _MAX_UPLOAD_BYTES

    client = _client_with_scripted_calls([_JOB_DESCRIPTION_CALL])
    job = client.post("/api/jobs", json={"job_posting": "role"}).json()

    oversized = b"x" * (_MAX_UPLOAD_BYTES + 1)
    resp = client.post(
        f"/api/jobs/{job['id']}/documents/upload",
        files={"file": ("resume.txt", oversized, "text/plain")},
    )
    assert resp.status_code == 413


def test_uploadDocumentForJob_unsupportedType_returns422() -> None:
    client = _client_with_scripted_calls([_JOB_DESCRIPTION_CALL])
    job = client.post("/api/jobs", json={"job_posting": "role"}).json()

    resp = client.post(
        f"/api/jobs/{job['id']}/documents/upload",
        files={"file": ("image.png", b"\x89PNG", "image/png")},
    )
    assert resp.status_code == 422


def test_generateForJob_noScoredDraftProduced_returns500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same facade-layer edge case as ``test_generate_noScoredDraftProduced_returns500``,
    for the job-scoped generation route.
    """
    from trustresume.models import WorkflowState

    facade = TrustResumeApp(
        connection=connect(":memory:"),
        chroma_client=chromadb.EphemeralClient(),
        embedder=FakeEmbeddings(),
        model=FakeToolCallingChatModel(messages=iter([_JOB_DESCRIPTION_CALL])),
        chroma_collection_name=f"test-{uuid.uuid4().hex}",
    )
    test_client = TestClient(create_app(facade))
    job_id = test_client.post("/api/jobs", json={"job_posting": "role"}).json()["id"]
    monkeypatch.setattr(
        facade, "generate_for_job", lambda *, user_id, job_id: WorkflowState(user_id=user_id)
    )

    resp = test_client.post(f"/api/jobs/{job_id}/generate")
    assert resp.status_code == 500
    assert "no scored draft" in resp.json()["detail"]
