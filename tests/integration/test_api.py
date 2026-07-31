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

from tests.fakes import FakeEmbeddings, FakeToolCallingChatModel
from trustresume.api import TrustResumeApp, create_app
from trustresume.storage import connect

from .test_app_service import _FULL_GENERATION


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
