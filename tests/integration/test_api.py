"""Integration tests for the FastAPI backend (M7).

Drives the real HTTP layer via ``TestClient`` over a fully in-memory facade
(fake embedder + a scripted chat model), so the endpoints, request/response
schemas, and the facade wiring are all exercised without a browser, network,
or AWS.
"""

from __future__ import annotations

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
    )
    return TestClient(create_app(facade))


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
