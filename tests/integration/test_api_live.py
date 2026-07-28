"""Live-server API regression suite (M7).

Unlike ``test_api.py`` (which drives the app in-process via ``TestClient``),
this boots a **real uvicorn server** in a subprocess and calls it over HTTP —
proving the actual deployment entry point (``build_served_app``), env-var
configuration, ASGI stack, and JSON contracts all work as a client (Postman,
curl) would see them.

Runs in offline LLM mode (``TRUSTRESUME_LLM=test``, ``AutoStructuredFakeChatModel``
— see ``api/test_provider.py``) with temp SQLite + Chroma, so it needs no AWS
credentials and leaves nothing behind.

These tests are marked ``live`` and **deselected by default** (the default
``pytest`` run uses ``-m 'not live'`` — see pyproject) because spawning a
server is slower and heavier than the in-process suite. Run them explicitly:

    pytest -m live
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from collections.abc import Iterator

import httpx
import pytest

pytestmark = pytest.mark.live

BOOT_TIMEOUT_S = 60.0


def _free_port() -> int:
    """Grab an OS-assigned free port, then release it for uvicorn to bind."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def base_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """Boot a real uvicorn server in offline mode; yield its base URL."""
    port = _free_port()
    tmp = tmp_path_factory.mktemp("live_api")
    env = {
        "TRUSTRESUME_LLM": "test",
        "TRUSTRESUME_DB_PATH": str(tmp / "trustresume.db"),
        "TRUSTRESUME_CHROMA_PATH": str(tmp / "chroma"),
        # Inherit PATH etc. so the venv's uvicorn/python resolve.
        "PATH": _env_path(),
    }
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "trustresume.api.server:build_served_app",
            "--factory",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        env={**_base_env(), **env},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        _wait_until_healthy(url, proc)
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _base_env() -> dict[str, str]:
    import os

    return dict(os.environ)


def _env_path() -> str:
    import os

    return os.environ.get("PATH", "")


def _wait_until_healthy(url: str, proc: subprocess.Popen[bytes]) -> None:
    """Poll /api/health until the server answers or we time out / it dies."""
    deadline = time.monotonic() + BOOT_TIMEOUT_S
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            out = proc.stdout.read().decode() if proc.stdout else ""
            raise RuntimeError(f"uvicorn exited early (code {proc.returncode}):\n{out}")
        try:
            r = httpx.get(f"{url}/api/health", timeout=2.0)
            if r.status_code == 200:
                return
        except httpx.TransportError:
            pass
        time.sleep(0.3)
    raise RuntimeError("server did not become healthy within timeout")


# --- the regression checks -------------------------------------------------


def test_health(base_url: str) -> None:
    r = httpx.get(f"{base_url}/api/health", timeout=5.0)
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_openapi_importableByPostman(base_url: str) -> None:
    r = httpx.get(f"{base_url}/openapi.json", timeout=5.0)
    assert r.status_code == 200
    paths = r.json()["paths"]
    for expected in ("/api/health", "/api/documents", "/api/generate", "/api/ping"):
        assert expected in paths


def test_fullFlow_ingestThenGenerate(base_url: str) -> None:
    # Starts empty.
    r = httpx.get(f"{base_url}/api/documents", timeout=5.0)
    assert r.status_code == 200
    assert r.json() == []

    # Ingest a document.
    r = httpx.post(
        f"{base_url}/api/documents",
        json={
            "filename": "cv.txt",
            "text": "Built Python services on AWS. Led a team of five engineers.",
            "document_type": "RESUME",
        },
        timeout=30.0,
    )
    assert r.status_code == 201
    assert r.json()["filename"] == "cv.txt"

    # It now appears in the list.
    r = httpx.get(f"{base_url}/api/documents", timeout=5.0)
    assert len(r.json()) == 1

    # Generate — full pipeline over HTTP, driven by the offline
    # AutoStructuredFakeChatModel (synthesizes valid structured output for
    # whatever schema each agent binds — see api/test_provider.py).
    r = httpx.post(
        f"{base_url}/api/generate",
        json={"job_posting": "Senior Python Engineer with strong AWS experience."},
        timeout=60.0,
    )
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {
        "draft",
        "trust_score",
        "ats_score",
        "passed",
        "exhausted",
        "iterations",
        "hallucinations",
        "missing_keywords",
    }
    assert 0 <= body["trust_score"] <= 100
    assert 0 <= body["ats_score"] <= 100
    assert isinstance(body["passed"], bool)


def test_addDocument_validationError(base_url: str) -> None:
    r = httpx.post(f"{base_url}/api/documents", json={"filename": "x", "text": ""}, timeout=5.0)
    assert r.status_code == 422


def test_generate_requiresJobPosting(base_url: str) -> None:
    r = httpx.post(f"{base_url}/api/generate", json={"job_posting": ""}, timeout=5.0)
    assert r.status_code == 422


def test_ping_probesConfiguredProvider(base_url: str) -> None:
    """Ping probes whichever provider the server is configured for.

    This server was booted with ``TRUSTRESUME_LLM=test`` (see ``base_url``),
    so ping resolves the same offline ``test`` provider as the rest of the
    app — no AWS/network needed. Both outcomes are correct contracts:
    - 200 → the smoke test's tool-calling agent gets a real (synthesized)
      answer from the offline model.
    - 502 → the smoke test's own agent loop hits a shape the fake doesn't
      satisfy (e.g. more turns than the fake's plain-message fallback
      supports); reported as a clean HTTP error, not a crash.
    The point of the test is that ping never crashes the server, whatever the env.
    """
    r = httpx.get(f"{base_url}/api/ping", timeout=60.0)
    assert r.status_code in (200, 502)
    if r.status_code == 200:
        body = r.json()
        assert body["backend"] == "test"
        assert body["response"]  # non-empty model answer
    else:
        assert "detail" in r.json()
