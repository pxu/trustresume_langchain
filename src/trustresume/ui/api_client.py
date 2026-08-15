"""Thin ``requests`` wrapper over the TrustResume FastAPI backend.

Kept separate from ``streamlit_app.py`` so the HTTP calls are plain functions
— callable, and testable, without importing ``streamlit`` or running a script.
Every function raises via ``response.raise_for_status()`` on a non-2xx
response; the Streamlit app is responsible for catching and displaying that.
"""

from __future__ import annotations

from typing import Any, cast

import requests

DEFAULT_TIMEOUT = 300  # seconds — a capped-out run (4 drafts, real provider) took ~181s in practice
#: Mirrors ``api.server.USER_ID_HEADER``; duplicated rather than imported so
#: the UI package keeps its one-way dependency on the backend (HTTP only, no
#: Python imports from ``trustresume.api``).
USER_ID_HEADER = "X-User-Id"


class TrustResumeClient:
    """A session-backed client for one TrustResume API base URL."""

    def __init__(
        self, base_url: str, *, timeout: float = DEFAULT_TIMEOUT, user_id: str | None = None
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        # Set once on the session rather than per call: every route is
        # user-scoped (ADR-0001), so a client that forgot the header on one
        # method would silently read the demo user's data instead of its own.
        # Omitted entirely when no id is given, which the backend reads as
        # "the demo user".
        if user_id:
            self._session.headers[USER_ID_HEADER] = user_id

    def health(self) -> dict[str, Any]:
        resp = self._session.get(f"{self.base_url}/api/health", timeout=self.timeout)
        resp.raise_for_status()
        return cast(dict[str, Any], resp.json())

    def list_documents(self) -> list[dict[str, Any]]:
        resp = self._session.get(f"{self.base_url}/api/documents", timeout=self.timeout)
        resp.raise_for_status()
        return cast(list[dict[str, Any]], resp.json())

    def upload_document(self, *, filename: str, data: bytes, document_type: str) -> dict[str, Any]:
        resp = self._session.post(
            f"{self.base_url}/api/documents/upload",
            files={"file": (filename, data)},
            data={"document_type": document_type},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return cast(dict[str, Any], resp.json())

    def delete_document(self, document_id: str) -> None:
        resp = self._session.delete(
            f"{self.base_url}/api/documents/{document_id}", timeout=self.timeout
        )
        resp.raise_for_status()

    def search(self, *, query: str, limit: int = 5) -> dict[str, Any]:
        resp = self._session.post(
            f"{self.base_url}/api/search",
            json={"query": query, "limit": limit},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return cast(dict[str, Any], resp.json())

    def generate(self, *, job_posting: str, max_iterations: int | None = None) -> dict[str, Any]:
        """``max_iterations`` overrides the server's config-resolved default for this call only."""
        body: dict[str, Any] = {"job_posting": job_posting}
        if max_iterations is not None:
            body["max_iterations"] = max_iterations
        resp = self._session.post(
            f"{self.base_url}/api/generate",
            json=body,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return cast(dict[str, Any], resp.json())

    def create_job(self, *, job_posting: str) -> dict[str, Any]:
        resp = self._session.post(
            f"{self.base_url}/api/jobs",
            json={"job_posting": job_posting},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return cast(dict[str, Any], resp.json())

    def list_jobs(self) -> list[dict[str, Any]]:
        resp = self._session.get(f"{self.base_url}/api/jobs", timeout=self.timeout)
        resp.raise_for_status()
        return cast(list[dict[str, Any]], resp.json())

    def upload_document_for_job(
        self, *, job_id: str, filename: str, data: bytes, document_type: str
    ) -> dict[str, Any]:
        resp = self._session.post(
            f"{self.base_url}/api/jobs/{job_id}/documents/upload",
            files={"file": (filename, data)},
            data={"document_type": document_type},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return cast(dict[str, Any], resp.json())

    def list_documents_for_job(self, *, job_id: str) -> list[dict[str, Any]]:
        resp = self._session.get(
            f"{self.base_url}/api/jobs/{job_id}/documents", timeout=self.timeout
        )
        resp.raise_for_status()
        return cast(list[dict[str, Any]], resp.json())

    def generate_for_job(self, *, job_id: str, max_iterations: int | None = None) -> dict[str, Any]:
        """``max_iterations`` overrides the server's config-resolved default for this call only."""
        body = {"max_iterations": max_iterations} if max_iterations is not None else {}
        resp = self._session.post(
            f"{self.base_url}/api/jobs/{job_id}/generate", json=body, timeout=self.timeout
        )
        resp.raise_for_status()
        return cast(dict[str, Any], resp.json())

    def list_resumes_for_job(self, *, job_id: str) -> list[dict[str, Any]]:
        resp = self._session.get(f"{self.base_url}/api/jobs/{job_id}/resumes", timeout=self.timeout)
        resp.raise_for_status()
        return cast(list[dict[str, Any]], resp.json())

    def download_resume_pdf(self, *, resume_id: str) -> bytes:
        resp = self._session.get(
            f"{self.base_url}/api/resumes/{resume_id}/pdf", timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.content

    def download_resume_markdown(self, *, resume_id: str) -> str:
        resp = self._session.get(
            f"{self.base_url}/api/resumes/{resume_id}/markdown", timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.text
