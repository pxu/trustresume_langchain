"""Unit tests for the Streamlit UI's REST client.

Mocks ``requests.Session`` directly rather than adding a mocking dependency
(e.g. ``responses``) just for this — the client is a thin wrapper, so
asserting on the call args/kwargs it builds is enough to cover it.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from trustresume.ui.api_client import TrustResumeClient


def _mock_response(json_body: object, *, ok: bool = True) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = json_body
    if ok:
        resp.raise_for_status.return_value = None
    else:
        error = requests.HTTPError("boom")
        error.response = resp
        resp.raise_for_status.side_effect = error
    return resp


@patch("trustresume.ui.api_client.requests.Session")
def test_health_getsExpectedUrl(session_cls: MagicMock) -> None:
    session = session_cls.return_value
    session.get.return_value = _mock_response({"status": "ok"})

    client = TrustResumeClient("http://api:8000")
    result = client.health()

    assert result == {"status": "ok"}
    session.get.assert_called_once_with("http://api:8000/api/health", timeout=client.timeout)


@patch("trustresume.ui.api_client.requests.Session")
def test_baseUrl_trailingSlashStripped(session_cls: MagicMock) -> None:
    session = session_cls.return_value
    session.get.return_value = _mock_response({"status": "ok"})

    client = TrustResumeClient("http://api:8000/")
    client.health()

    session.get.assert_called_once_with("http://api:8000/api/health", timeout=client.timeout)


@patch("trustresume.ui.api_client.requests.Session")
def test_uploadDocument_sendsMultipartFileAndFormField(session_cls: MagicMock) -> None:
    session = session_cls.return_value
    session.post.return_value = _mock_response(
        {"id": "d1", "filename": "r.txt", "document_type": "RESUME"}
    )

    client = TrustResumeClient("http://api:8000")
    result = client.upload_document(filename="r.txt", data=b"hello", document_type="RESUME")

    assert result["id"] == "d1"
    _, kwargs = session.post.call_args
    assert kwargs["files"] == {"file": ("r.txt", b"hello")}
    assert kwargs["data"] == {"document_type": "RESUME"}


@patch("trustresume.ui.api_client.requests.Session")
def test_search_sendsQueryAndLimit(session_cls: MagicMock) -> None:
    session = session_cls.return_value
    session.post.return_value = _mock_response({"query": "python", "chunks": []})

    client = TrustResumeClient("http://api:8000")
    client.search(query="python", limit=3)

    _, kwargs = session.post.call_args
    assert kwargs["json"] == {"query": "python", "limit": 3}


@patch("trustresume.ui.api_client.requests.Session")
def test_generate_postsJobPosting(session_cls: MagicMock) -> None:
    session = session_cls.return_value
    session.post.return_value = _mock_response({"draft": {}, "trust_score": 90.0})

    client = TrustResumeClient("http://api:8000")
    client.generate(job_posting="Senior Engineer")

    _, kwargs = session.post.call_args
    assert kwargs["json"] == {"job_posting": "Senior Engineer"}


@patch("trustresume.ui.api_client.requests.Session")
def test_listDocuments_getsExpectedUrlAndReturnsBody(session_cls: MagicMock) -> None:
    session = session_cls.return_value
    session.get.return_value = _mock_response([{"id": "d1", "filename": "r.txt"}])

    client = TrustResumeClient("http://api:8000")
    result = client.list_documents()

    assert result == [{"id": "d1", "filename": "r.txt"}]
    session.get.assert_called_once_with("http://api:8000/api/documents", timeout=client.timeout)


@patch("trustresume.ui.api_client.requests.Session")
def test_deleteDocument_sendsDeleteToExpectedUrl(session_cls: MagicMock) -> None:
    session = session_cls.return_value
    session.delete.return_value = _mock_response(None)

    client = TrustResumeClient("http://api:8000")
    client.delete_document("d1")

    session.delete.assert_called_once_with(
        "http://api:8000/api/documents/d1", timeout=client.timeout
    )


@patch("trustresume.ui.api_client.requests.Session")
def test_listDocuments_raisesOnServerError(session_cls: MagicMock) -> None:
    session = session_cls.return_value
    session.get.return_value = _mock_response({"detail": "boom"}, ok=False)

    client = TrustResumeClient("http://api:8000")
    with pytest.raises(requests.HTTPError):
        client.list_documents()
