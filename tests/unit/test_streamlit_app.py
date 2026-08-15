"""Unit tests for the Streamlit UI, using ``streamlit.testing.v1.AppTest``.

``AppTest.from_file`` runs the real script end to end (widget rendering,
button clicks, form submission) without a browser. ``requests.Session`` is
mocked at the ``trustresume.ui.api_client`` import site — the same seam
``test_ui_api_client.py`` uses — so no real HTTP call happens; this file is
about the Streamlit rendering logic (which widgets appear, what a click
triggers), not the HTTP client itself.

A prior browser-based manual check (see PR history) caught a real bug here —
``streamlit run`` executes this file as a script with no package context, so
the module's `from .api_client import ...` relative import crashed. This
suite exists so that class of regression is caught by ``pytest``, not only by
a human clicking through the app.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

APP_PATH = "src/trustresume/ui/streamlit_app.py"


@pytest.fixture(autouse=True)
def _clear_client_cache() -> None:
    """``_client`` is ``@st.cache_resource``-cached by base URL — without
    clearing it, a later test would reuse an earlier test's mocked
    ``TrustResumeClient`` instance instead of building a fresh one.
    """
    st.cache_resource.clear()


def _ok_response(json_body: object) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = json_body
    resp.raise_for_status.return_value = None
    return resp


@pytest.fixture
def mock_session() -> MagicMock:
    """A ``requests.Session`` double wired for the app's startup GETs
    (``health``, the Documents tab's ``list_documents``, the Jobs tab's
    ``list_jobs``); each test layers its own ``session.post``/other
    ``session.get`` return values on top. Dispatches by URL substring since
    the app now issues more than one kind of listing GET.
    """
    with patch("trustresume.ui.api_client.requests.Session") as session_cls:
        session = session_cls.return_value
        health_resp = _ok_response({"status": "ok"})
        docs_resp = _ok_response([])
        jobs_resp = _ok_response([])

        def _get(url: str, **kw: object) -> MagicMock:
            if "health" in url:
                return health_resp
            if "/api/jobs" in url:
                return jobs_resp
            return docs_resp

        session.get.side_effect = _get
        yield session


def test_app_rendersFourTabsWithoutError(mock_session: MagicMock) -> None:
    at = AppTest.from_file(APP_PATH)
    at.run()

    assert not at.exception
    assert [t.label for t in at.tabs] == [
        "📁 Documents",
        "✨ Generate",
        "💼 Jobs",
        "🔍 Search",
    ]
    assert at.sidebar.success[0].value == "Backend connected"


def test_app_backendUnreachable_showsSidebarError() -> None:
    with patch("trustresume.ui.api_client.requests.Session") as session_cls:
        session_cls.return_value.get.side_effect = requests.ConnectionError("down")
        at = AppTest.from_file(APP_PATH)
        at.run()

    assert not at.exception
    assert "unreachable" in at.sidebar.error[0].value


def test_documentsTab_listsUploadedDocuments(mock_session: MagicMock) -> None:
    docs_resp = _ok_response([{"id": "d1", "filename": "resume.txt", "document_type": "RESUME"}])
    health_resp = _ok_response({"status": "ok"})
    jobs_resp = _ok_response([])
    mock_session.get.side_effect = lambda url, **kw: (
        health_resp if "health" in url else jobs_resp if "/api/jobs" in url else docs_resp
    )
    at = AppTest.from_file(APP_PATH)
    at.run()

    assert not at.exception
    docs_tab = at.tabs[0]
    assert any("resume.txt" in w.value and "RESUME" in w.value for w in docs_tab.markdown)
    assert any(b.label == "Delete" for b in docs_tab.button)


def test_documentsTab_deleteButton_removesDocumentAndReruns(mock_session: MagicMock) -> None:
    # list_documents() is called both before and after the delete — its
    # response must reflect the deletion only on the *second* call, the same
    # way a real backend would; a plain fixed return value can't distinguish
    # "before" from "after" across the delete-triggered rerun.
    docs = [{"id": "d1", "filename": "resume.txt", "document_type": "RESUME"}]
    health_resp = _ok_response({"status": "ok"})
    jobs_resp = _ok_response([])
    mock_session.get.side_effect = lambda url, **kw: (
        health_resp
        if "health" in url
        else jobs_resp
        if "/api/jobs" in url
        else _ok_response([] if mock_session.delete.called else docs)
    )
    mock_session.delete.return_value = _ok_response(None)

    at = AppTest.from_file(APP_PATH)
    at.run()
    docs_tab = at.tabs[0]
    delete_button = next(b for b in docs_tab.button if b.label == "Delete")
    delete_button.click().run()

    assert not at.exception
    mock_session.delete.assert_called_once_with(
        "http://localhost:8000/api/documents/d1", timeout=300
    )
    assert at.tabs[0].info[0].value == "No documents uploaded yet."


def test_documentsTab_deleteButton_httpError_showsErrorMessage(mock_session: MagicMock) -> None:
    docs = [{"id": "d1", "filename": "resume.txt", "document_type": "RESUME"}]
    docs_resp = _ok_response(docs)
    health_resp = _ok_response({"status": "ok"})
    jobs_resp = _ok_response([])
    mock_session.get.side_effect = lambda url, **kw: (
        health_resp if "health" in url else jobs_resp if "/api/jobs" in url else docs_resp
    )
    error_resp = MagicMock()
    error_resp.text = "cannot delete"
    http_error = requests.HTTPError("500 error")
    http_error.response = error_resp
    mock_session.delete.side_effect = http_error

    at = AppTest.from_file(APP_PATH)
    at.run()
    docs_tab = at.tabs[0]
    delete_button = next(b for b in docs_tab.button if b.label == "Delete")
    delete_button.click().run()

    assert not at.exception
    assert "Delete failed" in at.tabs[0].error[0].value


def test_documentsTab_uploadSuccess_showsSuccessMessage(mock_session: MagicMock) -> None:
    mock_session.post.return_value = _ok_response(
        {"id": "d1", "filename": "resume.txt", "document_type": "RESUME"}
    )

    at = AppTest.from_file(APP_PATH)
    at.run()
    docs_tab = at.tabs[0]
    docs_tab.file_uploader[0].upload("resume.txt", b"Built Python services.").run()
    docs_tab.button[-1].click().run()

    assert not at.exception
    assert at.tabs[0].success[0].value == "Ingested resume.txt as RESUME."


def test_documentsTab_uploadHttpError_showsErrorMessage(mock_session: MagicMock) -> None:
    # The widget itself restricts extensions to .txt/.md/.docx (its `type=`
    # kwarg), so a server-side rejection is exercised with an accepted
    # extension whose *content* the backend rejects, not the widget.
    error_resp = MagicMock()
    error_resp.text = "empty file"
    http_error = requests.HTTPError("422 error")
    http_error.response = error_resp
    mock_session.post.side_effect = http_error

    at = AppTest.from_file(APP_PATH)
    at.run()
    docs_tab = at.tabs[0]
    docs_tab.file_uploader[0].upload("empty.txt", b"").run()
    docs_tab.button[-1].click().run()

    assert not at.exception
    assert "Upload failed" in at.tabs[0].error[0].value


def test_documentsTab_uploadWithNoFileChosen_warns(mock_session: MagicMock) -> None:
    at = AppTest.from_file(APP_PATH)
    at.run()

    docs_tab = at.tabs[0]
    docs_tab.button[-1].click().run()  # form submit, no file attached

    assert not at.exception
    assert at.tabs[0].warning[0].value == "Choose a file first."


def test_generateTab_success_showsScoresAndDraft(mock_session: MagicMock) -> None:
    mock_session.post.return_value = _ok_response(
        {
            "draft": {
                "summary": "Backend engineer with Python and AWS experience.",
                "sections": [{"heading": "Skills", "bullets": ["Python", "AWS"]}],
                "iteration": 0,
            },
            "trust_score": 95.0,
            "ats_score": 100.0,
            "passed": True,
            "exhausted": False,
            "iterations": 0,
            "hallucinations": [],
            "missing_keywords": [],
        }
    )

    at = AppTest.from_file(APP_PATH)
    at.run()
    gen_tab = at.tabs[1]
    gen_tab.text_area[0].set_value("Senior Python Engineer").run()
    gen_tab.button[0].click().run()

    assert not at.exception
    metrics = {m.label: m.value for m in at.tabs[1].metric}
    assert metrics == {"Trust score": "95", "ATS score": "100", "Iterations": "0"}
    assert at.tabs[1].success[0].value == "Passed the quality gate."
    assert any("Python" in m.value for m in at.tabs[1].markdown)


def test_generateTab_exhaustedWithoutPassing_showsWarningAndHallucinations(
    mock_session: MagicMock,
) -> None:
    mock_session.post.return_value = _ok_response(
        {
            "draft": {"summary": "", "sections": [], "iteration": 3},
            "trust_score": 0.0,
            "ats_score": 100.0,
            "passed": False,
            "exhausted": True,
            "iterations": 3,
            "hallucinations": [{"text": "Knows Kubernetes", "category": "SKILL"}],
            "missing_keywords": ["terraform"],
        }
    )

    at = AppTest.from_file(APP_PATH)
    at.run()
    gen_tab = at.tabs[1]
    gen_tab.text_area[0].set_value("Senior Python Engineer").run()
    gen_tab.button[0].click().run()

    assert not at.exception
    assert "rewrite cap" in at.tabs[1].warning[0].value
    assert any("Kubernetes" in m.value for m in at.tabs[1].markdown)
    assert "terraform" in at.tabs[1].markdown[-1].value


def test_generateTab_backendError_showsErrorMessage(mock_session: MagicMock) -> None:
    error_resp = MagicMock()
    error_resp.text = "workflow produced no scored draft"
    http_error = requests.HTTPError("500 error")
    http_error.response = error_resp
    mock_session.post.side_effect = http_error

    at = AppTest.from_file(APP_PATH)
    at.run()
    gen_tab = at.tabs[1]
    gen_tab.text_area[0].set_value("Senior Python Engineer").run()
    gen_tab.button[0].click().run()

    assert not at.exception
    assert "Generation failed" in at.tabs[1].error[0].value


def test_generateTab_connectionError_showsBackendUnreachable(mock_session: MagicMock) -> None:
    # A connection-level failure (no response at all), distinct from HTTPError.
    mock_session.post.side_effect = requests.ConnectionError("no route to host")

    at = AppTest.from_file(APP_PATH)
    at.run()
    gen_tab = at.tabs[1]
    gen_tab.text_area[0].set_value("Senior Python Engineer").run()
    gen_tab.button[0].click().run()

    assert not at.exception
    assert "Could not reach the backend" in at.tabs[1].error[0].value


def test_searchTab_returnsRankedChunks(mock_session: MagicMock) -> None:
    mock_session.post.return_value = _ok_response(
        {
            "query": "python aws",
            "chunks": [
                {
                    "chunk_id": "c1",
                    "document_type": "RESUME",
                    "source_document": "resume.txt",
                    "text": "Built Python services on AWS.",
                    "score": 0.9,
                }
            ],
        }
    )

    at = AppTest.from_file(APP_PATH)
    at.run()
    search_tab = at.tabs[3]
    search_tab.text_input[0].set_value("python aws").run()
    search_tab.button[0].click().run()

    assert not at.exception
    captions = [c.value for c in at.tabs[3].caption]
    assert any("RESUME" in c and "resume.txt" in c for c in captions)
    assert "Built Python services on AWS." in at.tabs[3].markdown[-1].value


def test_searchTab_noResults_showsInfo(mock_session: MagicMock) -> None:
    mock_session.post.return_value = _ok_response({"query": "nothing", "chunks": []})

    at = AppTest.from_file(APP_PATH)
    at.run()
    search_tab = at.tabs[3]
    search_tab.text_input[0].set_value("nothing matches").run()
    search_tab.button[0].click().run()

    assert not at.exception
    assert at.tabs[3].info[0].value == "No matching evidence found."


def test_searchTab_connectionError_showsBackendUnreachable(mock_session: MagicMock) -> None:
    mock_session.post.side_effect = requests.ConnectionError("no route to host")

    at = AppTest.from_file(APP_PATH)
    at.run()
    search_tab = at.tabs[3]
    search_tab.text_input[0].set_value("python").run()
    search_tab.button[0].click().run()

    assert not at.exception
    assert "Could not reach the backend" in at.tabs[3].error[0].value


def test_searchTab_backendError_showsErrorMessage(mock_session: MagicMock) -> None:
    error_resp = MagicMock()
    error_resp.text = "boom"
    http_error = requests.HTTPError("500 error")
    http_error.response = error_resp
    mock_session.post.side_effect = http_error

    at = AppTest.from_file(APP_PATH)
    at.run()
    search_tab = at.tabs[3]
    search_tab.text_input[0].set_value("python").run()
    search_tab.button[0].click().run()

    assert not at.exception
    assert "Search failed" in at.tabs[3].error[0].value


# --- Jobs tab ----------------------------------------------------------------


def test_jobsTab_noJobsYet_showsInfo(mock_session: MagicMock) -> None:
    at = AppTest.from_file(APP_PATH)
    at.run()

    jobs_tab = at.tabs[2]
    assert any(i.value == "No jobs created yet." for i in jobs_tab.info)


def test_jobsTab_createJob_success_reruns(mock_session: MagicMock) -> None:
    mock_session.post.return_value = _ok_response({"id": "j1", "title": "Engineer"})

    at = AppTest.from_file(APP_PATH)
    at.run()
    jobs_tab = at.tabs[2]
    jobs_tab.text_area[0].set_value("Senior Python Engineer role").run()
    jobs_tab.button[0].click().run()

    assert not at.exception
    _, kwargs = mock_session.post.call_args
    assert kwargs["json"] == {"job_posting": "Senior Python Engineer role"}


def test_jobsTab_createJob_noPostingChosen_warns(mock_session: MagicMock) -> None:
    at = AppTest.from_file(APP_PATH)
    at.run()
    jobs_tab = at.tabs[2]
    jobs_tab.button[0].click().run()

    assert not at.exception
    assert at.tabs[2].warning[0].value == "Paste a job posting first."


def test_jobsTab_createJob_httpError_showsErrorMessage(mock_session: MagicMock) -> None:
    error_resp = MagicMock()
    error_resp.text = "cannot create"
    http_error = requests.HTTPError("500 error")
    http_error.response = error_resp
    mock_session.post.side_effect = http_error

    at = AppTest.from_file(APP_PATH)
    at.run()
    jobs_tab = at.tabs[2]
    jobs_tab.text_area[0].set_value("Senior Python Engineer role").run()
    jobs_tab.button[0].click().run()

    assert not at.exception
    assert "Could not create job" in at.tabs[2].error[0].value


def test_jobsTab_createJob_connectionError_showsBackendUnreachable(
    mock_session: MagicMock,
) -> None:
    mock_session.post.side_effect = requests.ConnectionError("down")

    at = AppTest.from_file(APP_PATH)
    at.run()
    jobs_tab = at.tabs[2]
    jobs_tab.text_area[0].set_value("Senior Python Engineer role").run()
    jobs_tab.button[0].click().run()

    assert not at.exception
    assert "Could not reach the backend" in at.tabs[2].error[0].value


def test_jobsTab_withExistingJob_rendersUploadGenerateAndPastResumes(
    mock_session: MagicMock,
) -> None:
    jobs_resp = _ok_response([{"id": "j1", "summary": "Senior Python Engineer at Acme"}])
    resumes_resp = _ok_response(
        [
            {
                "id": "r1",
                "trust_score": 95.0,
                "ats_score": 90.0,
                "iteration": 0,
                "passed": True,
                "created_at": "2026-01-01T00:00:00",
            }
        ]
    )
    health_resp = _ok_response({"status": "ok"})
    docs_resp = _ok_response([])
    pdf_resp = MagicMock()
    pdf_resp.raise_for_status.return_value = None
    pdf_resp.content = b"%PDF-1.4 fake"
    md_resp = MagicMock()
    md_resp.raise_for_status.return_value = None
    md_resp.text = "# Summary"

    def _get(url: str, **kw: object) -> MagicMock:
        if "health" in url:
            return health_resp
        if url.endswith("/resumes"):
            return resumes_resp
        if url.endswith("/pdf"):
            return pdf_resp
        if url.endswith("/markdown"):
            return md_resp
        if "/api/jobs" in url:
            return jobs_resp
        return docs_resp

    mock_session.get.side_effect = _get

    at = AppTest.from_file(APP_PATH)
    at.run()
    jobs_tab = at.tabs[2]

    assert not at.exception
    # AppTest reports the selectbox's options already run through
    # format_func — verified empirically — so this checks the displayed
    # label (the job's summary), not the raw job id.
    assert jobs_tab.selectbox[0].options == ["Senior Python Engineer at Acme"]
    # Past resume is rendered (st.write -> markdown element) with its scores,
    # plus both download buttons (a distinct AppTest element category from
    # plain st.button — verified empirically).
    assert any("Trust 95" in w.value for w in jobs_tab.markdown)
    download_labels = {b.label for b in jobs_tab.download_button}
    assert "Download PDF" in download_labels
    assert "Download Markdown" in download_labels


def test_jobsTab_uploadDocumentForJob_success_showsSuccessMessage(
    mock_session: MagicMock,
) -> None:
    jobs_resp = _ok_response([{"id": "j1", "summary": "Senior Python Engineer at Acme"}])
    health_resp = _ok_response({"status": "ok"})
    docs_resp = _ok_response([])
    empty_resumes_resp = _ok_response([])

    def _get(url: str, **kw: object) -> MagicMock:
        if "health" in url:
            return health_resp
        if url.endswith("/resumes"):
            return empty_resumes_resp
        if "/api/jobs" in url:
            return jobs_resp
        return docs_resp

    mock_session.get.side_effect = _get
    mock_session.post.return_value = _ok_response(
        {"id": "d1", "filename": "resume.txt", "document_type": "RESUME"}
    )

    at = AppTest.from_file(APP_PATH)
    at.run()
    jobs_tab = at.tabs[2]
    jobs_tab.file_uploader[0].upload("resume.txt", b"Built Python services.").run()
    # First form's submit is "Create job"; the upload form's submit follows it.
    upload_button = next(b for b in jobs_tab.button if b.label == "Upload")
    upload_button.click().run()

    assert not at.exception
    assert any("Linked resume.txt to this job" in s.value for s in at.tabs[2].success)


def test_jobsTab_uploadDocumentForJob_noFileChosen_warns(mock_session: MagicMock) -> None:
    jobs_resp = _ok_response([{"id": "j1", "summary": "Senior Python Engineer at Acme"}])
    health_resp = _ok_response({"status": "ok"})
    docs_resp = _ok_response([])
    empty_resumes_resp = _ok_response([])

    def _get(url: str, **kw: object) -> MagicMock:
        if "health" in url:
            return health_resp
        if url.endswith("/resumes"):
            return empty_resumes_resp
        if "/api/jobs" in url:
            return jobs_resp
        return docs_resp

    mock_session.get.side_effect = _get

    at = AppTest.from_file(APP_PATH)
    at.run()
    jobs_tab = at.tabs[2]
    upload_button = next(b for b in jobs_tab.button if b.label == "Upload")
    upload_button.click().run()

    assert not at.exception
    assert at.tabs[2].warning[0].value == "Choose a file first."


def test_jobsTab_generateForJob_success_rendersResult(mock_session: MagicMock) -> None:
    jobs_resp = _ok_response([{"id": "j1", "summary": "Senior Python Engineer at Acme"}])
    health_resp = _ok_response({"status": "ok"})
    docs_resp = _ok_response([])
    empty_resumes_resp = _ok_response([])
    generate_resp = _ok_response(
        {
            "draft": {"summary": "Backend engineer.", "sections": [], "iteration": 0},
            "trust_score": 95.0,
            "ats_score": 90.0,
            "passed": True,
            "exhausted": False,
            "iterations": 0,
            "hallucinations": [],
            "missing_keywords": [],
            "resume_id": "r1",
        }
    )

    def _get(url: str, **kw: object) -> MagicMock:
        if "health" in url:
            return health_resp
        if url.endswith("/resumes"):
            return empty_resumes_resp
        if "/api/jobs" in url:
            return jobs_resp
        return docs_resp

    mock_session.get.side_effect = _get
    mock_session.post.return_value = generate_resp

    at = AppTest.from_file(APP_PATH)
    at.run()
    jobs_tab = at.tabs[2]
    generate_button = next(b for b in jobs_tab.button if b.label == "Generate for this job")
    generate_button.click().run()

    assert not at.exception
    assert any("Passed the quality gate." in s.value for s in at.tabs[2].success)


def test_jobsTab_generateForJob_httpError_showsErrorMessage(mock_session: MagicMock) -> None:
    jobs_resp = _ok_response([{"id": "j1", "summary": "Senior Python Engineer at Acme"}])
    health_resp = _ok_response({"status": "ok"})
    docs_resp = _ok_response([])
    empty_resumes_resp = _ok_response([])

    def _get(url: str, **kw: object) -> MagicMock:
        if "health" in url:
            return health_resp
        if url.endswith("/resumes"):
            return empty_resumes_resp
        if "/api/jobs" in url:
            return jobs_resp
        return docs_resp

    mock_session.get.side_effect = _get
    error_resp = MagicMock()
    error_resp.text = "cannot generate"
    http_error = requests.HTTPError("500 error")
    http_error.response = error_resp
    mock_session.post.side_effect = http_error

    at = AppTest.from_file(APP_PATH)
    at.run()
    jobs_tab = at.tabs[2]
    generate_button = next(b for b in jobs_tab.button if b.label == "Generate for this job")
    generate_button.click().run()

    assert not at.exception
    assert "Generation failed" in at.tabs[2].error[0].value


def test_jobsTab_generateForJob_connectionError_showsBackendUnreachable(
    mock_session: MagicMock,
) -> None:
    jobs_resp = _ok_response([{"id": "j1", "summary": "Senior Python Engineer at Acme"}])
    health_resp = _ok_response({"status": "ok"})
    docs_resp = _ok_response([])
    empty_resumes_resp = _ok_response([])

    def _get(url: str, **kw: object) -> MagicMock:
        if "health" in url:
            return health_resp
        if url.endswith("/resumes"):
            return empty_resumes_resp
        if "/api/jobs" in url:
            return jobs_resp
        return docs_resp

    mock_session.get.side_effect = _get
    mock_session.post.side_effect = requests.ConnectionError("down")

    at = AppTest.from_file(APP_PATH)
    at.run()
    jobs_tab = at.tabs[2]
    generate_button = next(b for b in jobs_tab.button if b.label == "Generate for this job")
    generate_button.click().run()

    assert not at.exception
    assert "Could not reach the backend" in at.tabs[2].error[0].value


def test_jobsTab_listResumesForJob_connectionError_showsBackendUnreachable(
    mock_session: MagicMock,
) -> None:
    jobs_resp = _ok_response([{"id": "j1", "summary": "Senior Python Engineer at Acme"}])
    health_resp = _ok_response({"status": "ok"})
    docs_resp = _ok_response([])

    def _get(url: str, **kw: object) -> MagicMock:
        if "health" in url:
            return health_resp
        if url.endswith("/resumes"):
            raise requests.ConnectionError("down")
        if "/api/jobs" in url:
            return jobs_resp
        return docs_resp

    mock_session.get.side_effect = _get

    at = AppTest.from_file(APP_PATH)
    at.run()

    assert not at.exception
    assert any("Could not reach the backend" in e.value for e in at.tabs[2].error)


def test_jobsTab_backendUnreachableListingJobs_showsError(mock_session: MagicMock) -> None:
    # Health succeeds (so the sidebar/other tabs render normally) but the
    # Jobs tab's own list_jobs() call fails — isolates that tab's specific
    # try/except from the app-wide health-check one in main().
    health_resp = _ok_response({"status": "ok"})

    def _get(url: str, **kw: object) -> MagicMock:
        if "health" in url:
            return health_resp
        raise requests.ConnectionError("down")

    mock_session.get.side_effect = _get

    at = AppTest.from_file(APP_PATH)
    at.run()

    assert not at.exception
    assert any("Could not reach the backend" in e.value for e in at.tabs[2].error)


def _generation_with_usage(usage: dict[str, object] | None) -> dict[str, object]:
    return {
        "draft": {"summary": "s", "sections": [], "iteration": 0},
        "trust_score": 95.0,
        "ats_score": 100.0,
        "passed": True,
        "exhausted": False,
        "iterations": 0,
        "hallucinations": [],
        "missing_keywords": [],
        "usage": usage,
    }


def _run_generate(at: AppTest) -> None:
    gen_tab = at.tabs[1]
    gen_tab.text_area[0].set_value("Senior Python Engineer").run()
    gen_tab.button[0].click().run()


def test_generateTab_usageWithKnownPrice_showsCostCaption(mock_session: MagicMock) -> None:
    mock_session.post.return_value = _ok_response(
        _generation_with_usage(
            {
                "llm_calls": 4,
                "input_tokens": 1000,
                "output_tokens": 500,
                "total_tokens": 1500,
                "duration_ms": 8200.0,
                "cost_usd": 0.0123,
            }
        )
    )

    at = AppTest.from_file(APP_PATH)
    at.run()
    _run_generate(at)

    assert not at.exception
    caption = next(c.value for c in at.tabs[1].caption if "LLM calls" in c.value)
    assert "4 LLM calls" in caption
    assert "1,500 tokens" in caption
    assert "8.2s" in caption
    assert "$0.0123" in caption


def test_generateTab_usageWithoutPrice_showsTokensButNoDollarFigure(
    mock_session: MagicMock,
) -> None:
    """The offline default: real token counts, honestly no cost (ADR-0012).

    Showing "$0.0000" here would claim the run was free rather than unpriced.
    """
    mock_session.post.return_value = _ok_response(
        _generation_with_usage(
            {
                "llm_calls": 4,
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "duration_ms": 120.0,
                "cost_usd": None,
            }
        )
    )

    at = AppTest.from_file(APP_PATH)
    at.run()
    _run_generate(at)

    assert not at.exception
    caption = next(c.value for c in at.tabs[1].caption if "LLM calls" in c.value)
    assert "150 tokens" in caption
    assert "$" not in caption


def test_generateTab_noUsageInResponse_rendersNoCaption(mock_session: MagicMock) -> None:
    """An older backend (or an unmeasured run) must not break the page."""
    mock_session.post.return_value = _ok_response(_generation_with_usage(None))

    at = AppTest.from_file(APP_PATH)
    at.run()
    _run_generate(at)

    assert not at.exception
    assert not [c for c in at.tabs[1].caption if "LLM calls" in c.value]
