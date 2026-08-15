"""Streamlit frontend for TrustResume.

A thin client over the FastAPI backend (``api/server.py``) — no direct
imports from ``trustresume.api``/``orchestration``/etc., so this process can
point at a backend running anywhere (same host, Docker Compose service,
remote deployment) via ``TRUSTRESUME_API_URL``. All business logic — RAG
retrieval, the quality loop, trust scoring — lives in the backend; this file
only renders forms and results.

Run (with the backend already up, e.g. ``docker compose up`` or the offline
``TRUSTRESUME_LLM_PROVIDER=test uvicorn ...`` command from the README)::

    streamlit run src/trustresume/ui/streamlit_app.py

Four tabs, matching the things a candidate does with the app: Documents
(upload career evidence), Generate (run the full RAG + multi-agent pipeline
against a one-off job posting, no persistence), Jobs (persist a job, upload
job-scoped documents, generate against it repeatedly, browse/download past
resumes), Search (inspect retrieval directly).
"""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

from trustresume.models import DocumentType
from trustresume.ui.api_client import TrustResumeClient

DEFAULT_API_URL = os.getenv("TRUSTRESUME_API_URL", "http://localhost:8000")
DEFAULT_USER_ID = os.getenv("TRUSTRESUME_USER_ID", "")


@st.cache_resource
def _client(base_url: str, user_id: str) -> TrustResumeClient:
    """One cached client per (backend, user) pair.

    ``user_id`` is part of the cache key on purpose: switching users in the
    sidebar must hand back a client carrying the *new* ``X-User-Id``, not a
    cached one still scoped to the previous user.
    """
    return TrustResumeClient(base_url, user_id=user_id or None)


def _error_detail(exc: requests.HTTPError) -> str:
    """The backend's response body, or the exception text if none was captured."""
    return exc.response.text if exc.response is not None else str(exc)


def _render_documents_tab(client: TrustResumeClient) -> None:
    st.subheader("Upload candidate evidence")
    st.caption(
        "Résumés, project reports, STAR stories, certifications — anything the "
        "Resume Writer can ground a draft in. Parsed and embedded server-side."
    )

    with st.form("upload_form", clear_on_submit=True):
        upload = st.file_uploader("File (.txt, .md, .docx)", type=["txt", "md", "docx"])
        document_type = st.selectbox(
            "Document type", [t.value for t in DocumentType], index=len(DocumentType) - 1
        )
        submitted = st.form_submit_button("Upload")

    if submitted:
        if upload is None:
            st.warning("Choose a file first.")
        else:
            try:
                result = client.upload_document(
                    filename=upload.name, data=upload.getvalue(), document_type=document_type
                )
            except requests.HTTPError as exc:
                st.error(f"Upload failed: {_error_detail(exc)}")
            else:
                st.success(f"Ingested {result['filename']} as {result['document_type']}.")

    st.divider()
    st.subheader("Your documents")
    try:
        docs = client.list_documents()
    except requests.RequestException as exc:
        st.error(f"Could not reach the backend: {exc}")
        return

    if not docs:
        st.info("No documents uploaded yet.")
    else:
        for doc in docs:
            col_info, col_delete = st.columns([5, 1])
            col_info.write(f"**{doc['filename']}** · {doc['document_type']}")
            if col_delete.button("Delete", key=f"delete-{doc['id']}"):
                try:
                    client.delete_document(doc["id"])
                except requests.HTTPError as exc:
                    st.error(f"Delete failed: {_error_detail(exc)}")
                else:
                    st.rerun()


def _render_usage(usage: dict[str, Any] | None) -> None:
    """One line of "what that run cost", under the scores.

    Shown as a caption rather than a metric row: it's operator information,
    not something the candidate is being scored on. ``cost_usd`` is absent
    when the model in use has no configured price
    (``config/pricing.json``) — the line then reports tokens and time only,
    rather than implying the run was free.
    """
    if not usage:
        return
    parts = [
        f"{usage['llm_calls']} LLM calls",
        f"{usage['total_tokens']:,} tokens",
        f"{usage['duration_ms'] / 1000:.1f}s",
    ]
    if usage.get("cost_usd") is not None:
        parts.append(f"${usage['cost_usd']:.4f}")
    st.caption(" · ".join(parts))


def _render_generation_result(result: dict[str, Any]) -> None:
    """Shared rendering for a ``GenerateResponse``-shaped dict.

    Used by both the one-off Generate tab and the Jobs tab's
    "Generate for this job" action — factored out so the two don't
    duplicate the same score/draft/hallucination/keyword panel.
    """
    col1, col2, col3 = st.columns(3)
    col1.metric("Trust score", f"{result['trust_score']:.0f}")
    col2.metric("ATS score", f"{result['ats_score']:.0f}")
    col3.metric("Iterations", result["iterations"])

    if result["passed"]:
        st.success("Passed the quality gate.")
    else:
        # The quality loop always runs every draft to the iteration cap (no
        # early exit on a pass), so a failing result here already reflects
        # the best-scoring draft across every iteration, not just the last.
        st.warning("Did not pass the quality gate — showing the best-scoring draft anyway.")

    _render_usage(result.get("usage"))

    draft = result["draft"]
    st.subheader("Draft")
    if draft["summary"]:
        st.write(draft["summary"])
    for section in draft["sections"]:
        st.markdown(f"**{section['heading']}**")
        for bullet in section["bullets"]:
            st.markdown(f"- {bullet}")

    if result["hallucinations"]:
        st.subheader("⚠️ Unsupported claims (flagged by the Trust Harness)")
        for claim in result["hallucinations"]:
            st.markdown(f"- *{claim['category']}*: {claim['text']}")

    if result["missing_keywords"]:
        st.subheader("Missing ATS keywords")
        st.write(", ".join(result["missing_keywords"]))


def _has_any_documents(client: TrustResumeClient) -> bool:
    """Best-effort check for the Generate tab's disabled state.

    Fails open (``True``) on a backend error rather than blocking the whole
    tab on a fetch that isn't the point of this render — a real problem
    surfaces naturally when the user actually clicks Generate.
    """
    try:
        return bool(client.list_documents())
    except requests.RequestException:
        return True


def _has_eligible_documents_for_job(client: TrustResumeClient, *, job_id: str) -> bool:
    """Same fail-open check as :func:`_has_any_documents`, scoped to one job's eligible pool."""
    try:
        return bool(client.list_documents_for_job(job_id=job_id))
    except requests.RequestException:
        return True


def _render_generate_tab(client: TrustResumeClient) -> None:
    st.subheader("Generate a tailored, evidence-checked resume")
    st.caption("One-off: not persisted. To save the result and reuse the job, use the Jobs tab.")
    job_posting = st.text_area("Job posting", height=200, placeholder="Paste the job posting…")
    max_iterations = st.number_input(
        "Rewrite attempts after the first draft",
        min_value=0,
        max_value=10,
        value=1,
        help=(
            "The quality loop always runs this many rewrites (no early exit "
            "on a pass) and ships the best-scoring draft. Higher costs more "
            "LLM calls for a chance at a better draft, not a guarantee."
        ),
    )
    has_documents = _has_any_documents(client)
    if not has_documents:
        st.info("Upload at least one document on the Documents tab before generating.")
    if st.button("Generate", type="primary", disabled=not job_posting.strip() or not has_documents):
        with st.spinner("Running the pipeline — job analysis, retrieval, writing, verification…"):
            try:
                result = client.generate(
                    job_posting=job_posting, max_iterations=int(max_iterations)
                )
            except requests.HTTPError as exc:
                st.error(f"Generation failed: {_error_detail(exc)}")
                return
            except requests.RequestException as exc:
                st.error(f"Could not reach the backend: {exc}")
                return
        _render_generation_result(result)


def _render_search_tab(client: TrustResumeClient) -> None:
    st.subheader("Search your own evidence")
    st.caption(
        "Runs the same semantic search the Evidence Retrieval agent uses during "
        "generation — inspect what the RAG pipeline would retrieve for a query."
    )
    query = st.text_input("Query", placeholder="e.g. Python AWS backend experience")
    limit = st.slider("Max results", min_value=1, max_value=20, value=5)

    if st.button("Search", disabled=not query.strip()):
        try:
            result = client.search(query=query, limit=limit)
        except requests.HTTPError as exc:
            st.error(f"Search failed: {_error_detail(exc)}")
            return
        except requests.RequestException as exc:
            st.error(f"Could not reach the backend: {exc}")
            return

        chunks = result["chunks"]
        if not chunks:
            st.info("No matching evidence found.")
        for chunk in chunks:
            score = f"{chunk['score']:.3f}" if chunk["score"] is not None else "n/a"
            with st.container(border=True):
                st.caption(f"{chunk['document_type']} · {chunk['source_document']} · score {score}")
                st.write(chunk["text"])


def _render_jobs_tab(client: TrustResumeClient) -> None:
    st.subheader("Create a job")
    st.caption(
        "Persisted, unlike the one-off Generate tab: get an id you can upload "
        "job-scoped documents against, generate against repeatedly, and browse "
        "past resumes for."
    )
    with st.form("create_job_form", clear_on_submit=True):
        job_posting = st.text_area("Job posting", height=150, placeholder="Paste the job posting…")
        create_submitted = st.form_submit_button("Create job")

    if create_submitted:
        if not job_posting.strip():
            st.warning("Paste a job posting first.")
        else:
            try:
                client.create_job(job_posting=job_posting)
            except requests.HTTPError as exc:
                st.error(f"Could not create job: {_error_detail(exc)}")
            except requests.RequestException as exc:
                st.error(f"Could not reach the backend: {exc}")
            else:
                st.rerun()

    st.divider()

    try:
        jobs = client.list_jobs()
    except requests.RequestException as exc:
        st.error(f"Could not reach the backend: {exc}")
        return
    if not jobs:
        st.info("No jobs created yet.")
        return

    labels = {j["id"]: (j["summary"] or j["id"]) for j in jobs}
    job_id = st.selectbox("Job", options=list(labels), format_func=lambda jid: labels[jid])
    if job_id is None:
        return

    st.subheader("Job-scoped documents")
    st.caption(
        "Uploaded here, these are used for this job in addition to your generic "
        "(unlinked) document pool — see the Documents tab for that pool."
    )
    with st.form(f"job_upload_form_{job_id}", clear_on_submit=True):
        upload = st.file_uploader("File (.txt, .md, .docx)", type=["txt", "md", "docx"])
        document_type = st.selectbox(
            "Document type",
            [t.value for t in DocumentType],
            index=len(DocumentType) - 1,
            key=f"job_doc_type_{job_id}",
        )
        upload_submitted = st.form_submit_button("Upload")

    if upload_submitted:
        if upload is None:
            st.warning("Choose a file first.")
        else:
            try:
                client.upload_document_for_job(
                    job_id=job_id,
                    filename=upload.name,
                    data=upload.getvalue(),
                    document_type=document_type,
                )
            except requests.HTTPError as exc:
                st.error(f"Upload failed: {_error_detail(exc)}")
            else:
                st.success(f"Linked {upload.name} to this job.")

    st.divider()

    max_iterations = st.number_input(
        "Rewrite attempts after the first draft",
        min_value=0,
        max_value=10,
        value=1,
        key=f"job_max_iterations_{job_id}",
        help=(
            "The quality loop always runs this many rewrites (no early exit "
            "on a pass) and ships the best-scoring draft. Higher costs more "
            "LLM calls for a chance at a better draft, not a guarantee."
        ),
    )
    has_eligible_documents = _has_eligible_documents_for_job(client, job_id=job_id)
    if not has_eligible_documents:
        st.info(
            "No eligible documents for this job yet — upload one above, or add an "
            "unlinked document on the Documents tab (the generic pool counts too)."
        )
    if st.button("Generate for this job", type="primary", disabled=not has_eligible_documents):
        with st.spinner("Running the pipeline — job analysis, retrieval, writing, verification…"):
            try:
                result = client.generate_for_job(job_id=job_id, max_iterations=int(max_iterations))
            except requests.HTTPError as exc:
                st.error(f"Generation failed: {_error_detail(exc)}")
                return
            except requests.RequestException as exc:
                st.error(f"Could not reach the backend: {exc}")
                return
        _render_generation_result(result)

    st.divider()
    st.subheader("Past resumes for this job")
    try:
        resumes = client.list_resumes_for_job(job_id=job_id)
    except requests.RequestException as exc:
        st.error(f"Could not reach the backend: {exc}")
        return
    if not resumes:
        st.info("No resumes generated for this job yet.")
        return
    for resume in resumes:
        with st.container(border=True):
            st.write(
                f"Trust {resume['trust_score']:.0f} · ATS {resume['ats_score']:.0f} · "
                f"iteration {resume['iteration']} · "
                f"{'passed' if resume['passed'] else 'did not pass'} · {resume['created_at']}"
            )
            col_pdf, col_md = st.columns(2)
            col_pdf.download_button(
                "Download PDF",
                data=client.download_resume_pdf(resume_id=resume["id"]),
                file_name=f"resume-{resume['id']}.pdf",
                mime="application/pdf",
                key=f"pdf-{resume['id']}",
            )
            col_md.download_button(
                "Download Markdown",
                data=client.download_resume_markdown(resume_id=resume["id"]),
                file_name=f"resume-{resume['id']}.md",
                mime="text/markdown",
                key=f"md-{resume['id']}",
            )


def main() -> None:
    st.set_page_config(page_title="TrustResume", page_icon="📄", layout="centered")
    st.title("TrustResume")
    st.caption("Evidence-based, ATS-friendly resume generation with trust verification.")

    base_url = st.sidebar.text_input("Backend URL", value=DEFAULT_API_URL)
    user_id = st.sidebar.text_input(
        "User id",
        value=DEFAULT_USER_ID,
        help=(
            "Sent as X-User-Id. Every document, job, and resume is scoped to "
            "it (ADR-0001) — change it to see a completely separate workspace. "
            "Leave blank for the demo user."
        ),
    )
    client = _client(base_url, user_id.strip())

    try:
        client.health()
    except requests.RequestException:
        st.sidebar.error(f"Backend unreachable at {base_url}")
    else:
        st.sidebar.success("Backend connected")

    tab_documents, tab_generate, tab_jobs, tab_search = st.tabs(
        ["📁 Documents", "✨ Generate", "💼 Jobs", "🔍 Search"]
    )
    with tab_documents:
        _render_documents_tab(client)
    with tab_generate:
        _render_generate_tab(client)
    with tab_jobs:
        _render_jobs_tab(client)
    with tab_search:
        _render_search_tab(client)


if __name__ == "__main__":
    main()
