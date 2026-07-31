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

Three tabs, matching the three things a candidate does with the app:
Documents (upload career evidence), Generate (run the full RAG + multi-agent
pipeline against a job posting), Search (inspect retrieval directly).
"""

from __future__ import annotations

import os

import requests
import streamlit as st

from trustresume.models import DocumentType
from trustresume.ui.api_client import TrustResumeClient

DEFAULT_API_URL = os.getenv("TRUSTRESUME_API_URL", "http://localhost:8000")


@st.cache_resource
def _client(base_url: str) -> TrustResumeClient:
    return TrustResumeClient(base_url)


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


def _render_generate_tab(client: TrustResumeClient) -> None:
    st.subheader("Generate a tailored, evidence-checked resume")
    job_posting = st.text_area("Job posting", height=200, placeholder="Paste the job posting…")
    if st.button("Generate", type="primary", disabled=not job_posting.strip()):
        with st.spinner("Running the pipeline — job analysis, retrieval, writing, verification…"):
            try:
                result = client.generate(job_posting=job_posting)
            except requests.HTTPError as exc:
                st.error(f"Generation failed: {_error_detail(exc)}")
                return
            except requests.RequestException as exc:
                st.error(f"Could not reach the backend: {exc}")
                return

        col1, col2, col3 = st.columns(3)
        col1.metric("Trust score", f"{result['trust_score']:.0f}")
        col2.metric("ATS score", f"{result['ats_score']:.0f}")
        col3.metric("Iterations", result["iterations"])

        if result["passed"]:
            st.success("Passed the quality gate.")
        elif result["exhausted"]:
            st.warning("Hit the rewrite cap without passing — showing the last draft anyway.")

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


def main() -> None:
    st.set_page_config(page_title="TrustResume", page_icon="📄", layout="centered")
    st.title("TrustResume")
    st.caption("Evidence-based, ATS-friendly resume generation with trust verification.")

    base_url = st.sidebar.text_input("Backend URL", value=DEFAULT_API_URL)
    client = _client(base_url)

    try:
        client.health()
    except requests.RequestException:
        st.sidebar.error(f"Backend unreachable at {base_url}")
    else:
        st.sidebar.success("Backend connected")

    tab_documents, tab_generate, tab_search = st.tabs(["📁 Documents", "✨ Generate", "🔍 Search"])
    with tab_documents:
        _render_documents_tab(client)
    with tab_generate:
        _render_generate_tab(client)
    with tab_search:
        _render_search_tab(client)


if __name__ == "__main__":
    main()
