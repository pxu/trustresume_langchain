"""End-to-end integration test for TrustResumeApp (M7).

Exercises the whole stack wired together — ingestion → retrieval → the agents
→ orchestrator → persistence — with in-memory stores, the deterministic fake
embedder, and a scripted chat model driving the LLM agents. No network, no
model download, no AWS.

Unlike pydantic-ai's ``TestModel()``, which auto-synthesizes valid structured
output for whatever schema an agent requests, LangChain's fakes only replay a
pre-scripted message sequence — so each test scripts exactly the LLM calls its
code path will make, in order, with content engineered to pass the default
quality gate (Trust >= 90, ATS >= 85) on the first attempt. That keeps the
call count deterministic and low (one ``generate()`` with a fresh candidate
profile is 4 calls: Job Description, Candidate Profile, Resume Writer, Trust
Harness — ATS Evaluation is deterministic, no LLM call).
"""

from __future__ import annotations

import chromadb
import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages.tool import ToolCall

from tests.fakes import FakeEmbeddings, FakeToolCallingChatModel
from trustresume.api.app_service import TrustResumeApp, build_default_app
from trustresume.models import DocumentType
from trustresume.storage import connect

# Scripted so the pipeline passes the default gate on the first attempt:
# job.keywords + the draft's content overlap fully (ATS = 100), and the one
# claim is SUPPORTED (Trust = 100).
_JOB_DESCRIPTION_CALL = AIMessage(
    content="",
    tool_calls=[
        ToolCall(
            name="JobDescription",
            args={
                "raw_text": "ignored — overwritten by the agent",
                "title": "Senior Python Engineer",
                "keywords": ["python", "aws"],
            },
            id="call_jd",
        )
    ],
)
_CANDIDATE_PROFILE_CALL = AIMessage(
    content="",
    tool_calls=[
        ToolCall(
            name="CandidateProfile",
            args={"name": "Ada", "skills": ["python", "aws"]},
            id="call_cp",
        )
    ],
)
_RESUME_DRAFT_CALL = AIMessage(
    content="",
    tool_calls=[
        ToolCall(
            name="ResumeDraft",
            args={
                "summary": "Experienced python and aws engineer.",
                "sections": [{"heading": "Skills", "bullets": ["python", "aws"]}],
                "iteration": 0,
            },
            id="call_resume",
        )
    ],
)
_TRUST_CALL = AIMessage(
    content="",
    tool_calls=[
        ToolCall(
            name="_ClaimExtraction",
            args={
                "claims": [
                    {
                        "text": "Built python services on AWS",
                        "category": "SKILL",
                        "status": "SUPPORTED",
                    }
                ]
            },
            id="call_trust",
        )
    ],
)

# One full `generate()` call with a fresh (not-yet-cached) candidate profile.
_FULL_GENERATION = [_JOB_DESCRIPTION_CALL, _CANDIDATE_PROFILE_CALL, _RESUME_DRAFT_CALL, _TRUST_CALL]
# A `generate()` call that hits a warm candidate-profile cache (no CP call).
_CACHED_PROFILE_GENERATION = [_JOB_DESCRIPTION_CALL, _RESUME_DRAFT_CALL, _TRUST_CALL]


def _app_with_scripted_calls(*call_groups: list[AIMessage]) -> TrustResumeApp:
    messages = [msg for group in call_groups for msg in group]
    model = FakeToolCallingChatModel(messages=iter(messages))
    return TrustResumeApp(
        connection=connect(":memory:"),
        chroma_client=chromadb.EphemeralClient(),
        embedder=FakeEmbeddings(),
        model=model,
    )


@pytest.fixture
def app() -> TrustResumeApp:
    """An in-memory app scripted for exactly one full `generate()` call."""
    return _app_with_scripted_calls(_FULL_GENERATION)


def test_buildDefaultApp_unknownProvider_raises() -> None:
    from trustresume.api.model_factory import LLMConfig

    with pytest.raises(ValueError, match="unknown LLM provider"):
        build_default_app(llm_config=LLMConfig(provider="nope"))


def test_ensureUser_isIdempotent(app: TrustResumeApp) -> None:
    assert app.ensure_user("Ada", user_id="u1") == "u1"
    # Second call must not raise (user already exists).
    assert app.ensure_user("Ada", user_id="u1") == "u1"


def test_addDocument_thenListed(app: TrustResumeApp) -> None:
    app.ensure_user("Ada", user_id="u1")
    app.add_document(
        user_id="u1",
        filename="resume.txt",
        text="Built Python services on AWS. Led a team of five.",
        document_type=DocumentType.RESUME,
    )
    docs = app.list_documents("u1")
    assert len(docs) == 1
    assert docs[0]["filename"] == "resume.txt"


def test_generate_runsPipelineAndPersists(app: TrustResumeApp) -> None:
    app.ensure_user("Ada", user_id="u1")
    app.add_document(
        user_id="u1",
        filename="resume.txt",
        text="Built Python services on AWS. Led a team of five engineers.",
        document_type=DocumentType.RESUME,
    )

    state = app.generate(user_id="u1", job_posting="Senior Python Engineer with AWS")

    # The orchestrator produced a scored draft and the workflow state is whole.
    assert state.job is not None
    assert state.candidate_profile is not None
    assert state.evidence is not None
    assert state.current_draft is not None
    assert state.current_trust is not None
    assert state.current_ats is not None
    # Scripted to pass on the first attempt.
    assert state.passed is True
    assert state.iteration == 0


def test_generate_reusesCachedCandidateProfileAcrossCalls() -> None:
    app = _app_with_scripted_calls(
        _FULL_GENERATION,  # generate() #1 — no cache yet, computes the profile
        _CACHED_PROFILE_GENERATION,  # generate() #2 — cache hit, no CP call
        _FULL_GENERATION,  # generate() #3 — re-ingest flagged it stale, recomputes
    )
    app.ensure_user("Ada", user_id="u1")
    app.add_document(user_id="u1", filename="resume.txt", text="Built Python services on AWS.")

    app.generate(user_id="u1", job_posting="Backend Engineer")
    first = app._candidate_profiles.get("u1")
    assert first is not None and first.stale is False

    # A second generation against a different job must not recompute the
    # candidate profile — it's job-independent and nothing was re-ingested.
    app.generate(user_id="u1", job_posting="Data Engineer")
    second = app._candidate_profiles.get("u1")
    assert second is not None
    assert second.updated_at == first.updated_at

    # Re-ingesting a document flags the cache stale; the next generation
    # recomputes and bumps updated_at.
    app.add_document(user_id="u1", filename="resume2.txt", text="Also led a team of five.")
    app.generate(user_id="u1", job_posting="Data Engineer")
    third = app._candidate_profiles.get("u1")
    assert third is not None
    assert third.updated_at != first.updated_at


def test_generate_isolatesUsers(app: TrustResumeApp) -> None:
    # User u2 has no documents; generation still works but retrieves nothing
    # of u1's — verifying isolation carries through the full pipeline.
    app.ensure_user("Ada", user_id="u1")
    app.ensure_user("Bob", user_id="u2")
    app.add_document(user_id="u1", filename="r.txt", text="Secret Python work for u1")

    state = app.generate(user_id="u2", job_posting="Python role")
    assert state.evidence is not None
    assert state.evidence.chunks == []  # u2 sees none of u1's evidence
