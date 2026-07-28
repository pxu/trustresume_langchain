"""Unit tests for the agents.

LLM-backed agents (Job Description, Candidate Profile, Resume Writer, Trust
Harness) are exercised with a ``FakeToolCallingChatModel`` scripted via
``scripted_tool_call`` — the LangChain analog of pydantic-ai's ``TestModel``
+ ``custom_output_args`` — so the structured output is deterministic and no
network/model is needed. The deterministic agents (Evidence Retrieval, ATS
Evaluation) are tested against their real in-process collaborators.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

import chromadb
import pytest

from tests.fakes import FakeEmbeddings, scripted_tool_call
from trustresume.agents import (
    ATSEvaluationAgent,
    CandidateProfileAgent,
    EvidenceRetrievalAgent,
    JobDescriptionAgent,
    ResumeWriterAgent,
    TrustHarnessAgent,
)
from trustresume.models import (
    ClaimStatus,
    DocumentType,
    EvidenceChunk,
    EvidenceSet,
    JobDescription,
    ResumeDraft,
    ResumeSection,
    SeniorityLevel,
)
from trustresume.retrieval import ChromaVectorStore

_T = TypeVar("_T")


def run(awaitable: Awaitable[_T]) -> _T:
    """Drive an agent's async ``run`` to completion without an async plugin.

    Keeps the test suite dependency-free — pytest-asyncio/anyio aren't declared
    dependencies, and each agent call is a self-contained coroutine.
    """
    return asyncio.run(awaitable)  # type: ignore[arg-type]


# --- Job Description --------------------------------------------------------


def test_jobDescriptionAgent_structuresPostingAndPreservesRawText() -> None:
    tm = scripted_tool_call(
        "JobDescription",
        {
            "raw_text": "ignored — overwritten",
            "title": "Senior Backend Engineer",
            "seniority": "SENIOR",
            "required_skills": ["Python", "AWS"],
            "keywords": ["python", "aws"],
        },
    )
    agent = JobDescriptionAgent(tm)
    posting = "We seek a Senior Backend Engineer with Python and AWS."
    job = run(agent.run(posting))

    assert job.title == "Senior Backend Engineer"
    assert job.seniority is SeniorityLevel.SENIOR
    # raw_text is force-preserved from the input, not the model's echo.
    assert job.raw_text == posting


# --- Candidate Profile ------------------------------------------------------


def test_candidateProfileAgent_structuresCandidateText() -> None:
    tm = scripted_tool_call(
        "CandidateProfile",
        {
            "name": "Jordan Rivera",
            "summary": "Backend engineer with 5 years of experience.",
            "skills": ["Python", "AWS"],
            "certifications": ["AWS Certified Solutions Architect"],
        },
    )
    agent = CandidateProfileAgent(tm)
    candidate_text = "Jordan Rivera — backend engineer, Python, AWS, AWS SA cert."
    profile = run(agent.run(candidate_text))

    assert profile.name == "Jordan Rivera"
    assert profile.skills == ["Python", "AWS"]
    assert profile.certifications == ["AWS Certified Solutions Architect"]


# --- Evidence Retrieval (deterministic) ------------------------------------


def test_retrievalAgent_buildsQueryAndReturnsUserChunks(
    fake_embedder: FakeEmbeddings,
) -> None:
    store = ChromaVectorStore(chromadb.EphemeralClient(), fake_embedder, collection_name="test-t1")
    store.upsert_chunks(
        [
            EvidenceChunk(
                chunk_id="c1",
                user_id="u1",
                document_id="d1",
                document_type=DocumentType.RESUME,
                text="Python and AWS backend work",
            )
        ]
    )
    agent = EvidenceRetrievalAgent(store, top_k=5)
    job = JobDescription(raw_text="jd", required_skills=["Python", "AWS"], keywords=["python"])

    evidence = run(agent.run(user_id="u1", job=job))
    assert evidence.user_id == "u1"
    assert "Python" in evidence.query
    assert [c.chunk_id for c in evidence.chunks] == ["c1"]


def test_retrievalAgent_fallsBackToRawTextWhenNoStructuredFields(
    fake_embedder: FakeEmbeddings,
) -> None:
    store = ChromaVectorStore(chromadb.EphemeralClient(), fake_embedder, collection_name="test-t2")
    agent = EvidenceRetrievalAgent(store)
    job = JobDescription(raw_text="raw posting body")

    evidence = run(agent.run(user_id="u1", job=job))
    assert evidence.query == "raw posting body"


# --- Resume Writer ---------------------------------------------------------


def test_resumeAgent_generatesDraftAndStampsIteration() -> None:
    tm = scripted_tool_call(
        "ResumeDraft",
        {
            "summary": "Backend engineer with Python and AWS experience.",
            "sections": [{"heading": "Skills", "bullets": ["Python", "AWS"]}],
            "iteration": 0,
        },
    )
    agent = ResumeWriterAgent(tm)
    job = JobDescription(raw_text="jd", required_skills=["Python"])
    evidence = EvidenceSet(
        user_id="u1",
        query="python",
        chunks=[EvidenceChunk(chunk_id="c1", user_id="u1", document_id="d1", text="Python work")],
    )
    draft = run(agent.run(job=job, evidence=evidence, feedback="add AWS", iteration=2))

    assert draft.summary.startswith("Backend engineer")
    # iteration is stamped by the agent, overriding whatever the model returned.
    assert draft.iteration == 2


# --- Trust Harness ---------------------------------------------------------


def test_trustAgent_scoresFromClassifiedClaims() -> None:
    tm = scripted_tool_call(
        "_ClaimExtraction",
        {
            "claims": [
                {"text": "Knows Python", "category": "SKILL", "status": "SUPPORTED"},
                {"text": "Knows K8s", "category": "SKILL", "status": "UNSUPPORTED"},
            ]
        },
    )
    agent = TrustHarnessAgent(tm)
    draft = ResumeDraft(
        summary="Engineer",
        sections=[ResumeSection(heading="Skills", bullets=["Python", "Kubernetes"])],
        iteration=1,
    )
    evidence = EvidenceSet(user_id="u1", query="python", chunks=[])
    report = run(agent.run(draft=draft, evidence=evidence))

    # Deterministic rubric: 1 SUPPORTED + 1 UNSUPPORTED = 50.0, computed in code.
    assert report.score == pytest.approx(50.0)
    assert report.iteration == 1
    assert len(report.hallucinations) == 1
    assert report.hallucinations[0].status is ClaimStatus.UNSUPPORTED


# --- ATS Evaluation (deterministic) ----------------------------------------


def test_atsAgent_computesKeywordCoverage() -> None:
    agent = ATSEvaluationAgent()
    draft = ResumeDraft(
        summary="Built Python services",
        sections=[ResumeSection(heading="Skills", bullets=["AWS Lambda"])],
    )
    job = JobDescription(raw_text="jd", keywords=["Python", "AWS", "Kubernetes"])

    report = run(agent.run(draft=draft, job=job))
    assert set(report.matched_keywords) == {"Python", "AWS"}
    assert report.missing_keywords == ["Kubernetes"]
    assert report.score == pytest.approx(round(100.0 * 2 / 3, 2))


def test_atsAgent_noKeywords_scoresFull() -> None:
    agent = ATSEvaluationAgent()
    report = run(agent.run(draft=ResumeDraft(summary="x"), job=JobDescription(raw_text="jd")))
    assert report.score == 100.0
