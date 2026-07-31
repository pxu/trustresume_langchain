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


def test_retrievalAgent_documentIdsFilter_scopesRetrieval(
    fake_embedder: FakeEmbeddings,
) -> None:
    """document_ids, when given, must be threaded straight through to the
    retriever — this agent has no logic of its own for it, only pass-through.
    """
    store = ChromaVectorStore(chromadb.EphemeralClient(), fake_embedder, collection_name="test-t1b")
    store.upsert_chunks(
        [
            EvidenceChunk(
                chunk_id="c1",
                user_id="u1",
                document_id="d1",
                document_type=DocumentType.RESUME,
                text="Python and AWS backend work",
            ),
            EvidenceChunk(
                chunk_id="c2",
                user_id="u1",
                document_id="d2",
                document_type=DocumentType.RESUME,
                text="Python and AWS backend work too",
            ),
        ]
    )
    agent = EvidenceRetrievalAgent(store, top_k=5)
    job = JobDescription(raw_text="jd", required_skills=["Python", "AWS"])

    scoped = run(agent.run(user_id="u1", job=job, document_ids=["d1"]))
    unscoped = run(agent.run(user_id="u1", job=job))

    assert [c.chunk_id for c in scoped.chunks] == ["c1"]
    assert {c.chunk_id for c in unscoped.chunks} == {"c1", "c2"}


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
        "_DraftExtraction",
        {
            "summary": "Backend engineer with Python and AWS experience.",
            "sections": [{"heading": "Skills", "bullets": ["Python", "AWS"]}],
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


def test_resumeAgent_fullJobDetails_includesPreferredSkillsAndResponsibilities() -> None:
    """Covers the prompt-formatting branches other tests' bare JobDescriptions skip."""
    tm = scripted_tool_call("_DraftExtraction", {"summary": "Engineer.", "sections": []})
    agent = ResumeWriterAgent(tm)
    job = JobDescription(
        raw_text="jd",
        preferred_skills=["Terraform"],
        responsibilities=["Own the deployment pipeline"],
    )
    evidence = EvidenceSet(user_id="u1", query="terraform", chunks=[])

    draft = run(agent.run(job=job, evidence=evidence))
    assert draft.summary == "Engineer."


def test_resumeAgent_noEvidenceOrJobDetails_stillPromptsWithoutError() -> None:
    """Empty evidence/job fields exercise the agent's fallback prompt text."""
    tm = scripted_tool_call("_DraftExtraction", {"summary": "Generalist engineer.", "sections": []})
    agent = ResumeWriterAgent(tm)
    job = JobDescription(raw_text="jd")  # no title/skills/keywords/responsibilities
    evidence = EvidenceSet(user_id="u1", query="anything", chunks=[])  # no chunks

    draft = run(agent.run(job=job, evidence=evidence))

    assert draft.summary == "Generalist engineer."
    assert draft.iteration == 0


def test_resumeAgent_emptySectionHeading_relabeledNotDropped() -> None:
    """Regression test: real models (observed with Bedrock Claude) sometimes
    emit a section with an empty ``heading`` — bullets grouped under no real
    heading. ``ResumeDraft.sections[].heading`` requires ``min_length=1``, so
    binding it directly to the model crashes structured-output parsing and
    loses the whole draft (a real failure hit end-to-end against Bedrock, not
    a hypothetical). The agent binds a lenient private schema instead and
    relabels empty headings rather than dropping the bullets.
    """
    tm = scripted_tool_call(
        "_DraftExtraction",
        {
            "summary": "Engineer.",
            "sections": [
                {"heading": "Skills", "bullets": ["Python"]},
                {"heading": "", "bullets": ["Led a migration project"]},
                {"heading": "", "bullets": []},  # no bullets — dropped
            ],
        },
    )
    agent = ResumeWriterAgent(tm)
    job = JobDescription(raw_text="jd")
    evidence = EvidenceSet(user_id="u1", query="anything", chunks=[])

    draft = run(agent.run(job=job, evidence=evidence))

    assert [s.heading for s in draft.sections] == ["Skills", "Additional Information"]
    assert draft.sections[1].bullets == ["Led a migration project"]


def test_resumeAgent_whitespaceOnlySectionHeading_relabeledNotDropped() -> None:
    # A heading of "   " is truthy (`or` alone wouldn't relabel it) but is
    # exactly as un-real as an empty one — must be relabeled the same way.
    tm = scripted_tool_call(
        "_DraftExtraction",
        {
            "summary": "Engineer.",
            "sections": [{"heading": "   ", "bullets": ["Led a migration project"]}],
        },
    )
    agent = ResumeWriterAgent(tm)
    job = JobDescription(raw_text="jd")
    evidence = EvidenceSet(user_id="u1", query="anything", chunks=[])

    draft = run(agent.run(job=job, evidence=evidence))

    assert [s.heading for s in draft.sections] == ["Additional Information"]


def test_resumeAgent_bulletlessGroupHeading_dropped() -> None:
    """Regression test: a real generated resume showed "Professional
    Experience" rendered as its own heading with no bullets, immediately
    followed by a per-employer section — the model using it as a bare group
    label. ``ResumeSection`` is flat (no nested sub-sections), so a section
    with a heading but zero bullets carries no content and would otherwise
    render as a stray, duplicate-looking heading right above the real one.
    """
    tm = scripted_tool_call(
        "_DraftExtraction",
        {
            "summary": "Engineer.",
            "sections": [
                {"heading": "Professional Experience", "bullets": []},
                {"heading": "Senior Engineer | Acme Corp", "bullets": ["Shipped things."]},
            ],
        },
    )
    agent = ResumeWriterAgent(tm)
    job = JobDescription(raw_text="jd")
    evidence = EvidenceSet(user_id="u1", query="anything", chunks=[])

    draft = run(agent.run(job=job, evidence=evidence))

    assert [s.heading for s in draft.sections] == ["Senior Engineer | Acme Corp"]


@pytest.mark.parametrize("heading", ["Summary", "Professional Summary", "Profile", "Objective"])
def test_resumeAgent_summaryEmittedAsSection_foldedIntoSummaryField(heading: str) -> None:
    """Regression test: real models (observed with Bedrock Claude) sometimes
    put the professional summary in a "Summary"/"Professional Summary" style
    section instead of the dedicated `summary` field, despite the prompt and
    the field description saying not to — the same "prompt says X, code
    enforces X" pattern as the empty-heading handling above. Only the FIRST
    section is treated as a possible summary (a later section with the same
    heading is a real, separate one — e.g. a resume that legitimately has an
    "Objective" section further down isn't this case).
    """
    tm = scripted_tool_call(
        "_DraftExtraction",
        {
            "summary": "",
            "sections": [
                {"heading": heading, "bullets": ["Senior engineer.", "12 years experience."]},
                {"heading": "Skills", "bullets": ["Python"]},
            ],
        },
    )
    agent = ResumeWriterAgent(tm)
    job = JobDescription(raw_text="jd")
    evidence = EvidenceSet(user_id="u1", query="anything", chunks=[])

    draft = run(agent.run(job=job, evidence=evidence))

    assert draft.summary == "Senior engineer. 12 years experience."
    assert [s.heading for s in draft.sections] == ["Skills"]


def test_resumeAgent_summaryFieldAlreadySet_sectionLeftAlone() -> None:
    """A later section that happens to be titled 'Summary' but isn't the
    first section (or the model already used the `summary` field correctly)
    must not be folded away — only the fallback case is special-cased.
    """
    tm = scripted_tool_call(
        "_DraftExtraction",
        {
            "summary": "Already set correctly.",
            "sections": [{"heading": "Skills", "bullets": ["Python"]}],
        },
    )
    agent = ResumeWriterAgent(tm)
    job = JobDescription(raw_text="jd")
    evidence = EvidenceSet(user_id="u1", query="anything", chunks=[])

    draft = run(agent.run(job=job, evidence=evidence))

    assert draft.summary == "Already set correctly."
    assert [s.heading for s in draft.sections] == ["Skills"]


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


def test_trustAgent_citationToUnknownChunkId_isDropped() -> None:
    # A citation the model invents (or copies from an unrelated run) rather
    # than an id it actually retrieved must not survive into the report — an
    # unchecked citation would be indistinguishable from a real one to
    # anything reading the report for an audit trail (ADR-0004).
    real_chunk = EvidenceChunk(
        chunk_id="real-1",
        user_id="u1",
        document_id="d1",
        document_type=DocumentType.RESUME,
        source_document="resume.txt",
        text="Built services with Python.",
    )
    tm = scripted_tool_call(
        "_ClaimExtraction",
        {
            "claims": [
                {
                    "text": "Knows Python",
                    "category": "SKILL",
                    "status": "SUPPORTED",
                    "evidence_chunk_ids": ["real-1", "hallucinated-id"],
                }
            ]
        },
    )
    agent = TrustHarnessAgent(tm)
    draft = ResumeDraft(
        summary="Engineer",
        sections=[ResumeSection(heading="Skills", bullets=["Python"])],
    )
    evidence = EvidenceSet(user_id="u1", query="python", chunks=[real_chunk])

    report = run(agent.run(draft=draft, evidence=evidence))

    assert report.claims[0].evidence_chunk_ids == ["real-1"]


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
