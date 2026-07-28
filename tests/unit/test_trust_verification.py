"""Unit tests for the extracted Trust Harness logic (M6).

These test the reusable functions directly, without an LLM — the agent test in
test_agents.py covers the LLM-wrapper path.
"""

from __future__ import annotations

from trustresume.models import (
    ClaimCategory,
    ClaimStatus,
    EvidenceChunk,
    EvidenceSet,
    ResumeDraft,
    ResumeSection,
    VerifiedClaim,
)
from trustresume.trust_verification import (
    build_prompt,
    build_trust_report,
    format_draft,
    format_evidence,
)


def test_formatDraft_rendersSummaryAndSections() -> None:
    draft = ResumeDraft(
        summary="Engineer",
        sections=[ResumeSection(heading="Skills", bullets=["Python", "AWS"])],
    )
    rendered = format_draft(draft)
    assert "Engineer" in rendered
    assert "# Skills" in rendered
    assert "- Python" in rendered


def test_formatDraft_emptyDraft() -> None:
    assert format_draft(ResumeDraft()) == "(empty draft)"


def test_formatEvidence_citesChunkIds() -> None:
    evidence = EvidenceSet(
        user_id="u1",
        query="q",
        chunks=[EvidenceChunk(chunk_id="c1", user_id="u1", document_id="d1", text="Python work")],
    )
    rendered = format_evidence(evidence)
    assert "[c1]" in rendered
    assert "Python work" in rendered


def test_formatEvidence_noChunks() -> None:
    assert format_evidence(EvidenceSet(user_id="u1", query="q")) == "(no candidate evidence)"


def test_buildPrompt_includesBothSections() -> None:
    draft = ResumeDraft(summary="Engineer")
    evidence = EvidenceSet(user_id="u1", query="q")
    prompt = build_prompt(draft, evidence)
    assert "## Resume draft" in prompt
    assert "## Candidate evidence" in prompt


def test_buildTrustReport_appliesRubricAndIteration() -> None:
    claims = [
        VerifiedClaim(
            text="Knows Python", category=ClaimCategory.SKILL, status=ClaimStatus.SUPPORTED
        ),
        VerifiedClaim(
            text="Knows K8s", category=ClaimCategory.SKILL, status=ClaimStatus.UNSUPPORTED
        ),
    ]
    report = build_trust_report(claims, iteration=2)
    assert report.score == 50.0  # rubric: 1 supported of 2
    assert report.iteration == 2
    assert len(report.hallucinations) == 1
