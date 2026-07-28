"""Unit tests for the extracted ATS scoring logic (M6)."""

from __future__ import annotations

import pytest

from trustresume.evaluation import dedupe, draft_text, required_skill_coverage, score_keywords
from trustresume.models import JobDescription, ResumeDraft, ResumeSection


def test_dedupe_caseInsensitiveOrderPreserving() -> None:
    assert dedupe(["Python", "python", "AWS", "aws", "Go"]) == ["Python", "AWS", "Go"]


def test_draftText_flattensAndLowercases() -> None:
    draft = ResumeDraft(
        summary="Built Services",
        sections=[ResumeSection(heading="Skills", bullets=["Python"])],
    )
    text = draft_text(draft)
    assert "built services" in text
    assert "python" in text


def test_scoreKeywords_partialCoverage() -> None:
    draft = ResumeDraft(summary="Built Python services on AWS")
    job = JobDescription(raw_text="jd", keywords=["Python", "AWS", "Kubernetes"])
    report = score_keywords(draft, job)

    assert set(report.matched_keywords) == {"Python", "AWS"}
    assert report.missing_keywords == ["Kubernetes"]
    assert report.score == pytest.approx(round(100.0 * 2 / 3, 2))


def test_scoreKeywords_fallsBackToRequiredSkills() -> None:
    draft = ResumeDraft(summary="Python developer")
    job = JobDescription(raw_text="jd", required_skills=["Python", "Rust"])
    report = score_keywords(draft, job)
    assert report.matched_keywords == ["Python"]
    assert report.missing_keywords == ["Rust"]


def test_scoreKeywords_noKeywords_scoresFull() -> None:
    report = score_keywords(ResumeDraft(summary="x"), JobDescription(raw_text="jd"))
    assert report.score == 100.0


def test_requiredSkillCoverage_exactTokenMatch_caseInsensitive() -> None:
    job = JobDescription(raw_text="jd", required_skills=["Python", "AWS", "Kubernetes"])
    result = required_skill_coverage(job, ["python", "aws"])
    assert result["matched"] == ["Python", "AWS"]
    assert result["missing"] == ["Kubernetes"]
    assert result["coverage"] == pytest.approx(2 / 3)


def test_requiredSkillCoverage_noSubstringMatch() -> None:
    """Unlike score_keywords, a short required skill must match a whole token."""
    job = JobDescription(raw_text="jd", required_skills=["SQL"])
    result = required_skill_coverage(job, ["PostgreSQL"])
    assert result["matched"] == []
    assert result["missing"] == ["SQL"]
    assert result["coverage"] == 0.0


def test_requiredSkillCoverage_noRequiredSkills_scoresFull() -> None:
    job = JobDescription(raw_text="jd")
    result = required_skill_coverage(job, ["Python"])
    assert result["coverage"] == 1.0
    assert result["matched"] == []
    assert result["missing"] == []
