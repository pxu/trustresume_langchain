"""Unit tests for building a semantic-search query from a job description."""

from __future__ import annotations

from trustresume.models import JobDescription
from trustresume.retrieval.query import build_query, per_skill_queries, query_terms


def test_queryTerms_ordersTitleThenSkillsThenKeywords() -> None:
    job = JobDescription(
        raw_text="jd",
        title="Backend Engineer",
        required_skills=["Python"],
        preferred_skills=["AWS"],
        keywords=["kubernetes"],
    )
    assert query_terms(job) == ["Backend Engineer", "Python", "AWS", "kubernetes"]


def test_queryTerms_noTitle_omitsIt() -> None:
    job = JobDescription(raw_text="jd", required_skills=["Python"])
    assert query_terms(job) == ["Python"]


def test_buildQuery_joinsTermsWithSpaces() -> None:
    job = JobDescription(raw_text="jd", title="Engineer", required_skills=["Python"])
    assert build_query(job) == "Engineer Python"


def test_buildQuery_noStructuredFields_fallsBackToRawText() -> None:
    job = JobDescription(raw_text="the original posting body")
    assert build_query(job) == "the original posting body"


def test_perSkillQueries_dedupesRequiredAndPreferred() -> None:
    job = JobDescription(
        raw_text="jd",
        required_skills=["Python", "AWS"],
        preferred_skills=["AWS", "Kubernetes"],
    )
    assert per_skill_queries(job) == ["Python", "AWS", "Kubernetes"]


def test_perSkillQueries_excludesKeywords() -> None:
    job = JobDescription(raw_text="jd", required_skills=["Python"], keywords=["REST API"])
    assert per_skill_queries(job) == ["Python"]


def test_perSkillQueries_noSkills_returnsEmpty() -> None:
    job = JobDescription(raw_text="jd")
    assert per_skill_queries(job) == []
