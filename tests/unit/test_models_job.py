"""Unit tests for the JobDescription schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trustresume.models import JobDescription, SeniorityLevel


def test_jobDescription_minimalInput_appliesDefaults() -> None:
    job = JobDescription(raw_text="We need a backend engineer.")

    assert job.title is None
    assert job.company is None
    assert job.seniority is SeniorityLevel.UNKNOWN
    assert job.required_skills == []
    assert job.preferred_skills == []
    assert job.responsibilities == []
    assert job.keywords == []


def test_jobDescription_fullInput_roundTrips() -> None:
    job = JobDescription(
        raw_text="Senior Backend Engineer at Acme",
        title="Senior Backend Engineer",
        company="Acme",
        seniority=SeniorityLevel.SENIOR,
        required_skills=["Python", "AWS"],
        keywords=["python", "aws"],
    )

    dumped = job.model_dump()
    assert dumped["seniority"] == "SENIOR"
    assert JobDescription(**dumped) == job


def test_jobDescription_emptyRawText_raises() -> None:
    with pytest.raises(ValidationError):
        JobDescription(raw_text="")


def test_jobDescription_unknownField_forbidden() -> None:
    with pytest.raises(ValidationError):
        JobDescription(raw_text="x", nonsense="y")  # type: ignore[call-arg]
