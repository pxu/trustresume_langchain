"""Unit tests for the ResumeDraft schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trustresume.models import ResumeDraft, ResumeSection


def test_resumeDraft_defaults() -> None:
    draft = ResumeDraft()

    assert draft.summary == ""
    assert draft.sections == []
    assert draft.iteration == 0


def test_resumeDraft_negativeIteration_raises() -> None:
    with pytest.raises(ValidationError):
        ResumeDraft(iteration=-1)


def test_resumeSection_requiresHeading() -> None:
    with pytest.raises(ValidationError):
        ResumeSection(heading="")


def test_resumeDraft_withSections() -> None:
    draft = ResumeDraft(
        summary="Backend engineer.",
        sections=[ResumeSection(heading="Skills", bullets=["Python", "AWS"])],
        iteration=2,
    )

    assert draft.sections[0].heading == "Skills"
    assert draft.iteration == 2
