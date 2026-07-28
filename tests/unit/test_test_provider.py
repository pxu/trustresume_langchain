"""Unit tests for the offline ``"test"`` provider's auto-synthesizing fake model.

Verifies it can drive ``with_structured_output`` for each of the app's actual
agent schemas — not just a toy schema — since this is what makes
``TRUSTRESUME_LLM_PROVIDER=test`` able to run the full pipeline without
credentials.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

from trustresume.agents.trust_agent import _ClaimExtraction
from trustresume.api.test_provider import AutoStructuredFakeChatModel
from trustresume.models import CandidateProfile, JobDescription, ResumeDraft

_T = TypeVar("_T")


def run(awaitable: Awaitable[_T]) -> _T:
    return asyncio.run(awaitable)  # type: ignore[arg-type]


def test_noToolsBound_returnsPlainMessage() -> None:
    model = AutoStructuredFakeChatModel()
    result = run(model.ainvoke("hello"))
    assert result.content == "ok"


def test_structuredOutput_jobDescription_satisfiesRequiredField() -> None:
    model = AutoStructuredFakeChatModel()
    job = run(model.with_structured_output(JobDescription).ainvoke("some posting"))
    assert isinstance(job, JobDescription)
    assert job.raw_text  # required, minLength=1 — must be non-empty
    assert job.required_skills == []


def test_structuredOutput_candidateProfile_allFieldsOptional() -> None:
    model = AutoStructuredFakeChatModel()
    profile = run(model.with_structured_output(CandidateProfile).ainvoke("some candidate text"))
    assert isinstance(profile, CandidateProfile)
    assert profile.skills == []


def test_structuredOutput_resumeDraft_appliesDefaults() -> None:
    model = AutoStructuredFakeChatModel()
    draft = run(model.with_structured_output(ResumeDraft).ainvoke("write a resume"))
    assert isinstance(draft, ResumeDraft)
    assert draft.sections == []


def test_structuredOutput_claimExtraction_emptyClaimsIsValid() -> None:
    model = AutoStructuredFakeChatModel()
    extraction = run(model.with_structured_output(_ClaimExtraction).ainvoke("verify this draft"))
    assert isinstance(extraction, _ClaimExtraction)
    assert extraction.claims == []
