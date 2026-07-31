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
from trustresume.api.test_provider import (
    AutoStructuredFakeChatModel,
    _synthesize_object,
    _synthesize_value,
)
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


# --- _synthesize_value / _synthesize_object: direct unit tests -------------
#
# The agent schemas above only exercise string/array fields (all optional).
# These hit the branches no real agent schema needs today (enum, integer,
# number, boolean, anyOf union) so the synthesizer stays correct if a future
# schema does use them.


def test_synthesizeValue_enum_returnsFirstOption() -> None:
    assert _synthesize_value({"enum": ["SUPPORTED", "UNSUPPORTED"]}) == "SUPPORTED"


def test_synthesizeValue_string_emptyUnlessMinLengthRequires() -> None:
    assert _synthesize_value({"type": "string"}) == ""
    assert _synthesize_value({"type": "string", "minLength": 1}) == "test"


def test_synthesizeValue_integer_usesMinimumOrZero() -> None:
    assert _synthesize_value({"type": "integer"}) == 0
    assert _synthesize_value({"type": "integer", "minimum": 5}) == 5


def test_synthesizeValue_number_usesMinimumOrZero() -> None:
    assert _synthesize_value({"type": "number"}) == 0.0
    assert _synthesize_value({"type": "number", "minimum": 1.5}) == 1.5


def test_synthesizeValue_boolean_returnsFalse() -> None:
    assert _synthesize_value({"type": "boolean"}) is False


def test_synthesizeValue_array_returnsEmptyList() -> None:
    assert _synthesize_value({"type": "array"}) == []


def test_synthesizeValue_object_recursesIntoSynthesizeObject() -> None:
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string", "minLength": 1}},
        "required": ["name"],
    }
    assert _synthesize_value(schema) == {"name": "test"}


def test_synthesizeValue_anyOf_prefersFirstNonNullBranch() -> None:
    schema = {"anyOf": [{"type": "null"}, {"type": "string", "minLength": 1}]}
    assert _synthesize_value(schema) == "test"


def test_synthesizeValue_anyOf_allNull_returnsNone() -> None:
    assert _synthesize_value({"anyOf": [{"type": "null"}]}) is None


def test_synthesizeValue_default_usedWhenNoRecognizedType() -> None:
    assert _synthesize_value({"default": "fallback"}) == "fallback"


def test_synthesizeValue_unrecognizedSchema_returnsNone() -> None:
    assert _synthesize_value({}) is None


def test_synthesizeObject_onlyFillsRequiredFields() -> None:
    schema = {
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "age": {"type": "integer"},
            "optional_field": {"type": "string"},
        },
        "required": ["name", "age"],
    }
    result = _synthesize_object(schema)
    assert result == {"name": "test", "age": 0}
    assert "optional_field" not in result
