"""Unit tests for ``agents/base.py``'s ``ensure_type``/``with_structured_retry``.

Every LLM-backed agent uses ``ensure_type`` to narrow ``with_structured_output``'s
untyped return, and ``with_structured_retry`` to survive a transient provider
failure — these tests cover both helpers directly rather than only through
each agent (which only ever exercises the success path).
"""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.runnables import Runnable

from tests.fakes import FakeToolCallingChatModel
from trustresume.agents.base import ensure_type, with_structured_retry
from trustresume.models import JobDescription


def test_ensureType_matchingType_returnsItUnchanged() -> None:
    job = JobDescription(raw_text="jd")
    assert ensure_type(job, JobDescription) is job


def test_ensureType_wrongType_raisesTypeErrorWithBothTypeNames() -> None:
    with pytest.raises(TypeError, match="expected JobDescription.*got str"):
        ensure_type("not a job description", JobDescription)


def test_ensureType_none_raisesTypeError() -> None:
    with pytest.raises(TypeError, match="got NoneType"):
        ensure_type(None, JobDescription)


class _FlakyThenSucceedsModel(FakeToolCallingChatModel):
    """Raises a transient error on its first ``n_failures`` calls, then
    replays its scripted message — simulates a Bedrock throttle/network blip
    partway through a call that otherwise would have succeeded.

    Private attributes (leading underscore) rather than plain instance
    attributes: the base class is a Pydantic model, which rejects
    ``self.attempts = 0``-style assignment of fields it doesn't declare.
    """

    _n_failures: int = 0
    _attempts: int = 0

    def __init__(self, *, n_failures: int, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._n_failures = n_failures
        self._attempts = 0

    async def ainvoke(self, *args: object, **kwargs: object) -> object:  # type: ignore[override]
        self._attempts += 1
        if self._attempts <= self._n_failures:
            raise RuntimeError("transient provider error")
        return await super().ainvoke(*args, **kwargs)


def _model(n_failures: int) -> _FlakyThenSucceedsModel:
    return _FlakyThenSucceedsModel(
        n_failures=n_failures,
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(name="JobDescription", args={"raw_text": "jd"}, id="call_1")
                    ],
                )
            ]
        ),
    )


def test_withStructuredRetry_transientFailureThenSuccess_returnsResult() -> None:
    structured = with_structured_retry(_model(n_failures=2), JobDescription)
    assert isinstance(structured, Runnable)

    result = asyncio.run(structured.ainvoke("some job posting"))

    assert isinstance(result, JobDescription)
    assert result.raw_text == "jd"


def test_withStructuredRetry_exhaustsAttempts_raises() -> None:
    structured = with_structured_retry(_model(n_failures=5), JobDescription)

    with pytest.raises(RuntimeError, match="transient provider error"):
        asyncio.run(structured.ainvoke("some job posting"))
