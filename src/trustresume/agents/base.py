"""Shared plumbing for the agents.

Per ADR-0002/0003 every agent is a *pure input -> output step*: it receives
everything it needs as arguments, returns a model from ``trustresume.models``,
and never calls another agent or reaches into orchestrator state. The
orchestrator sequences them.

The model is injected, not hard-wired. In production the orchestrator passes
a Bedrock/OpenAI/Google chat model (``api/model_factory.py``); in tests a
``GenericFakeChatModel`` scripted with the exact structured-output tool call
expected. Unlike pydantic-ai's ``Agent.override(model=...)``, LangChain has no
swap-the-model-after-construction hook, so every agent requires its model at
construction time rather than defaulting to ``None``.

Milestone M4 (agents).
"""

from __future__ import annotations

from typing import TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

# Re-exported so agents keep importing prompt-injection defense from this
# module (their one shared entry point) even though the definitions
# themselves live in trustresume.prompting — kept framework-independent
# there so trust_verification/ (ADR-0004: zero LangChain dependency) can use
# them too without pulling in langchain_core.
from trustresume.prompting import UNTRUSTED_INPUT_NOTICE, wrap_untrusted

# What every agent constructor accepts for its model. Kept in one alias so the
# whole package presents a consistent injection point.
ModelInput = BaseChatModel

__all__ = [
    "ModelInput",
    "UNTRUSTED_INPUT_NOTICE",
    "wrap_untrusted",
    "with_structured_retry",
    "ensure_type",
]

_T = TypeVar("_T")


# A generation makes 5-9+ sequential LLM calls (one per agent per quality-loop
# iteration); without a retry, a single transient failure partway through
# (a Bedrock throttle, a momentary network blip) loses every call already
# made and fails the whole run. Neither ChatBedrockConverse/ChatOpenAI/
# ChatGoogleGenerativeAI take a retry constructor arg, so this wraps the
# bound structured-output runnable instead — ``Runnable.with_retry`` composes
# with ``with_structured_output`` cleanly (retries the whole
# invoke-and-parse), unlike wrapping the raw model (which returns a
# ``RunnableRetry`` with no ``with_structured_output`` method).
def with_structured_retry(model: BaseChatModel, schema: type[_T]) -> Runnable[object, _T]:
    """``model.with_structured_output(schema)``, retried up to 3 attempts."""
    return model.with_structured_output(schema).with_retry(stop_after_attempt=3)  # type: ignore[return-value]


def ensure_type(result: object, expected: type[_T]) -> _T:
    """Narrow ``with_structured_output``'s ``Any`` return to the bound schema.

    LangChain's ``Runnable.ainvoke`` types its return as ``Any`` regardless of
    what schema ``with_structured_output`` bound, so every agent needs a
    runtime check to get a typed result back — and to actually notice if a
    future LangChain/provider version ever returns something else (a raw
    dict, an unparsed message) instead of raising cleanly. A bare
    ``assert isinstance(...)`` would silently disappear under ``python -O``,
    turning that into a confusing ``AttributeError`` deep in the caller
    instead of a clear error right here.
    """
    if not isinstance(result, expected):
        raise TypeError(
            f"expected {expected.__name__} from with_structured_output, got {type(result).__name__}"
        )
    return result
