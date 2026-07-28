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

from langchain_core.language_models import BaseChatModel

# What every agent constructor accepts for its model. Kept in one alias so the
# whole package presents a consistent injection point.
ModelInput = BaseChatModel
