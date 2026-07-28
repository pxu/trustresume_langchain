"""FastAPI backend.

Wraps the :class:`app_service.TrustResumeApp` facade (which wires every
milestone together) behind an HTTP API. ``server.create_app`` builds the
FastAPI instance around an injected facade; ``build_default_app`` assembles
the production facade (file stores + an LLM chosen by
:class:`model_factory.LLMConfig` — Bedrock, OpenAI, Gemini, or an offline
``test`` model).

Milestone M7 (API).
"""

from __future__ import annotations

from .app_service import TrustResumeApp, build_default_app
from .server import create_app

__all__ = ["TrustResumeApp", "build_default_app", "create_app"]
