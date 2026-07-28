"""Trust Harness: claim extraction, evidence validation, hallucination detection, scoring.

The reusable verification workflow behind the Trust Harness agent (ADR-0004,
RQ3). The agent is a thin LLM wrapper; the prompt, draft/evidence formatting,
and report assembly live here so they can be reused and tested independently.

Milestone M6 (trust verification + evaluation).
"""

from __future__ import annotations

from .verifier import (
    SYSTEM_PROMPT,
    build_prompt,
    build_trust_report,
    format_draft,
    format_evidence,
)

__all__ = [
    "SYSTEM_PROMPT",
    "build_prompt",
    "build_trust_report",
    "format_draft",
    "format_evidence",
]
