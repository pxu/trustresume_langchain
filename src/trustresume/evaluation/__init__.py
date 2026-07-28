"""ATS evaluation: keyword coverage, experience relevance, and skill alignment scoring.

The reusable scoring behind the ATS Evaluation agent. Deterministic keyword
coverage lives here as a pure function; the agent is a thin wrapper over it.
``required_skill_coverage`` is a standalone cross-check (exact-token, not
substring) — not called by the agent or the orchestrator; it exists for
callers who want to compare against ``score_keywords``' results directly
(e.g. `notebooks/week3_baseline_model.py`).

Milestone M6 (trust verification + evaluation).
"""

from __future__ import annotations

from .scorer import dedupe, draft_text, required_skill_coverage, score_keywords

__all__ = ["score_keywords", "draft_text", "dedupe", "required_skill_coverage"]
