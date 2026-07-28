"""ATS Evaluation agent — draft + job -> ``ATSReport``.

Fifth step in the pipeline (ADR-0002). Scores how well the draft would fare
against an Applicant Tracking System via keyword coverage against the job's
target keywords.

Like the Trust Harness, the **score is computed deterministically in code**,
not self-reported by an LLM — reproducible and cheap. The scoring itself lives
in ``trustresume.evaluation`` (M6); this agent is a thin wrapper that keeps the
uniform ``run(...)`` shape the orchestrator calls.

Milestone M4 (agents); scoring logic extracted in M6.
"""

from __future__ import annotations

from trustresume.evaluation import score_keywords
from trustresume.models import ATSReport, JobDescription, ResumeDraft


class ATSEvaluationAgent:
    """Scores a draft's ATS keyword coverage against a job description."""

    async def run(self, *, draft: ResumeDraft, job: JobDescription) -> ATSReport:
        """Compute keyword coverage and return an :class:`ATSReport`."""
        return score_keywords(draft, job)
