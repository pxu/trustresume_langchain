"""Resume Writer agent — job + evidence (+ feedback) -> ``ResumeDraft``.

Third step in the pipeline (ADR-0002). A pure generation step: it writes a
tailored draft grounded in the retrieved evidence and never scores itself —
the Trust Harness and ATS Evaluation judge it afterward (ADR-0004). On a
quality-loop rewrite the orchestrator passes ``feedback`` and the previous
draft's ``iteration``; the agent must address the feedback and must not add
claims the evidence doesn't support.

The job/evidence/feedback are composed into the prompt (rather than exposed as
tools) so generation is deterministic and easy to test.

Milestone M4 (agents).
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from trustresume.models import EvidenceSet, JobDescription, ResumeDraft

from .base import ModelInput

_SYSTEM_PROMPT = """\
You are an expert resume writer. Write a concise, ATS-friendly resume draft \
tailored to the target job, using ONLY the candidate evidence provided. Every \
skill, achievement, and experience you state must be grounded in that \
evidence — never invent skills, employers, dates, or certifications the \
evidence does not support. Prefer the job's keywords where the evidence \
genuinely backs them. If feedback from a previous attempt is provided, address \
each point. Return a structured draft with a summary and sections of bullets."""


def _format_evidence(evidence: EvidenceSet) -> str:
    if not evidence.chunks:
        return "(no candidate evidence retrieved)"
    return "\n".join(f"- [{c.document_type.value}] {c.text}" for c in evidence.chunks)


def _format_job(job: JobDescription) -> str:
    lines = [f"Title: {job.title or '(unspecified)'}", f"Seniority: {job.seniority.value}"]
    if job.required_skills:
        lines.append(f"Required skills: {', '.join(job.required_skills)}")
    if job.preferred_skills:
        lines.append(f"Preferred skills: {', '.join(job.preferred_skills)}")
    if job.keywords:
        lines.append(f"Keywords to target: {', '.join(job.keywords)}")
    if job.responsibilities:
        lines.append("Responsibilities:\n" + "\n".join(f"- {r}" for r in job.responsibilities))
    return "\n".join(lines)


class ResumeWriterAgent:
    """Wraps a LangChain structured-output call that drafts an evidence-grounded resume."""

    def __init__(self, model: ModelInput) -> None:
        self._structured = model.with_structured_output(ResumeDraft)

    async def run(
        self,
        *,
        job: JobDescription,
        evidence: EvidenceSet,
        feedback: str | None = None,
        iteration: int = 0,
    ) -> ResumeDraft:
        """Generate a draft; ``iteration`` is stamped onto the result."""
        prompt = (
            f"## Target job\n{_format_job(job)}\n\n"
            f"## Candidate evidence\n{_format_evidence(evidence)}"
        )
        if feedback:
            prompt += f"\n\n## Feedback to address from the previous draft\n{feedback}"

        result = await self._structured.ainvoke(
            [SystemMessage(_SYSTEM_PROMPT), HumanMessage(prompt)]
        )
        assert isinstance(result, ResumeDraft)
        return result.model_copy(update={"iteration": iteration})
