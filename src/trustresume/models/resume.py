"""Resume Writer output: a structured draft resume.

The Resume Writer agent produces a ``ResumeDraft`` from the job description and
retrieved evidence. It is a pure generation step — it never scores itself; the
Trust Harness and ATS Evaluation judge it afterward (ADR-0004).

Milestone M1 (shared models).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ResumeSection(BaseModel):
    """A titled block of resume content (e.g. 'Experience', 'Skills')."""

    model_config = ConfigDict(extra="forbid")

    heading: str = Field(..., min_length=1, description="Section heading.")
    bullets: list[str] = Field(
        default_factory=list,
        description="Bullet points / lines under this section.",
    )


class ResumeDraft(BaseModel):
    """A single generated resume draft.

    ``iteration`` records which pass of the quality loop produced it (0 = the
    initial generation), so the orchestrator can track loop progress and the
    UI can show which draft is being exported (ADR-0005).
    """

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(default="", description="Professional summary / headline paragraph.")
    sections: list[ResumeSection] = Field(
        default_factory=list,
        description="Ordered resume sections.",
    )
    iteration: int = Field(
        default=0,
        ge=0,
        description="Quality-loop iteration that produced this draft (0 = initial).",
    )
