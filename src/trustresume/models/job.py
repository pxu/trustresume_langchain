"""Job Description output: a structured view of a target job description.

Produced by the Job Description agent (the first step in the pipeline) from
raw job-posting text, and consumed by every downstream step — retrieval uses
the skills/keywords as query terms, the Resume Writer targets the
requirements, and ATS Evaluation scores keyword coverage against them.

Milestone M1 (shared models).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .enums import SeniorityLevel


class JobDescription(BaseModel):
    """A parsed, structured job description.

    ``raw_text`` is preserved verbatim so later steps can fall back to the
    source; the remaining fields are the agent's extraction of it.
    """

    model_config = ConfigDict(extra="forbid")

    raw_text: str = Field(..., min_length=1, description="Original job-posting text, verbatim.")
    title: str | None = Field(None, description="Job title, e.g. 'Senior Backend Engineer'.")
    company: str | None = Field(None, description="Hiring company, if stated.")
    seniority: SeniorityLevel = Field(
        default=SeniorityLevel.UNKNOWN,
        description="Coarse seniority inferred from the posting.",
    )
    required_skills: list[str] = Field(
        default_factory=list,
        description="Skills/technologies the posting requires.",
    )
    preferred_skills: list[str] = Field(
        default_factory=list,
        description="Nice-to-have skills the posting mentions.",
    )
    responsibilities: list[str] = Field(
        default_factory=list,
        description="Key responsibilities/duties extracted from the posting.",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="ATS-relevant keywords to target, deduped across skills/responsibilities.",
    )
