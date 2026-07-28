"""Candidate Profile output: a job-independent, structured view of a candidate.

Produced by the Candidate Profile agent from a user's already-ingested
documents (chunks already sitting in ``storage/``/``retrieval/`` — this agent
never touches a raw upload). Unlike :class:`~trustresume.models.job.JobDescription`,
a candidate has no required-vs-preferred split: these are the skills,
certifications, and experience the candidate actually has, independent of any
one job posting. Because it doesn't vary per job, it's computed once per user
and cached (``orchestration/candidate_profile_service.py``'s
``CandidateProfileService``), not regenerated on every ``generate()`` call.

Milestone M1 (shared models).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CandidateProfile(BaseModel):
    """A parsed, structured candidate background.

    Deliberately separate from :class:`~trustresume.models.job.JobDescription`
    — a resume has skills the candidate actually has, not a required/preferred
    split, so reusing the job shape here would be semantically wrong.
    """

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(None, description="Candidate name, if stated.")
    summary: str | None = Field(None, description="One-line professional summary.")
    skills: list[str] = Field(
        default_factory=list,
        description="Skills/technologies the candidate has hands-on experience with.",
    )
    certifications: list[str] = Field(
        default_factory=list,
        description="Certifications the candidate holds.",
    )
