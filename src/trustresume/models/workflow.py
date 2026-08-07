"""Orchestrator workflow state and the quality-gate configuration.

The orchestrator owns all control flow and holds ``WorkflowState`` as the
single inspectable record of a run: the job description, retrieved evidence,
current draft, the scores for each iteration, and the iteration count. The
LangGraph orchestrator (``orchestration/orchestrator.py``) runs its own
internal graph state and converts the final result back into this model, so
``WorkflowState`` stays the stable public contract every other layer
(``api/``) depends on. ``QualityGate`` holds the pass thresholds and
iteration cap (ADR-0005) - the 90 / 85 / 3 values are placeholders to be
validated empirically, so they live in one editable place rather than
scattered as magic numbers.

Milestone M1 (shared models).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .candidate import CandidateProfile
from .evidence import EvidenceSet
from .job import JobDescription
from .resume import ResumeDraft
from .trust import ATSReport, TrustReport
from .usage import RunUsage


class QualityGate(BaseModel):
    """Pass/fail thresholds and the hard iteration cap for the quality loop.

    Defaults are the ADR-0005 placeholders. A draft passes only when *both*
    scores meet their threshold; the loop stops on a pass or after
    ``max_iterations`` rewrites, whichever comes first.
    """

    model_config = ConfigDict(extra="forbid")

    min_trust_score: float = Field(default=90.0, ge=0, le=100)
    min_ats_score: float = Field(default=85.0, ge=0, le=100)
    max_iterations: int = Field(
        default=3,
        ge=1,
        description="Maximum number of rewrite iterations after the initial pass.",
    )

    def passes(self, trust: TrustReport, ats: ATSReport) -> bool:
        """True when both scores meet their thresholds."""
        return trust.score >= self.min_trust_score and ats.score >= self.min_ats_score


class WorkflowState(BaseModel):
    """The complete, inspectable state of one resume-generation run.

    Every field except ``user_id`` and ``gate`` starts empty and is filled in
    by the orchestrator as it sequences the agents. Reports are kept as lists
    parallel to the drafts so the full history of the quality loop is
    preserved for the write-up and the UI.
    """

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(..., min_length=1, description="User this run belongs to (ADR-0001).")
    gate: QualityGate = Field(
        default_factory=QualityGate, description="Quality-gate config for this run."
    )
    job_id: str | None = Field(
        None,
        description=(
            "The persisted job this run was generated for, if any — None for "
            "the legacy raw-job_posting-string generate() path with no "
            "persisted Job entity behind it."
        ),
    )
    resume_id: str | None = Field(
        None,
        description=(
            "The id of the persisted resume row for this run's final draft, "
            "set once TrustResumeApp._persist has written it. None until "
            "then (or if persistence produced no scored draft to save)."
        ),
    )

    job: JobDescription | None = Field(None, description="Job Description output.")
    candidate_profile: CandidateProfile | None = Field(
        None,
        description=(
            "Candidate Profile output - cached and job-independent, so it may "
            "have been reused from a prior run rather than freshly generated."
        ),
    )
    evidence: EvidenceSet | None = Field(None, description="Evidence Retrieval output.")
    drafts: list[ResumeDraft] = Field(
        default_factory=list, description="Draft per iteration, in order."
    )
    trust_reports: list[TrustReport] = Field(
        default_factory=list, description="Trust report per draft."
    )
    ats_reports: list[ATSReport] = Field(default_factory=list, description="ATS report per draft.")
    iteration: int = Field(
        default=0, ge=0, description="Current iteration (0 = initial generation)."
    )
    usage: RunUsage | None = Field(
        None,
        description=(
            "Tokens, estimated cost, and wall-clock time for this run, set by "
            "the orchestrator. None for a state built by hand (tests, a "
            "caller assembling a state outside a real run)."
        ),
    )

    @property
    def current_draft(self) -> ResumeDraft | None:
        """The most recent draft, if any."""
        return self.drafts[-1] if self.drafts else None

    @property
    def current_trust(self) -> TrustReport | None:
        """The most recent Trust report, if any."""
        return self.trust_reports[-1] if self.trust_reports else None

    @property
    def current_ats(self) -> ATSReport | None:
        """The most recent ATS report, if any."""
        return self.ats_reports[-1] if self.ats_reports else None

    @property
    def passed(self) -> bool:
        """Whether the latest draft passes the quality gate.

        False when either report is missing - an unscored draft has not passed.
        """
        if self.current_trust is None or self.current_ats is None:
            return False
        return self.gate.passes(self.current_trust, self.current_ats)

    @property
    def is_exhausted(self) -> bool:
        """Whether the loop has hit its iteration cap (ADR-0005)."""
        return self.iteration >= self.gate.max_iterations

    @property
    def should_continue(self) -> bool:
        """Whether the orchestrator should run another rewrite iteration.

        Continue only when there is a scored draft that failed the gate and the
        cap has not been reached.
        """
        if self.current_draft is None:
            return True  # nothing generated yet
        return not self.passed and not self.is_exhausted
