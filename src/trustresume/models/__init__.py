"""Shared Pydantic schemas passed between the orchestrator and the agents.

Everything else in TrustResume imports its data contracts from here, so the
whole package agrees on shapes. Modules are split by domain (job, evidence,
resume, trust, workflow) but re-exported flat for convenience::

    from trustresume.models import JobDescription, TrustReport, WorkflowState

Milestone M1 (shared models) — first thing built; everything else imports it.
"""

from __future__ import annotations

from .candidate import CandidateProfile
from .enums import (
    ClaimCategory,
    ClaimStatus,
    DocumentType,
    SeniorityLevel,
)
from .evidence import EvidenceChunk, EvidenceSet
from .job import JobDescription
from .resume import ResumeDraft, ResumeSection
from .trust import ATSReport, TrustReport, VerifiedClaim
from .usage import ModelUsage, NodeTiming, RunUsage
from .workflow import QualityGate, WorkflowState

__all__ = [
    # enums
    "ClaimStatus",
    "ClaimCategory",
    "DocumentType",
    "SeniorityLevel",
    # job
    "JobDescription",
    # candidate
    "CandidateProfile",
    # evidence
    "EvidenceChunk",
    "EvidenceSet",
    # resume
    "ResumeDraft",
    "ResumeSection",
    # trust / evaluation
    "VerifiedClaim",
    "TrustReport",
    "ATSReport",
    # usage / telemetry
    "ModelUsage",
    "NodeTiming",
    "RunUsage",
    # workflow / orchestration
    "QualityGate",
    "WorkflowState",
]
