"""The pipeline agents: Job Description, Evidence Retrieval, Resume Writer,
Trust Harness, ATS Evaluation — plus Candidate Profile, a cached,
job-independent sixth agent.

Each is a pure input -> output step (ADR-0002/0003): it takes what it needs as
arguments, returns a ``trustresume.models`` object, and never calls another
agent — the orchestrator (M5) sequences them. LLM-backed agents take an
injectable chat model (``base.ModelInput``); Evidence Retrieval and ATS
Evaluation are deterministic and take their collaborators/inputs directly.
Scores are always computed in code, never self-reported by an LLM (ADR-0004).

Candidate Profile is the one exception to "the orchestrator calls agents
directly": its output doesn't vary per job, so ``CandidateProfileService``
(``orchestration/``) caches it and only calls this agent when the cache is
missing or stale.

Milestone M4 (agents).
"""

from __future__ import annotations

from .base import ModelInput
from .candidate_profile_agent import CandidateProfileAgent
from .evaluation_agent import ATSEvaluationAgent
from .job_description_agent import JobDescriptionAgent
from .resume_agent import ResumeWriterAgent
from .retrieval_agent import EvidenceRetrievalAgent
from .trust_agent import TrustHarnessAgent

__all__ = [
    "ModelInput",
    "JobDescriptionAgent",
    "CandidateProfileAgent",
    "EvidenceRetrievalAgent",
    "ResumeWriterAgent",
    "TrustHarnessAgent",
    "ATSEvaluationAgent",
]
