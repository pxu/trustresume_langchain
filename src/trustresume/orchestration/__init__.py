"""Agent Orchestration Layer.

The single owner of control flow: sequences the agents via a LangGraph
``StateGraph``, holds/reconstructs the ``WorkflowState``, and runs the
quality-improvement loop (ADR-0005). ``feedback`` derives the rewrite
instructions when a draft fails the gate. ``CandidateProfileService``
resolves the cached Candidate Profile the orchestrator attaches to state once
per generation.

Milestone M5 (orchestration).
"""

from __future__ import annotations

from .candidate_profile_service import CandidateProfileService
from .feedback import build_feedback
from .orchestrator import Orchestrator

__all__ = ["Orchestrator", "build_feedback", "CandidateProfileService"]
