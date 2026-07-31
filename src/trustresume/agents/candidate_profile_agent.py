"""Candidate Profile agent — candidate document text -> structured ``CandidateProfile``.

A pure LLM extraction step, same shape as ``JobDescriptionAgent``: it takes
text and returns structured fields, does no retrieval, and has no side
effects of its own. Unlike Job Description analysis, its output is
job-independent — the candidate's skills/experience don't change per job
posting — so it is not called on every generation. ``CandidateProfileService``
(in ``orchestration/``) owns the cache-check and document assembly around it
and decides when it actually needs to run.

Milestone M4 (agents).
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from trustresume.models import CandidateProfile

from .base import ModelInput, ensure_type, with_structured_retry

_SYSTEM_PROMPT = """\
You are a resume analyst. Extract a structured summary of the candidate \
background you are given: their name if stated, a one-line professional \
summary, the skills/technologies they have hands-on experience with, and \
any certifications. Only extract what the document actually states — do \
not invent experience."""


class CandidateProfileAgent:
    """Wraps a LangChain structured-output call that structures a candidate's background."""

    def __init__(self, model: ModelInput) -> None:
        self._structured = with_structured_retry(model, CandidateProfile)

    async def run(self, candidate_text: str) -> CandidateProfile:
        """Analyze concatenated candidate document text into a :class:`CandidateProfile`."""
        result = await self._structured.ainvoke(
            [SystemMessage(_SYSTEM_PROMPT), HumanMessage(candidate_text)]
        )
        return ensure_type(result, CandidateProfile)
