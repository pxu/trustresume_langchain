"""Job Description agent — raw job posting -> structured ``JobDescription``.

First step in the pipeline (ADR-0002). A pure LLM extraction step: it reads the
posting text and returns the structured fields downstream steps rely on
(skills/keywords for retrieval and ATS scoring, requirements for the writer).
It performs no retrieval and has no side effects.

Milestone M4 (agents).
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from trustresume.models import JobDescription

from .base import ModelInput

_SYSTEM_PROMPT = """\
You are a job-posting analyst. Extract a structured summary of the job \
description you are given. Identify the title, hiring company, seniority \
level, required vs. preferred skills, key responsibilities, and the \
ATS-relevant keywords a candidate should target. Only extract what the posting \
actually states — do not invent requirements."""


class JobDescriptionAgent:
    """Wraps a LangChain structured-output call that structures a job posting."""

    def __init__(self, model: ModelInput) -> None:
        self._structured = model.with_structured_output(JobDescription)

    async def run(self, job_posting: str) -> JobDescription:
        """Analyze a raw job posting into a :class:`JobDescription`.

        ``raw_text`` is force-set from the input so the verbatim source is
        preserved even if the model paraphrases it.
        """
        result = await self._structured.ainvoke(
            [SystemMessage(_SYSTEM_PROMPT), HumanMessage(job_posting)]
        )
        assert isinstance(result, JobDescription)
        return result.model_copy(update={"raw_text": job_posting})
