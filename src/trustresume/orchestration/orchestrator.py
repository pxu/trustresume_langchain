"""The orchestrator — the single owner of control flow, built on LangGraph.

The original (pydantic-ai) version of this project hand-rolled the control
flow deliberately, so it stayed readable for the write-up (its own ADR-0003)
— but explicitly flagged migrating to a graph framework later as an isolated
change. This is that migration: the flow and its public contract are
unchanged, only the execution engine underneath ``run()`` is now a LangGraph
``StateGraph``.

The flow, once per generation:

    analyze_job -> load_candidate_profile -> retrieve_evidence -> write_resume
    -> score_trust -> score_ats -> (conditional: rewrite or end)

then the quality loop (ADR-0005): while the draft fails the gate
(Trust >= 90 AND ATS >= 85) and we're under the iteration cap, build specific
feedback and re-run write_resume -> score_trust -> score_ats. Job analysis,
candidate-profile resolution, and evidence retrieval happen once — the job
and the candidate's evidence don't change between rewrites, only the draft
does.

``Orchestrator``'s constructor and ``run()`` signature are identical to the
original: every other layer (``api/app_service.py``) depends on the public
``Orchestrator(...)``/``run(...) -> WorkflowState`` contract, not on how the
loop is implemented internally, so nothing above this module needed to
change (this is the "isolated change" ADR-0003 anticipated).

Milestone M5 (orchestration).
"""

from __future__ import annotations

import logging
import operator
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from trustresume.agents import (
    ATSEvaluationAgent,
    EvidenceRetrievalAgent,
    JobDescriptionAgent,
    ResumeWriterAgent,
    TrustHarnessAgent,
)
from trustresume.models import (
    ATSReport,
    CandidateProfile,
    EvidenceSet,
    JobDescription,
    QualityGate,
    ResumeDraft,
    TrustReport,
    WorkflowState,
)

from .candidate_profile_service import CandidateProfileService
from .feedback import build_feedback

logger = logging.getLogger(__name__)


# LangGraph's internal working state. Kept private to this module — every
# other layer only ever sees the ``WorkflowState`` the orchestrator converts
# this into on return. ``drafts``/``trust_reports``/``ats_reports`` are
# append-only (mirroring ``WorkflowState``'s parallel lists exactly); every
# other field is last-write-wins, LangGraph's default merge behavior.
class _GraphState(TypedDict):
    user_id: str
    job_posting: str
    gate: QualityGate
    job: JobDescription | None
    candidate_profile: CandidateProfile | None
    evidence: EvidenceSet | None
    drafts: Annotated[list[ResumeDraft], operator.add]
    trust_reports: Annotated[list[TrustReport], operator.add]
    ats_reports: Annotated[list[ATSReport], operator.add]
    iteration: int
    feedback: str | None


class Orchestrator:
    """Sequences the agents and runs the quality-improvement loop via LangGraph."""

    def __init__(
        self,
        *,
        job_description_agent: JobDescriptionAgent,
        candidate_profile_service: CandidateProfileService,
        retrieval_agent: EvidenceRetrievalAgent,
        resume_agent: ResumeWriterAgent,
        trust_agent: TrustHarnessAgent,
        evaluation_agent: ATSEvaluationAgent,
    ) -> None:
        self._job = job_description_agent
        self._candidate_profile = candidate_profile_service
        self._retrieval = retrieval_agent
        self._resume = resume_agent
        self._trust = trust_agent
        self._evaluation = evaluation_agent
        self._graph = self._build_graph()

    def _build_graph(self) -> Any:  # CompiledStateGraph — a bare generic under mypy
        builder = StateGraph(_GraphState)
        builder.add_node("analyze_job", self._analyze_job)
        builder.add_node("load_candidate_profile", self._load_candidate_profile)
        builder.add_node("retrieve_evidence", self._retrieve_evidence)
        builder.add_node("write_resume", self._write_resume)
        builder.add_node("score_trust", self._score_trust)
        builder.add_node("score_ats", self._score_ats)
        builder.add_node("prepare_rewrite", self._prepare_rewrite)

        builder.add_edge(START, "analyze_job")
        builder.add_edge("analyze_job", "load_candidate_profile")
        builder.add_edge("load_candidate_profile", "retrieve_evidence")
        builder.add_edge("retrieve_evidence", "write_resume")
        builder.add_edge("write_resume", "score_trust")
        builder.add_edge("score_trust", "score_ats")
        builder.add_conditional_edges(
            "score_ats", self._route, {"rewrite": "prepare_rewrite", "end": END}
        )
        builder.add_edge("prepare_rewrite", "write_resume")
        return builder.compile()

    # --- nodes --------------------------------------------------------------

    async def _analyze_job(self, state: _GraphState) -> dict[str, object]:
        job = await self._job.run(state["job_posting"])
        return {"job": job}

    async def _load_candidate_profile(self, state: _GraphState) -> dict[str, object]:
        profile = await self._candidate_profile.get_or_refresh(state["user_id"])
        return {"candidate_profile": profile}

    async def _retrieve_evidence(self, state: _GraphState) -> dict[str, object]:
        assert state["job"] is not None
        evidence = await self._retrieval.run(user_id=state["user_id"], job=state["job"])
        return {"evidence": evidence}

    async def _write_resume(self, state: _GraphState) -> dict[str, object]:
        assert state["job"] is not None and state["evidence"] is not None
        draft = await self._resume.run(
            job=state["job"],
            evidence=state["evidence"],
            feedback=state["feedback"],
            iteration=state["iteration"],
        )
        return {"drafts": [draft]}

    async def _score_trust(self, state: _GraphState) -> dict[str, object]:
        assert state["evidence"] is not None
        trust = await self._trust.run(draft=state["drafts"][-1], evidence=state["evidence"])
        logger.info(
            "trust scored",
            extra={
                "user_id": state["user_id"],
                "iteration": state["iteration"],
                "trust_score": trust.score,
                "hallucinations": len(trust.hallucinations),
            },
        )
        return {"trust_reports": [trust]}

    async def _score_ats(self, state: _GraphState) -> dict[str, object]:
        assert state["job"] is not None
        ats = await self._evaluation.run(draft=state["drafts"][-1], job=state["job"])
        logger.info(
            "ats scored",
            extra={
                "user_id": state["user_id"],
                "iteration": state["iteration"],
                "ats_score": ats.score,
                "missing_keywords": len(ats.missing_keywords),
            },
        )
        return {"ats_reports": [ats]}

    async def _prepare_rewrite(self, state: _GraphState) -> dict[str, object]:
        feedback = build_feedback(state["trust_reports"][-1], state["ats_reports"][-1])
        next_iteration = state["iteration"] + 1
        logger.info(
            "preparing rewrite",
            extra={"user_id": state["user_id"], "next_iteration": next_iteration},
        )
        return {"iteration": next_iteration, "feedback": feedback}

    def _route(self, state: _GraphState) -> Literal["rewrite", "end"]:
        """Mirrors ``WorkflowState.should_continue`` exactly.

        Called right after ``score_ats``, so ``iteration`` here is still the
        iteration of the draft that was just scored (the increment happens in
        ``_prepare_rewrite``, only on the "rewrite" branch) — the same
        pre-increment value the original's ``is_exhausted`` check reads. With
        the default gate (``max_iterations=3``) this yields up to 4 total
        drafts (iterations 0-3), not 3.
        """
        gate = state["gate"]
        passed = gate.passes(state["trust_reports"][-1], state["ats_reports"][-1])
        is_exhausted = state["iteration"] >= gate.max_iterations
        decision: Literal["rewrite", "end"] = "end" if (passed or is_exhausted) else "rewrite"
        logger.info(
            "quality gate routed",
            extra={
                "user_id": state["user_id"],
                "iteration": state["iteration"],
                "passed": passed,
                "is_exhausted": is_exhausted,
                "decision": decision,
            },
        )
        return decision

    # --- public API -----------------------------------------------------------

    async def run(
        self,
        *,
        user_id: str,
        job_posting: str,
        gate: QualityGate | None = None,
    ) -> WorkflowState:
        """Generate a resume for ``user_id`` against ``job_posting``.

        Returns the full :class:`WorkflowState` — every draft, every score,
        and the final pass/fail — so the caller can inspect the whole run, not
        just the final draft.
        """
        resolved_gate = gate or QualityGate()
        logger.info("generation run started", extra={"user_id": user_id})
        initial: _GraphState = {
            "user_id": user_id,
            "job_posting": job_posting,
            "gate": resolved_gate,
            "job": None,
            "candidate_profile": None,
            "evidence": None,
            "drafts": [],
            "trust_reports": [],
            "ats_reports": [],
            "iteration": 0,
            "feedback": None,
        }
        # Scale the recursion limit with the gate's iteration cap: 6 one-time
        # + first-generation steps, then 4 steps per rewrite pass, plus a
        # safety margin — so a caller-supplied gate with a higher cap than the
        # default doesn't hit LangGraph's default limit (25) prematurely.
        recursion_limit = 6 + 4 * resolved_gate.max_iterations + 10
        result = await self._graph.ainvoke(initial, config={"recursion_limit": recursion_limit})

        final_trust = result["trust_reports"][-1].score if result["trust_reports"] else None
        final_ats = result["ats_reports"][-1].score if result["ats_reports"] else None
        logger.info(
            "generation run finished",
            extra={
                "user_id": user_id,
                "iterations": result["iteration"],
                "final_trust_score": final_trust,
                "final_ats_score": final_ats,
            },
        )
        return WorkflowState(
            user_id=result["user_id"],
            gate=result["gate"],
            job=result["job"],
            candidate_profile=result["candidate_profile"],
            evidence=result["evidence"],
            drafts=result["drafts"],
            trust_reports=result["trust_reports"],
            ats_reports=result["ats_reports"],
            iteration=result["iteration"],
        )
