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
import time
from collections.abc import AsyncIterator, Coroutine
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal, Protocol, TypedDict

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
    NodeTiming,
    QualityGate,
    ResumeDraft,
    TrustReport,
    WorkflowState,
)
from trustresume.telemetry import track_usage

from .candidate_profile_service import CandidateProfileService
from .feedback import build_feedback

logger = logging.getLogger(__name__)


def _recursion_limit(gate: QualityGate) -> int:
    """Scale LangGraph's recursion limit with the gate's iteration cap.

    6 one-time + first-generation steps, then 4 steps per rewrite pass, plus a
    safety margin — so a caller-supplied gate with a higher cap than the
    default doesn't hit LangGraph's default limit (25) prematurely.
    """
    return 6 + 4 * gate.max_iterations + 10


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
    job_id: str | None
    document_ids: list[str] | None
    candidate_profile: CandidateProfile | None
    evidence: EvidenceSet | None
    drafts: Annotated[list[ResumeDraft], operator.add]
    trust_reports: Annotated[list[TrustReport], operator.add]
    ats_reports: Annotated[list[ATSReport], operator.add]
    # Append-only like the three above: nodes inside the quality loop execute
    # once per iteration, and "how long did the 3rd rewrite take" is exactly
    # the question a per-node total would erase.
    timings: Annotated[list[NodeTiming], operator.add]
    iteration: int
    feedback: str | None


class _Node(Protocol):
    """A graph node: async, takes the state, returns a state update.

    A ``Protocol`` rather than a ``Callable[...]`` alias because LangGraph's
    own node protocol declares its parameter by name (``state``), so a
    positional-only ``Callable`` isn't assignable to it — the wrapper below
    has to keep that parameter name to type-check against ``add_node``.
    """

    def __call__(self, state: _GraphState) -> Coroutine[Any, Any, dict[str, object]]: ...


def _timed(name: str, node: _Node) -> _Node:
    """Wrap a node so its wall-clock duration lands in ``state["timings"]``.

    Applied once at graph-construction time rather than inside each node body:
    seven nodes would otherwise each need the same start/stop boilerplate, and
    a node added later would silently go unmeasured. The wrapper only *adds* a
    key to whatever the node returned, so node logic is untouched.

    Timing lives in graph state (not in the ``UsageTracker`` callback) because
    only the graph knows where node boundaries are — callbacks see LLM calls,
    which is a different, finer-grained thing. A node that raises isn't timed:
    the run is failing anyway, and reporting a partial duration as if it were
    a completed step would be misleading.
    """

    async def run_timed(state: _GraphState) -> dict[str, object]:
        started = time.perf_counter()
        result = await node(state)
        duration_ms = (time.perf_counter() - started) * 1000
        return {**result, "timings": [NodeTiming(node=name, duration_ms=duration_ms)]}

    return run_timed


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
        checkpoint_path: str | None = None,
    ) -> None:
        self._job = job_description_agent
        self._candidate_profile = candidate_profile_service
        self._retrieval = retrieval_agent
        self._resume = resume_agent
        self._trust = trust_agent
        self._evaluation = evaluation_agent
        # Durable execution (ADR-0015), opt-in. ``None`` (the default) keeps
        # the original behavior exactly: one graph compiled once, no
        # checkpointer, ``run(run_id=...)`` and ``resume`` unavailable. When a
        # path is given, each run compiles a fresh graph bound to a saver
        # scoped to that call's event loop (see ``_compiled``) — the app drives
        # every generation through its own ``asyncio.run``, so a saver held for
        # the app's lifetime would be pinned to a dead loop by the second call.
        self._checkpoint_path = checkpoint_path
        # The uncompiled builder is kept so a checkpointed run can recompile
        # against a per-call saver; the no-checkpoint path compiles once here.
        self._builder = self._build_builder()
        self._graph = self._builder.compile()

    def _build_builder(self) -> Any:  # StateGraph — a bare generic under mypy
        builder = StateGraph(_GraphState)
        builder.add_node("analyze_job", _timed("analyze_job", self._analyze_job))
        builder.add_node(
            "load_candidate_profile", _timed("load_candidate_profile", self._load_candidate_profile)
        )
        builder.add_node("retrieve_evidence", _timed("retrieve_evidence", self._retrieve_evidence))
        builder.add_node("write_resume", _timed("write_resume", self._write_resume))
        builder.add_node("score_trust", _timed("score_trust", self._score_trust))
        builder.add_node("score_ats", _timed("score_ats", self._score_ats))
        builder.add_node("prepare_rewrite", _timed("prepare_rewrite", self._prepare_rewrite))

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
        return builder

    # --- nodes --------------------------------------------------------------

    async def _analyze_job(self, state: _GraphState) -> dict[str, object]:
        """Extract the job description, unless one was already supplied.

        A caller re-using a persisted ``JobDescription`` (``run(job=...)``)
        pre-populates ``state["job"]`` before the graph starts — this node
        then becomes a no-op rather than re-running the extraction, without
        the graph's shape (nodes/edges) changing at all. Keeping one fixed,
        unconditionally-built graph rather than branching the graph itself
        is deliberate: it means every other node, `_route`'s exhaustion
        check, and `_prepare_rewrite`'s increment are untouched by this.
        """
        if state["job"] is not None:
            return {}
        job = await self._job.run(state["job_posting"])
        return {"job": job}

    async def _load_candidate_profile(self, state: _GraphState) -> dict[str, object]:
        profile = await self._candidate_profile.get_or_refresh(state["user_id"])
        return {"candidate_profile": profile}

    async def _retrieve_evidence(self, state: _GraphState) -> dict[str, object]:
        assert state["job"] is not None
        evidence = await self._retrieval.run(
            user_id=state["user_id"], job=state["job"], document_ids=state["document_ids"]
        )
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
        job_posting: str | None = None,
        job: JobDescription | None = None,
        job_id: str | None = None,
        document_ids: list[str] | None = None,
        gate: QualityGate | None = None,
        run_id: str | None = None,
    ) -> WorkflowState:
        """Generate a resume for ``user_id`` against a job posting.

        Exactly one of ``job_posting`` (the legacy path: raw text,
        re-extracted here every call) or ``job`` (a pre-extracted, typically
        persisted ``JobDescription`` — re-extraction is skipped) must be
        given. ``job_id``/``document_ids`` are carried through for job-scoped
        retrieval and persistence but change no control flow.

        ``run_id`` is the LangGraph ``thread_id`` this run's checkpoints are
        keyed by (ADR-0015). It only has an effect when the orchestrator was
        built with a ``checkpoint_path``; otherwise it is ignored (there is no
        checkpointer to write to). Pass the same ``run_id`` to :meth:`resume`
        to continue a run that crashed part-way.

        Returns the full :class:`WorkflowState` — every draft, every score,
        and the final pass/fail — so the caller can inspect the whole run, not
        just the final draft.
        """
        if (job_posting is None) == (job is None):
            raise ValueError("exactly one of job_posting or job must be given")

        resolved_gate = gate or QualityGate()
        logger.info("generation run started", extra={"user_id": user_id, "run_id": run_id})
        initial: _GraphState = {
            "user_id": user_id,
            "job_posting": job_posting or "",
            "gate": resolved_gate,
            "job": job,
            "job_id": job_id,
            "document_ids": document_ids,
            "candidate_profile": None,
            "evidence": None,
            "drafts": [],
            "trust_reports": [],
            "ats_reports": [],
            "timings": [],
            "iteration": 0,
            "feedback": None,
        }
        checkpointed = self._checkpoint_path is not None and run_id is not None
        async with self._compiled(checkpointed=checkpointed) as graph:
            return await self._invoke(
                graph,
                initial,
                run_id=run_id,
                recursion_limit=_recursion_limit(resolved_gate),
                user_id=user_id,
            )

    async def resume(self, *, run_id: str) -> WorkflowState | None:
        """Continue a checkpointed run from its last completed node.

        Requires the orchestrator to have been built with a ``checkpoint_path``
        (raises ``RuntimeError`` otherwise). Returns ``None`` when no checkpoint
        exists for ``run_id`` — the run never started, or its checkpoints were
        pruned/deleted — so the caller can 404 rather than silently start a
        blank generation. Resuming a run that already reached the end replays
        no nodes and just returns its final state.

        Caveat: the token/cost ``usage`` on the returned state reflects only
        the nodes executed *during this resume* — LangGraph checkpoints graph
        state (drafts, scores, per-node timings), not the ``UsageTracker``'s
        accumulation, which is per-invocation. Latency ``timings`` are complete
        (they live in graph state); tokens/cost from before the crash are not
        re-counted. Documented in ADR-0015 rather than papered over.
        """
        if self._checkpoint_path is None:
            raise RuntimeError("resume requires the orchestrator built with a checkpoint_path")
        async with self._compiled(checkpointed=True) as graph:
            config: dict[str, Any] = {"configurable": {"thread_id": run_id}}
            snapshot = await graph.aget_state(config)
            if not snapshot.values:
                logger.info("resume found no checkpoint", extra={"run_id": run_id})
                return None
            gate = snapshot.values["gate"]
            user_id = snapshot.values["user_id"]
            logger.info("generation run resuming", extra={"user_id": user_id, "run_id": run_id})
            # ``None`` input tells LangGraph to continue from the checkpoint
            # rather than re-seed the state from scratch.
            return await self._invoke(
                graph,
                None,
                run_id=run_id,
                recursion_limit=_recursion_limit(gate),
                user_id=user_id,
            )

    # --- invocation helpers ---------------------------------------------------

    @asynccontextmanager
    async def _compiled(self, *, checkpointed: bool) -> AsyncIterator[Any]:
        """Yield a graph to invoke — the cached one, or a per-call checkpointed one.

        A checkpointed graph is compiled fresh against an ``AsyncSqliteSaver``
        opened for the duration of this call (and thus this event loop), then
        torn down; the durable state lives in the on-disk SQLite file, so
        nothing needs to outlive the ``with`` block. The saver import is lazy
        so the ``durable`` extra is only required when checkpointing is
        actually used.
        """
        if not checkpointed:
            yield self._graph
            return
        assert self._checkpoint_path is not None  # implied by checkpointed=True
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        async with AsyncSqliteSaver.from_conn_string(self._checkpoint_path) as saver:
            yield self._builder.compile(checkpointer=saver)

    async def _invoke(
        self,
        graph: Any,
        graph_input: _GraphState | None,
        *,
        run_id: str | None,
        recursion_limit: int,
        user_id: str,
    ) -> WorkflowState:
        """Invoke ``graph``, track usage, and convert the result to ``WorkflowState``.

        Shared by :meth:`run` (``graph_input`` is a fresh initial state) and
        :meth:`resume` (``graph_input`` is ``None``, so LangGraph continues
        from the thread's checkpoint).
        """
        config: dict[str, Any] = {"recursion_limit": recursion_limit}
        if run_id is not None:
            config["configurable"] = {"thread_id": run_id}
        # One tracker per invocation, handed to LangGraph as a callback: it
        # propagates down to every agent's model call, so token accounting
        # needs no cooperation from (and no coupling to) the agents themselves.
        with track_usage() as tracker:
            config["callbacks"] = [tracker]
            result = await graph.ainvoke(graph_input, config=config)
        usage = tracker.finalize(timings=result["timings"])

        final_trust = result["trust_reports"][-1].score if result["trust_reports"] else None
        final_ats = result["ats_reports"][-1].score if result["ats_reports"] else None
        logger.info(
            "generation run finished",
            extra={
                "user_id": user_id,
                "run_id": run_id,
                "iterations": result["iteration"],
                "final_trust_score": final_trust,
                "final_ats_score": final_ats,
                **usage.log_fields(),
            },
        )
        return WorkflowState(
            user_id=result["user_id"],
            gate=result["gate"],
            job_id=result["job_id"],
            job=result["job"],
            candidate_profile=result["candidate_profile"],
            evidence=result["evidence"],
            drafts=result["drafts"],
            trust_reports=result["trust_reports"],
            ats_reports=result["ats_reports"],
            iteration=result["iteration"],
            usage=usage,
        )
