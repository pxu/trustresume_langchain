# ADR-0003: LangGraph orchestrator, replacing the hand-rolled MVP loop

## Status
Accepted. Supersedes the original repo's ADR-0003 ("Hand-rolled orchestrator
for the MVP, not LangGraph/CrewAI/AutoGen"), which explicitly named this as
the anticipated next step: "If/when the orchestration logic outgrows a
hand-rolled loop … migrating to LangGraph or similar becomes a deliberate,
isolated change — the orchestrator already lives in one module." This repo
is that migration.

## Context
The original MVP deliberately avoided a framework so control flow (agent
sequencing, the quality-loop condition, feedback threading between rewrites)
stayed fully visible for the capstone write-up. That reasoning was scoped to
"while the system's behavior and failure modes are still being understood" —
once the pipeline (job analysis → retrieval → write → verify → evaluate, with
a bounded rewrite loop) is well-understood and stable, the case for a
hand-rolled `while` loop weakens, and a graph framework's built-in
primitives (typed state, conditional edges, cycles) become a better fit than
custom code for the same behavior.

## Decision
Rebuild `Orchestrator` on a LangGraph `StateGraph`, keeping its public
contract identical: same constructor (the same six collaborators), same
`async def run(*, user_id, job_posting, gate=None) -> WorkflowState`. Callers
(`api/app_service.py`) needed zero changes — exactly the "isolated change"
the original ADR-0003 predicted.

The graph: `analyze_job → load_candidate_profile → retrieve_evidence →
write_resume → score_trust → score_ats`, then a conditional edge mirroring
`WorkflowState.should_continue` exactly (`gate.passes(...)` or
`iteration >= gate.max_iterations` → end; otherwise a `prepare_rewrite` node
increments `iteration`, builds feedback via the unchanged `build_feedback`,
and loops back to `write_resume`). `drafts`/`trust_reports`/`ats_reports` are
modeled as `Annotated[list[X], operator.add]` — LangGraph's standard
append-only reducer — the direct analog of `WorkflowState`'s parallel lists.

Internally, `Orchestrator.run()` builds a private `_GraphState` dict, invokes
the compiled graph via `ainvoke`, and converts the result back into a
`WorkflowState` on return — the graph's shape is an implementation detail
`WorkflowState` and every other layer stays ignorant of.

## Consequences
- The quality loop's exact iteration counting (an unintuitive but
  load-bearing detail — `max_iterations=3` yields **4** total drafts,
  iterations 0–3, not 3) had to be reproduced precisely as a conditional-edge
  check evaluated *before* the increment; this is exercised directly by
  `tests/unit/test_orchestrator.py::test_orchestrator_failsToCap_stopsAndExportsRealScores`.
- Control flow is now expressed as graph nodes/edges rather than a single
  readable `while` loop — a deliberate trade of some at-a-glance readability
  for LangGraph's built-in state management, cycle support, and (not yet
  used here, but available for free) streaming/checkpointing/visualization.
- The orchestrator's recursion limit is scaled from the gate's
  `max_iterations` (`6 + 4 * max_iterations + margin`) rather than left at
  LangGraph's default (25), so a caller-supplied gate with a higher cap than
  the default doesn't fail with a recursion-limit error instead of a normal
  "capped out" result.
