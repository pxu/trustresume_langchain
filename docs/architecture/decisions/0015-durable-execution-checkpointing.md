# ADR-0015: Durable execution via LangGraph checkpointing (demonstrative)

## Status
Accepted, scoped as **demonstrative**. ADR-0003 migrated the orchestrator to a
LangGraph `StateGraph` and noted checkpointing was "available for free" from
that move but "not yet used here" — this ADR turns it on for one purpose only,
and is explicit that it is a learning artifact rather than a reliability
requirement.

## Context
A single generation makes several paid LLM calls: Job Description + Candidate
Profile (extraction), then up to four rewrite passes each running the Resume
Writer and the Trust Harness (ATS is deterministic, no LLM call). With the
default gate that is a dozen-plus model calls in one `run()`.

If the process dies part-way — an unhandled exception in a node, an OOM, a
deploy — every already-completed, already-*paid-for* node is discarded, and the
next attempt starts from `analyze_job` again. This isn't hypothetical: the
em-dash `render_pdf` crash documented in `export/artifacts.py` destroyed whole
generations *after* every LLM call had been billed.

LangGraph persists per-node checkpoints when a graph is compiled with a
checkpointer, keyed by a `thread_id`, so a run can resume from its last
completed super-step instead of restarting. The mechanism was already latent in
the `StateGraph` from ADR-0003; nothing but a checkpointer and a stable id was
missing.

At personal-project scale the reliability win is **marginal** — a run is
seconds to tens of seconds, and a crash is rare — so this is *not* justified as
a necessity. It is adopted as a working demonstration of durable execution: a
concrete, testable exercise of a core LangGraph capability the port otherwise
name-drops but never uses.

## Decision
Add an **opt-in** checkpointer to `Orchestrator`, threaded through
`TrustResumeApp` and exposed over one HTTP route.

- **`Orchestrator(checkpoint_path=None)`.** `None` (the default) is behavior-
  identical to before: one graph compiled once in `__init__`, no checkpointer,
  `run(run_id=...)` ignored, `resume` unavailable. Mirrors ADR-0013's opt-in
  role tiering — existing callers and every test are untouched.
- **Per-call saver, not a long-lived one.** The app drives every generation
  through its own `asyncio.run` (see `api/app_service.py`), so a fresh event
  loop is created per call. An `AsyncSqliteSaver` holds an `aiosqlite`
  connection bound to the loop it was opened in, so a saver kept for the app's
  lifetime would be pinned to a dead loop by the second call. Instead
  `Orchestrator._compiled` opens the saver inside each `run`/`resume` (via
  `AsyncSqliteSaver.from_conn_string`), compiles a graph against it, and tears
  it down when the call returns — the durable state lives in the on-disk SQLite
  file, so nothing needs to outlive the `with` block. The saver import is lazy,
  so the `durable` extra is only required when checkpointing is actually used.
- **Its own SQLite file, separate from `trustresume.db`.** The checkpoint
  store is managed by LangGraph's own `.setup()`, so it deliberately never
  touches this repo's "no migration tooling" schema constraint (see CLAUDE.md)
  — its tables are LangGraph's problem, not `storage/schema.py`'s.
- **`run_id` minted before the run.** `TrustResumeApp.generate` /
  `generate_for_job` create a `uuid4` `run_id` *before* calling the
  orchestrator and pass it as the `thread_id`. It has to exist pre-run because
  `resume_id` is only assigned post-run in `_persist`, and a crash-resume needs
  a stable key that survives the crash.
- **`resume` is fail-closed and user-scoped.** `TrustResumeApp.resume_run`
  collapses three cases to an indistinguishable "not found": durable execution
  disabled, no checkpoint for that `run_id`, and a checkpoint owned by a
  different user — so a caller can't probe other users' run ids (ADR-0001's
  isolation still holds at this new surface). The route is
  `POST /api/runs/{run_id}/resume`, 404 on any of the three.
- **Enabled by `TRUSTRESUME_CHECKPOINT_PATH`** in `build_served_app`; unset or
  empty leaves it off (the default), using the same or-`None` idiom as
  `TRUSTRESUME_OUTPUT_DIR`.

## Consequences
- One new optional dependency (`langgraph-checkpoint-sqlite`, pulling in
  `aiosqlite`), gated behind a `durable` extra so the default install and CI's
  `dev`/`providers`/`ui` extras stay lean. `uv.lock` is regenerated to include
  it.
- **Resume usage accounting is partial by design.** LangGraph checkpoints
  *graph state* (drafts, trust/ATS reports, per-node `timings`), not the
  `UsageTracker`'s token/cost accumulation — the tracker is per-invocation
  (ADR-0012). So the `usage` on a resumed run's `WorkflowState` reflects only
  the nodes that executed *during the resume*; latency `timings` are complete
  (they live in state), but tokens/cost spent before the crash are not
  re-counted. Documented in `Orchestrator.resume`'s docstring rather than
  papered over.
- **The HTTP client can't yet discover the `run_id` of a crashed run.** It is
  minted server-side and only surfaced in logs, so today's resume story is
  "an operator reads the `run_id` from the structured logs and calls the
  route." Making resume genuinely client-driven means letting the client
  supply a `run_id` / idempotency key on the generate request — a small,
  deliberately-deferred follow-up, noted here so the scope isn't mistaken for
  a finished feature.
- Reminder that a checkpoint DB created before a LangGraph checkpoint-schema
  change has the same "delete the file" caveat as the rest of the project — but
  it's LangGraph's schema, and the file is disposable (it holds resumable runs,
  not source-of-truth data), so this is even lower-stakes than `trustresume.db`.

## Alternatives considered
- **A long-lived `AsyncSqliteSaver` compiled once in `__init__`.** Rejected:
  incompatible with the `asyncio.run`-per-call architecture (loop-bound
  connection). Would have forced either an app-wide event loop or a switch to
  the synchronous `SqliteSaver`, which in turn runs its checkpoint writes in a
  thread-pool under an async graph and trips SQLite's `check_same_thread`.
- **Checkpointing into `trustresume.db`.** Rejected: it would entangle
  LangGraph's auto-managed checkpoint tables with `storage/schema.py`'s
  migration-free `CREATE TABLE IF NOT EXISTS` regime, exactly the coupling the
  separate-file decision avoids.
- **Deriving `thread_id` from `resume_id`.** Impossible — `resume_id` doesn't
  exist until after the run persists, which is precisely the moment a
  crash-resume can't rely on.
- **Human-in-the-loop interrupts / cross-turn threads.** These are the features
  that would justify checkpointing as a *requirement* rather than a demo. The
  quality gate is deterministic and each generation is independent, so neither
  applies here; noted so this ADR isn't read as a general endorsement of
  turning checkpointing on everywhere.
- **Not doing it.** The honest default for a personal project. Chosen against
  only because durable execution is a headline LangGraph capability the port
  claims to be learning, and a demonstrable version costs little and stays
  fully opt-in.
