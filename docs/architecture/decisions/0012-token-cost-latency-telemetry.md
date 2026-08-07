# ADR-0012: Per-run token, cost, and latency accounting

## Status
Accepted. New — no equivalent decision in the original repo.

## Context
One generation makes **5–9+ sequential LLM calls** (one per LLM-backed agent,
times up to four quality-loop iterations). The system could not report how many
tokens that consumed, what it cost, or where the time went. Structured logging
(`logging_config.py`) recorded iteration counts and scores, but nothing about
resource use.

That's a gap in three directions at once:

- **Operational.** "Why was last month's bill that size" and "which step is
  slow" had no answer.
- **Engineering.** ADR-0013's model tiering is a cost/latency trade — proposing
  it without a way to measure the effect is guesswork.
- **Evaluation.** ADR-0011's harness reports quality; quality per dollar is the
  number that actually decides a configuration.

The technical obstacle is specific: every LLM-backed agent calls
`model.with_structured_output(Schema)`, whose result is the *parsed Pydantic
object*. The underlying `AIMessage` — and with it `usage_metadata` — is
consumed inside the chain and never reaches the agent. Token counts are simply
not available at the call site.

## Decision
Collect usage in a **callback handler** (`telemetry.UsageTracker`), attached
once per run by the orchestrator via LangGraph's `config={"callbacks": [...]}`.
Callbacks propagate to nested runnables, so one tracker sees every agent's model
call and no agent needs to know it exists.

- **`models/usage.py`** holds `RunUsage`/`ModelUsage`/`NodeTiming` —
  framework-free, in `models/` like every other cross-layer contract, so
  storage and the API serialize them without a LangChain dependency.
- **Per-node latency** is measured by `_timed()`, applied once when the graph
  is built rather than inside seven node bodies — a node added later is
  measured automatically. Timings are a *list*, appended per execution, so
  "was the third rewrite slower than the first" stays answerable.
- **Pricing lives in `config/pricing.json`**, not in source. List prices change
  often enough that hard-coding them guarantees they're wrong later.
- **An unpriced model yields `cost_usd = None`, never a partial total.** A sum
  that silently omits the most expensive model reads like a real number and is
  wrong, usually in the cheap direction.

We deliberately did **not** reuse `langchain_core`'s `UsageMetadataCallbackHandler`:
it drops usage entirely when a provider omits `response_metadata["model_name"]`
(silently reporting zero cost is the exact failure this exists to prevent), and
it counts tokens but not *calls* — "how many LLM round-trips did one generation
make" is a headline number for a pipeline that makes 5–9 of them. `UsageTracker`
attributes nameless calls to `"unknown"` and counts a call even when usage
metadata is absent.

Usage is threaded all the way out: `WorkflowState.usage` → `GenerateResponse` /
`ResumeDetail` → four flattened columns on `generated_resumes` (so "what did
resumes cost me last month" is one `SUM()`, not a JSON scan) → a caption in the
Streamlit UI. Those columns are **nullable**: an unmeasured run is not a free
one, and `0` would claim otherwise.

## Consequences
- `AutoStructuredFakeChatModel` and the scripted test double now emit
  `usage_metadata` and a `model_name` (a deterministic ~4-chars-per-token
  estimate). Without that, every token assertion in the offline suite would be
  vacuously zero — the telemetry path would be untestable in exactly the mode
  the whole suite runs in. Neither fake's model id appears in
  `config/pricing.json`, so offline runs report real token counts with an
  honest `cost_usd: null`, exercising the unpriced path for free.
- `config/pricing.json` ships with OpenAI entries only. Bedrock/Gemini rates
  are deliberately absent rather than guessed — add yours from your own rate
  card. Until then, Bedrock runs report tokens and `cost_usd: null`.
- Adding usage columns is a schema change, and this project has no migrations
  (see `CLAUDE.md`): delete `trustresume.db` or `docker compose down -v` first.
- The tracker sees LLM calls, not node boundaries; the graph sees node
  boundaries, not LLM calls. Both halves are needed, which is why
  `UsageTracker.finalize(timings=...)` takes the graph's timings as a
  parameter instead of collecting them itself.

## Alternatives considered
- **`with_structured_output(include_raw=True)`.** Would surface the raw
  message, but changes every agent's return handling and couples token
  accounting to each agent — the opposite of the callback's "agents don't know
  it exists" property.
- **LangSmith tracing.** Better UI, and worth adding later, but it's a hosted
  dependency that would put an account and network egress between this project
  and a number it can compute locally. It also wouldn't populate the database
  column that makes cost queryable.
- **Hard-coding prices in Python.** Rejected above: a stale price in source is
  a wrong number presented confidently.
