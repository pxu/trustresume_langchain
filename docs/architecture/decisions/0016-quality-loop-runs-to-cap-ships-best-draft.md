# ADR-0016: Quality loop runs to its full cap and ships the best-scoring draft

## Status
Accepted. Deviates deliberately from the *original* repo's ADR-0005 ("the
quality loop stops the instant a draft passes both thresholds"), which this
port had carried over unchanged until now.

## Context
The original design conflated two different questions into one stopping
rule: "has this draft passed?" and "should we keep trying?" The moment a
draft passed, the loop ended — even if `max_iterations` allowed more
attempts, and even though a later attempt, given the same evidence, might
have scored higher (better ATS keyword coverage, in particular) without
sacrificing Trust.

The request that started this: generate a résumé, and if it doesn't pass,
retry up to some *configurable* number of times, keeping whichever attempt
actually scores best — not necessarily the first one that happened to pass,
and not necessarily the last one generated either. That reframes "the
quality loop" from "retry until success or exhaustion" to "always explore
`N` drafts, then pick the best."

This is a real behavior change, not just a config knob: under the old rule,
`max_iterations` was a *ceiling* that a lucky early pass could avoid ever
reaching. Under the new rule, `max_iterations` is the number of rewrites
that happen on *every* run, unconditionally — which directly multiplies the
LLM calls (and cost) of the common case where the first draft is already
good. That cost implication shaped several of the decisions below.

## Decision

**1. `_route` no longer checks `gate.passes(...)`.** In
`orchestration/orchestrator.py`, the conditional edge that used to be

```python
decision = "end" if (passed or is_exhausted) else "rewrite"
```

is now

```python
decision = "end" if is_exhausted else "rewrite"
```

`passed` is still computed and logged (for observability), but it no longer
affects control flow. Every run now generates exactly `max_iterations + 1`
drafts — not "up to" that many. The iteration-counting subtlety ADR-0003
already documented (the check reads `iteration` *before* `prepare_rewrite`
increments it) is unchanged; only the extra `passed or` clause is gone.

**2. `WorkflowState` gained a `final_*` family, distinct from `current_*`.**
`current_draft`/`current_trust`/`current_ats`/`passed` still mean "the most
recent iteration" — the loop's own routing and the iteration-history views
reason about the latest draft, and that concept still needs a name.
`final_index`/`final_draft`/`final_trust`/`final_ats`/`final_passed` are new:
they identify the draft this run actually *ships*, chosen by:

```python
key=lambda i: (
    gate.passes(trust_reports[i], ats_reports[i]),  # passing always outranks failing
    ats_reports[i].score,                             # among same-side drafts, higher ATS
    i,                                                  # ties go to the later iteration
)
```

Ranking rules, in order:
- A draft that passes the gate always beats one that doesn't, regardless of
  raw scores. This was checked first for a reason: an earlier design of this
  ranking put Trust ahead of ATS as the *tiebreak*, which is fine on its own,
  but the "passing beats failing" rule has to be the outermost comparison or
  a failing draft with an unusually high secondary score could outrank a
  draft that genuinely passed — never acceptable for a system whose whole
  premise is "don't ship what didn't verify."
- Among drafts on the same side of that line (all passing, or all failing),
  **ATS**, not Trust, breaks the tie. Every passing draft already cleared
  the Trust threshold, so comparing Trust further adds no signal among
  passers — ATS keyword tailoring is what actually differentiates them. This
  was an explicit product choice, not an oversight: an earlier iteration of
  this design ranked by Trust first, then ATS, reasoning that Trust is the
  project's core anti-hallucination guarantee. That reasoning is sound for
  *comparing a passing draft to a failing one* — which is why "passed" is
  still the first-order check — but doesn't hold for ranking *within* the
  passing set, where Trust has already done its job.
- In the all-failed case, ranking is still ATS-only, with **no Trust-based
  fallback**. This is a known, documented limitation: a failing draft with
  severe hallucinations but strong keyword overlap can outrank one with
  fewer hallucinations but weaker tailoring. Accepted as a deliberate
  tradeoff (see Alternatives), not fixed, because nothing in this run
  ships anyway — the persisted draft still carries its real (bad) Trust
  score and a rejection reason built from it.
- `is_exhausted` (`iteration >= max_iterations`) is now `True` for every
  run that completes normally — there is no other way for the loop to end.
  It remains a meaningful concept mid-loop (it drives `should_continue`'s
  own logic) and on hand-built `WorkflowState`s in tests, but as an outcome
  signal on a finished run it is now tautological.

**3. `GenerateResponse.exhausted` and the exported `evaluation.json`'s
`"exhausted"` key are removed.** Both were downstream projections of
`is_exhausted`, and once that is always `True` on a real run, a field that
cannot vary carries no information. `passed` (really: `final_passed`, since
that's what `GenerateResponse.from_state` now reads) is the field to check
instead. Streamlit's warning banner collapsed from an `if passed / elif
exhausted` branch to `if passed / else`, for the same reason.

**4. `QualityGate.max_iterations`'s Pydantic floor relaxed from `ge=1` to
`ge=0`.** Under the old design, `max_iterations=0` was nearly pointless — a
gate that never rewrites is barely distinguishable from one that rewrites
once and immediately exhausts, since an early pass would end either at
iteration 0 anyway. Under the new design, `0` is a real, useful value:
"generate exactly one draft, no exploration, no extra cost." Several
integration tests that scripted "exactly one round of LLM calls" now pass
`gate=QualityGate(max_iterations=0)` explicitly to get that back.

**5. `max_iterations` became config/env-driven, not a hard-coded default.**
Because every default-gate run now pays for `max_iterations + 1` drafts
unconditionally, the default value itself became a real cost/quality
tradeoff an operator should be able to tune without editing code.
`model_factory.load_quality_gate()` mirrors `LLMConfig.load()`'s exact
precedence: `TRUSTRESUME_QUALITY_MAX_ITERATIONS` env var > gitignored
`config/quality_gate.local.json` > committed `config/quality_gate.json` >
`QualityGate`'s own Pydantic default (`3`). The committed default is `1`
(two total drafts) — deliberately *not* raised to match the old effective
default of `3` (four drafts), for the cost reason above and the empirical
finding below. `TrustResumeApp` resolves this once as `default_gate` at
construction; `build_default_app` calls `load_quality_gate()` when the
caller doesn't supply one. A per-call `gate=` to `generate()`/
`generate_for_job()` still overrides it for that call only.

**6. The override reaches the HTTP API and the Streamlit UI, not just
Python callers.** `GenerateRequest` and the new `GenerateForJobRequest`
(the latter previously had no request body schema at all) both gained an
optional `max_iterations: int | None`, threaded through to the facade as a
`QualityGate`. `GenerateForJobRequest`'s body is itself optional, so
existing callers that post no body are unaffected. `TrustResumeClient` and
both Streamlit generate actions (Generate tab, Jobs tab) gained a "Rewrite
attempts after the first draft" number input reaching the same parameter,
defaulting to `1` to match the committed config.

## Consequences

- **Every default-gate generation costs more, unconditionally.** Before,
  a lucky first-draft pass meant one round of LLM calls. Now, every run
  makes at least `max_iterations + 1` rounds no matter what — set by
  `config/quality_gate.json`, currently `2`. This is the central tradeoff
  of this ADR, and it is why the default was kept low rather than restored
  to the old effective cap of `4`.
- **A real empirical check, not just a design argument.** One comparison
  against Bedrock (`scripts/manual_rag_test.py`, one job posting, two sample
  résumés): with `max_iterations=0` (one draft), the shipped draft scored
  Trust 94.29 / ATS 45.16. With `max_iterations=1` (the committed default,
  two drafts), draft 0 scored Trust 90.0 / ATS 43.75 and the rewrite (draft
  1) scored Trust 94.44 / ATS **40.62** — the extra rewrite's ATS went
  *down*, and `final_index` correctly shipped draft 0 over the regressed
  draft 1 rather than whatever ran last. Neither run passed the gate for
  this job/candidate pair (the posting wanted skills — Django, TypeScript,
  React, agentic systems — the sample résumés genuinely don't have, and the
  Trust Harness correctly refused to fabricate them: zero hallucinations in
  both runs). This is **n=1, not a rigorous study** — one job, one candidate,
  one provider call each. It directionally supports keeping the default low
  rather than raising it, but does not prove extra rewrites are useless in
  general; a job/candidate pair closer to a real pass, or a larger sample,
  could tell a different story. Running `python -m trustresume.evals` with a
  larger labeled set, or repeating this comparison across several real
  job/résumé pairs, would raise confidence either way.
- **Two parallel accessor families on `WorkflowState` is real cognitive
  overhead.** `current_*` and `final_*` coexist and mean different things;
  a future contributor reaching for the "obvious" `current_draft` when they
  mean "the draft this run shipped" will get the *latest* iteration, not
  the *best* one, whenever the two diverge (any all-failed run, or any run
  where a later rewrite regressed after an earlier pass). Every real
  consumer (`_persist`, `GenerateResponse.from_state`, the exported
  artifacts, Streamlit) was audited and switched to `final_*`; `scripts/
  manual_rag_test.py` had the same bug and was fixed for the same reason.
- **No Trust-based fallback when every draft fails** is a known, accepted
  gap (see Decision, point 2) — flagged here so it reads as a considered
  tradeoff, not a bug someone should "fix" without revisiting the reasoning
  above.
- **The HTTP/UI override is coarse (one integer), not the full
  `QualityGate`.** `min_trust_score`/`min_ats_score` are not config- or
  request-driven — only `max_iterations` is, by deliberate scope choice.
  Making the thresholds themselves tunable per call is a larger, separate
  feature this ADR does not cover.

## Alternatives considered
- **Keep early-stop-on-pass; only make `max_iterations` configurable.**
  This is a smaller, less invasive change, but it doesn't deliver what was
  actually asked for: a lucky early pass would still end the loop
  immediately, so "configurable retry count" would only ever matter on the
  failure path, never as a way to find a *better* passing draft. Rejected
  because it answers a different question than the one asked.
- **Rank the all-failed case by Trust, falling back to ATS only on ties.**
  Considered and rejected in favor of pure ATS: this repo's earlier working
  version of `final_index` did exactly this (Trust before ATS, unconditionally,
  including as the all-failed tiebreak), and reasoned that Trust deserved
  priority everywhere because it is the project's core guarantee. That
  reasoning holds for "passing beats failing" but not for ranking *within*
  a side — see Decision, point 2 — and the product decision landed on ATS
  for both the passing-tiebreak and the all-failed case, accepting the
  documented gap rather than adding a second ranking rule for one branch.
- **Restore the default `max_iterations` to `3` (four total drafts),
  matching the old effective behavior.** Rejected pending the cost/quality
  question actually being answered — see Consequences. Kept low (`1`) as
  the more conservative default; raising it later is a one-line config
  change, not a code change, once there's real evidence it pays for itself.
- **Leave `max_iterations` as a hard-coded Pydantic default, expose the
  override only via `TrustResumeApp.generate(gate=...)`.** This was the
  first version of this change, and it left a real gap: an operator
  running the shipped API/UI had no way to change the default cost behavior
  without editing Python. `config/quality_gate.json` (mirroring the
  existing `LLMConfig`/`config/llm.json` pattern) closes that gap without
  requiring a code change to retune.
- **Not doing this at all.** Rejected because the ask was specific and the
  mechanism (a ranking function plus removing one clause from `_route`) is
  small; the harder part — validating whether it's worth the extra cost —
  is explicitly called out above as unfinished, not skipped.
