# ADR-0013: Pin sampling temperature; tier models by agent role

## Status
Accepted. New — the original repo built one model with provider defaults and
shared it across every agent.

## Context
Two related gaps in `api/model_factory.py`.

**1. Temperature was never set.** None of the three provider branches passed
`temperature`, so each provider's own default applied — and that default is
outside this project's control, differs by provider, and differs by model
generation (`langchain_aws` even drops the parameter entirely for models that
no longer accept it). Three of the four LLM steps are structured
extraction/classification (Job Description, Candidate Profile, Trust Harness),
where sampling variance shows up directly as a Trust score that differs
between identical runs. The project's headline claim is *deterministic,
reproducible scoring* — "the score is computed by code, never self-reported by
an LLM" — and that claim was resting on an unpinned sampling parameter.

**2. One model served every agent.** `TrustResumeApp.__init__` took a single
`model` and handed the same object to all four LLM agents. But the four jobs
have genuinely different cost/quality profiles: pulling fields out of text into
a fixed schema is not the same task as writing prose a human will read, which
is not the same task as adjudicating whether a claim is supported. Paying
top-tier prices for schema-filling is waste; economizing on the verifier is
the one place economizing is actively harmful.

## Decision
**Pin temperature explicitly, defaulting to `0`,** and pass it to every
provider branch. `0` because the majority of calls are extraction; the writer
can be raised per role if more varied prose is wanted.

**Introduce three roles**, defined by the *job* rather than by the agent name,
so a seventh agent picks a fitting role instead of needing a new config key:

| Role | Agents | Why |
|---|---|---|
| `extraction` | Job Description, Candidate Profile | Text → fixed schema. The cheapest model that reliably fills a schema is usually enough. |
| `writer` | Resume Writer | The one genuinely generative step, and the only place a stronger model changes what a user reads. |
| `verifier` | Trust Harness | The project's core correctness claim. The last place to economize. |

Config gains `temperature` and a `roles` map; each role entry may override the
model, the temperature, or both, inheriting anything it omits. Env vars
(`TRUSTRESUME_LLM_<ROLE>_MODEL` / `_TEMPERATURE`) win over the file, matching
the precedence the rest of `LLMConfig.load` already uses.

Tiering is **opt-in**: with no `roles` configured, all three resolve to the
same model and temperature, so the default deployment and every existing test
behave exactly as before. `TrustResumeApp` takes optional per-role models that
each fall back to the shared `model` argument.

A malformed `temperature` or `roles` section is logged and ignored rather than
raising — a config typo shouldn't take the server down at startup, and the
fallback (deterministic sampling) is the safe direction.

## Consequences
- Identical inputs now produce identical extraction and classification output
  for providers that honor `temperature=0`, so the reproducibility claim is
  backed by a setting rather than by luck.
- Cost and latency become tunable without touching code, and ADR-0012's
  telemetry reports per-model token splits — so the effect of a tiering change
  is measurable rather than asserted. `RunUsage.models` is a *list* precisely
  so a tiered run shows the split.
- `build_default_app` constructs three model objects instead of one. These are
  API clients, not loaded weights, so the cost is negligible; with no overrides
  they're three identically-configured clients.
- Temperature is passed even to models that reject the parameter;
  `ChatBedrockConverse` drops it automatically in that case, so pinning is safe
  across model generations.

## Alternatives considered
- **Per-agent model config** (one key per agent) rather than three roles.
  Rejected: it makes adding an agent a config-schema change, and the real
  distinction is the kind of work, not the class name.
- **Temperature `0` with no override.** Rejected: résumé prose is the one place
  sampling variety is arguably useful, so the writer keeps an escape hatch.
- **A separate "cheap model" boolean.** Rejected as a false binary — it can't
  express "cheap extraction, strong verifier, default writer", which is exactly
  the configuration this is for.
