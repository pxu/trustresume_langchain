"""Per-run resource accounting: tokens, cost, and wall-clock time.

A single generation makes 5-9+ sequential LLM calls (one per LLM-backed agent
per quality-loop iteration), so "what did this run cost, and where did the
time go" is not answerable by looking at any one call. These models carry that
answer out of the orchestrator alongside the drafts and scores, so the API can
return it, the UI can show it, and the eval harness can report cost-per-run
next to quality metrics.

Deliberately in ``models/`` (framework-free) rather than next to the callback
that populates them (``trustresume.telemetry``, which imports LangChain): the
storage and API layers persist and serialize these without taking a LangChain
dependency, exactly like every other cross-layer contract here.

Added post-port; no equivalent in the original.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelUsage(BaseModel):
    """Token counts attributed to one model id within a run.

    Per-model rather than a single flat total because a run may legitimately
    span two models once agents are tiered by role (a cheap extraction model
    plus a stronger writer/verifier — see ``api/model_factory.py``), and the
    whole point of tiering is being able to see the split.
    """

    model_config = ConfigDict(extra="forbid")

    model: str = Field(..., min_length=1, description="Model id the provider reported.")
    input_tokens: int = Field(0, ge=0, description="Prompt tokens sent to this model.")
    output_tokens: int = Field(0, ge=0, description="Completion tokens returned by this model.")
    calls: int = Field(0, ge=0, description="Number of LLM calls attributed to this model.")

    @property
    def total_tokens(self) -> int:
        """Input + output tokens for this model."""
        return self.input_tokens + self.output_tokens


class NodeUsage(BaseModel):
    """Tokens for one LLM call, attributed to the orchestrator node that made it.

    One entry per successful call, in execution order — mirroring
    ``NodeTiming``'s per-execution (not per-node-aggregated) shape, because the
    same node name recurs once per quality-loop iteration with different token
    counts each time. Populated from LangGraph's own ``langgraph_node`` run
    metadata (see ``telemetry.UsageTracker``), so it costs nothing extra to
    compute and needs no cooperation from the agents themselves.
    """

    model_config = ConfigDict(extra="forbid")

    node: str = Field(..., min_length=1, description="Orchestrator node that made this call.")
    model: str = Field(..., min_length=1, description="Model id the provider reported.")
    input_tokens: int = Field(0, ge=0)
    output_tokens: int = Field(0, ge=0)
    cost_usd: float | None = Field(
        None, description="Estimated USD cost of this one call, or None if the model is unpriced."
    )


class NodeTiming(BaseModel):
    """One execution of one orchestrator node.

    A *list* of these (not a per-node total) is what the graph accumulates,
    because a node inside the quality loop runs once per iteration and the
    per-iteration shape is exactly what you want when asking "is the third
    rewrite slower than the first?". :meth:`RunUsage.duration_by_node` folds
    them into totals when that's what you want instead.
    """

    model_config = ConfigDict(extra="forbid")

    node: str = Field(..., min_length=1, description="Orchestrator node name.")
    duration_ms: float = Field(..., ge=0, description="Wall-clock duration of this execution.")


class RunUsage(BaseModel):
    """What one generation consumed: tokens, cost, and time.

    ``cost_usd`` is ``None`` when *any* model in the run has no configured
    price (``config/pricing.json``) — deliberately null rather than a partial
    sum, since a total that silently omits the most expensive model is worse
    than an honest "unknown". See ``trustresume.telemetry.estimate_cost``.
    """

    model_config = ConfigDict(extra="forbid")

    models: list[ModelUsage] = Field(
        default_factory=list, description="Token usage per model id, in first-seen order."
    )
    timings: list[NodeTiming] = Field(
        default_factory=list, description="One entry per node execution, in execution order."
    )
    node_calls: list[NodeUsage] = Field(
        default_factory=list,
        description="One entry per LLM call attributed to its orchestrator node, in call order.",
    )
    total_duration_ms: float = Field(
        0.0, ge=0, description="Wall-clock duration of the whole run, measured by the caller."
    )
    cost_usd: float | None = Field(
        None, description="Estimated USD cost, or None if any model's price is unknown."
    )

    @property
    def input_tokens(self) -> int:
        """Prompt tokens across every model in the run."""
        return sum(m.input_tokens for m in self.models)

    @property
    def output_tokens(self) -> int:
        """Completion tokens across every model in the run."""
        return sum(m.output_tokens for m in self.models)

    @property
    def total_tokens(self) -> int:
        """Input + output tokens across every model in the run."""
        return self.input_tokens + self.output_tokens

    @property
    def llm_calls(self) -> int:
        """How many LLM calls the run made, across every model."""
        return sum(m.calls for m in self.models)

    def duration_by_node(self) -> dict[str, float]:
        """Total milliseconds per node name, summed over repeated executions."""
        totals: dict[str, float] = {}
        for timing in self.timings:
            totals[timing.node] = totals.get(timing.node, 0.0) + timing.duration_ms
        return totals

    def log_fields(self) -> dict[str, object]:
        """A flat, JSON-loggable summary — safe to splat into ``logging``'s ``extra``.

        Deliberately flat scalars (no nested dicts, no reserved ``LogRecord``
        attribute names like ``module``/``filename``) so it can be passed
        straight to ``logger.info(..., extra=...)`` — see
        ``logging_config.py`` for why that matters.
        """
        return {
            "llm_calls": self.llm_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "duration_ms": round(self.total_duration_ms, 1),
        }
