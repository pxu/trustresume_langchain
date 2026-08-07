"""LLM usage capture: tokens per model, estimated cost, per-node timing.

Why a callback and not a return value: every LLM-backed agent calls
``model.with_structured_output(Schema)``, whose result is the *parsed Pydantic
object* — the underlying ``AIMessage`` (and with it ``usage_metadata``) is
consumed inside the chain and never reaches the agent. A callback handler is
the only place the raw message is still visible, so that is where token counts
have to be collected.

``UsageTracker`` deliberately does not reuse ``langchain_core``'s
``UsageMetadataCallbackHandler``, for two reasons: it drops usage entirely
when a provider omits ``response_metadata["model_name"]`` (silently reporting
zero cost is the one failure mode this module exists to prevent), and it
counts tokens but not *calls* — "how many LLM round-trips did one generation
make" is a headline number for a pipeline that makes 5-9+ of them.

Pricing lives in ``config/pricing.json``, not in this file: list prices change
often enough that hard-coding them in source guarantees they are wrong later,
and a wrong cost number is worse than no cost number. An unpriced model yields
``cost_usd=None`` (see :func:`estimate_cost`) rather than a partial total.

Added post-port; no equivalent in the original.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from trustresume.models import ModelUsage, NodeTiming, RunUsage

logger = logging.getLogger(__name__)

DEFAULT_PRICING_PATH = Path(__file__).resolve().parents[2] / "config" / "pricing.json"

#: Model id reported when a provider omits ``model_name`` from its response
#: metadata. Tokens are still counted under this key (rather than dropped);
#: it has no price, so a run containing it reports ``cost_usd=None``.
UNKNOWN_MODEL = "unknown"


class UsageTracker(BaseCallbackHandler):
    """Accumulates token usage and call counts across every LLM call in a run.

    Pass one instance per generation via ``config={"callbacks": [tracker]}``:
    LangChain propagates callbacks down through nested runnables, so a single
    tracker attached to the LangGraph invocation sees every agent's model call
    without any agent knowing it exists.

    Thread-safe (the lock mirrors ``UsageMetadataCallbackHandler``'s): a
    handler instance can be invoked from LangChain's executor threads, and
    nothing guarantees those are the caller's thread.
    """

    def __init__(self, pricing: dict[str, tuple[float, float]] | None = None) -> None:
        super().__init__()
        self._lock = threading.Lock()
        # Insertion-ordered so ``usage()`` reports models in first-seen order,
        # which for a tiered run means "extraction model first" — stable
        # output makes the eval harness's diffs readable.
        self._by_model: dict[str, ModelUsage] = {}
        self._pricing = pricing if pricing is not None else {}
        self._elapsed_ms = 0.0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Record one completed LLM call's tokens against its model id."""
        message = _final_message(response)
        if message is None:
            return
        usage = message.usage_metadata
        model = str(message.response_metadata.get("model_name") or UNKNOWN_MODEL)
        # A call with no usage metadata at all still counts as a call — the
        # round-trip happened, and a silently missing call is exactly the
        # blind spot this class exists to close.
        input_tokens = int(usage["input_tokens"]) if usage else 0
        output_tokens = int(usage["output_tokens"]) if usage else 0
        with self._lock:
            current = self._by_model.get(model)
            if current is None:
                self._by_model[model] = ModelUsage(
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    calls=1,
                )
            else:
                self._by_model[model] = ModelUsage(
                    model=model,
                    input_tokens=current.input_tokens + input_tokens,
                    output_tokens=current.output_tokens + output_tokens,
                    calls=current.calls + 1,
                )

    def usage(self) -> list[ModelUsage]:
        """A snapshot of per-model usage so far, in first-seen order."""
        with self._lock:
            return list(self._by_model.values())

    def finalize(self, *, timings: list[NodeTiming] | None = None) -> RunUsage:
        """Fold everything captured so far into a :class:`RunUsage`.

        Call this *after* the :func:`track_usage` block exits — the elapsed
        wall-clock time is stamped on exit, so finalizing early reports
        ``total_duration_ms=0``. ``timings`` comes from the orchestrator's
        graph state (the tracker never sees node boundaries; it only sees LLM
        calls), which is why it's a parameter rather than something collected
        here.
        """
        models = self.usage()
        return RunUsage(
            models=models,
            timings=list(timings or []),
            total_duration_ms=self._elapsed_ms,
            cost_usd=estimate_cost(models, self._pricing),
        )


def _final_message(response: LLMResult) -> AIMessage | None:
    """The ``AIMessage`` of a completed call, or ``None`` if there isn't one."""
    for generations in response.generations:
        for generation in generations:
            if isinstance(generation, ChatGeneration) and isinstance(generation.message, AIMessage):
                return generation.message
    return None


def load_pricing(path: str | Path | None = None) -> dict[str, tuple[float, float]]:
    """Load ``{model_id: (usd_per_1M_input, usd_per_1M_output)}`` from JSON.

    Path precedence: the ``path`` argument, then ``$TRUSTRESUME_PRICING``,
    then ``config/pricing.json``. A missing or malformed file is *not* an
    error — it yields an empty table, so cost simply reports as unknown and
    a generation never fails because of a pricing-config problem.
    """
    pricing_path = Path(path or os.getenv("TRUSTRESUME_PRICING") or DEFAULT_PRICING_PATH)
    if not pricing_path.is_file():
        return {}
    try:
        raw = json.loads(pricing_path.read_text(encoding="utf-8"))
        # A top-level array or scalar would make ``.get`` raise AttributeError,
        # which is not one of the caught types below — and this runs on every
        # generation, so it would turn a bad config file into a 500 on every
        # request rather than the documented "cost reports as unknown".
        entries = raw.get("models", {}) if isinstance(raw, dict) else {}
        return {
            str(model): (float(prices["input_per_1m"]), float(prices["output_per_1m"]))
            for model, prices in entries.items()
        }
    except (json.JSONDecodeError, AttributeError, KeyError, TypeError, ValueError):
        logger.warning("pricing config unreadable; cost will report as unknown")
        return {}


def _price_for(model: str, pricing: dict[str, tuple[float, float]]) -> tuple[float, float] | None:
    """Price for ``model``: exact id first, then longest matching id fragment.

    Provider model ids are versioned and often region-prefixed
    (``global.anthropic.claude-opus-4-6-v1``,
    ``gpt-4o-2024-08-06``), so requiring an exact match would mean re-editing
    the price table on every provider version bump. Substring matching keyed
    on the *longest* configured id avoids the other trap — ``gpt-4o`` must not
    shadow ``gpt-4o-mini`` just because it was declared first.
    """
    if model in pricing:
        return pricing[model]
    matches = [key for key in pricing if key in model]
    if not matches:
        return None
    return pricing[max(matches, key=len)]


def estimate_cost(usage: list[ModelUsage], pricing: dict[str, tuple[float, float]]) -> float | None:
    """Total USD for ``usage``, or ``None`` if any model has no known price.

    All-or-nothing on purpose: a partial total that quietly omits whichever
    model isn't in the price table reads like a real number and would be
    wrong, usually in the cheap direction. ``None`` forces the reader to
    either add the price or accept that cost is unknown.
    """
    if not usage:
        # Deliberately not 0.0. An empty tracker means "no LLM call was
        # observed", which is a *measurement* outcome, not a bill: if a
        # provider ever stops emitting ``on_llm_end``, a run that really cost
        # money would otherwise be persisted and displayed as free — the exact
        # failure this module exists to prevent, and the reason the database
        # columns are nullable. A caller that genuinely made no calls loses
        # nothing by reporting unknown.
        return None
    total = 0.0
    for entry in usage:
        price = _price_for(entry.model, pricing)
        if price is None:
            return None
        input_per_1m, output_per_1m = price
        total += entry.input_tokens / 1_000_000 * input_per_1m
        total += entry.output_tokens / 1_000_000 * output_per_1m
    return round(total, 6)


@contextmanager
def track_usage(pricing: dict[str, tuple[float, float]] | None = None) -> Iterator[UsageTracker]:
    """Yield a :class:`UsageTracker` and stamp the block's wall-clock time on exit.

    The tracker is yielded so the caller can hand it to LangChain as a
    callback; call :meth:`UsageTracker.finalize` after the block to get the
    :class:`RunUsage`. Timing brackets the whole block, so it includes
    non-LLM work (retrieval, scoring, graph overhead) — that total is what a
    user actually waits on, and splitting it out per node is what
    ``RunUsage.timings`` is for.

    The timer is stamped in ``finally``, so a run that raises still reports
    how long it ran before failing.
    """
    tracker = UsageTracker(load_pricing() if pricing is None else pricing)
    started = time.perf_counter()
    try:
        yield tracker
    finally:
        tracker._elapsed_ms = (time.perf_counter() - started) * 1000
