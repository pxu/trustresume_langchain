"""Unit tests for LLM usage capture, pricing, and cost estimation."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import TypedDict

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from trustresume.api.test_provider import FAKE_MODEL_NAME, AutoStructuredFakeChatModel
from trustresume.models import ModelUsage, NodeTiming, RunUsage
from trustresume.telemetry import (
    UNKNOWN_MODEL,
    UsageTracker,
    estimate_cost,
    load_pricing,
    track_usage,
)

_RUN_ID = uuid.uuid4()


class _Schema(BaseModel):
    a: str = ""


def _llm_result(
    *, model: str | None, input_tokens: int | None = 10, output_tokens: int = 5
) -> LLMResult:
    """An ``LLMResult`` shaped the way a chat provider returns one."""
    usage = (
        None
        if input_tokens is None
        else {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
    )
    message = AIMessage(
        content="hi",
        usage_metadata=usage,  # type: ignore[arg-type]
        response_metadata={"model_name": model} if model else {},
    )
    return LLMResult(generations=[[ChatGeneration(message=message)]])


# --- UsageTracker ---------------------------------------------------------


def test_usageTracker_multipleCalls_accumulatesPerModel() -> None:
    tracker = UsageTracker()
    tracker.on_llm_end(_llm_result(model="m1", input_tokens=10, output_tokens=5))
    tracker.on_llm_end(_llm_result(model="m1", input_tokens=20, output_tokens=7))
    tracker.on_llm_end(_llm_result(model="m2", input_tokens=1, output_tokens=1))

    usage = tracker.usage()
    assert [u.model for u in usage] == ["m1", "m2"]  # first-seen order
    assert usage[0] == ModelUsage(model="m1", input_tokens=30, output_tokens=12, calls=2)
    assert usage[1].calls == 1


def test_usageTracker_missingModelName_countsUnderUnknownNotDropped() -> None:
    """A provider that omits ``model_name`` must not silently zero the bill."""
    tracker = UsageTracker()
    tracker.on_llm_end(_llm_result(model=None, input_tokens=10, output_tokens=5))

    usage = tracker.usage()
    assert len(usage) == 1
    assert usage[0].model == UNKNOWN_MODEL
    assert usage[0].total_tokens == 15


def test_usageTracker_noUsageMetadata_stillCountsTheCall() -> None:
    tracker = UsageTracker()
    tracker.on_llm_end(_llm_result(model="m1", input_tokens=None))

    usage = tracker.usage()
    assert usage[0].calls == 1
    assert usage[0].total_tokens == 0


def test_usageTracker_nonChatGeneration_ignored() -> None:
    tracker = UsageTracker()
    tracker.on_llm_end(LLMResult(generations=[[]]))

    assert tracker.usage() == []


def test_usageTracker_callWithNoRunId_stillCountsButUnattributed() -> None:
    """Direct callers (existing tests, ad-hoc scripts) have no run_id at all."""
    tracker = UsageTracker()
    tracker.on_llm_end(_llm_result(model="m1"))

    assert tracker.usage()[0].calls == 1
    assert tracker.node_calls() == []


# --- per-node attribution (via a real LangGraph invocation) ----------------


class _GraphState(TypedDict):
    done: int


def test_usageTracker_realGraphInvocation_attributesTokensToTheirNode() -> None:
    """LangGraph tags every LLM call's metadata with the node that made it.

    Runs a tiny two-node graph, each calling the offline fake model once, and
    checks the tracker split those two calls back out by node — this is the
    mechanism ``metrics.json``'s per-step breakdown depends on.
    """
    model = AutoStructuredFakeChatModel()
    tracker = UsageTracker()

    async def node_a(state: _GraphState) -> dict[str, int]:
        await model.with_structured_output(_Schema).ainvoke(
            "prompt a", config={"callbacks": [tracker]}
        )
        return {"done": 1}

    async def node_b(state: _GraphState) -> dict[str, int]:
        await model.with_structured_output(_Schema).ainvoke(
            "prompt b", config={"callbacks": [tracker]}
        )
        return {"done": 2}

    builder = StateGraph(_GraphState)
    builder.add_node("node_a", node_a)
    builder.add_node("node_b", node_b)
    builder.add_edge(START, "node_a")
    builder.add_edge("node_a", "node_b")
    builder.add_edge("node_b", END)
    graph = builder.compile()

    asyncio.run(graph.ainvoke({"done": 0}, config={"callbacks": [tracker]}))

    calls = tracker.node_calls()
    assert [c.node for c in calls] == ["node_a", "node_b"]
    assert all(c.model == FAKE_MODEL_NAME for c in calls)
    assert all(c.input_tokens > 0 for c in calls)


def test_usageTracker_nodeCalls_carryPerCallCostWhenPriced() -> None:
    tracker = UsageTracker(pricing={"m1": (1.0, 1.0)})
    tracker.on_chat_model_start({}, [], run_id=_RUN_ID, metadata={"langgraph_node": "write_resume"})
    tracker.on_llm_end(
        _llm_result(model="m1", input_tokens=1_000_000, output_tokens=2_000_000), run_id=_RUN_ID
    )

    calls = tracker.node_calls()
    assert calls[0].cost_usd == 3.0


def test_usageTracker_nodeCalls_unpricedModel_costIsNoneNotZero() -> None:
    tracker = UsageTracker(pricing={})
    tracker.on_chat_model_start({}, [], run_id=_RUN_ID, metadata={"langgraph_node": "write_resume"})
    tracker.on_llm_end(_llm_result(model="mystery"), run_id=_RUN_ID)

    assert tracker.node_calls()[0].cost_usd is None


# --- pricing / cost -------------------------------------------------------


def test_estimateCost_knownPrices_sumsPerMillionTokens() -> None:
    usage = [ModelUsage(model="gpt-4o", input_tokens=1_000_000, output_tokens=500_000, calls=1)]
    assert estimate_cost(usage, {"gpt-4o": (2.5, 10.0)}) == 7.5


def test_estimateCost_unknownModel_returnsNoneNotPartialTotal() -> None:
    """A partial total reads like a real number and would understate the bill."""
    usage = [
        ModelUsage(model="gpt-4o", input_tokens=1_000_000, output_tokens=0, calls=1),
        ModelUsage(model="mystery-model", input_tokens=1_000_000, output_tokens=0, calls=1),
    ]
    assert estimate_cost(usage, {"gpt-4o": (2.5, 10.0)}) is None


def test_estimateCost_noObservedCalls_isUnknownNotFree() -> None:
    """ "No LLM call was seen" is a measurement outcome, not a $0 bill.

    If a provider ever stops emitting on_llm_end, a run that really cost money
    would otherwise be persisted and shown to the user as free.
    """
    assert estimate_cost([], {}) is None


def test_estimateCost_versionedModelId_matchesByLongestFragment() -> None:
    """'gpt-4o' must not shadow 'gpt-4o-mini' for a versioned mini id."""
    pricing = {"gpt-4o": (2.5, 10.0), "gpt-4o-mini": (0.15, 0.6)}
    usage = [
        ModelUsage(model="gpt-4o-mini-2024-07-18", input_tokens=1_000_000, output_tokens=0, calls=1)
    ]
    assert estimate_cost(usage, pricing) == 0.15


def test_estimateCost_bedrockInferenceProfilePrefix_stillMatches() -> None:
    """A region/profile prefix and a -v suffix must both be seen through."""
    pricing = {"claude-opus-4": (15.0, 75.0)}
    usage = [
        ModelUsage(
            model="global.anthropic.claude-opus-4-v1", input_tokens=1_000_000, output_tokens=0
        )
    ]
    assert estimate_cost(usage, pricing) == 15.0


def test_estimateCost_laterModelVersion_isNotPricedAsTheEarlierOne() -> None:
    """A new release must report unknown, not silently inherit an earlier price.

    Unguarded substring matching would let a 'claude-opus-4' row also price
    'claude-opus-4-6' — a different model with its own (unconfigured) rate.
    """
    pricing = {"claude-opus-4": (15.0, 75.0)}
    usage = [
        ModelUsage(
            model="global.anthropic.claude-opus-4-6-v1", input_tokens=1_000_000, output_tokens=0
        )
    ]
    assert estimate_cost(usage, pricing) is None


def test_estimateCost_versionKeyMatchesAtEndOfId_stillPriced() -> None:
    """A version-ending key with nothing after it in the id is a real match,
    not a guarded-against extension. Uses a prefixed id so the match goes
    through substring matching rather than short-circuiting on an exact hit.
    """
    pricing = {"claude-opus-4": (15.0, 75.0)}
    usage = [ModelUsage(model="anthropic.claude-opus-4", input_tokens=1_000_000, output_tokens=0)]
    assert estimate_cost(usage, pricing) == 15.0


def test_estimateCost_versionKeyImmediatelyFollowedByDigit_rejected() -> None:
    """'claude-opus-4' must not match inside 'claude-opus-42' (no separator, still a digit)."""
    pricing = {"claude-opus-4": (15.0, 75.0)}
    usage = [ModelUsage(model="claude-opus-42", input_tokens=1_000_000, output_tokens=0)]
    assert estimate_cost(usage, pricing) is None


def test_loadPricing_readsModelsSection(tmp_path: Path) -> None:
    path = tmp_path / "pricing.json"
    path.write_text(
        json.dumps({"models": {"m1": {"input_per_1m": 1.0, "output_per_1m": 2.0}}}),
        encoding="utf-8",
    )
    assert load_pricing(path) == {"m1": (1.0, 2.0)}


def test_loadPricing_missingFile_returnsEmptyTable(tmp_path: Path) -> None:
    assert load_pricing(tmp_path / "nope.json") == {}


def test_loadPricing_malformedFile_returnsEmptyTableNotRaise(tmp_path: Path) -> None:
    """A broken pricing config must never fail a generation."""
    path = tmp_path / "pricing.json"
    path.write_text("{not json", encoding="utf-8")
    assert load_pricing(path) == {}


def test_loadPricing_committedConfigParses() -> None:
    """The repo's own config/pricing.json is valid and non-empty."""
    assert load_pricing() != {}


def test_loadPricing_committedConfig_hasNoPriceForTheOfflineFake() -> None:
    """Offline runs must report real tokens with an honest cost_usd=None."""
    assert (
        estimate_cost(
            [ModelUsage(model=FAKE_MODEL_NAME, input_tokens=1, output_tokens=1, calls=1)],
            load_pricing(),
        )
        is None
    )


def test_loadPricing_committedConfig_pricesTheRealBedrockDefaultModel() -> None:
    """The default deploy target (config/llm.json's Bedrock model) must be
    priced explicitly, not "unknown", or every sample/eval run reports null cost.
    """
    from trustresume.api.model_factory import BEDROCK_DEFAULT_MODEL

    assert estimate_cost(
        [ModelUsage(model=BEDROCK_DEFAULT_MODEL, input_tokens=1_000_000, output_tokens=0)],
        load_pricing(),
    ) == pytest.approx(15.0)


# --- track_usage / finalize ----------------------------------------------


def test_trackUsage_realModelCalls_capturesTokensAndCalls() -> None:
    """End-to-end through the offline provider: callbacks must reach the tracker."""
    model = AutoStructuredFakeChatModel()

    async def run() -> RunUsage:
        with track_usage(pricing={FAKE_MODEL_NAME: (1.0, 1.0)}) as tracker:
            for _ in range(3):
                await model.with_structured_output(_Schema).ainvoke(
                    "a prompt long enough to count", config={"callbacks": [tracker]}
                )
        return tracker.finalize(timings=[NodeTiming(node="write_resume", duration_ms=2.0)])

    usage = asyncio.run(run())
    assert usage.llm_calls == 3
    assert usage.input_tokens > 0
    assert usage.total_duration_ms > 0
    assert usage.cost_usd is not None
    assert usage.duration_by_node() == {"write_resume": 2.0}


def test_trackUsage_blockRaises_stillStampsDuration() -> None:
    tracker_holder: list[UsageTracker] = []
    try:
        with track_usage(pricing={}) as tracker:
            tracker_holder.append(tracker)
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert tracker_holder[0].finalize().total_duration_ms > 0


def test_runUsage_durationByNode_sumsRepeatedExecutions() -> None:
    usage = RunUsage(
        timings=[
            NodeTiming(node="write_resume", duration_ms=10.0),
            NodeTiming(node="score_trust", duration_ms=4.0),
            NodeTiming(node="write_resume", duration_ms=6.0),
        ]
    )
    assert usage.duration_by_node() == {"write_resume": 16.0, "score_trust": 4.0}


def test_runUsage_logFields_areFlatScalarsSafeForLoggingExtra() -> None:
    usage = RunUsage(
        models=[ModelUsage(model="m1", input_tokens=3, output_tokens=2, calls=1)],
        total_duration_ms=12.34,
    )
    fields = usage.log_fields()
    assert fields["total_tokens"] == 5
    assert fields["duration_ms"] == 12.3
    assert all(isinstance(v, int | float | str) or v is None for v in fields.values())


def test_loadPricing_nonObjectJson_returnsEmptyTableNotAttributeError(tmp_path: Path) -> None:
    """A top-level array would make ``.get`` raise, on *every* generation."""
    path = tmp_path / "pricing.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    assert load_pricing(path) == {}
