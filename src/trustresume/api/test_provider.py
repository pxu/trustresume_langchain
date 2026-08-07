"""Offline ``"test"`` LLM provider: a chat model that needs no credentials.

``GenericFakeChatModel`` (LangChain's built-in fake) only replays a
pre-scripted message sequence and doesn't implement ``bind_tools`` at all, so
it can't drive ``with_structured_output`` — which every LLM-backed agent in
this app uses. That would make ``TRUSTRESUME_LLM_PROVIDER=test`` unable to
actually run ``/api/generate`` end to end, breaking the documented
walk-up-and-try-it-without-credentials use case (see ``server.py``'s
``build_served_app`` docstring).

``AutoStructuredFakeChatModel`` fixes that generically: whatever schema an
agent's ``with_structured_output(Schema)`` binds, it synthesizes a minimal
valid instance (empty strings/lists, the first enum value, zero for numbers)
for that schema's *required* fields and lets Pydantic fill in the rest from
its own defaults — so any of the app's agents can drive it without being
individually scripted.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.messages.tool import ToolCall
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool


def _synthesize_value(schema: dict[str, Any]) -> Any:
    """A minimal value satisfying a single JSON Schema node."""
    if "enum" in schema:
        return schema["enum"][0]
    json_type = schema.get("type")
    if json_type == "string":
        # Non-empty only when required to be (``minLength``) — otherwise "".
        return "test" if schema.get("minLength", 0) else ""
    if json_type == "integer":
        return int(schema.get("minimum", 0))
    if json_type == "number":
        return float(schema.get("minimum", 0.0))
    if json_type == "boolean":
        return False
    if json_type == "array":
        return []
    if json_type == "object":
        return _synthesize_object(schema)
    if "anyOf" in schema:
        # An untagged union (e.g. ``str | None``) without its own default:
        # prefer the first non-null branch.
        for branch in schema["anyOf"]:
            if branch.get("type") != "null":
                return _synthesize_value(branch)
        return None
    if "default" in schema:
        return schema["default"]
    return None


def _synthesize_object(schema: dict[str, Any]) -> dict[str, Any]:
    """A dict with only the schema's *required* fields filled in.

    Optional fields are omitted entirely — Pydantic validation fills them
    from the model's own field defaults, so there's no need to reconstruct a
    default value from the JSON Schema (which, e.g., doesn't represent
    ``default_factory=list`` the same way a plain ``default`` would).
    """
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    return {name: _synthesize_value(properties[name]) for name in required if name in properties}


#: Model id this fake reports as ``response_metadata["model_name"]``. It is
#: deliberately absent from ``config/pricing.json``, so an offline run reports
#: real token counts with ``cost_usd=None`` — the honest answer, and a live
#: exercise of the unpriced-model path.
FAKE_MODEL_NAME = "trustresume-test-fake"


class AutoStructuredFakeChatModel(BaseChatModel):
    """Offline chat model that auto-fills whatever tool schema is bound to it."""

    @property
    def _llm_type(self) -> str:
        return FAKE_MODEL_NAME

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> Runnable[Any, Any]:
        formatted = [convert_to_openai_tool(t) for t in tools]
        return self.bind(tools=formatted, **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        tools = kwargs.get("tools")
        if not tools:
            # No schema bound (e.g. a plain chat call, like the poc smoke
            # test's tool-calling agent probing before it picks a tool).
            return ChatResult(
                generations=[ChatGeneration(message=self._message("ok", messages, None))]
            )

        function = tools[0]["function"]
        args = _synthesize_object(function["parameters"])
        tool_call = ToolCall(name=function["name"], args=args, id="auto-1")
        return ChatResult(
            generations=[ChatGeneration(message=self._message("", messages, tool_call))]
        )

    def _message(
        self, content: str, prompt: list[BaseMessage], tool_call: ToolCall | None
    ) -> AIMessage:
        """An ``AIMessage`` carrying synthetic-but-plausible token accounting.

        Real usage numbers matter here even though the model is fake: the
        telemetry path (``trustresume.telemetry.UsageTracker``) reads
        ``usage_metadata`` and ``response_metadata["model_name"]`` off this
        message, and a fake that omitted them would make token/cost reporting
        untestable in exactly the offline mode the whole test suite runs in.
        Counts are a deterministic ~4-chars-per-token estimate of the real
        prompt and response, so they scale with input size the way real ones
        do — they are *not* a real tokenizer's output and shouldn't be
        compared against a provider's billing.
        """
        prompt_chars = sum(len(str(message.content)) for message in prompt)
        response_chars = len(content) + (len(str(tool_call["args"])) if tool_call else 0)
        input_tokens = max(1, prompt_chars // 4)
        output_tokens = max(1, response_chars // 4)
        return AIMessage(
            content=content,
            tool_calls=[tool_call] if tool_call else [],
            usage_metadata=UsageMetadata(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
            response_metadata={"model_name": FAKE_MODEL_NAME},
        )
