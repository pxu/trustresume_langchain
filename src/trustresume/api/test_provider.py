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


class AutoStructuredFakeChatModel(BaseChatModel):
    """Offline chat model that auto-fills whatever tool schema is bound to it."""

    @property
    def _llm_type(self) -> str:
        return "trustresume-test-fake"

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
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

        function = tools[0]["function"]
        args = _synthesize_object(function["parameters"])
        message = AIMessage(
            content="", tool_calls=[ToolCall(name=function["name"], args=args, id="auto-1")]
        )
        return ChatResult(generations=[ChatGeneration(message=message)])
