"""Provider-agnostic LLM smoke test.

A minimal end-to-end check that confirms two things work together for
whichever provider is currently configured (``config/llm.json`` /
``config/llm.local.json`` / env vars — see
:class:`trustresume.api.model_factory.LLMConfig`):

1. The configured provider/credentials can actually reach the model.
2. LangGraph's prebuilt tool-calling agent can drive that model and get a
   tool call back (the same mechanism ``with_structured_output`` rides on
   for the real agents in ``trustresume.agents``).

This is throwaway validation code, not part of the TrustResume pipeline. Run it
directly (see this package's README) — importing it has no side effects
because the LLM call lives inside ``main()``.

Note: the offline ``test`` provider (``GenericFakeChatModel``) has no real
model behind it and won't produce a meaningful tool-calling answer for this
prompt — use ``test`` to validate the rest of the app, not this script.
"""

from typing import Any

from langchain.agents import create_agent

from trustresume.api.model_factory import LLMConfig, build_model


def get_weather(city: str) -> dict[str, Any]:
    """Get current weather for a city."""
    return {"city": city, "temp_f": 72, "condition": "sunny"}


def convert_temp(fahrenheit: float) -> float:
    """Convert Fahrenheit to Celsius."""
    return round((fahrenheit - 32) * 5 / 9, 1)


def build_agent(config: LLMConfig | None = None) -> Any:
    """Construct a LangGraph tool-calling agent backed by whatever provider is configured.

    Uses the shared :func:`trustresume.api.model_factory.build_model` (same
    config resolution as the app) instead of wiring a provider SDK by hand, so
    this smoke test and the app construct models the same way. Defaults to
    :meth:`LLMConfig.from_env` when no config is given.
    """
    resolved_config = config or LLMConfig.from_env()
    model = build_model(resolved_config)
    return create_agent(model, tools=[get_weather, convert_temp])


SMOKE_TEST_PROMPT = "What's the weather in Paris and Tokyo, in Celsius?"


def run_smoke_test(prompt: str = SMOKE_TEST_PROMPT, config: LLMConfig | None = None) -> str:
    """Run the smoke-test prompt through the configured provider and return the answer.

    Returns the text so callers other than the CLI (e.g. the ``/api/ping``
    endpoint) can relay it. Constructs the agent on each call — this is a
    liveness probe, not a hot path.
    """
    agent = build_agent(config)
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    return str(result["messages"][-1].content)


def main() -> None:
    """Run the smoke-test prompt and print the model's answer."""
    config = LLMConfig.from_env()
    print(f"provider={config.provider} model={config.model_name()}")
    print(run_smoke_test(config=config))


if __name__ == "__main__":
    main()
