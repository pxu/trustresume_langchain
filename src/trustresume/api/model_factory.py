"""Provider-agnostic LangChain chat model construction.

Agents take an injected ``langchain_core.language_models.BaseChatModel``
(``agents/base.py``'s ``ModelInput``); this module builds one from
``config/llm.json`` (a gitignored ``config/llm.local.json`` overlays it, env
vars win over both) so the rest of the app never hard-codes a provider.

Milestone M7 (api) — kept alongside ``app_service.py`` since both are part of
the app's wiring layer, not the agents themselves.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.language_models import BaseChatModel

BEDROCK_DEFAULT_MODEL = "global.anthropic.claude-opus-4-6-v1"
BEDROCK_DEFAULT_PROFILE = "twdc-bedrock-central"
BEDROCK_DEFAULT_REGION = "us-west-2"

_DEFAULT_MODELS = {"bedrock": BEDROCK_DEFAULT_MODEL, "openai": "gpt-4o", "google": "gemini-1.5-pro"}
_KNOWN_PROVIDERS = frozenset({"bedrock", "openai", "google", "test"})
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "llm.json"
_CONFIG_KEYS = frozenset({"provider", "model", "api_key", "aws_profile", "aws_region"})


@dataclass(frozen=True)
class LLMConfig:
    """Resolved LLM provider configuration.

    ``model=None`` means "use the provider default" (see ``model_name``).
    ``api_key`` is optional and only meaningful for ``openai``/``google`` —
    Bedrock authenticates via AWS credentials (``aws_profile``/``aws_region``),
    not an API key.
    """

    provider: str = "bedrock"
    model: str | None = None
    # repr=False so a stray `repr(config)`/`str(config)` (logging, an
    # uncaught exception's traceback locals) can't leak the key in cleartext.
    api_key: str | None = field(default=None, repr=False)
    aws_profile: str = BEDROCK_DEFAULT_PROFILE
    aws_region: str = BEDROCK_DEFAULT_REGION

    @classmethod
    def load(cls, path: str | Path | None = None) -> LLMConfig:
        """Resolve config with precedence: env > llm.local.json > llm.json > default.

        ``path`` (or ``$TRUSTRESUME_LLM_CONFIG``, or ``config/llm.json`` by
        default) is read first; a sibling ``llm.local.json`` next to it is
        read second and overlays same-named keys. Environment variables win
        over both file sources.
        """
        config_path = Path(path or os.getenv("TRUSTRESUME_LLM_CONFIG") or DEFAULT_CONFIG_PATH)
        file_values: dict[str, object] = {}
        for candidate in (config_path, config_path.with_name("llm.local.json")):
            if candidate.is_file():
                raw = json.loads(candidate.read_text(encoding="utf-8"))
                file_values.update({k: v for k, v in raw.items() if k in _CONFIG_KEYS})

        def opt(key: str, env: str) -> str | None:
            if env in os.environ:
                return os.environ[env]
            value = file_values.get(key)
            return str(value) if value is not None else None

        provider = os.getenv(
            "TRUSTRESUME_LLM_PROVIDER",
            os.getenv("TRUSTRESUME_LLM", str(file_values.get("provider", "bedrock"))),
        ).lower()
        return cls(
            provider=provider,
            model=opt("model", "TRUSTRESUME_LLM_MODEL"),
            api_key=opt("api_key", "TRUSTRESUME_LLM_API_KEY"),
            aws_profile=os.getenv(
                "TRUSTRESUME_AWS_PROFILE",
                str(file_values.get("aws_profile", BEDROCK_DEFAULT_PROFILE)),
            ),
            aws_region=os.getenv(
                "TRUSTRESUME_AWS_REGION",
                str(file_values.get("aws_region", BEDROCK_DEFAULT_REGION)),
            ),
        )

    @classmethod
    def from_env(cls) -> LLMConfig:
        """No-arg form of :meth:`load`, reading the default config path."""
        return cls.load()

    def model_name(self) -> str:
        """The effective model id: ``model`` if set, else the provider's default."""
        return self.model or _DEFAULT_MODELS.get(self.provider, "")


def build_model(config: LLMConfig) -> BaseChatModel:
    """Construct a LangChain chat model for ``config.provider``.

    Each provider's SDK is imported lazily inside its branch so installing
    only the providers you use keeps the default install lean.
    """
    if config.provider not in _KNOWN_PROVIDERS:
        raise ValueError(
            f"unknown LLM provider {config.provider!r}; expected one of {sorted(_KNOWN_PROVIDERS)}"
        )
    if config.provider == "bedrock":
        return build_bedrock_model(
            model_id=config.model_name(),
            profile_name=config.aws_profile,
            region_name=config.aws_region,
        )
    if config.provider == "openai":
        from langchain_openai import ChatOpenAI

        if config.api_key:
            return ChatOpenAI(model=config.model_name(), api_key=config.api_key)
        return ChatOpenAI(model=config.model_name())
    if config.provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if config.api_key:
            return ChatGoogleGenerativeAI(model=config.model_name(), google_api_key=config.api_key)
        return ChatGoogleGenerativeAI(model=config.model_name())
    # config.provider == "test"
    from .test_provider import AutoStructuredFakeChatModel

    return AutoStructuredFakeChatModel()


def build_bedrock_model(
    *,
    model_id: str = BEDROCK_DEFAULT_MODEL,
    profile_name: str = BEDROCK_DEFAULT_PROFILE,
    region_name: str = BEDROCK_DEFAULT_REGION,
) -> BaseChatModel:
    """Build a Bedrock Converse chat model from an explicit boto3 session/profile."""
    import boto3
    from langchain_aws import ChatBedrockConverse

    session = boto3.Session(profile_name=profile_name, region_name=region_name)
    client = session.client("bedrock-runtime")
    return ChatBedrockConverse(model_id=model_id, client=client)
