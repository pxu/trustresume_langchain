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
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

BEDROCK_DEFAULT_MODEL = "global.anthropic.claude-opus-4-6-v1"
BEDROCK_DEFAULT_PROFILE = "twdc-bedrock-central"
BEDROCK_DEFAULT_REGION = "us-west-2"

_DEFAULT_MODELS = {"bedrock": BEDROCK_DEFAULT_MODEL, "openai": "gpt-4o", "google": "gemini-1.5-pro"}
_KNOWN_PROVIDERS = frozenset({"bedrock", "openai", "google", "test"})
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "llm.json"
_CONFIG_KEYS = frozenset(
    {"provider", "model", "api_key", "aws_profile", "aws_region", "temperature", "roles"}
)

#: The three jobs an LLM does in this pipeline, as distinct cost/quality
#: profiles rather than as agent names:
#:
#: * ``extraction`` — Job Description + Candidate Profile agents. Pulling
#:   fields out of text into a fixed schema; the cheapest model that can hold
#:   a schema is usually enough.
#: * ``writer`` — Resume Writer. The one genuinely generative step, and the
#:   only place a stronger model visibly changes the output a user reads.
#: * ``verifier`` — Trust Harness. Classifies each claim against evidence;
#:   this is the project's core correctness claim, so it is the *last* place
#:   to economize.
#:
#: Grouping by job rather than by agent means adding a seventh agent doesn't
#: require a new config key — it picks whichever role fits.
AGENT_ROLES = ("extraction", "writer", "verifier")

#: Sampling temperature default. ``0`` because three of the four LLM steps are
#: structured extraction/classification, where sampling variance shows up
#: directly as an unreproducible Trust score. Override per role (see
#: :class:`RoleConfig`) for the writer if you want more varied prose.
DEFAULT_TEMPERATURE = 0.0


@dataclass(frozen=True)
class RoleConfig:
    """Per-role overrides layered on top of the base :class:`LLMConfig`.

    ``None`` for either field means "inherit the base config's value", so a
    role entry can override only the model, only the temperature, or both.
    """

    model: str | None = None
    temperature: float | None = None


@dataclass(frozen=True)
class LLMConfig:
    """Resolved LLM provider configuration.

    ``model=None`` means "use the provider default" (see ``model_name``).
    ``api_key`` is optional and only meaningful for ``openai``/``google`` —
    Bedrock authenticates via AWS credentials (``aws_profile``/``aws_region``),
    not an API key.

    ``temperature`` and ``roles`` control sampling and per-role model tiering:
    ``build_model(config, role="extraction")`` resolves the model id and
    temperature for that role, falling back to the base values.
    """

    provider: str = "bedrock"
    model: str | None = None
    # repr=False so a stray `repr(config)`/`str(config)` (logging, an
    # uncaught exception's traceback locals) can't leak the key in cleartext.
    api_key: str | None = field(default=None, repr=False)
    aws_profile: str = BEDROCK_DEFAULT_PROFILE
    aws_region: str = BEDROCK_DEFAULT_REGION
    temperature: float = DEFAULT_TEMPERATURE
    roles: Mapping[str, RoleConfig] = field(default_factory=dict)

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
            temperature=_float_or(
                opt("temperature", "TRUSTRESUME_LLM_TEMPERATURE"), DEFAULT_TEMPERATURE
            ),
            roles=_parse_roles(file_values.get("roles")),
        )

    @classmethod
    def from_env(cls) -> LLMConfig:
        """No-arg form of :meth:`load`, reading the default config path."""
        return cls.load()

    def model_name(self, role: str | None = None) -> str:
        """The effective model id for ``role``.

        Resolution order: the role's own override, then this config's
        ``model``, then the provider default. ``role=None`` (or a role with no
        entry) resolves to the base model, so every existing caller is
        unaffected.
        """
        override = self.roles.get(role or "", RoleConfig()).model
        return override or self.model or _DEFAULT_MODELS.get(self.provider, "")

    def temperature_for(self, role: str | None = None) -> float:
        """The effective sampling temperature for ``role`` (role override, else base)."""
        override = self.roles.get(role or "", RoleConfig()).temperature
        return self.temperature if override is None else override


def _float_or(value: str | None, fallback: float) -> float:
    """``float(value)``, or ``fallback`` when unset or unparseable.

    A typo'd temperature must not crash startup with a bare ``ValueError``
    from deep inside config loading — the safe default (deterministic
    sampling) is a better failure mode than no app at all.
    """
    if value is None:
        return fallback
    try:
        return float(value)
    except ValueError:
        logger.warning("ignoring unparseable temperature %r; using %s", value, fallback)
        return fallback


def _parse_roles(raw: object) -> dict[str, RoleConfig]:
    """Parse the config file's ``roles`` mapping, ignoring anything malformed.

    Env-var overrides (``TRUSTRESUME_LLM_<ROLE>_MODEL``) are applied on top,
    so a deployment can retier models without editing a file — the same
    "env wins over file" precedence the rest of :meth:`LLMConfig.load` uses.
    """
    parsed: dict[str, RoleConfig] = {}
    entries = raw if isinstance(raw, dict) else {}
    for role in AGENT_ROLES:
        entry = entries.get(role)
        entry = entry if isinstance(entry, dict) else {}
        model = os.getenv(f"TRUSTRESUME_LLM_{role.upper()}_MODEL") or entry.get("model")
        raw_temp = os.getenv(f"TRUSTRESUME_LLM_{role.upper()}_TEMPERATURE")
        if raw_temp is None and entry.get("temperature") is not None:
            raw_temp = str(entry["temperature"])
        temperature = None if raw_temp is None else _float_or(raw_temp, DEFAULT_TEMPERATURE)
        if model or temperature is not None:
            parsed[role] = RoleConfig(model=str(model) if model else None, temperature=temperature)
    return parsed


def build_model(config: LLMConfig, *, role: str | None = None) -> BaseChatModel:
    """Construct a LangChain chat model for ``config.provider`` and ``role``.

    Each provider's SDK is imported lazily inside its branch so installing
    only the providers you use keeps the default install lean.

    ``role`` (one of :data:`AGENT_ROLES`) picks up any per-role model and
    temperature overrides; omitting it builds the base model, so pre-tiering
    callers behave exactly as before. Temperature is always passed explicitly
    rather than left unset — an unset temperature means "whatever this
    provider/model defaults to today", which is not a property a
    reproducibility claim can rest on.
    """
    if config.provider not in _KNOWN_PROVIDERS:
        raise ValueError(
            f"unknown LLM provider {config.provider!r}; expected one of {sorted(_KNOWN_PROVIDERS)}"
        )
    model_id = config.model_name(role)
    temperature = config.temperature_for(role)
    if config.provider == "bedrock":
        return build_bedrock_model(
            model_id=model_id,
            profile_name=config.aws_profile,
            region_name=config.aws_region,
            temperature=temperature,
        )
    if config.provider == "openai":
        from langchain_openai import ChatOpenAI

        if config.api_key:
            return ChatOpenAI(model=model_id, api_key=config.api_key, temperature=temperature)
        return ChatOpenAI(model=model_id, temperature=temperature)
    if config.provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if config.api_key:
            return ChatGoogleGenerativeAI(
                model=model_id, google_api_key=config.api_key, temperature=temperature
            )
        return ChatGoogleGenerativeAI(model=model_id, temperature=temperature)
    # config.provider == "test"
    from .test_provider import AutoStructuredFakeChatModel

    return AutoStructuredFakeChatModel()


def build_bedrock_model(
    *,
    model_id: str = BEDROCK_DEFAULT_MODEL,
    profile_name: str = BEDROCK_DEFAULT_PROFILE,
    region_name: str = BEDROCK_DEFAULT_REGION,
    temperature: float = DEFAULT_TEMPERATURE,
) -> BaseChatModel:
    """Build a Bedrock Converse chat model from an explicit boto3 session/profile.

    ``temperature`` is passed through to ``ChatBedrockConverse``, which drops
    it automatically for models that no longer accept the parameter — so
    pinning it here is safe across model generations.
    """
    import boto3
    from langchain_aws import ChatBedrockConverse

    session = boto3.Session(profile_name=profile_name, region_name=region_name)
    client = session.client("bedrock-runtime")
    return ChatBedrockConverse(model_id=model_id, client=client, temperature=temperature)
