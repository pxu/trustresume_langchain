"""Unit tests for the provider-agnostic LLM factory.

Verifies config parsing and provider dispatch without contacting any provider —
only the offline ``test`` model is actually built (it needs no keys). The other
providers are checked for correct model-name defaulting and error handling.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trustresume.api.model_factory import (
    BEDROCK_DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    LLMConfig,
    RoleConfig,
    build_model,
    load_quality_gate,
)
from trustresume.api.test_provider import AutoStructuredFakeChatModel

# Env vars that influence config loading — cleared before each test so the
# real environment / shell doesn't leak into these assertions.
_CONFIG_ENV_VARS = (
    "TRUSTRESUME_LLM_PROVIDER",
    "TRUSTRESUME_LLM",
    "TRUSTRESUME_LLM_MODEL",
    "TRUSTRESUME_LLM_API_KEY",
    "TRUSTRESUME_AWS_PROFILE",
    "TRUSTRESUME_AWS_REGION",
    "TRUSTRESUME_LLM_CONFIG",
    "TRUSTRESUME_LLM_TEMPERATURE",
    "TRUSTRESUME_LLM_EXTRACTION_MODEL",
    "TRUSTRESUME_LLM_WRITER_MODEL",
    "TRUSTRESUME_LLM_VERIFIER_MODEL",
    "TRUSTRESUME_LLM_EXTRACTION_TEMPERATURE",
    "TRUSTRESUME_LLM_WRITER_TEMPERATURE",
    "TRUSTRESUME_LLM_VERIFIER_TEMPERATURE",
    "TRUSTRESUME_QUALITY_GATE_CONFIG",
    "TRUSTRESUME_QUALITY_MAX_ITERATIONS",
)


@pytest.fixture(autouse=True)
def _clear_config_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _CONFIG_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _write(path: Path, **values: object) -> Path:
    path.write_text(json.dumps(values), encoding="utf-8")
    return path


def test_load_missingFile_defaultsToBedrock(tmp_path: Path) -> None:
    config = LLMConfig.load(tmp_path / "nope.json")
    assert config.provider == "bedrock"
    assert config.model_name() == BEDROCK_DEFAULT_MODEL


def test_load_readsProviderModelAndApiKeyFromJson(tmp_path: Path) -> None:
    path = _write(tmp_path / "llm.json", provider="openai", model="gpt-4o-mini", api_key="sk-abc")
    config = LLMConfig.load(path)
    assert config.provider == "openai"
    assert config.model_name() == "gpt-4o-mini"
    assert config.api_key == "sk-abc"


def test_load_ignoresCommentKeys(tmp_path: Path) -> None:
    path = tmp_path / "llm.json"
    path.write_text(json.dumps({"$comment": "docs", "provider": "test"}), encoding="utf-8")
    assert LLMConfig.load(path).provider == "test"


def test_load_localOverlayWins(tmp_path: Path) -> None:
    _write(tmp_path / "llm.json", provider="bedrock", model=None)
    _write(tmp_path / "llm.local.json", provider="openai", api_key="sk-local")
    config = LLMConfig.load(tmp_path / "llm.json")
    assert config.provider == "openai"  # overlay overrides base
    assert config.api_key == "sk-local"


def test_load_envOverridesJson(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write(tmp_path / "llm.json", provider="openai", api_key="sk-file")
    monkeypatch.setenv("TRUSTRESUME_LLM_PROVIDER", "google")
    monkeypatch.setenv("TRUSTRESUME_LLM_API_KEY", "g-env")
    config = LLMConfig.load(path)
    assert config.provider == "google"  # env beats file
    assert config.api_key == "g-env"


def test_load_honorsLegacyProviderVar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRUSTRESUME_LLM", "test")
    assert LLMConfig.load(tmp_path / "nope.json").provider == "test"


def test_load_providerLowercased(tmp_path: Path) -> None:
    path = _write(tmp_path / "llm.json", provider="OpenAI")
    assert LLMConfig.load(path).provider == "openai"


def test_modelName_perProviderDefaults() -> None:
    assert LLMConfig(provider="openai").model_name() == "gpt-4o"
    assert LLMConfig(provider="google").model_name() == "gemini-1.5-pro"
    assert LLMConfig(provider="bedrock").model_name() == BEDROCK_DEFAULT_MODEL


def test_buildModel_testProvider_returnsFakeChatModel() -> None:
    model = build_model(LLMConfig(provider="test"))
    assert isinstance(model, AutoStructuredFakeChatModel)


def test_buildModel_unknownProvider_raises() -> None:
    with pytest.raises(ValueError, match="unknown LLM provider"):
        build_model(LLMConfig(provider="llama"))


def test_buildModel_openai_withConfiguredKey(monkeypatch: pytest.MonkeyPatch) -> None:
    # No OPENAI_API_KEY in env — the configured api_key must satisfy the provider.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    model = build_model(LLMConfig(provider="openai", model="gpt-4o", api_key="sk-not-real"))
    assert model.model_name == "gpt-4o"  # type: ignore[attr-defined]


def test_buildModel_openai_fallsBackToEnvKey(monkeypatch: pytest.MonkeyPatch) -> None:
    # No configured key; the SDK reads OPENAI_API_KEY from the environment.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    model = build_model(LLMConfig(provider="openai", model="gpt-4o"))
    assert model.model_name == "gpt-4o"  # type: ignore[attr-defined]


def test_buildModel_google_withConfiguredKey(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    model = build_model(LLMConfig(provider="google", model="gemini-1.5-flash", api_key="g-x"))
    assert model.model == "gemini-1.5-flash"  # type: ignore[attr-defined]


def test_buildModel_google_fallsBackToEnvKey(monkeypatch: pytest.MonkeyPatch) -> None:
    # No configured key; the SDK reads GOOGLE_API_KEY from the environment.
    monkeypatch.setenv("GOOGLE_API_KEY", "g-env-not-real")
    model = build_model(LLMConfig(provider="google", model="gemini-1.5-flash"))
    assert model.model == "gemini-1.5-flash"  # type: ignore[attr-defined]


# --- temperature + per-role model tiering ---------------------------------


def test_temperature_defaultsToZeroForReproducibleScoring() -> None:
    """Extraction/classification must not vary run to run by default."""
    assert LLMConfig().temperature == DEFAULT_TEMPERATURE == 0.0
    assert LLMConfig().temperature_for("writer") == 0.0


def test_load_readsTemperatureAndRoleOverridesFromJson(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "llm.json",
        provider="openai",
        model="gpt-4o",
        temperature=0.1,
        roles={
            "extraction": {"model": "gpt-4o-mini"},
            "writer": {"temperature": 0.7},
        },
    )
    config = LLMConfig.load(path)

    assert config.temperature == 0.1
    assert config.model_name("extraction") == "gpt-4o-mini"  # role model override
    assert config.temperature_for("extraction") == 0.1  # inherits base temperature
    assert config.model_name("writer") == "gpt-4o"  # inherits base model
    assert config.temperature_for("writer") == 0.7  # role temperature override
    assert config.model_name("verifier") == "gpt-4o"  # no entry: all inherited


def test_load_roleEnvVarsOverrideFile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    roles = {"writer": {"model": "from-file"}}
    path = _write(tmp_path / "llm.json", provider="openai", roles=roles)
    monkeypatch.setenv("TRUSTRESUME_LLM_WRITER_MODEL", "from-env")
    assert LLMConfig.load(path).model_name("writer") == "from-env"


def test_load_unparseableTemperature_fallsBackToDefaultNotCrash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A typo in config must not take the whole app down at startup."""
    monkeypatch.setenv("TRUSTRESUME_LLM_TEMPERATURE", "warm")
    assert LLMConfig.load(tmp_path / "nope.json").temperature == DEFAULT_TEMPERATURE


def test_load_malformedRolesSection_ignoredNotCrash(tmp_path: Path) -> None:
    path = _write(tmp_path / "llm.json", provider="openai", roles="not-a-mapping")
    assert LLMConfig.load(path).roles == {}


def test_modelName_unknownRole_fallsBackToBaseModel() -> None:
    config = LLMConfig(provider="openai", model="gpt-4o", roles={"writer": RoleConfig(model="x")})
    assert config.model_name("reranker") == "gpt-4o"


def test_buildModel_openai_passesResolvedRoleModelAndTemperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    config = LLMConfig(
        provider="openai",
        model="gpt-4o",
        temperature=0.0,
        roles={"writer": RoleConfig(model="gpt-4o-mini", temperature=0.7)},
    )
    writer = build_model(config, role="writer")
    verifier = build_model(config, role="verifier")

    assert writer.model_name == "gpt-4o-mini"  # type: ignore[attr-defined]
    assert writer.temperature == 0.7  # type: ignore[attr-defined]
    assert verifier.model_name == "gpt-4o"  # type: ignore[attr-defined]
    assert verifier.temperature == 0.0  # type: ignore[attr-defined]


@patch("boto3.Session")
def test_buildModel_bedrock_pinsTemperatureExplicitly(session_cls: MagicMock) -> None:
    """Unset temperature would mean 'whatever the provider defaults to today'."""
    model = build_model(LLMConfig(provider="bedrock", model="claude-x"), role="extraction")
    assert model.temperature == 0.0  # type: ignore[attr-defined]


@patch("boto3.Session")
def test_buildModel_bedrock_dispatchesToBuildBedrockModel(session_cls: MagicMock) -> None:
    # boto3.Session mocked out — this test is about dispatch/argument wiring,
    # not a real AWS round trip (constructing ChatBedrockConverse doesn't call
    # AWS itself, but resolving a profile does read local AWS config, which
    # varies by machine/CI — mocking keeps this deterministic). Note:
    # ChatBedrockConverse itself opens a second internal Session/client (to
    # probe streaming support), so we assert on our own call, not "called once".
    session = session_cls.return_value
    config = LLMConfig(
        provider="bedrock", model="claude-x", aws_profile="myprofile", aws_region="eu-west-1"
    )
    model = build_model(config)

    assert session_cls.call_args_list[0].kwargs == {
        "profile_name": "myprofile",
        "region_name": "eu-west-1",
    }
    session.client.assert_any_call("bedrock-runtime")
    assert model.model_id == "claude-x"  # type: ignore[attr-defined]


@patch("boto3.Session")
def test_buildBedrockModel_usesModuleDefaultsWhenArgsOmitted(session_cls: MagicMock) -> None:
    from trustresume.api.model_factory import (
        BEDROCK_DEFAULT_PROFILE,
        BEDROCK_DEFAULT_REGION,
        build_bedrock_model,
    )

    model = build_bedrock_model()

    assert session_cls.call_args_list[0].kwargs == {
        "profile_name": BEDROCK_DEFAULT_PROFILE,
        "region_name": BEDROCK_DEFAULT_REGION,
    }
    assert model.model_id == BEDROCK_DEFAULT_MODEL  # type: ignore[attr-defined]


# --- load_quality_gate (mirrors LLMConfig.load's precedence) ---------------


def test_loadQualityGate_missingFile_usesPydanticDefault(tmp_path: Path) -> None:
    gate = load_quality_gate(tmp_path / "nope.json")
    assert gate.max_iterations == 3  # QualityGate's own Field(default=3, ...)


def test_loadQualityGate_readsMaxIterationsFromJson(tmp_path: Path) -> None:
    path = _write(tmp_path / "quality_gate.json", max_iterations=1)
    assert load_quality_gate(path).max_iterations == 1


def test_loadQualityGate_ignoresCommentKeys(tmp_path: Path) -> None:
    path = tmp_path / "quality_gate.json"
    path.write_text(json.dumps({"$comment": "docs", "max_iterations": 0}), encoding="utf-8")
    assert load_quality_gate(path).max_iterations == 0


def test_loadQualityGate_fileWithoutMaxIterationsKey_usesPydanticDefault(
    tmp_path: Path,
) -> None:
    path = tmp_path / "quality_gate.json"
    path.write_text(json.dumps({"$comment": "docs only, no max_iterations"}), encoding="utf-8")
    assert load_quality_gate(path).max_iterations == 3


def test_loadQualityGate_localOverlayWins(tmp_path: Path) -> None:
    _write(tmp_path / "quality_gate.json", max_iterations=2)
    _write(tmp_path / "quality_gate.local.json", max_iterations=5)
    assert load_quality_gate(tmp_path / "quality_gate.json").max_iterations == 5


def test_loadQualityGate_envOverridesJson(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write(tmp_path / "quality_gate.json", max_iterations=2)
    monkeypatch.setenv("TRUSTRESUME_QUALITY_MAX_ITERATIONS", "7")
    assert load_quality_gate(path).max_iterations == 7


def test_loadQualityGate_unparseableEnvValue_fallsBackToPydanticDefault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write(tmp_path / "quality_gate.json", max_iterations=2)
    monkeypatch.setenv("TRUSTRESUME_QUALITY_MAX_ITERATIONS", "not-a-number")
    # A typo'd env var must not crash startup. Falls all the way to the
    # Pydantic default rather than back down to the file value — matching
    # LLMConfig.load's own precedent (once the env var is present, it wins or
    # falls back to the ultimate default; the file isn't consulted again).
    assert load_quality_gate(path).max_iterations == 3


def test_loadQualityGate_unparseableFileValue_fallsBackToPydanticDefault(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path / "quality_gate.json", max_iterations="lots")
    assert load_quality_gate(path).max_iterations == 3


def test_loadQualityGate_outOfRangeValue_fallsBackToPydanticDefault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # max_iterations has a ge=0 floor; a negative value is parseable as an int
    # but still invalid — this must be caught too, not just non-numeric input.
    monkeypatch.setenv("TRUSTRESUME_QUALITY_MAX_ITERATIONS", "-1")
    assert load_quality_gate(tmp_path / "nope.json").max_iterations == 3


def test_loadQualityGate_customPathEnvVar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write(tmp_path / "custom_gate.json", max_iterations=4)
    monkeypatch.setenv("TRUSTRESUME_QUALITY_GATE_CONFIG", str(path))
    assert load_quality_gate().max_iterations == 4
