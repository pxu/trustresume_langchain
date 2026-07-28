"""Unit tests for the provider-agnostic LLM factory.

Verifies config parsing and provider dispatch without contacting any provider —
only the offline ``test`` model is actually built (it needs no keys). The other
providers are checked for correct model-name defaulting and error handling.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trustresume.api.model_factory import (
    BEDROCK_DEFAULT_MODEL,
    LLMConfig,
    build_model,
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
