import pytest

from app.llm import (
    DEFAULT_MODELS,
    DEFAULT_OLLAMA_BASE_URL,
    LLMConfig,
    LLMConfigError,
    build_llm,
    load_config,
)
from app.triage import message_text

ENV_VARS = (
    "TRIAGE_LLM_PROVIDER",
    "TRIAGE_LLM_MODEL",
    "OLLAMA_BASE_URL",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_default_config_is_local_ollama() -> None:
    config = load_config()

    assert config == LLMConfig(
        provider="ollama",
        model=DEFAULT_MODELS["ollama"],
        ollama_base_url=DEFAULT_OLLAMA_BASE_URL,
    )


def test_gemini_config_uses_gemini_default_model(monkeypatch) -> None:
    monkeypatch.setenv("TRIAGE_LLM_PROVIDER", "Gemini")

    config = load_config()

    assert config.provider == "gemini"
    assert config.model == DEFAULT_MODELS["gemini"]


def test_model_override(monkeypatch) -> None:
    monkeypatch.setenv("TRIAGE_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("TRIAGE_LLM_MODEL", "gemini-custom")

    assert load_config().model == "gemini-custom"


def test_unknown_provider_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("TRIAGE_LLM_PROVIDER", "openai")

    with pytest.raises(LLMConfigError, match="Unknown TRIAGE_LLM_PROVIDER"):
        load_config()


def test_gemini_without_api_key_fails_clearly() -> None:
    with pytest.raises(LLMConfigError, match="GOOGLE_API_KEY"):
        build_llm(LLMConfig(provider="gemini", model="gemini-x"))


def test_gemini_builds_json_model_with_key(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    llm = build_llm(LLMConfig(provider="gemini", model="gemini-x"))

    assert type(llm).__name__ == "ChatGoogleGenerativeAI"
    assert llm.model.endswith("gemini-x")
    assert llm.response_mime_type == "application/json"


def test_ollama_builds_json_model() -> None:
    llm = build_llm(LLMConfig(provider="ollama", model="qwen3:8b"))

    assert type(llm).__name__ == "ChatOllama"
    assert llm.format == "json"


def test_message_text_accepts_string_and_parts() -> None:
    assert message_text('{"a": 1}') == '{"a": 1}'
    assert (
        message_text([{"type": "text", "text": '{"a":'}, {"type": "text", "text": " 1}"}])
        == '{"a": 1}'
    )


def test_message_text_rejects_non_text() -> None:
    with pytest.raises(TypeError):
        message_text([{"type": "image", "url": "x"}])
