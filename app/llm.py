"""Model configuration: choose the LLM provider and model from environment variables.

Environment variables:
    TRIAGE_LLM_PROVIDER  "ollama" (default) or "gemini"
    TRIAGE_LLM_MODEL     model name; defaults depend on the provider
    OLLAMA_BASE_URL      Ollama server URL (default http://127.0.0.1:11434)
    GOOGLE_API_KEY       Gemini API key (GEMINI_API_KEY is also accepted)
"""

import os
from dataclasses import dataclass
from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel

PROVIDERS = ("ollama", "gemini")

DEFAULT_MODELS = {
    "ollama": "qwen3:8b",
    # Flash-Lite has a much larger free-tier daily quota than Flash, which
    # matters when an eval run makes dozens of requests.
    "gemini": "gemini-3.5-flash-lite",
}

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"


class LLMConfigError(ValueError):
    """Raised when the model configuration is invalid or incomplete."""


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL


def load_config() -> LLMConfig:
    """Read the model configuration from environment variables."""
    provider = os.environ.get("TRIAGE_LLM_PROVIDER", "ollama").strip().lower()
    if provider not in PROVIDERS:
        raise LLMConfigError(
            f"Unknown TRIAGE_LLM_PROVIDER {provider!r}; expected one of {PROVIDERS}."
        )

    model = os.environ.get("TRIAGE_LLM_MODEL", "").strip() or DEFAULT_MODELS[provider]
    base_url = os.environ.get("OLLAMA_BASE_URL", "").strip() or DEFAULT_OLLAMA_BASE_URL
    return LLMConfig(provider=provider, model=model, ollama_base_url=base_url)


def gemini_api_key() -> str | None:
    return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")


def build_llm(config: LLMConfig) -> BaseChatModel:
    """Create a chat model that is asked to return JSON."""
    if config.provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config.model,
            temperature=0,
            base_url=config.ollama_base_url,
            format="json",
        )

    api_key = gemini_api_key()
    if not api_key:
        raise LLMConfigError(
            "TRIAGE_LLM_PROVIDER is 'gemini' but no API key was found. "
            "Set GOOGLE_API_KEY (or GEMINI_API_KEY)."
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    # Temperature is left at the model default: Google's Gemini 3 guide
    # recommends against lowering it, as that can degrade output.
    return ChatGoogleGenerativeAI(
        model=config.model,
        google_api_key=api_key,
        response_mime_type="application/json",
        # Retries with backoff absorb free-tier rate-limit (429) errors.
        max_retries=6,
    )


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """Return the configured chat model, created once per process."""
    return build_llm(load_config())
