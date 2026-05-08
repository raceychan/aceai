"""Provider credential rules for the AceAI app layer."""

from aceai.agent.provider_catalog import api_key_env, auth_mode
from aceai.llm.openai_codex import (
    CODEX_CLI_AUTH_SENTINEL,
    OPENAI_CODEX_PROVIDER_NAME,
)


def default_api_key_for_provider(provider: str) -> str:
    if provider == OPENAI_CODEX_PROVIDER_NAME:
        return CODEX_CLI_AUTH_SENTINEL
    return ""


def api_key_placeholder(provider: str) -> str:
    if provider == OPENAI_CODEX_PROVIDER_NAME:
        return CODEX_CLI_AUTH_SENTINEL
    return api_key_env(provider)


def provider_uses_api_key(provider: str) -> bool:
    return auth_mode(provider) == "api_key"


def provider_uses_subscription(provider: str) -> bool:
    return auth_mode(provider) == "subscription"
