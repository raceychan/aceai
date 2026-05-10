"""Provider credential rules for the AceAI app layer."""

import json
import os
from pathlib import Path

from aceai.agent.provider_catalog import api_key_env, auth_mode
from aceai.llm.errors import AceAIConfigurationError
from aceai.llm.anthropic import ANTHROPIC_OAUTH_PROVIDER_NAME
from aceai.llm.openai_codex import OPENAI_CODEX_PROVIDER_NAME

CODEX_CLI_AUTH_SENTINEL = "codex-cli"


def default_api_key_for_provider(provider: str) -> str:
    if provider == OPENAI_CODEX_PROVIDER_NAME:
        return CODEX_CLI_AUTH_SENTINEL
    if provider == ANTHROPIC_OAUTH_PROVIDER_NAME:
        auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
        if auth_token is not None:
            return auth_token
    return ""


def api_key_placeholder(provider: str) -> str:
    if provider == OPENAI_CODEX_PROVIDER_NAME:
        return CODEX_CLI_AUTH_SENTINEL
    if provider == ANTHROPIC_OAUTH_PROVIDER_NAME:
        return "CLAUDE_CODE_OAUTH_TOKEN or ANTHROPIC_AUTH_TOKEN"
    return api_key_env(provider)


def provider_uses_api_key(provider: str) -> bool:
    return auth_mode(provider) == "api_key"


def provider_uses_subscription(provider: str) -> bool:
    return auth_mode(provider) == "subscription"


def resolve_provider_api_key(provider: str, api_key: str) -> str:
    if provider == OPENAI_CODEX_PROVIDER_NAME:
        return resolve_codex_access_token(api_key)
    if provider == ANTHROPIC_OAUTH_PROVIDER_NAME:
        return resolve_anthropic_oauth_token(api_key)
    if api_key == "":
        raise AceAIConfigurationError(f"{provider} API key is required")
    return api_key


def resolve_anthropic_oauth_token(api_key: str) -> str:
    if api_key != "":
        return api_key
    auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if auth_token is not None and auth_token != "":
        return auth_token
    raise AceAIConfigurationError("Anthropic OAuth bearer token is required")


def resolve_codex_access_token(api_key: str) -> str:
    if api_key == CODEX_CLI_AUTH_SENTINEL:
        return read_codex_cli_access_token()
    if api_key == "":
        raise AceAIConfigurationError("OpenAI Codex access token is required")
    return api_key


def read_codex_cli_access_token() -> str:
    auth_path = codex_auth_path()
    if not auth_path.is_file():
        raise AceAIConfigurationError(
            "Codex CLI auth is missing. Run `codex login` or configure an "
            "OpenAI Codex access token directly."
        )
    data = json.loads(auth_path.read_text(encoding="utf-8"))
    tokens = data["tokens"]
    access_token = tokens["access_token"]
    if type(access_token) is not str:
        raise TypeError("Codex CLI access_token must be str")
    if access_token == "":
        raise AceAIConfigurationError("Codex CLI access_token is empty")
    return access_token


def codex_auth_path() -> Path:
    codex_home = os.environ.get("CODEX_HOME")
    if codex_home is None:
        return Path.home() / ".codex" / "auth.json"
    if codex_home == "":
        raise AceAIConfigurationError("CODEX_HOME must not be empty")
    return Path(codex_home).expanduser() / "auth.json"
