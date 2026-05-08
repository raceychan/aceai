"""OpenAI Codex provider backed by the ChatGPT Codex Responses endpoint."""

import json
import os
from pathlib import Path
from typing import Any, AsyncGenerator

from openai import AsyncOpenAI

from aceai.llm.errors import AceAIConfigurationError
from aceai.llm.models import LLMInput, LLMResponse, LLMSegment, LLMStreamEvent
from aceai.llm.openai import OpenAI, OpenAIMeta, OpenAIPayload


OPENAI_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
OPENAI_CODEX_PROVIDER_NAME = "codex"
CODEX_CLI_AUTH_SENTINEL = "codex-cli"
OPENAI_CODEX_DEFAULT_INSTRUCTIONS = "You are AceAI, a concise coding agent."


class OpenAICodex(OpenAI):
    """OpenAI Codex provider using ChatGPT/Codex OAuth credentials."""

    def __init__(
        self,
        *,
        api_key: str,
        default_meta: OpenAIMeta,
        base_url: str = OPENAI_CODEX_BASE_URL,
    ):
        super().__init__(
            client=AsyncOpenAI(
                api_key=_resolve_codex_access_token(api_key),
                base_url=base_url,
            ),
            default_meta=default_meta,
            provider_name=OPENAI_CODEX_PROVIDER_NAME,
        )

    def _build_response_kwargs(self, payload: OpenAIPayload) -> dict[str, Any]:
        kwargs = super()._build_response_kwargs(payload)
        kwargs["instructions"] = OPENAI_CODEX_DEFAULT_INSTRUCTIONS
        kwargs["store"] = False
        if "max_output_tokens" in kwargs:
            del kwargs["max_output_tokens"]
        return kwargs

    async def stream(self, request: LLMInput) -> AsyncGenerator[LLMStreamEvent, None]:
        text = ""
        async for event in super().stream(request):
            if event.event_type == "response.output_text.delta":
                if isinstance(event.text_delta, str):
                    text += event.text_delta
            elif event.event_type == "response.completed":
                response = event.response
                if (
                    isinstance(response, LLMResponse)
                    and response.text == ""
                    and text != ""
                ):
                    segments = [LLMSegment(type="text", content=text)]
                    segments.extend(response.segments)
                    patched_response = LLMResponse(
                        id=response.id,
                        model=response.model,
                        text=text,
                        tool_calls=response.tool_calls,
                        usage=response.usage,
                        segments=segments,
                        provider_meta=response.provider_meta,
                        status=response.status,
                        reasoning=response.reasoning,
                        reasoning_content=response.reasoning_content,
                    )
                    event = LLMStreamEvent(
                        event_type="response.completed",
                        response=patched_response,
                        segments=patched_response.segments,
                        provider_meta=patched_response.provider_meta,
                    )
            yield event


def _resolve_codex_access_token(api_key: str) -> str:
    if api_key == CODEX_CLI_AUTH_SENTINEL:
        return _read_codex_cli_access_token()
    if api_key == "":
        raise AceAIConfigurationError("OpenAI Codex access token is required")
    return api_key


def _read_codex_cli_access_token() -> str:
    auth_path = _codex_auth_path()
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


def _codex_auth_path() -> Path:
    codex_home = os.environ.get("CODEX_HOME")
    if codex_home is None:
        return Path.home() / ".codex" / "auth.json"
    if codex_home == "":
        raise AceAIConfigurationError("CODEX_HOME must not be empty")
    return Path(codex_home).expanduser() / "auth.json"
