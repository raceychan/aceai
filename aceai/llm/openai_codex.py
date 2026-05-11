"""OpenAI Codex provider backed by the ChatGPT Codex Responses endpoint."""

from typing import Any, AsyncGenerator

from openai import AsyncOpenAI

from aceai.llm.errors import AceAIConfigurationError, AceAIRuntimeError
from aceai.llm.models import LLMInput, LLMResponse, LLMSegment, LLMStreamEvent
from aceai.llm.openai import OpenAI, OpenAIMeta, OpenAIPayload


OPENAI_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
OPENAI_CODEX_PROVIDER_NAME = "codex"
OPENAI_CODEX_CLIENT_TIMEOUT_SECONDS = 300.0


class OpenAICodex(OpenAI):
    """OpenAI Codex provider using a resolved ChatGPT/Codex access token."""

    def __init__(
        self,
        *,
        api_key: str,
        default_meta: OpenAIMeta,
        base_url: str = OPENAI_CODEX_BASE_URL,
        instructions: str = "",
    ):
        if api_key == "":
            raise AceAIConfigurationError("OpenAI Codex access token is required")
        super().__init__(
            client=AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=OPENAI_CODEX_CLIENT_TIMEOUT_SECONDS,
            ),
            default_meta=default_meta,
            provider_name=OPENAI_CODEX_PROVIDER_NAME,
        )
        self._instructions = instructions

    def _build_response_kwargs(self, payload: OpenAIPayload) -> dict[str, Any]:
        kwargs = super()._build_response_kwargs(payload)
        if self._instructions != "":
            kwargs["instructions"] = self._instructions
        kwargs["store"] = False
        if "max_output_tokens" in kwargs:
            del kwargs["max_output_tokens"]
        return kwargs

    async def complete(self, request: LLMInput) -> LLMResponse:
        response: LLMResponse | None = None
        async for event in self.stream(request):
            if event.event_type == "response.completed" and event.response is not None:
                response = event.response
        if response is None:
            raise AceAIRuntimeError("OpenAI Codex stream did not complete")
        return response

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
