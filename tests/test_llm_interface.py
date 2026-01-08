import io
from typing import AsyncIterator

import pytest

from aceai.errors import AceAIImplementationError
from aceai.llm.models import (
    LLMMessage,
    LLMProviderBase,
    LLMRequest,
    LLMResponse,
    LLMStreamEvent,
    LLMToolCall,
    LLMToolCallMessage,
)


class PassthroughProvider(LLMProviderBase):
    async def complete(self, request: LLMRequest, *, trace_ctx=None) -> LLMResponse:
        return await super().complete(request, trace_ctx=trace_ctx)

    def stream(self, request: LLMRequest, *, trace_ctx=None) -> AsyncIterator[LLMStreamEvent]:
        return super().stream(request, trace_ctx=trace_ctx)

    async def stt(
        self,
        filename: str,
        file,
        *,
        model: str,
        prompt: str | None = None,
        trace_ctx=None,
    ) -> str:
        return await super().stt(
            filename,
            file,
            model=model,
            prompt=prompt,
            trace_ctx=trace_ctx,
        )


def test_llm_message_inplace_merge_with_structured_message() -> None:
    base = LLMMessage.build("assistant", "Hi")
    other = LLMMessage.build("assistant", " again")

    base |= other

    assert [part["data"] for part in base.content] == ["Hi", " again"]


def test_llm_message_inplace_merge_requires_matching_roles() -> None:
    base = LLMMessage.build("user", "Hi")
    other = LLMMessage.build("assistant", "Nope")

    with pytest.raises(ValueError):
        base |= other


def test_llm_message_asdict_includes_optional_fields() -> None:
    tool_call = LLMToolCall(name="calc", arguments="{}", call_id="call-1")
    message = LLMToolCallMessage.from_content("call tool", tool_calls=[tool_call])

    as_dict = message.asdict()

    assert as_dict["role"] == "assistant"
    assert as_dict["content"][0]["data"] == "call tool"
    assert as_dict["tool_calls"][0]["name"] == "calc"


@pytest.mark.anyio
async def test_llm_provider_base_async_methods_raise_not_implemented() -> None:
    provider = PassthroughProvider()
    request: LLMRequest = {"messages": [LLMMessage(role="system", content="hi")]}

    with pytest.raises(AceAIImplementationError):
        await provider.complete(request)

    with pytest.raises(NotImplementedError):
        await provider.stt("file.wav", io.BytesIO(b"_"), model="whisper", prompt=None)


def test_llm_provider_base_sync_interfaces_raise() -> None:
    provider = PassthroughProvider()
    request: LLMRequest = {"messages": [LLMMessage(role="system", content="hi")]}

    with pytest.raises(AceAIImplementationError):
        provider.stream(request)
