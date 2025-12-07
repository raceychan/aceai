import io
from typing import AsyncIterator

import pytest

from aceai.errors import AceAIImplementationError, AceAIValidationError
from aceai.llm.models import (
    LLMMessage,
    LLMProviderBase,
    LLMRequest,
    LLMResponse,
    LLMStreamEvent,
    LLMToolCall,
    LLMToolCallMessage,
    LLMToolUseMessage,
)


class PassthroughProvider(LLMProviderBase):
    async def complete(self, request: LLMRequest) -> LLMResponse:
        return await super().complete(request)

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        return super().stream(request)

    @property
    def default_model(self) -> str:
        return super().default_model

    @property
    def default_stream_model(self) -> str:
        return super().default_stream_model

    async def stt(self, filename: str, file, *, model: str) -> str:
        return await super().stt(filename, file, model=model)


def test_llm_message_inplace_merge_with_string() -> None:
    message = LLMMessage(role="user", content="Hello")

    message |= " there"

    assert message.content == "Hello there"


def test_llm_message_inplace_merge_with_structured_message() -> None:
    base = LLMMessage(role="assistant", content="Hi")
    other = LLMMessage(role="assistant", content=" again")

    base |= other

    assert base.content == "Hi again"


def test_llm_message_inplace_merge_requires_matching_roles() -> None:
    base = LLMMessage(role="user", content="Hi")
    other = LLMMessage(role="assistant", content="Nope")

    with pytest.raises(AceAIValidationError):
        base |= other


def test_llm_message_asdict_includes_optional_fields() -> None:
    tool_call = LLMToolCall(name="calc", arguments="{}", call_id="call-1")
    message = LLMToolCallMessage(
        role="assistant",
        content="call tool",
        name="tool-call",
        tool_calls=[tool_call],
    )

    as_dict = message.asdict()

    assert as_dict["role"] == "assistant"
    assert as_dict["name"] == "tool-call"
    assert as_dict["tool_calls"][0]["name"] == "calc"


def test_llm_tool_use_message_asdict_includes_tool_call_id() -> None:
    message = LLMToolUseMessage(role="tool", content="done", call_id="call-42")

    as_dict = message.asdict()

    assert as_dict["tool_call_id"] == "call-42"
    assert as_dict["content"] == "done"


@pytest.mark.anyio
async def test_llm_provider_base_async_methods_raise_not_implemented() -> None:
    provider = PassthroughProvider()
    request: LLMRequest = {"messages": [LLMMessage(role="system", content="hi")]}

    with pytest.raises(AceAIImplementationError):
        await provider.complete(request)

    with pytest.raises(AceAIImplementationError):
        await provider.stt("file.wav", io.BytesIO(b"_"), model="whisper")


def test_llm_provider_base_sync_interfaces_raise() -> None:
    provider = PassthroughProvider()
    request: LLMRequest = {"messages": [LLMMessage(role="system", content="hi")]}

    with pytest.raises(AceAIImplementationError):
        provider.stream(request)

    with pytest.raises(AceAIImplementationError):
        _ = provider.default_model

    with pytest.raises(AceAIImplementationError):
        _ = provider.default_stream_model
