import pytest

from aceai.llm.deepseek import DEEPSEEK_BASE_URL, DeepSeek
from aceai.llm.models import LLMMessage, LLMToolCall, LLMToolCallMessage


class RecordingCompletions:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return ChatCompletionResponse()


class RecordingClient:
    def __init__(self) -> None:
        self.chat = ChatResource()


class ChatResource:
    def __init__(self) -> None:
        self.completions = RecordingCompletions()


class ChatCompletionResponse:
    id = "chatcmpl-test"
    model = "deepseek-v4-flash"
    usage = None
    choices = []

    def __init__(self) -> None:
        self.choices = [ChatChoice()]


class ChatChoice:
    finish_reason = "stop"
    message = None

    def __init__(self) -> None:
        self.message = ChatMessage()


class ChatMessage:
    content = "hello from deepseek"
    tool_calls = None
    model_extra = {"reasoning_content": "thinking about hello"}


class StreamingCompletions:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return StreamingResponse()


class StreamingClient:
    def __init__(self) -> None:
        self.chat = StreamingChatResource()


class StreamingChatResource:
    def __init__(self) -> None:
        self.completions = StreamingCompletions()


class StreamingResponse:
    def __aiter__(self):
        return self._events()

    async def _events(self):
        yield StreamChunk(choices=[StreamChoice(delta=StreamDelta(content="hi"))])
        yield StreamChunk(
            choices=[],
            usage=StreamUsage(
                prompt_tokens=10,
                completion_tokens=3,
                total_tokens=13,
                prompt_tokens_details=StreamPromptTokenDetails(cached_tokens=4),
            ),
        )


class StreamChunk:
    def __init__(self, *, choices, usage=None) -> None:
        self.choices = choices
        self.usage = usage


class StreamChoice:
    def __init__(self, *, delta) -> None:
        self.delta = delta


class StreamDelta:
    tool_calls = None
    model_extra = None

    def __init__(self, *, content=None) -> None:
        self.content = content


class StreamUsage:
    def __init__(
        self,
        *,
        prompt_tokens,
        completion_tokens,
        total_tokens,
        prompt_tokens_details,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.prompt_tokens_details = prompt_tokens_details
        self.prompt_cache_hit_tokens = prompt_tokens_details.cached_tokens
        self.prompt_cache_miss_tokens = (
            prompt_tokens - prompt_tokens_details.cached_tokens
        )


class StreamPromptTokenDetails:
    def __init__(self, *, cached_tokens) -> None:
        self.cached_tokens = cached_tokens


def test_deepseek_provider_uses_openai_compatible_base_url() -> None:
    provider = DeepSeek(
        api_key="test-key",
        default_meta={"model": "deepseek-v4-flash"},
    )

    assert str(provider._client.base_url) == DEEPSEEK_BASE_URL
    assert provider._provider_name == "deepseek"


@pytest.mark.anyio
async def test_deepseek_complete_uses_chat_completions() -> None:
    provider = DeepSeek(
        api_key="test-key",
        default_meta={"model": "deepseek-v4-flash"},
    )
    client = RecordingClient()
    provider._client = client

    response = await provider.complete(
        {
            "messages": [
                LLMMessage.build(role="user", content="hello"),
            ],
        }
    )

    assert client.chat.completions.kwargs == {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": "hello"}],
    }
    assert response.text == "hello from deepseek"
    assert response.reasoning_content == "thinking about hello"
    assert response.provider_meta[0].provider_name == "deepseek"


@pytest.mark.anyio
async def test_deepseek_complete_passes_reasoning_effort() -> None:
    provider = DeepSeek(
        api_key="test-key",
        default_meta={"model": "deepseek-v4-pro"},
    )
    client = RecordingClient()
    provider._client = client

    await provider.complete(
        {
            "messages": [
                LLMMessage.build(role="user", content="hello"),
            ],
            "metadata": {
                "reasoning": {
                    "effort": "max",
                    "summary": "auto",
                }
            },
        }
    )

    assert client.chat.completions.kwargs is not None
    assert client.chat.completions.kwargs["reasoning_effort"] == "max"
    assert client.chat.completions.kwargs["extra_body"] == {
        "thinking": {"type": "enabled"}
    }


def test_deepseek_formats_reasoning_content_for_tool_sub_turn() -> None:
    provider = DeepSeek(
        api_key="test-key",
        default_meta={"model": "deepseek-v4-flash"},
    )
    call = LLMToolCall(
        name="lookup",
        arguments='{"q":"aceai"}',
        call_id="call-1",
    )
    messages = provider._format_messages_for_chat(
        [
            LLMToolCallMessage.from_content(
                content="",
                tool_calls=[call],
                reasoning_content="need to look this up",
            )
        ]
    )

    assert messages == [
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "need to look this up",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"q":"aceai"}',
                    },
                }
            ],
        }
    ]


@pytest.mark.anyio
async def test_deepseek_stream_requests_and_maps_usage() -> None:
    provider = DeepSeek(
        api_key="test-key",
        default_meta={"model": "deepseek-v4-flash"},
    )
    client = StreamingClient()
    provider._client = client

    events = [
        event
        async for event in provider.stream(
            {
                "messages": [
                    LLMMessage.build(role="user", content="hello"),
                ],
            }
        )
    ]

    assert client.chat.completions.kwargs is not None
    assert client.chat.completions.kwargs["stream"] is True
    assert client.chat.completions.kwargs["stream_options"] == {"include_usage": True}
    final_response = events[-1].response
    assert final_response.usage.input_tokens == 10
    assert final_response.usage.cached_input_tokens == 4
    assert final_response.usage.cache_miss_input_tokens == 6
    assert final_response.usage.input_cache_hit_rate == 0.4
    assert final_response.usage.output_tokens == 3
    assert final_response.usage.total_tokens == 13
