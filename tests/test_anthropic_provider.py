import httpx
import pytest

from aceai.core.tools import tool
from aceai.core.tools._tool_sig import Annotated, spec
from aceai.llm.anthropic import (
    ANTHROPIC_API_URL,
    ANTHROPIC_API_VERSION,
    ANTHROPIC_OAUTH_PROVIDER_NAME,
    Anthropic,
)
from agent_core.provider_auth import (
    api_key_placeholder,
    default_api_key_for_provider,
    resolve_provider_api_key,
)
from aceai.llm.errors import AceAIConfigurationError
from aceai.llm.models import (
    LLMHostedToolSpec,
    LLMMessage,
    LLMResponseFormat,
    LLMToolCall,
    LLMToolCallMessage,
    LLMToolUseMessage,
)


class RecordingAsyncClient:
    def __init__(self) -> None:
        self.post_calls: list[dict[str, object]] = []
        self.stream_calls: list[dict[str, object]] = []
        self.response_payload: dict[str, object] = {
            "id": "msg-test",
            "model": "claude-sonnet-4-20250514",
            "content": [{"type": "text", "text": "hello from claude"}],
            "usage": {
                "input_tokens": 10,
                "cache_read_input_tokens": 2,
                "output_tokens": 3,
            },
            "stop_reason": "end_turn",
        }
        self.stream_lines: list[str] = []

    async def post(self, url: str, *, headers, json):
        self.post_calls.append(
            {
                "url": url,
                "headers": headers,
                "json": json,
            }
        )
        return httpx.Response(
            200,
            json=self.response_payload,
            request=httpx.Request("POST", url),
        )

    def stream(self, method: str, url: str, *, headers, json):
        self.stream_calls.append(
            {
                "method": method,
                "url": url,
                "headers": headers,
                "json": json,
            }
        )
        return StreamingResponse(self.stream_lines)


class StreamingResponse:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self):
        for line in self._lines:
            yield line


@pytest.fixture
def anthropic_echo_spec():
    @tool
    def echo(
        message: Annotated[str, spec(description="Echo message")],
    ) -> str:
        """Echo message."""
        return message

    return echo.tool_spec


@pytest.mark.anyio
async def test_anthropic_complete_uses_messages_api(anthropic_echo_spec) -> None:
    client = RecordingAsyncClient()
    provider = Anthropic(
        api_key="test-key",
        default_meta={"model": "claude-sonnet-4-20250514"},
        client=client,
    )

    response = await provider.complete(
        {
            "messages": [
                LLMMessage.build(role="system", content="Be concise."),
                LLMMessage.build(role="user", content="hello"),
            ],
            "max_tokens": 128,
            "tools": [anthropic_echo_spec],
            "tool_choice": "auto",
        }
    )

    call = client.post_calls[0]
    assert call["url"] == ANTHROPIC_API_URL
    assert call["headers"] == {
        "anthropic-version": ANTHROPIC_API_VERSION,
        "content-type": "application/json",
        "x-api-key": "test-key",
    }
    assert call["json"] == {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 128,
        "system": [
            {
                "type": "text",
                "text": "Be concise.",
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]}
        ],
        "tools": [
            {
                "name": "echo",
                "description": "Echo message.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Echo message",
                        }
                    },
                    "required": ["message"],
                    "additionalProperties": False,
                },
            }
        ],
        "tool_choice": {"type": "auto"},
    }
    assert response.text == "hello from claude"
    assert response.usage.input_tokens == 10
    assert response.usage.cached_input_tokens == 2
    assert response.provider_meta[0].provider_name == "anthropic"


def test_anthropic_prompt_cache_marks_last_system_block() -> None:
    provider = Anthropic(
        api_key="test-key",
        default_meta={"model": "claude-sonnet-4-20250514"},
    )

    payload = provider._build_messages_request(
        provider.request_to_payload(
            {
                "messages": [
                    LLMMessage.build(role="system", content="First."),
                    LLMMessage.build(role="system", content="Second."),
                    LLMMessage.build(role="user", content="hello"),
                ],
            }
        )
    )

    assert payload["system"] == [
        {"type": "text", "text": "First."},
        {
            "type": "text",
            "text": "Second.",
            "cache_control": {"type": "ephemeral"},
        },
    ]


def test_anthropic_prompt_cache_marks_last_tool_when_system_is_absent(
    anthropic_echo_spec,
) -> None:
    provider = Anthropic(
        api_key="test-key",
        default_meta={"model": "claude-sonnet-4-20250514"},
    )

    payload = provider._build_messages_request(
        provider.request_to_payload(
            {
                "messages": [LLMMessage.build(role="user", content="hello")],
                "tools": [anthropic_echo_spec],
            }
        )
    )

    assert payload["tools"][0]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_prompt_cache_marks_history_when_no_system_or_tools() -> None:
    provider = Anthropic(
        api_key="test-key",
        default_meta={"model": "claude-sonnet-4-20250514"},
    )

    payload = provider._build_messages_request(
        provider.request_to_payload(
            {
                "messages": [
                    LLMMessage.build(role="user", content="first"),
                    LLMMessage.build(role="assistant", content="second"),
                    LLMMessage.build(role="user", content="latest"),
                ],
            }
        )
    )

    assert payload["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "first"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "second",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {"role": "user", "content": [{"type": "text", "text": "latest"}]},
    ]


def test_anthropic_oauth_uses_bearer_header() -> None:
    provider = Anthropic(
        api_key="oauth-token",
        default_meta={"model": "claude-sonnet-4-20250514"},
        provider_name=ANTHROPIC_OAUTH_PROVIDER_NAME,
        auth_mode="oauth",
    )

    assert provider._headers() == {
        "anthropic-version": ANTHROPIC_API_VERSION,
        "content-type": "application/json",
        "authorization": "Bearer oauth-token",
    }


def test_anthropic_oauth_resolves_env_fallback(monkeypatch) -> None:
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "fallback-token")

    assert default_api_key_for_provider("anthropic-oauth") == "fallback-token"
    assert resolve_provider_api_key("anthropic-oauth", "") == "fallback-token"
    assert (
        api_key_placeholder("anthropic-oauth")
        == "CLAUDE_CODE_OAUTH_TOKEN or ANTHROPIC_AUTH_TOKEN"
    )


def test_anthropic_formats_tool_sub_turn() -> None:
    provider = Anthropic(
        api_key="test-key",
        default_meta={"model": "claude-sonnet-4-20250514"},
    )
    call = LLMToolCall(
        name="lookup",
        arguments='{"q":"aceai"}',
        call_id="toolu_1",
    )
    payload = provider._build_messages_request(
        provider.request_to_payload(
            {
                "messages": [
                    LLMToolCallMessage.from_content(
                        content="",
                        tool_calls=[call],
                    ),
                    LLMToolUseMessage.from_content(
                        content="result",
                        name="lookup",
                        call_id="toolu_1",
                    ),
                ],
            }
        )
    )

    assert payload["messages"] == [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "lookup",
                    "input": {"q": "aceai"},
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "result",
                }
            ],
        },
    ]


def test_anthropic_rejects_hosted_tools() -> None:
    provider = Anthropic(
        api_key="test-key",
        default_meta={"model": "claude-sonnet-4-20250514"},
    )

    with pytest.raises(AceAIConfigurationError):
        provider._build_messages_request(
            provider.request_to_payload(
                {
                    "messages": [LLMMessage.build(role="user", content="hello")],
                    "tools": [
                        LLMHostedToolSpec(
                            provider_name="openai",
                            native_name="web_search",
                        )
                    ],
                }
            )
        )


def test_anthropic_rejects_structured_response_format() -> None:
    provider = Anthropic(
        api_key="test-key",
        default_meta={"model": "claude-sonnet-4-20250514"},
    )

    with pytest.raises(Exception, match="text response_format"):
        provider._build_messages_request(
            provider.request_to_payload(
                {
                    "messages": [LLMMessage.build(role="user", content="hello")],
                    "response_format": LLMResponseFormat(type="json_object"),
                }
            )
        )


@pytest.mark.anyio
async def test_anthropic_stream_maps_text_and_tool_call() -> None:
    client = RecordingAsyncClient()
    client.stream_lines = [
        "event: message_start",
        'data: {"type":"message_start","message":{"id":"msg-stream","model":"claude-sonnet-4-20250514","usage":{"input_tokens":5}}}',
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}',
        'data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"lookup","input":{}}}',
        'data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"q\\":"}}',
        'data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"\\"aceai\\"}"}}',
        'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":2}}',
        'data: {"type":"message_stop"}',
    ]
    provider = Anthropic(
        api_key="test-key",
        default_meta={"model": "claude-sonnet-4-20250514"},
        client=client,
    )

    events = [
        event
        async for event in provider.stream(
            {"messages": [LLMMessage.build(role="user", content="hello")]}
        )
    ]

    assert client.stream_calls[0]["json"]["stream"] is True
    assert events[0].text_delta == "hi"
    assert events[1].tool_call_delta.arguments_delta == '{"q":'
    final_response = events[-1].response
    assert final_response.text == "hi"
    assert final_response.tool_calls[0].arguments == '{"q":"aceai"}'
    assert final_response.tool_calls[0].call_id == "toolu_1"
    assert final_response.status == "tool_use"
