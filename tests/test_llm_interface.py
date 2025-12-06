import pytest

from aceai.llm.models import LLMMessage, LLMToolCall, LLMToolCallMessage


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

    with pytest.raises(ValueError):
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
