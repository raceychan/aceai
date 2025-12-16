import pytest

from aceai.llm.models import LLMMessage, LLMToolCall, LLMToolCallMessage


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
    message = LLMToolCallMessage.build("call tool", tool_calls=[tool_call])

    as_dict = message.asdict()

    assert as_dict["role"] == "assistant"
    assert as_dict["content"][0]["data"] == "call tool"
    assert as_dict["tool_calls"][0]["name"] == "calc"
