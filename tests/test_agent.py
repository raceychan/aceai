import pytest

from aceai.agent.base import AgentBase, ToolExecutionFailure
from aceai.errors import AceAIConfigurationError
from aceai.llm.models import LLMToolCall


def test_build_agent_base() -> None:
    agent = AgentBase(
        prompt="You are a helpful assistant.",
        default_model="gpt-4",
        llm_service=None,  # type: ignore
        executor=None,  # type: ignore
    )
    assert agent.system_message.content[0]["data"] == "You are a helpful assistant."
    assert agent.default_model == "gpt-4"
    assert agent.max_steps == 5
    assert hasattr(agent, "run")


def test_agent_base_add_instruction_updates_system_message() -> None:
    agent = AgentBase(
        prompt="Initial",
        default_model="gpt-4",
        llm_service=None,  # type: ignore[arg-type]
        executor=None,  # type: ignore[arg-type]
    )
    agent.add_instruction(" + More")
    agent.add_instruction(" + More")
    assert agent.system_message.content[0]["data"] == "Initial + More"


def test_agent_base_add_instruction_rejects_empty_string() -> None:
    agent = AgentBase(
        prompt="Initial",
        default_model="gpt-4",
        llm_service=None,  # type: ignore[arg-type]
        executor=None,  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="Empty Instruction"):
        agent.add_instruction("")


def test_agent_base_requires_positive_max_steps() -> None:
    with pytest.raises(AceAIConfigurationError):
        AgentBase(
            prompt="Prompt",
            default_model="gpt-4",
            llm_service=None,  # type: ignore[arg-type]
            executor=None,  # type: ignore[arg-type]
            max_steps=0,
        )


def test_agent_base_rejects_chunk_size_argument() -> None:
    with pytest.raises(TypeError):
        AgentBase(
            prompt="Prompt",
            default_model="gpt-4",
            llm_service=None,  # type: ignore[arg-type]
            executor=None,  # type: ignore[arg-type]
            delta_chunk_size=1,
        )


def test_agent_base_rejects_reasoning_log_argument() -> None:
    with pytest.raises(TypeError):
        AgentBase(
            prompt="Prompt",
            default_model="gpt-4",
            llm_service=None,  # type: ignore[arg-type]
            executor=None,  # type: ignore[arg-type]
            reasoning_log_max_chars=10,
        )


def test_tool_execution_failure_preserves_original_error() -> None:
    tool_call = LLMToolCall(name="calc", arguments="{}", call_id="call-1")
    cause = ValueError("explode")

    failure = ToolExecutionFailure(tool_call=tool_call, error=cause)

    assert str(failure) == "explode"
    assert failure.tool_call is tool_call
    assert failure.original_error is cause
