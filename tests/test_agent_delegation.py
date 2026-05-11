import json
from pathlib import Path

import pytest

from aceai.agent.features.delegation import (
    DEV_READ_ONLY_TOOLSET,
    NO_CHILD_TOOLSET,
    _build_child_agent,
    build_delegate_to_subagent_tool,
)
from aceai.agent.features.tools import (
    default_agent_tools,
    read_text_file,
    run_shell_command,
    search_text,
)
from aceai.core import ToolExecutionError
from aceai.llm.interface import UNSET
from aceai.llm.models import (
    LLMHostedToolSpec,
    LLMResponse,
    LLMStreamEvent,
    LLMToolCall,
)


class RecordingDelegationLLMService:
    def __init__(self, streams: list[list[LLMStreamEvent]]) -> None:
        self._streams = [list(stream) for stream in streams]
        self.stream_calls: list[dict] = []

    async def stream(self, **request):
        if not self._streams:
            raise AssertionError("RecordingDelegationLLMService has no stream fixture")
        self.stream_calls.append(request)
        for event in self._streams.pop(0):
            yield event

    async def complete(self, **request) -> LLMResponse:
        raise AssertionError("delegate_to_subagent child agent should stream")


def completed_stream(response: LLMResponse) -> list[LLMStreamEvent]:
    return [
        LLMStreamEvent(
            event_type="response.completed",
            response=response,
        )
    ]


def test_delegate_to_subagent_defaults_child_max_steps_to_unset() -> None:
    assert build_delegate_to_subagent_tool.__kwdefaults__["child_max_steps"] is UNSET


@pytest.mark.anyio
async def test_delegate_to_subagent_runs_child_agent_with_generated_instructions() -> None:
    llm_service = RecordingDelegationLLMService(
        [
            completed_stream(
                LLMResponse(
                    text=(
                        "Summary:\nReviewed the issue.\n\n"
                        "Evidence:\nUsed the provided context.\n\n"
                        "Risks:\nNo repo access was needed."
                    )
                )
            )
        ]
    )
    delegate_to_subagent = build_delegate_to_subagent_tool(
        llm_service=llm_service,
        default_model="gpt-5.5",
        available_tools=[],
    )

    result = await delegate_to_subagent(
        task="Review the proposed architecture boundary.",
        instructions="Act as a skeptical architecture reviewer.",
        context_brief="The main agent separated run state from agent definition.",
        toolset=NO_CHILD_TOOLSET,
        allowed_tools=[],
    )

    payload = json.loads(result.output)
    truncated_payload = json.loads(result.truncated_output)
    assert payload["status"] == "completed"
    assert payload["final_answer"].startswith("Summary:\nReviewed")
    assert payload["summary"] == payload["final_answer"]
    assert payload["step_count"] == 1
    assert payload["important_evidence"] == []
    assert payload["tool_results"] == []
    assert payload["agent_id"].startswith("child-")
    assert truncated_payload["type"] == "subagent_handoff"
    assert truncated_payload["handoff"].startswith("Summary:\nReviewed")

    messages = llm_service.stream_calls[0]["messages"]
    system_text = messages[0].content[0]["data"]
    user_text = messages[1].content[0]["data"]
    assert "delegated child agent" in system_text
    assert "Act as a skeptical architecture reviewer." in system_text
    assert "Return a concise result" in system_text
    assert "Review the proposed architecture boundary." in user_text
    assert "The main agent separated run state" in user_text
    assert "tools" not in llm_service.stream_calls[0]


@pytest.mark.anyio
async def test_delegate_to_subagent_defaults_to_dev_read_only_toolset() -> None:
    llm_service = RecordingDelegationLLMService(
        [
            completed_stream(
                LLMResponse(
                    text=(
                        "Summary:\nChecked with read-only tools.\n\n"
                        "Evidence:\nTools were available.\n\n"
                        "Risks:\nNone."
                    )
                )
            )
        ]
    )
    delegate_to_subagent = build_delegate_to_subagent_tool(
        llm_service=llm_service,
        default_model="gpt-5.5",
        available_tools=[read_text_file, search_text, run_shell_command],
    )

    await delegate_to_subagent(
        task="Inspect the repo.",
        instructions="Use read-only tools if needed.",
        context_brief="Need source evidence.",
        allowed_tools=[],
    )

    first_call_tools = llm_service.stream_calls[0]["tools"]
    assert [tool.name for tool in first_call_tools] == ["read_text_file", "search_text"]


@pytest.mark.anyio
async def test_delegate_to_subagent_limits_child_tools_to_allowed_names(tmp_path: Path) -> None:
    target = tmp_path / "note.txt"
    target.write_text("delegated evidence", encoding="utf-8")
    llm_service = RecordingDelegationLLMService(
        [
            completed_stream(
                LLMResponse(
                    tool_calls=[
                        LLMToolCall(
                            name="read_text_file",
                            arguments='{"path":"' + str(target) + '"}',
                            call_id="call-read",
                        )
                    ]
                )
            ),
            completed_stream(
                LLMResponse(
                    text=(
                        "Summary:\nThe file says delegated evidence.\n\n"
                        "Evidence:\nread_text_file returned delegated evidence.\n\n"
                        "Risks:\nNone."
                    )
                )
            ),
        ]
    )
    delegate_to_subagent = build_delegate_to_subagent_tool(
        llm_service=llm_service,
        default_model="gpt-5.5",
        available_tools=[read_text_file, search_text],
    )

    result = await delegate_to_subagent(
        task="Read the note and report the content.",
        instructions="Use file evidence only.",
        context_brief="The target file is the provided note.",
        allowed_tools=["read_text_file"],
    )

    payload = json.loads(result.output)
    truncated_payload = json.loads(result.truncated_output)
    assert payload["status"] == "completed"
    assert payload["step_count"] == 2
    assert len(payload["tool_results"]) == 1
    assert payload["tool_results"][0]["tool_name"] == "read_text_file"
    assert payload["tool_results"][0]["call_id"] == "call-read"
    assert payload["tool_results"][0]["arguments"] == '{"path":"' + str(target) + '"}'
    assert "delegated evidence" in payload["tool_results"][0]["output"]
    assert payload["important_evidence"] == [payload["tool_results"][0]["output"]]
    assert truncated_payload["tool_result_count"] == 1
    assert truncated_payload["tool_names"] == ["read_text_file"]
    assert "delegated evidence" in truncated_payload["evidence"][0]

    first_call_tools = llm_service.stream_calls[0]["tools"]
    assert [tool.name for tool in first_call_tools] == ["read_text_file"]
    second_call_tools = llm_service.stream_calls[1]["tools"]
    assert [tool.name for tool in second_call_tools] == ["read_text_file"]


@pytest.mark.anyio
async def test_delegate_to_subagent_allows_child_hosted_tools() -> None:
    hosted_tool = LLMHostedToolSpec(
        provider_name="openai",
        native_name="web_search",
    )
    llm_service = RecordingDelegationLLMService(
        [
            completed_stream(
                LLMResponse(
                    text=(
                        "Summary:\nFound current news.\n\n"
                        "Evidence:\nUsed hosted web search.\n\n"
                        "Risks:\nNone."
                    )
                )
            )
        ]
    )
    delegate_to_subagent = build_delegate_to_subagent_tool(
        llm_service=llm_service,
        default_model="gpt-5.5",
        available_tools=[],
        available_hosted_tools=[hosted_tool],
    )

    result = await delegate_to_subagent(
        task="Search current Iran news.",
        instructions="Use hosted web search.",
        context_brief="Current date is 2026-05-08.",
        allowed_tools=["openai:web_search"],
    )

    payload = json.loads(result.output)
    assert payload["status"] == "completed"
    assert payload["final_answer"].startswith("Summary:\nFound current news.")
    assert llm_service.stream_calls[0]["tools"] == [hosted_tool]


def test_delegate_to_subagent_schema_exposes_toolset_and_allowed_tool_enums() -> None:
    hosted_tool = LLMHostedToolSpec(
        provider_name="openai",
        native_name="web_search",
    )
    delegate_to_subagent = build_delegate_to_subagent_tool(
        llm_service=RecordingDelegationLLMService([]),
        default_model="gpt-5.5",
        available_tools=[read_text_file, run_shell_command],
        available_hosted_tools=[hosted_tool],
    )

    schema = delegate_to_subagent.tool_spec.generate_schema()

    properties = schema["parameters"]["properties"]
    assert properties["toolset"]["enum"] == ["dev_read_only", "custom", "none"]
    assert properties["allowed_tools"]["items"]["enum"] == [
        "openai:web_search",
        "read_text_file",
    ]
    assert "functions. prefix" in properties["allowed_tools"]["description"]


def test_delegated_child_agent_inherits_context_window_tokens() -> None:
    agent = _build_child_agent(
        llm_service=RecordingDelegationLLMService([]),
        default_model="gpt-5.5",
        instructions="Report concisely.",
        tools=[],
        hosted_tools=[],
        child_max_steps=4,
        compress_threshold=0.5,
        context_window_tokens=1050000,
    )

    assert agent._compression_policy.threshold == 0.5
    assert agent._compression_policy.context_window_tokens == 1050000


@pytest.mark.anyio
async def test_delegate_to_subagent_rejects_approval_required_child_tools() -> None:
    llm_service = RecordingDelegationLLMService([])
    delegate_to_subagent = build_delegate_to_subagent_tool(
        llm_service=llm_service,
        default_model="gpt-5.5",
        available_tools=[run_shell_command],
    )

    with pytest.raises(
        ToolExecutionError,
        match="delegate_to_subagent cannot use approval-required child tools: run_shell_command",
    ):
        await delegate_to_subagent(
            task="Run a shell command.",
            instructions="Use the shell.",
            context_brief="No context.",
            allowed_tools=["run_shell_command"],
        )
    assert llm_service.stream_calls == []


@pytest.mark.anyio
async def test_delegate_to_subagent_rejects_unknown_child_tools() -> None:
    llm_service = RecordingDelegationLLMService([])
    delegate_to_subagent = build_delegate_to_subagent_tool(
        llm_service=llm_service,
        default_model="gpt-5.5",
        available_tools=[read_text_file],
    )

    with pytest.raises(
        ToolExecutionError,
        match="delegate_to_subagent received unknown child tool: missing_tool",
    ):
        await delegate_to_subagent(
            task="Use a missing tool.",
            instructions="Call the missing tool.",
            context_brief="No context.",
            toolset="custom",
            allowed_tools=["missing_tool"],
        )
    assert llm_service.stream_calls == []


@pytest.mark.anyio
async def test_delegate_to_subagent_rejects_custom_toolset_without_allowed_tools() -> None:
    llm_service = RecordingDelegationLLMService([])
    delegate_to_subagent = build_delegate_to_subagent_tool(
        llm_service=llm_service,
        default_model="gpt-5.5",
        available_tools=[read_text_file],
    )

    with pytest.raises(
        ToolExecutionError,
        match="custom toolset requires allowed_tools",
    ):
        await delegate_to_subagent(
            task="Read with custom tools.",
            instructions="Use file reads.",
            context_brief="No context.",
            toolset="custom",
            allowed_tools=[],
        )
    assert llm_service.stream_calls == []


def test_default_agent_tools_do_not_include_delegate_to_subagent() -> None:
    assert "delegate_to_subagent" not in {tool.name for tool in default_agent_tools()}
