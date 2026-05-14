import asyncio

import pytest

from aceai.core.agent import Agent
from aceai.llm.errors import AceAIRuntimeError, LLMContextWindowExceededError
from aceai.core.executor import ToolExecutionError
from aceai.core.run_state import ToolRunState
from aceai.core.skills import SkillRegistry
from aceai.core.events import (
    AgentEvent,
    ContextCompactionFailedEvent,
    ContextCompactionStartedEvent,
    ContextCompressedEvent,
    LLMMediaEvent,
    LLMHostedToolEvent,
    LLMOutputDeltaEvent,
    LLMReasoningEvent,
    LLMRetryingEvent,
    LLMStartedEvent,
    LLMToolCallDeltaEvent,
    RunCompletedEvent,
    RunFailedEvent,
    RunSuspendedEvent,
    StepCompletedEvent,
    ToolApprovalRequestedEvent,
    ToolApprovalResolvedEvent,
    ToolCompletedEvent,
    ToolFailedEvent,
    ToolStartedEvent,
)
from aceai.core.models import ToolApprovalDecision, ToolExecutionOutput
from aceai.llm import LLMResponse
from aceai.llm.interface import UNSET
from aceai.llm.models import (
    LLMGeneratedMedia,
    LLMHostedToolSpec,
    LLMMessage,
    LLMSegment,
    LLMStreamEvent,
    LLMToolCall,
    LLMToolCallMessage,
    LLMToolCallDelta,
    LLMToolUseMessage,
)


class StubExecutor:
    def __init__(
        self,
        results: dict[str, str] | None = None,
        approval_required: set[str] | None = None,
        hosted_tools: list[LLMHostedToolSpec] | None = None,
    ) -> None:
        self.tool_specs: list[object] = []
        self._results = results or {}
        self._approval_required = approval_required or set()
        self.calls: list[LLMToolCall] = []
        self._skill_registry = SkillRegistry()
        self._hosted_tools = hosted_tools if hosted_tools is not None else []

    @property
    def prompt_instructions(self) -> str:
        return ""

    @property
    def skill_registry(self) -> SkillRegistry:
        return self._skill_registry

    @property
    def hosted_tools(self) -> list[LLMHostedToolSpec]:
        return self._hosted_tools

    def select_tools(
        self, include: set[str] | None = None, exclude: set[str] | None = None
    ) -> list[object]:
        if include and exclude:
            raise ValueError("Cannot specify both include and exclude")
        return self.tool_specs

    def resolve_invocation(self, tool_call: LLMToolCall):
        return StubInvocation(
            call=tool_call,
            approval_required=tool_call.name in self._approval_required,
        )

    async def execute(
        self,
        invocation,
        *,
        tool_state: ToolRunState,
    ) -> str:
        tool_call = invocation.call
        self.calls.append(tool_call)
        return self._results[tool_call.name]


class StubInvocation:
    def __init__(self, call: LLMToolCall, approval_required: bool = False) -> None:
        self.call = call
        self.approval_required = approval_required
        self.tool = StubTool(call.name)


class StubTool:
    def __init__(self, name: str) -> None:
        self.name = name
        self.metadata = StubToolMetadata()


class StubToolMetadata:
    approval_policy = "test_policy"


class LargeToolSpec:
    name = "large_tool"

    def generate_schema(self) -> dict:
        return {
            "type": "function",
            "name": self.name,
            "description": "x" * 360,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "y" * 360,
                    },
                },
                "required": ["query"],
            },
        }


class StubLLMService:
    def __init__(self, streams: list[list[LLMStreamEvent]]) -> None:
        self._streams = [list(stream) for stream in streams]
        self.calls: list[dict] = []

    async def stream(self, **request):
        if not self._streams:
            raise AssertionError("StubLLMService has no remaining stream fixtures")
        self.calls.append(request)
        events = self._streams.pop(0)
        for event in events:
            yield event

    async def complete(self, **request) -> LLMResponse:
        raise AssertionError("Agent should not call complete() in streaming mode")


class RaisingExecutor(StubExecutor):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self._error = error

    async def execute(
        self,
        invocation,
        *,
        tool_state: ToolRunState,
    ) -> str:
        raise self._error


class ExplicitTruncatedOutputExecutor(StubExecutor):
    async def execute(
        self,
        invocation,
        *,
        tool_state: ToolRunState,
    ) -> ToolExecutionOutput:
        tool_call = invocation.call
        self.calls.append(tool_call)
        output = "audit-output"
        truncated_output = "truncated-output-line\n" * 5000
        return ToolExecutionOutput(output=output, truncated_output=truncated_output)


class BlockingExecutor(StubExecutor):
    def __init__(self, results: dict[str, str]) -> None:
        super().__init__(results)
        self.both_started = asyncio.Event()
        self.release = asyncio.Event()

    async def execute(
        self,
        invocation,
        *,
        tool_state: ToolRunState,
    ) -> str:
        tool_call = invocation.call
        self.calls.append(tool_call)
        if len(self.calls) == 2:
            self.both_started.set()
        await self.release.wait()
        return self._results[tool_call.name]


class RaisingStreamLLMService:
    def __init__(self, events: list[LLMStreamEvent], error: Exception) -> None:
        self._events = list(events)
        self._error = error
        self.calls: list[dict] = []

    async def stream(self, **request):
        self.calls.append(request)
        for event in self._events:
            yield event
        raise self._error

    async def complete(self, **request) -> LLMResponse:
        raise AssertionError("Agent should not call complete() in streaming mode")


class SimpleLLMService:
    def __init__(self, events: list[LLMStreamEvent]) -> None:
        self._events = list(events)
        self.calls: list[dict] = []

    async def stream(self, **request):
        self.calls.append(request)
        for event in self._events:
            yield event

    async def complete(self, **request) -> LLMResponse:
        raise AssertionError("Agent should not call complete() in streaming mode")


class CompressingLLMService:
    def __init__(self, stream_events: list[LLMStreamEvent]) -> None:
        self._stream_events = list(stream_events)
        self.complete_calls: list[dict] = []
        self.stream_calls: list[dict] = []

    async def stream(self, **request):
        self.stream_calls.append(request)
        for event in self._stream_events:
            yield event

    async def complete(self, **request) -> LLMResponse:
        self.complete_calls.append(request)
        return LLMResponse(text="Earlier discussion summary.")


class CompressingMultiStreamLLMService:
    def __init__(self, streams: list[list[LLMStreamEvent]]) -> None:
        self._streams = [list(stream) for stream in streams]
        self.complete_calls: list[dict] = []
        self.stream_calls: list[dict] = []

    async def stream(self, **request):
        if not self._streams:
            raise AssertionError("CompressingMultiStreamLLMService has no streams")
        self.stream_calls.append(request)
        events = self._streams.pop(0)
        for event in events:
            yield event

    async def complete(self, **request) -> LLMResponse:
        self.complete_calls.append(request)
        return LLMResponse(text="Earlier discussion summary.")


class ContextWindowThenRecoveringLLMService:
    def __init__(self, stream_events: list[LLMStreamEvent]) -> None:
        self._stream_events = list(stream_events)
        self.failures_remaining = 1
        self.complete_calls: list[dict] = []
        self.stream_calls: list[dict] = []

    async def stream(self, **request):
        self.stream_calls.append(request)
        if self.failures_remaining > 0:
            self.failures_remaining -= 1
            raise LLMContextWindowExceededError(
                "APIError: Your input exceeds the context window of this model. "
                "Please adjust your input and try again."
            )
        for event in self._stream_events:
            yield event

    async def complete(self, **request) -> LLMResponse:
        self.complete_calls.append(request)
        return LLMResponse(text="Earlier discussion summary.")


class ContextWindowDuringCompactionLLMService(ContextWindowThenRecoveringLLMService):
    async def complete(self, **request) -> LLMResponse:
        self.complete_calls.append(request)
        raise LLMContextWindowExceededError(
            "APIError: Your input exceeds the context window of this model. "
            "Please adjust your input and try again."
        )


def make_stream(
    *,
    response: LLMResponse,
    deltas: list[str] | None = None,
) -> list[LLMStreamEvent]:
    events: list[LLMStreamEvent] = []
    for chunk in deltas or []:
        events.append(
            LLMStreamEvent(
                event_type="response.output_text.delta",
                text_delta=chunk,
            )
        )
    events.append(
        LLMStreamEvent(
            event_type="response.completed",
            response=response,
        )
    )
    return events


async def collect_events(agent: Agent, question: str) -> list[AgentEvent]:
    return [event async for event in agent.run(question)]


@pytest.mark.anyio
async def test_agent_compresses_resume_history_before_llm_call() -> None:
    llm_service = CompressingLLMService(
        make_stream(response=LLMResponse(text="done"), deltas=["done"])
    )
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,
        executor=StubExecutor(),
        max_steps=1,
        compress_threshold=1,
    )
    history = [
        LLMMessage.build(role="user", content=f"history message {index}")
        for index in range(10)
    ]

    events = [event async for event in agent.resume("new question", history)]

    assert isinstance(events[-1], RunCompletedEvent)
    context_events = [
        event for event in events if isinstance(event, ContextCompressedEvent)
    ]
    assert len(context_events) == 1
    assert context_events[0].reason == "threshold"
    assert len(llm_service.complete_calls) == 1
    messages = llm_service.stream_calls[0]["messages"]
    assert messages[0].role == "system"
    assert messages[1].role == "system"
    assert (
        '<aceai_context_summary scope="prior_runs">' in messages[1].content[0]["data"]
    )
    assert "Earlier discussion summary." in messages[1].content[0]["data"]
    assert messages[-1].content[0]["data"] == "new question"
    assert "history message 0" not in "\n".join(
        message.content[0]["data"] for message in messages
    )


@pytest.mark.anyio
async def test_agent_preflight_compresses_before_full_context_window() -> None:
    llm_service = CompressingLLMService(
        make_stream(response=LLMResponse(text="done"), deltas=["done"])
    )
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,
        executor=StubExecutor(),
        max_steps=1,
        compress_threshold="100%",
        context_window_tokens=100,
    )
    history = [
        LLMMessage.build(role="user", content=f"history message {index}")
        for index in range(9)
    ]

    events = [event async for event in agent.resume("new question", history)]

    assert isinstance(events[-1], RunCompletedEvent)
    assert len(llm_service.complete_calls) > 1
    assert len([event for event in events if isinstance(event, ContextCompressedEvent)]) == 1
    messages = llm_service.stream_calls[0]["messages"]
    assert '<aceai_context_summary scope="prior_runs">' in messages[1].content[0]["data"]


@pytest.mark.anyio
async def test_agent_preflight_counts_tool_schema_budget() -> None:
    llm_service = CompressingLLMService(
        make_stream(response=LLMResponse(text="done"), deltas=["done"])
    )
    executor = StubExecutor()
    executor.tool_specs = [LargeToolSpec()]
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,
        executor=executor,
        max_steps=1,
        compress_threshold="100%",
        context_window_tokens=180,
    )
    history = [
        LLMMessage.build(role="user", content=f"history message {index}")
        for index in range(3)
    ]

    events = [event async for event in agent.resume("new question", history)]

    assert isinstance(events[-1], RunCompletedEvent)
    assert len(llm_service.complete_calls) == 1
    assert len([event for event in events if isinstance(event, ContextCompressedEvent)]) == 1


@pytest.mark.anyio
async def test_agent_compression_keeps_recent_run_as_unit() -> None:
    llm_service = CompressingLLMService(
        make_stream(response=LLMResponse(text="done"), deltas=["done"])
    )
    call = LLMToolCall(name="lookup", arguments="{}", call_id="call-lookup")
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,
        executor=StubExecutor(),
        max_steps=1,
        compress_threshold=1,
    )
    history = [
        LLMMessage.build(role="user", content="older question"),
        LLMMessage.build(role="assistant", content="older answer"),
        LLMMessage.build(role="user", content="history message 0"),
        LLMToolCallMessage.from_content(content=[], tool_calls=[call]),
        LLMToolUseMessage.from_content(
            name="lookup",
            call_id=call.call_id,
            content="lookup result",
        ),
        LLMMessage.build(role="user", content="history message 1"),
        LLMMessage.build(role="user", content="history message 2"),
        LLMMessage.build(role="user", content="history message 3"),
        LLMMessage.build(role="user", content="history message 4"),
        LLMMessage.build(role="user", content="history message 5"),
        LLMMessage.build(role="user", content="history message 6"),
    ]

    events = [event async for event in agent.resume("new question", history)]

    assert isinstance(events[-1], RunCompletedEvent)
    assert len(llm_service.complete_calls) == 1
    messages = llm_service.stream_calls[0]["messages"]
    message_text = "\n".join(
        part["data"]
        for message in messages
        for part in message.content
        if part["type"] == "text"
    )
    assert "Earlier discussion summary." in message_text
    assert "older question" not in message_text
    assert "history message 0" not in message_text
    assert "history message 6" not in message_text
    assert not any(isinstance(message, LLMToolCallMessage) for message in messages)
    assert not any(isinstance(message, LLMToolUseMessage) for message in messages)


@pytest.mark.anyio
async def test_agent_compression_rejects_history_without_run_boundary() -> None:
    llm_service = CompressingLLMService(
        make_stream(response=LLMResponse(text="done"), deltas=["done"])
    )
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,
        executor=StubExecutor(),
        max_steps=1,
        compress_threshold=1,
    )
    history = [
        LLMMessage.build(role="assistant", content=f"orphan message {index}")
        for index in range(10)
    ]

    with pytest.raises(
        AceAIRuntimeError,
        match="context compression requires user-message run boundaries",
    ):
        [event async for event in agent.resume("new question", history)]


@pytest.mark.anyio
async def test_agent_compression_rejects_tool_output_without_same_run_call() -> None:
    llm_service = CompressingLLMService(
        make_stream(response=LLMResponse(text="done"), deltas=["done"])
    )
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,
        executor=StubExecutor(),
        max_steps=1,
        compress_threshold=1,
    )
    history = [
        LLMMessage.build(role="user", content="previous question"),
        LLMToolUseMessage.from_content(
            name="lookup",
            call_id="missing-call",
            content="lookup result",
        ),
    ]

    with pytest.raises(
        AceAIRuntimeError,
        match="tool output without a tool call in the same step",
    ):
        [event async for event in agent.resume("new question", history)]


@pytest.mark.anyio
async def test_agent_validates_orphan_tool_output_before_llm_call() -> None:
    llm_service = StubLLMService(
        [make_stream(response=LLMResponse(text="should not be called"))]
    )
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,
        executor=StubExecutor(),
        max_steps=1,
        compress_threshold="100%",
    )
    history = [
        LLMMessage.build(role="user", content="previous question"),
        LLMToolUseMessage.from_content(
            name="lookup",
            call_id="missing-call",
            content="lookup result",
        ),
    ]

    with pytest.raises(
        AceAIRuntimeError,
        match="tool output without a tool call in the same step",
    ):
        [event async for event in agent.resume("new question", history)]

    assert llm_service.calls == []


@pytest.mark.anyio
async def test_agent_compresses_and_retries_current_step_after_context_window_error() -> (
    None
):
    llm_service = ContextWindowThenRecoveringLLMService(
        make_stream(response=LLMResponse(text="done"), deltas=["done"])
    )
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,
        executor=StubExecutor(),
        max_steps=1,
    )
    history = [
        LLMMessage.build(role="user", content=f"history message {index}")
        for index in range(10)
    ]

    events = [event async for event in agent.resume("new question", history)]

    assert isinstance(events[-1], RunCompletedEvent)
    assert events[-1].final_answer == "done"
    context_events = [
        event for event in events if isinstance(event, ContextCompressedEvent)
    ]
    assert len(context_events) == 1
    assert context_events[0].reason == "context_window_retry"
    assert len(llm_service.stream_calls) == 2
    assert len(llm_service.complete_calls) == 1
    first_messages = llm_service.stream_calls[0]["messages"]
    second_messages = llm_service.stream_calls[1]["messages"]
    assert "history message 0" in "\n".join(
        message.content[0]["data"] for message in first_messages
    )
    assert (
        '<aceai_context_summary scope="prior_runs">'
        in second_messages[1].content[0]["data"]
    )
    assert "Earlier discussion summary." in second_messages[1].content[0]["data"]
    assert "history message 0" not in "\n".join(
        message.content[0]["data"] for message in second_messages
    )


@pytest.mark.anyio
async def test_agent_chunks_large_context_compaction_inputs() -> None:
    llm_service = CompressingLLMService(
        make_stream(response=LLMResponse(text="done"), deltas=["done"])
    )
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,
        executor=StubExecutor(),
        max_steps=1,
        compress_threshold="60%",
        context_window_tokens=500,
    )
    history = [
        LLMMessage.build(
            role="user",
            content=f"history message {index}: " + ("detail " * 30),
        )
        for index in range(8)
    ]

    events = [event async for event in agent.resume("new question", history)]

    assert isinstance(events[-1], RunCompletedEvent)
    assert len(llm_service.complete_calls) > 1
    assert [event for event in events if isinstance(event, ContextCompressedEvent)]


@pytest.mark.anyio
async def test_agent_emits_context_compaction_failed_when_summary_exceeds_window() -> (
    None
):
    llm_service = ContextWindowDuringCompactionLLMService(
        make_stream(response=LLMResponse(text="done"), deltas=["done"])
    )
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,
        executor=StubExecutor(),
        max_steps=1,
    )
    history = [LLMMessage.build(role="user", content="older context")]

    events = [event async for event in agent.resume("new question", history)]

    assert isinstance(events[-1], RunFailedEvent)
    assert [event for event in events if isinstance(event, ContextCompactionStartedEvent)]
    failed_events = [
        event for event in events if isinstance(event, ContextCompactionFailedEvent)
    ]
    assert failed_events
    assert "exceeds the context window" in failed_events[0].error


@pytest.mark.anyio
async def test_agent_compresses_older_completed_steps_inside_current_run() -> None:
    first_call = LLMToolCall(
        name="first", arguments='{"value":"old"}', call_id="call-1"
    )
    second_call = LLMToolCall(
        name="second",
        arguments='{"value":"recent"}',
        call_id="call-2",
    )
    llm_service = CompressingMultiStreamLLMService(
        [
            make_stream(response=LLMResponse(text="", tool_calls=[first_call])),
            make_stream(response=LLMResponse(text="", tool_calls=[second_call])),
            make_stream(response=LLMResponse(text="done"), deltas=["done"]),
        ]
    )
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,
        executor=StubExecutor(
            {
                "first": "old tool result " * 30,
                "second": "recent tool result",
            }
        ),
        max_steps=3,
        compress_threshold=50,
        context_window_tokens=256,
    )

    events = await collect_events(agent, "Need two lookups")

    assert isinstance(events[-1], RunCompletedEvent)
    context_events = [
        event for event in events if isinstance(event, ContextCompressedEvent)
    ]
    assert len(context_events) == 1
    messages = llm_service.stream_calls[2]["messages"]
    message_text = "\n".join(
        part["data"]
        for message in messages
        for part in message.content
        if part["type"] == "text"
    )
    assert '<aceai_context_summary scope="current_run">' in message_text
    assert "Earlier discussion summary." in message_text
    assert any(
        isinstance(message, LLMToolCallMessage)
        and message.tool_calls[0].call_id == "call-2"
        for message in messages
    )
    assert not any(
        isinstance(message, LLMToolCallMessage)
        and message.tool_calls[0].call_id == "call-1"
        for message in messages
    )


@pytest.mark.anyio
async def test_agent_truncates_explicit_tool_truncated_output_before_history() -> None:
    call = LLMToolCall(name="lookup", arguments="{}", call_id="call-lookup")
    llm_service = CompressingMultiStreamLLMService(
        [
            make_stream(response=LLMResponse(text="", tool_calls=[call])),
            make_stream(response=LLMResponse(text="done"), deltas=["done"]),
        ]
    )
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,
        executor=ExplicitTruncatedOutputExecutor(),
        max_steps=2,
    )

    events = await collect_events(agent, "Need lookup")

    assert isinstance(events[-1], RunCompletedEvent)
    tool_events = [event for event in events if isinstance(event, ToolCompletedEvent)]
    assert tool_events
    event_truncated_text = tool_events[0].tool_result.truncated_output
    assert "tokens truncated" in event_truncated_text
    assert len(event_truncated_text) < len("truncated-output-line\n" * 5000)
    messages = llm_service.stream_calls[1]["messages"]
    tool_messages = [
        message for message in messages if isinstance(message, LLMToolUseMessage)
    ]
    assert len(tool_messages) == 1
    truncated_text = tool_messages[0].content[0]["data"]
    assert "tokens truncated" in truncated_text
    assert len(truncated_text) < len("truncated-output-line\n" * 5000)


@pytest.mark.anyio
async def test_agent_does_not_retry_context_window_error_without_compressible_context() -> (
    None
):
    llm_service = ContextWindowThenRecoveringLLMService(
        make_stream(response=LLMResponse(text="done"), deltas=["done"])
    )
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,
        executor=StubExecutor(),
        max_steps=1,
    )

    events = await collect_events(agent, "new question")

    assert isinstance(events[-1], RunFailedEvent)
    assert "Context compaction could not reduce this context-window retry" in events[
        -1
    ].error
    assert "no completed prior runs or completed current-run steps" in events[-1].error
    assert len(llm_service.stream_calls) == 1
    assert len(llm_service.complete_calls) == 0
    assert not [
        event for event in events if isinstance(event, ContextCompactionStartedEvent)
    ]


@pytest.mark.anyio
async def test_agent_allows_whitespace_question_and_calls_llm() -> None:
    streams = [
        make_stream(
            response=LLMResponse(text="  answer  "),
            deltas=["  answer  "],
        )
    ]
    llm_service = StubLLMService(streams)
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,
        executor=StubExecutor(),
    )

    events = await collect_events(agent, "   ")
    final_event = events[-1]

    assert isinstance(final_event, RunCompletedEvent)
    assert final_event.final_answer == "  answer  "
    assert len(llm_service.calls) == 1
    assert llm_service.calls[0]["messages"][-1].content[0]["data"] == "   "


@pytest.mark.anyio
async def test_agent_events_share_a_non_empty_run_id() -> None:
    streams = [
        make_stream(
            response=LLMResponse(text="answer"),
            deltas=["answer"],
        )
    ]
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService(streams),
        executor=StubExecutor(),
    )

    events = await collect_events(agent, "Question?")
    run_ids = {event.run_id for event in events}

    assert len(run_ids) == 1
    assert "" not in run_ids


def test_agent_creates_independent_run_contexts() -> None:
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([]),
        max_steps=2,
    )

    first_run = agent.create_run("first")
    second_run = agent.create_run("second")

    assert first_run is not second_run
    assert first_run.run_id != second_run.run_id
    assert first_run.question == "first"
    assert second_run.question == "second"
    assert first_run.context.context[-1].content[0]["data"] == "first"
    assert second_run.context.context[-1].content[0]["data"] == "second"


@pytest.mark.anyio
async def test_agent_rejects_run_context_from_different_agent() -> None:
    first_agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([]),
        agent_id="first",
    )
    second_agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([]),
        agent_id="second",
    )
    run = first_agent.create_run("hello")

    with pytest.raises(AceAIRuntimeError, match="different agent"):
        async for _ in second_agent.execute(run):
            pass
    with pytest.raises(AceAIRuntimeError, match="different agent"):
        async for _ in second_agent.resume_approval(
            run,
            ToolApprovalDecision(call_id="call", approved=True),
        ):
            pass


@pytest.mark.anyio
async def test_agent_returns_llm_text_without_tool_calls() -> None:
    streams = [
        make_stream(
            response=LLMResponse(text="  hello world  "),
            deltas=["  hello ", "world  "],
        )
    ]
    llm_service = StubLLMService(streams)
    executor = StubExecutor()
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,
        executor=executor,
    )

    events = await collect_events(agent, "What time is it?")

    assert isinstance(events[-1], RunCompletedEvent)
    assert events[-1].final_answer == "  hello world  "
    assert len(llm_service.calls) == 1


@pytest.mark.anyio
async def test_agent_passes_hosted_tools_from_executor() -> None:
    streams = [
        make_stream(
            response=LLMResponse(text="searched"),
            deltas=["searched"],
        )
    ]
    llm_service = StubLLMService(streams)
    hosted_tool = LLMHostedToolSpec(
        provider_name="openai",
        native_name="web_search",
    )
    executor = StubExecutor(hosted_tools=[hosted_tool])
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-5.5",
        llm_service=llm_service,
        executor=executor,
    )

    events = await collect_events(agent, "What changed today?")

    assert isinstance(events[-1], RunCompletedEvent)
    assert llm_service.calls[0]["tools"] == [hosted_tool]


@pytest.mark.anyio
async def test_agent_resume_includes_restored_history_before_current_question() -> None:
    streams = [
        make_stream(
            response=LLMResponse(text="answer"),
            deltas=["answer"],
        )
    ]
    llm_service = StubLLMService(streams)
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,
        executor=StubExecutor(),
    )

    events = [
        event
        async for event in agent.resume(
            "current",
            [
                LLMMessage.build(role="user", content="first"),
                LLMMessage.build(role="assistant", content="second"),
            ],
        )
    ]

    assert isinstance(events[-1], RunCompletedEvent)
    messages = llm_service.calls[0]["messages"]
    assert [message.role for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert messages[1].content[0]["data"] == "first"
    assert messages[2].content[0]["data"] == "second"
    assert messages[3].content[0]["data"] == "current"


@pytest.mark.anyio
async def test_agent_surfaces_media_events() -> None:
    media = LLMGeneratedMedia(type="image", mime_type="image/png", data=b"\x89")
    media_segment = LLMSegment(type="image", content="", media=media)
    streams = [
        [
            LLMStreamEvent(
                event_type="response.media",
                segments=[media_segment],
            ),
            LLMStreamEvent(
                event_type="response.completed",
                response=LLMResponse(text="done", segments=[media_segment]),
            ),
        ]
    ]
    llm_service = StubLLMService(streams)
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,
        executor=StubExecutor(),
    )

    events = await collect_events(agent, "Need an image")

    media_events = [event for event in events if isinstance(event, LLMMediaEvent)]
    assert media_events
    assert media_events[0].segments[0].media is media
    assert isinstance(events[-1], RunCompletedEvent)
    assert events[-1].final_answer == "done"


@pytest.mark.anyio
async def test_agent_handles_tool_call_and_continues_conversation() -> None:
    lookup_call = LLMToolCall(name="lookup", arguments="{}", call_id="tool-1")
    streams = [
        make_stream(
            response=LLMResponse(text="use lookup", tool_calls=[lookup_call]),
            deltas=["use ", "lookup"],
        ),
        make_stream(
            response=LLMResponse(text="answer after tool"),
            deltas=["answer after tool"],
        ),
    ]
    llm_service = StubLLMService(streams)
    executor = StubExecutor({"lookup": '{"value":42}'})
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,
        executor=executor,
        max_steps=3,
    )

    events = await collect_events(agent, "Need data")

    assert isinstance(events[-1], RunCompletedEvent)
    assert events[-1].final_answer == "answer after tool"
    assert len(llm_service.calls) == 2
    assert executor.calls[0].name == "lookup"
    tool_events = [event for event in events if isinstance(event, ToolStartedEvent)]
    assert tool_events
    assert tool_events[0].tool_name == "lookup"
    completed_steps = [
        event for event in events if isinstance(event, StepCompletedEvent)
    ]
    assert [event.step_index for event in completed_steps] == [0, 1]


@pytest.mark.anyio
async def test_agent_preserves_reasoning_content_during_tool_sub_turn() -> None:
    lookup_call = LLMToolCall(name="lookup", arguments="{}", call_id="tool-1")
    streams = [
        make_stream(
            response=LLMResponse(
                text="",
                tool_calls=[lookup_call],
                reasoning_content="need lookup first",
            )
        ),
        make_stream(response=LLMResponse(text="done"), deltas=["done"]),
    ]
    llm_service = StubLLMService(streams)
    executor = StubExecutor({"lookup": "tool result"})
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,
        executor=executor,
        max_steps=3,
    )

    events = await collect_events(agent, "Use lookup")

    assert isinstance(events[-1], RunCompletedEvent)
    assistant_message = llm_service.calls[1]["messages"][-2]
    assert assistant_message.reasoning_content == "need lookup first"


@pytest.mark.anyio
async def test_agent_does_not_complete_mid_step_after_tool_calls() -> None:
    lookup_call = LLMToolCall(name="lookup", arguments="{}", call_id="tool-1")
    streams = [
        make_stream(
            response=LLMResponse(text="use lookup", tool_calls=[lookup_call]),
            deltas=["use lookup"],
        ),
        make_stream(
            response=LLMResponse(text="done"),
            deltas=["done"],
        ),
    ]
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService(streams),
        executor=StubExecutor({"lookup": '{"value":42}'}),
        max_steps=2,
    )

    events = await collect_events(agent, "Finish up")

    assert isinstance(events[-1], RunCompletedEvent)
    assert events[-1].final_answer == "done"


@pytest.mark.anyio
async def test_agent_raises_after_exceeding_turn_limit() -> None:
    streams = [make_stream(response=LLMResponse(text=""), deltas=[])]
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService(streams),
        executor=StubExecutor(),
        max_steps=1,
    )

    events: list[AgentEvent] = []
    with pytest.raises(AceAIRuntimeError, match="exceeded maximum steps"):
        async for evt in agent.run("try again"):
            events.append(evt)

    assert isinstance(events[-1], RunFailedEvent)
    assert events[-1].error == "Agent exceeded maximum steps: 1 without answering"


@pytest.mark.anyio
async def test_agent_unset_max_steps_runs_until_answer() -> None:
    streams = [
        make_stream(response=LLMResponse(text=""), deltas=[]),
        make_stream(response=LLMResponse(text=""), deltas=[]),
        make_stream(response=LLMResponse(text="done"), deltas=["done"]),
    ]
    llm_service = StubLLMService(streams)
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,
        executor=StubExecutor(),
        max_steps=UNSET,
    )

    events = await collect_events(agent, "try again")

    assert len(llm_service.calls) == 3
    assert isinstance(events[-1], RunCompletedEvent)
    assert events[-1].final_answer == "done"


@pytest.mark.anyio
async def test_agent_can_return_structured_response() -> None:
    response = LLMResponse(text="final")
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([make_stream(response=response, deltas=["final"])]),
        executor=StubExecutor(),
    )

    events = await collect_events(agent, "Finish up")
    final_event = events[-1]

    assert isinstance(final_event, RunCompletedEvent)
    assert final_event.final_answer == "final"


@pytest.mark.anyio
async def test_agent_stream_emits_run_completed_event() -> None:
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService(
            [
                make_stream(
                    response=LLMResponse(text="  hello  "),
                    deltas=["  hello  "],
                )
            ]
        ),
        executor=StubExecutor(),
    )

    events = await collect_events(agent, "Question?")

    assert isinstance(events[0], LLMStartedEvent)
    assert isinstance(events[-1], RunCompletedEvent)
    assert events[-1].final_answer == "  hello  "


@pytest.mark.anyio
async def test_agent_emits_text_deltas_without_buffering_reasoning_log() -> None:
    stream = [
        LLMStreamEvent(
            event_type="response.output_text.delta",
            text_delta="foo",
        ),
        LLMStreamEvent(
            event_type="response.output_text.delta",
            text_delta="bar",
        ),
        LLMStreamEvent(
            event_type="response.completed",
            response=LLMResponse(text="foobar"),
        ),
    ]
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([stream]),
        executor=StubExecutor(),
    )

    events = await collect_events(agent, "Question?")
    deltas = [event for event in events if isinstance(event, LLMOutputDeltaEvent)]

    assert len(deltas) == 2
    assert deltas[0].text_delta == "foo"
    assert deltas[1].text_delta == "bar"
    final_event = events[-1]
    assert isinstance(final_event, RunCompletedEvent)
    assert final_event.step.reasoning_log == ""
    assert final_event.step.reasoning_log_truncated is False


@pytest.mark.anyio
async def test_reasoning_log_is_empty_when_no_deltas() -> None:
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService(
            [
                make_stream(
                    response=LLMResponse(text="final"),
                    deltas=[],
                )
            ]
        ),
        executor=StubExecutor(),
    )

    events = await collect_events(agent, "Question?")

    assert isinstance(events[-1], RunCompletedEvent)
    assert events[-1].step.reasoning_log == ""
    assert events[-1].final_answer == "final"


@pytest.mark.anyio
async def test_stream_error_emits_run_failed_event() -> None:
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService(
            [
                [
                    LLMStreamEvent(
                        event_type="response.error",
                        error="provider exploded",
                    )
                ]
            ]
        ),
        executor=StubExecutor(),
    )

    events = await collect_events(agent, "boom?")

    assert isinstance(events[-3], LLMStartedEvent)
    assert isinstance(events[-1], RunFailedEvent)
    assert events[-1].error == "provider exploded"


@pytest.mark.anyio
async def test_agent_emits_function_call_argument_deltas() -> None:
    stream = [
        LLMStreamEvent(
            event_type="response.function_call_arguments.delta",
            tool_call_delta=LLMToolCallDelta(id="tool-1", arguments_delta='{"a":'),
        ),
        LLMStreamEvent(
            event_type="response.output_text.delta",
            text_delta="hello ",
        ),
        LLMStreamEvent(
            event_type="response.output_text.delta",
            text_delta="world",
        ),
        LLMStreamEvent(
            event_type="response.completed",
            response=LLMResponse(text="hello world"),
        ),
    ]
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=SimpleLLMService(stream),
        executor=StubExecutor(),
    )

    events = await collect_events(agent, "Question?")
    tool_deltas = [
        event for event in events if isinstance(event, LLMToolCallDeltaEvent)
    ]

    assert len(tool_deltas) == 1
    assert tool_deltas[0].tool_call_delta.id == "tool-1"
    assert tool_deltas[0].text_delta == '{"a":'
    assert isinstance(events[-1], RunCompletedEvent)
    assert events[-1].final_answer == "hello world"


@pytest.mark.anyio
async def test_agent_emits_llm_retry_progress() -> None:
    stream = [
        LLMStreamEvent(
            event_type="response.retrying",
            error="RemoteProtocolError: peer closed",
            retry_count=1,
            retry_max=2,
            retry_delay_seconds=0.5,
        ),
        LLMStreamEvent(
            event_type="response.completed",
            response=LLMResponse(text="recovered"),
        ),
    ]
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=SimpleLLMService(stream),
        executor=StubExecutor(),
    )

    events = await collect_events(agent, "Question?")
    retry_events = [event for event in events if isinstance(event, LLMRetryingEvent)]

    assert len(retry_events) == 1
    assert retry_events[0].retry_count == 1
    assert retry_events[0].retry_max == 2
    assert retry_events[0].retry_delay_seconds == 0.5
    assert retry_events[0].error == "RemoteProtocolError: peer closed"


@pytest.mark.anyio
async def test_agent_uses_default_error_message_when_stream_error_missing_text() -> (
    None
):
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService(
            [
                [
                    LLMStreamEvent(
                        event_type="response.error",
                    )
                ]
            ]
        ),
        executor=StubExecutor(),
    )

    events = await collect_events(agent, "Boom?")

    assert isinstance(events[-3], LLMStartedEvent)
    assert isinstance(events[-1], RunFailedEvent)
    assert events[-1].error == "LLM streaming error"


@pytest.mark.anyio
async def test_agent_treats_failed_llm_completion_as_run_failure() -> None:
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService(
            [
                [
                    LLMStreamEvent(
                        event_type="response.completed",
                        response=LLMResponse(
                            text=(
                                "LLM request failed after retries. "
                                "Please try again later."
                            ),
                            status="failed",
                        ),
                    )
                ]
            ]
        ),
        executor=StubExecutor(),
    )

    events = await collect_events(agent, "Boom?")

    assert isinstance(events[-1], RunFailedEvent)
    assert events[-1].error == (
        "LLM request failed after retries. Please try again later."
    )


@pytest.mark.anyio
async def test_agent_errors_when_completion_event_lacks_response() -> None:
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService(
            [
                [
                    LLMStreamEvent(
                        event_type="response.completed",
                    )
                ]
            ]
        ),
        executor=StubExecutor(),
    )

    events: list[AgentEvent] = []
    with pytest.raises(
        AceAIRuntimeError, match="LLM stream completed without a response payload"
    ):
        async for evt in agent.run("Question?"):
            events.append(evt)

    assert isinstance(events[-1], LLMStartedEvent)
    assert not any(isinstance(evt, RunFailedEvent) for evt in events)


@pytest.mark.anyio
async def test_reasoning_segments_do_not_populate_reasoning_log() -> None:
    response = LLMResponse(
        text="done",
        segments=[
            LLMSegment(type="reasoning", content="first"),
            LLMSegment(type="reasoning", content="second"),
        ],
    )
    stream = make_stream(response=response, deltas=[])
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=SimpleLLMService(stream),
        executor=StubExecutor(),
    )

    events = await collect_events(agent, "Question?")
    final_event = events[-1]
    reasoning_events = [
        event for event in events if isinstance(event, LLMReasoningEvent)
    ]

    assert [event.segment.content for event in reasoning_events] == ["first", "second"]
    assert isinstance(final_event, RunCompletedEvent)
    assert final_event.step.reasoning_log == ""
    assert final_event.step.reasoning_log_truncated is False


@pytest.mark.anyio
async def test_agent_emits_streaming_reasoning_before_text() -> None:
    reasoning_segment = LLMSegment(
        type="reasoning",
        content="think first",
    )
    stream = [
        LLMStreamEvent(
            event_type="response.reasoning.delta",
            segments=[reasoning_segment],
        ),
        LLMStreamEvent(
            event_type="response.output_text.delta",
            text_delta="answer",
        ),
        LLMStreamEvent(
            event_type="response.completed",
            response=LLMResponse(
                text="answer",
                segments=[LLMSegment(type="reasoning", content="think first")],
            ),
        ),
    ]
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([stream]),
        executor=StubExecutor(),
    )

    events = await collect_events(agent, "Question")

    reasoning_index = next(
        index
        for index, event in enumerate(events)
        if isinstance(event, LLMReasoningEvent)
    )
    text_index = next(
        index
        for index, event in enumerate(events)
        if isinstance(event, LLMOutputDeltaEvent)
    )
    assert reasoning_index < text_index
    assert [event for event in events if isinstance(event, LLMReasoningEvent)][
        0
    ].segment is reasoning_segment
    assert len([event for event in events if isinstance(event, LLMReasoningEvent)]) == 1


@pytest.mark.anyio
async def test_agent_emits_hosted_tool_activity_before_text() -> None:
    hosted_tool_segment = LLMSegment(
        type="hosted_tool",
        content="Searching the web",
    )
    stream = [
        LLMStreamEvent(
            event_type="response.hosted_tool",
            segments=[hosted_tool_segment],
        ),
        LLMStreamEvent(
            event_type="response.output_text.delta",
            text_delta="answer",
        ),
        LLMStreamEvent(
            event_type="response.completed",
            response=LLMResponse(text="answer"),
        ),
    ]
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([stream]),
        executor=StubExecutor(),
    )

    events = await collect_events(agent, "Question")

    hosted_tool_index = next(
        index
        for index, event in enumerate(events)
        if isinstance(event, LLMHostedToolEvent)
    )
    text_index = next(
        index
        for index, event in enumerate(events)
        if isinstance(event, LLMOutputDeltaEvent)
    )
    assert hosted_tool_index < text_index
    assert [event for event in events if isinstance(event, LLMHostedToolEvent)][
        0
    ].segment is hosted_tool_segment


@pytest.mark.anyio
async def test_agent_returns_tool_execution_failure_to_model() -> None:
    call = LLMToolCall(name="calc", arguments="{}", call_id="calc-1")
    streams = [
        [
            LLMStreamEvent(
                event_type="response.completed",
                response=LLMResponse(text="use calc", tool_calls=[call]),
            )
        ],
        make_stream(response=LLMResponse(text="recovered"), deltas=["recovered"]),
    ]
    executor = RaisingExecutor(ToolExecutionError("no calc"))
    llm_service = StubLLMService(streams)
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,
        executor=executor,
    )

    events = await collect_events(agent, "Question?")

    tool_failed = [evt for evt in events if isinstance(evt, ToolFailedEvent)]
    assert tool_failed and tool_failed[0].error == "no calc"
    assert tool_failed[0].tool_result.error == "no calc"
    assert tool_failed[0].tool_result.output == "Tool execution failed: no calc"
    tool_message = llm_service.calls[1]["messages"][-1]
    assert tool_message.role == "tool"
    assert tool_message.call_id == "calc-1"
    assert tool_message.content[0]["data"] == "Tool execution failed: no calc"
    assert isinstance(events[-1], RunCompletedEvent)
    assert events[-1].final_answer == "recovered"


@pytest.mark.anyio
async def test_agent_reraises_unstructured_tool_failure() -> None:
    call = LLMToolCall(name="calc", arguments="{}", call_id="calc-1")
    stream = [
        LLMStreamEvent(
            event_type="response.completed",
            response=LLMResponse(text="use calc", tool_calls=[call]),
        )
    ]
    executor = RaisingExecutor(ValueError("broken executor"))
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([stream]),
        executor=executor,
    )

    events: list[AgentEvent] = []
    with pytest.raises(ValueError, match="broken executor"):
        async for event in agent.run("Question?"):
            events.append(event)

    assert isinstance(events[-1], RunFailedEvent)
    assert not [event for event in events if isinstance(event, ToolFailedEvent)]


@pytest.mark.anyio
async def test_agent_executes_all_tool_calls_in_step() -> None:
    first_call = LLMToolCall(name="lookup", arguments="{}", call_id="lookup-1")
    second_call = LLMToolCall(name="calc", arguments="{}", call_id="calc-1")
    streams = [
        [
            LLMStreamEvent(
                event_type="response.completed",
                response=LLMResponse(
                    text="tool outputs",
                    tool_calls=[first_call, second_call],
                ),
            )
        ],
        make_stream(
            response=LLMResponse(text="done"),
            deltas=["done"],
        ),
    ]
    executor = StubExecutor({"lookup": '{"value":42}', "calc": "3"})
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService(streams),
        executor=executor,
        max_steps=2,
    )

    events = await collect_events(agent, "Finish?")

    assert executor.calls == [first_call, second_call]
    assert isinstance(events[-1], RunCompletedEvent)
    assert events[-1].final_answer == "done"


@pytest.mark.anyio
async def test_agent_executes_approval_free_tool_calls_concurrently() -> None:
    first_call = LLMToolCall(name="lookup", arguments="{}", call_id="lookup-1")
    second_call = LLMToolCall(name="calc", arguments="{}", call_id="calc-1")
    streams = [
        [
            LLMStreamEvent(
                event_type="response.completed",
                response=LLMResponse(
                    text="tool outputs",
                    tool_calls=[first_call, second_call],
                ),
            )
        ],
        make_stream(
            response=LLMResponse(text="done"),
            deltas=["done"],
        ),
    ]
    executor = BlockingExecutor({"lookup": '{"value":42}', "calc": "3"})
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService(streams),
        executor=executor,
        max_steps=2,
    )

    events_task = asyncio.create_task(collect_events(agent, "Finish?"))
    await asyncio.wait_for(executor.both_started.wait(), timeout=0.2)
    executor.release.set()
    events = await events_task

    assert executor.calls == [first_call, second_call]
    assert isinstance(events[-1], RunCompletedEvent)
    assert events[-1].final_answer == "done"


@pytest.mark.anyio
async def test_agent_suspends_before_approval_required_tool() -> None:
    call = LLMToolCall(name="write_file", arguments='{"path":"x"}', call_id="write-1")
    streams = [
        [
            LLMStreamEvent(
                event_type="response.completed",
                response=LLMResponse(text="use write", tool_calls=[call]),
            )
        ],
    ]
    executor = StubExecutor(
        {"write_file": '{"ok":true}'},
        approval_required={"write_file"},
    )
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService(streams),
        executor=executor,
        max_steps=2,
    )
    run = agent.create_run("Write it")

    events = [event async for event in agent.execute(run)]

    requested = [
        event for event in events if isinstance(event, ToolApprovalRequestedEvent)
    ]
    suspended = [event for event in events if isinstance(event, RunSuspendedEvent)]
    assert requested
    assert suspended
    assert requested[0].request.call is call
    assert requested[0].request.tool_name == "write_file"
    assert run.status == "suspended"
    assert executor.calls == []


@pytest.mark.anyio
async def test_agent_resumes_approved_tool_and_continues_conversation() -> None:
    call = LLMToolCall(name="write_file", arguments='{"path":"x"}', call_id="write-1")
    streams = [
        [
            LLMStreamEvent(
                event_type="response.completed",
                response=LLMResponse(text="use write", tool_calls=[call]),
            )
        ],
        make_stream(response=LLMResponse(text="done"), deltas=["done"]),
    ]
    executor = StubExecutor(
        {"write_file": '{"ok":true}'},
        approval_required={"write_file"},
    )
    llm_service = StubLLMService(streams)
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,
        executor=executor,
        max_steps=2,
    )
    run = agent.create_run("Write it")
    suspend_events = [event async for event in agent.execute(run)]
    request = [
        event
        for event in suspend_events
        if isinstance(event, ToolApprovalRequestedEvent)
    ][0].request

    resume_events = [
        event
        async for event in agent.resume_approval(
            run, ToolApprovalDecision.approve(request)
        )
    ]

    resolved = [
        event for event in resume_events if isinstance(event, ToolApprovalResolvedEvent)
    ]
    assert resolved and resolved[0].decision.approved is True
    assert executor.calls == [call]
    assert isinstance(resume_events[-1], RunCompletedEvent)
    assert resume_events[-1].final_answer == "done"
    tool_message = llm_service.calls[1]["messages"][-1]
    assert tool_message.role == "tool"
    assert tool_message.content[0]["data"] == '{"ok":true}'


@pytest.mark.anyio
async def test_agent_reuses_approved_tool_name_in_same_run() -> None:
    first_call = LLMToolCall(
        name="write_file",
        arguments='{"path":"a"}',
        call_id="write-1",
    )
    second_call = LLMToolCall(
        name="write_file",
        arguments='{"path":"b"}',
        call_id="write-2",
    )
    streams = [
        [
            LLMStreamEvent(
                event_type="response.completed",
                response=LLMResponse(
                    text="use write",
                    tool_calls=[first_call, second_call],
                ),
            )
        ],
        make_stream(response=LLMResponse(text="done"), deltas=["done"]),
    ]
    executor = StubExecutor(
        {"write_file": '{"ok":true}'},
        approval_required={"write_file"},
    )
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService(streams),
        executor=executor,
        max_steps=2,
    )
    run = agent.create_run("Write both")
    suspend_events = [event async for event in agent.execute(run)]
    request = [
        event
        for event in suspend_events
        if isinstance(event, ToolApprovalRequestedEvent)
    ][0].request

    resume_events = [
        event
        async for event in agent.resume_approval(
            run, ToolApprovalDecision.approve(request)
        )
    ]

    assert executor.calls == [first_call, second_call]
    assert not [
        event for event in resume_events if isinstance(event, RunSuspendedEvent)
    ]
    assert isinstance(resume_events[-1], RunCompletedEvent)
    assert resume_events[-1].final_answer == "done"


@pytest.mark.anyio
async def test_agent_rejects_tool_and_returns_rejection_to_model() -> None:
    call = LLMToolCall(name="write_file", arguments='{"path":"x"}', call_id="write-1")
    streams = [
        [
            LLMStreamEvent(
                event_type="response.completed",
                response=LLMResponse(text="use write", tool_calls=[call]),
            )
        ],
        make_stream(response=LLMResponse(text="skipped"), deltas=["skipped"]),
    ]
    executor = StubExecutor(
        {"write_file": '{"ok":true}'},
        approval_required={"write_file"},
    )
    llm_service = StubLLMService(streams)
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,
        executor=executor,
        max_steps=2,
    )
    run = agent.create_run("Write it")
    suspend_events = [event async for event in agent.execute(run)]
    request = [
        event
        for event in suspend_events
        if isinstance(event, ToolApprovalRequestedEvent)
    ][0].request

    resume_events = [
        event
        async for event in agent.resume_approval(
            run,
            ToolApprovalDecision.reject(request, reason="not now"),
        )
    ]

    assert executor.calls == []
    failed = [event for event in resume_events if isinstance(event, ToolFailedEvent)]
    assert failed and failed[0].error == "not now"
    tool_message = llm_service.calls[1]["messages"][-1]
    assert tool_message.role == "tool"
    assert tool_message.content[0]["data"] == "Tool execution rejected: not now"
    assert isinstance(resume_events[-1], RunCompletedEvent)
    assert resume_events[-1].final_answer == "skipped"


@pytest.mark.anyio
async def test_suspended_agent_run_requires_approval_resume() -> None:
    call = LLMToolCall(name="write_file", arguments='{"path":"x"}', call_id="write-1")
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService(
            [
                [
                    LLMStreamEvent(
                        event_type="response.completed",
                        response=LLMResponse(text="use write", tool_calls=[call]),
                    )
                ],
            ]
        ),
        executor=StubExecutor(
            {"write_file": '{"ok":true}'},
            approval_required={"write_file"},
        ),
        max_steps=2,
    )
    run = agent.create_run("Write it")

    [event async for event in agent.execute(run)]

    with pytest.raises(AceAIRuntimeError, match="resume_approval"):
        async for _ in agent.execute(run):
            pass
    with pytest.raises(AceAIRuntimeError, match="does not match"):
        async for _ in agent.resume_approval(
            run,
            ToolApprovalDecision(call_id="other-call", approved=True),
        ):
            pass


@pytest.mark.anyio
async def test_agent_run_rethrows_when_no_steps_recorded() -> None:
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=RaisingStreamLLMService([], RuntimeError("preflight failure")),
        executor=StubExecutor(),
        max_steps=1,
    )

    with pytest.raises(RuntimeError, match="preflight failure"):
        async for _ in agent.run("Question?"):
            pass
