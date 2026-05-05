import pytest

from aceai.agent.app import AceAgentApp
from aceai.agent.session import SessionEvent, SessionStore
from aceai.core.base import AgentBase
from aceai.core.events import RunSuspendedEvent, ToolApprovalRequestedEvent
from aceai.llm import LLMResponse
from aceai.llm.models import LLMStreamEvent, LLMToolCall

from tests.test_agent_behavior import StubExecutor, StubLLMService, make_stream


@pytest.mark.anyio
async def test_agent_app_reuses_approved_tool_name_across_session_turns(tmp_path) -> None:
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
    llm_service = StubLLMService(
        [
            [
                LLMStreamEvent(
                    event_type="response.completed",
                    response=LLMResponse(text="use write", tool_calls=[first_call]),
                )
            ],
            make_stream(response=LLMResponse(text="first done"), deltas=["first done"]),
            [
                LLMStreamEvent(
                    event_type="response.completed",
                    response=LLMResponse(text="use write", tool_calls=[second_call]),
                )
            ],
            make_stream(response=LLMResponse(text="second done"), deltas=["second done"]),
        ]
    )
    executor = StubExecutor(
        {"write_file": '{"ok":true}'},
        approval_required={"write_file"},
    )
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,
        executor=executor,
        max_steps=2,
    )
    app = AceAgentApp(
        agent,
        provider_name="openai",
        selected_model="gpt-4o",
        session_store=SessionStore(tmp_path / "sessions"),
    )

    first_events = [event async for event in app.start_turn("Write one")]
    assert [event for event in first_events if isinstance(event, RunSuspendedEvent)]
    [event async for event in app.approve_tool()]

    second_events = [event async for event in app.start_turn("Write another")]

    assert executor.calls == [first_call, second_call]
    assert not [
        event for event in second_events if isinstance(event, ToolApprovalRequestedEvent)
    ]
    assert not [event for event in second_events if isinstance(event, RunSuspendedEvent)]


@pytest.mark.anyio
async def test_agent_app_seeds_approved_tool_cache_from_session_history(
    tmp_path,
) -> None:
    call = LLMToolCall(
        name="write_file",
        arguments='{"path":"b"}',
        call_id="write-2",
    )
    store = SessionStore(tmp_path / "sessions")
    metadata = store.create_session()
    store.append_event(
        metadata.session_id,
        SessionEvent(
            kind="tool_approval_resolved",
            payload={
                "content": "approved",
                "tool_name": "write_file",
                "tool_call_id": "write-1",
                "tool_call": {
                    "type": "function_call",
                    "name": "write_file",
                    "arguments": '{"path":"a"}',
                    "call_id": "write-1",
                },
            },
        ),
    )
    llm_service = StubLLMService(
        [
            [
                LLMStreamEvent(
                    event_type="response.completed",
                    response=LLMResponse(text="use write", tool_calls=[call]),
                )
            ],
            make_stream(response=LLMResponse(text="done"), deltas=["done"]),
        ]
    )
    executor = StubExecutor(
        {"write_file": '{"ok":true}'},
        approval_required={"write_file"},
    )
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,
        executor=executor,
        max_steps=2,
    )
    app = AceAgentApp(
        agent,
        provider_name="openai",
        selected_model="gpt-4o",
        session_store=store,
        session_id=metadata.session_id,
    )

    events = [event async for event in app.start_turn("Write another")]

    assert executor.calls == [call]
    assert not [event for event in events if isinstance(event, ToolApprovalRequestedEvent)]
    assert not [event for event in events if isinstance(event, RunSuspendedEvent)]
