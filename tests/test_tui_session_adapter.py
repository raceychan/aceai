from aceai.agent.session import SessionEvent, SessionMessage
from aceai.agent.tui.events import TUIEvent, user_message_event
from aceai.agent.tui.session_adapter import (
    session_messages_to_tui_events,
    tui_event_to_session_event,
)
from aceai.core.models import ToolExecutionResult
from aceai.llm.models import LLMToolCall, LLMUsage


def test_tui_event_to_session_event_keeps_storage_fields() -> None:
    usage = LLMUsage(input_tokens=1, output_tokens=2, total_tokens=3)
    event = TUIEvent(
        kind="llm_completed",
        step_index=0,
        step_id="step-1",
        title="llm completed",
        content="answer",
        usage=usage,
        raw_event=None,
    )

    session_event = tui_event_to_session_event(event)

    assert session_event == SessionEvent(
        kind="llm_completed",
        content="answer",
        usage=usage,
    )


def test_session_messages_to_tui_events_restores_display_transcript() -> None:
    created_at = "2026-05-04T00:00:00+00:00"
    events = session_messages_to_tui_events(
        [
            SessionMessage(kind="user", content="hello", created_at=created_at),
            SessionMessage(
                kind="assistant",
                content="answer",
                created_at=created_at,
                usage_input_tokens=1,
                usage_output_tokens=2,
                usage_total_tokens=3,
                cost_model="gpt-5.5",
                cost_input_usd=0.1,
                cost_cached_input_usd=0.0,
                cost_output_usd=0.2,
                cost_total_usd=0.3,
                cost_input_usd_per_million=1.0,
                cost_cached_input_usd_per_million=0.1,
                cost_output_usd_per_million=2.0,
                cost_pricing_source="test",
            ),
        ]
    )

    assert [event.kind for event in events] == ["user_message", "assistant_delta"]
    assert events[0].content == "hello"
    assert events[1].content == "answer"
    assert events[1].usage == LLMUsage(
        input_tokens=1,
        output_tokens=2,
        total_tokens=3,
    )
    assert events[1].cost is not None
    assert events[1].cost.total_cost_usd == 0.3


def test_session_messages_to_tui_events_restores_tool_message() -> None:
    events = session_messages_to_tui_events(
        [
            SessionMessage(
                kind="tool",
                content="completed",
                created_at="2026-05-04T00:00:00+00:00",
                tool_name="list_directory",
                tool_call_id="call-1",
                tool_arguments='{"path":"."}',
                tool_output='{"entries":[]}',
                status="completed",
            )
        ]
    )

    assert len(events) == 1
    event = events[0]
    assert event.kind == "tool_completed"
    assert event.tool_call == LLMToolCall(
        name="list_directory",
        arguments='{"path":"."}',
        call_id="call-1",
    )
    assert event.tool_result == ToolExecutionResult(
        call=event.tool_call,
        output='{"entries":[]}',
    )


def test_user_message_event_adapter_records_user_message() -> None:
    event = tui_event_to_session_event(user_message_event("hello"))

    assert event == SessionEvent(kind="user_message", content="hello")
