from aceai.agent.session import EventLog, SessionEvent
from aceai.agent.tui.events import TUIEvent
from aceai.agent.tui.session_adapter import tui_event_to_session_event
from aceai.agent.tui.session_replay import event_log_to_tui_events
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
        step_id="step-1",
        step_index=0,
        payload={
            "content": "answer",
            "usage": {
                "input_tokens": 1,
                "cached_input_tokens": None,
                "output_tokens": 2,
                "total_tokens": 3,
            },
        },
    )


def test_event_log_to_tui_events_restores_display_transcript() -> None:
    events = event_log_to_tui_events(
        EventLog(
            [
                SessionEvent(kind="user_message", payload={"content": "hello"}),
                SessionEvent(
                    kind="assistant_message",
                    payload={
                        "content": "answer",
                        "usage": {
                            "input_tokens": 1,
                            "cached_input_tokens": None,
                            "output_tokens": 2,
                            "total_tokens": 3,
                        },
                        "cost": {
                            "model": "gpt-5.5",
                            "input_cost_usd": 0.1,
                            "cached_input_cost_usd": 0.0,
                            "output_cost_usd": 0.2,
                            "total_cost_usd": 0.3,
                            "input_usd_per_million": 1.0,
                            "cached_input_usd_per_million": 0.1,
                            "output_usd_per_million": 2.0,
                            "pricing_source": "test",
                        },
                    },
                ),
            ]
        )
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


def test_event_log_to_tui_events_restores_tool_message() -> None:
    events = event_log_to_tui_events(
        EventLog(
            [
                SessionEvent(
                    kind="tool_result",
                    payload={
                        "content": "completed",
                        "tool_name": "list_directory",
                        "tool_call_id": "call-1",
                        "tool_arguments": '{"path":"."}',
                        "output": '{"entries":[]}',
                        "status": "completed",
                    },
                )
            ]
        )
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


def test_user_message_adapter_records_user_message() -> None:
    event = tui_event_to_session_event(TUIEvent.user_message("hello"))

    assert event == SessionEvent(
        kind="user_message",
        step_id=event.step_id,
        step_index=-1,
        payload={"content": "hello"},
    )
