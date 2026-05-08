import asyncio
from datetime import datetime, timezone
from io import StringIO

import pytest

from aceai.agent.ideas import Idea
from aceai.agent.session import EventLog, SessionEvent, SessionRecorder, SessionStore
from aceai.core.events import AgentEventBuilder
from aceai.core.models import AgentStep, ToolExecutionResult
from aceai.agent.tui import app as tui_app_module
from aceai.agent.tui.app import AceAITUI
from aceai.agent.tui.app import STREAM_DELTA_REFRESH_CHARS
from aceai.agent.tui.app import STREAM_DELTA_REFRESH_SECONDS
from aceai.agent.tui.demo import static_demo_events
from aceai.agent.tui.events import TUIEvent
from aceai.agent.tui.session_adapter import tui_event_to_session_event
from aceai.agent.tui.session_replay import event_log_to_tui_events
from aceai.agent.tui.setup import (
    SessionListWidget,
    _idea_body_text,
    _session_picker_renderables,
)
from aceai.agent.tui.state import (
    initial_state,
    reduce_events,
    reset_cache_rate,
    select_event,
)
from aceai.agent.tui.trajectory import TrajectoryScreen, _trajectory_renderables
from aceai.agent.tui.widgets import (
    CommandInput,
    DetailWidget,
    StreamWidget,
)
from aceai.llm.models import LLMResponse, LLMToolCall, LLMUsage
from rich.console import Console, Group
from textual.events import Click
from textual.widgets import Static


async def _wait_until(pilot, predicate, timeout: float = 0.2) -> None:
    async def wait_for_match() -> None:
        while not predicate():
            await pilot.pause()

    await asyncio.wait_for(wait_for_match(), timeout=timeout)
    assert predicate()


def test_reduce_events_tracks_run_completion() -> None:
    events = static_demo_events()

    state = reduce_events(events)

    assert state.status == "completed"
    assert state.final_answer == "Static TUI prototype is ready to inspect."
    assert state.error is None
    assert state.selected_event_id == events[-1].event_id


def test_reduce_events_tracks_step_and_tool_state() -> None:
    state = reduce_events(static_demo_events())

    assert len(state.steps) == 1
    step = state.steps[0]
    assert step.status == "completed"
    assert len(step.tools) == 1
    tool_state = step.tools[0]
    assert tool_state.name == "search_docs"
    assert tool_state.status == "completed"
    assert tool_state.arguments == '{"query":"aceai tui"}'
    assert tool_state.output == '{"matches":["spec/tui.md","docs/tui.md"]}'


def test_reduce_events_assigns_global_step_numbers() -> None:
    events = [
        TUIEvent.from_agent_event(
            AgentEventBuilder(step_index=0, step_id="run-1-step-1").llm_started()
        ),
        TUIEvent.from_agent_event(
            AgentEventBuilder(step_index=0, step_id="run-2-step-1").llm_started()
        ),
    ]

    state = reduce_events(events)

    assert [step.step_index for step in state.steps] == [0, 1]


def test_reduce_events_does_not_add_non_step_events_to_timeline() -> None:
    state = reduce_events(
        [
            TUIEvent.user_message("What changed?"),
            TUIEvent.session_notice("Sessions"),
        ]
    )

    assert state.steps == []
    assert state.events[0].kind == "user_message"
    assert state.events[1].kind == "session_notice"


def test_idea_picker_collapses_unselected_body_to_two_lines() -> None:
    idea = _idea(
        "title\nfirst line is visible\nsecond line is visible\nthird line is hidden"
    )

    collapsed = _idea_body_text(idea, expanded=False)
    expanded = _idea_body_text(idea, expanded=True)

    assert collapsed.plain.count("\n") == 1
    assert "first line is visible" in collapsed.plain
    assert "... more" in collapsed.plain
    assert "third line is hidden" not in collapsed.plain
    assert "third line is hidden" in expanded.plain


def test_idea_picker_short_body_keeps_two_line_height() -> None:
    collapsed = _idea_body_text(_idea("title"), expanded=False)
    title_only = _idea_body_text(_idea("title"), expanded=True)
    one_line = _idea_body_text(_idea("title\none line"), expanded=True)

    assert collapsed.plain == " \n "
    assert title_only.plain == " \n "
    assert one_line.plain == "one line\n "


def test_reduce_events_does_not_add_restored_transcript_to_timeline() -> None:
    events = event_log_to_tui_events(
        EventLog(
            [
                SessionEvent(kind="user_message", payload={"content": "question"}),
                SessionEvent(kind="assistant_message", payload={"content": "answer"}),
                SessionEvent(
                    kind="tool_result",
                    payload={
                        "content": "",
                        "tool_name": "read_text_file",
                        "tool_call_id": "call-1",
                        "tool_arguments": '{"path":"README.md"}',
                        "output": "contents",
                        "status": "completed",
                    },
                ),
            ]
        )
    )

    state = reduce_events(events)

    assert state.steps == []
    assert state.status == "idle"


def test_reduce_events_coalesces_consecutive_assistant_deltas() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    events = [
        TUIEvent.from_agent_event(builder.llm_text_delta(text_delta="hello ")),
        TUIEvent.from_agent_event(builder.llm_text_delta(text_delta="world")),
    ]

    state = reduce_events(events)

    assert len(state.events) == 1
    assert state.events[0].kind == "assistant_delta"
    assert state.events[0].content == "hello world"
    assert len(state.steps[0].events) == 1
    assert state.steps[0].events[0].content == "hello world"


def test_initial_state_is_idle() -> None:
    state = initial_state()

    assert state.status == "idle"
    assert state.steps == []
    assert state.events == []


def test_select_event_updates_selected_event_id() -> None:
    events = static_demo_events()
    state = reduce_events(events)
    tool_event = _first_event(events, "tool_completed")

    selected = select_event(state, tool_event.event_id)

    assert selected.selected_event_id == tool_event.event_id


def test_select_event_rejects_unknown_event_id() -> None:
    state = reduce_events(static_demo_events())

    with pytest.raises(ValueError, match="selected event does not exist"):
        select_event(state, "missing-event")


def test_reduce_events_tracks_usage_totals() -> None:
    first = AgentEventBuilder(step_index=0, step_id="step-1").llm_completed(
        step=AgentStep(
            llm_response=LLMResponse(
                text="first",
                usage=LLMUsage(
                    input_tokens=100,
                    cached_input_tokens=40,
                    output_tokens=20,
                    total_tokens=120,
                ),
            )
        )
    )
    second = AgentEventBuilder(step_index=1, step_id="step-2").llm_completed(
        step=AgentStep(
            llm_response=LLMResponse(
                text="second",
                usage=LLMUsage(input_tokens=150, output_tokens=30, total_tokens=180),
            )
        )
    )

    state = reduce_events(
        [TUIEvent.from_agent_event(first), TUIEvent.from_agent_event(second)]
    )

    assert state.usage.current_context_tokens == 150
    assert state.usage.session_input_tokens == 250
    assert state.usage.session_cached_input_tokens == 40
    assert state.usage.session_output_tokens == 50
    assert state.usage.session_total_tokens == 300


def test_reset_cache_rate_clears_current_cache_only() -> None:
    event = AgentEventBuilder(step_index=0, step_id="step-1").llm_completed(
        step=AgentStep(
            llm_response=LLMResponse(
                text="first",
                usage=LLMUsage(
                    input_tokens=100,
                    cached_input_tokens=40,
                    cache_miss_input_tokens=60,
                    input_cache_hit_rate=0.4,
                    output_tokens=20,
                    total_tokens=120,
                ),
            )
        )
    )
    state = reduce_events([TUIEvent.from_agent_event(event)])

    next_state = reset_cache_rate(state)

    assert next_state.usage.current_context_tokens == 100
    assert next_state.usage.current_cached_input_tokens == 0
    assert next_state.usage.current_input_cache_hit_rate == 0.0
    assert next_state.usage.session_cached_input_tokens == 40
    assert next_state.usage.session_total_tokens == 120


def test_reduce_events_tracks_cost_totals() -> None:
    first = AgentEventBuilder(step_index=0, step_id="step-1").llm_completed(
        step=AgentStep(
            llm_response=LLMResponse(
                text="first",
                model="gpt-5.5",
                usage=LLMUsage(
                    input_tokens=1_000,
                    cached_input_tokens=600,
                    output_tokens=100,
                    total_tokens=1_100,
                ),
            )
        )
    )
    second = AgentEventBuilder(step_index=1, step_id="step-2").llm_completed(
        step=AgentStep(
            llm_response=LLMResponse(
                text="second",
                model="gpt-5.4-mini",
                usage=LLMUsage(
                    input_tokens=1_000, output_tokens=100, total_tokens=1_100
                ),
            )
        )
    )

    state = reduce_events(
        [TUIEvent.from_agent_event(first), TUIEvent.from_agent_event(second)]
    )

    assert state.usage.current_cost_usd is not None
    assert state.usage.session_cost_usd is not None
    assert round(state.usage.current_cost_usd, 6) == 0.0012
    assert round(state.usage.session_cost_usd, 6) == 0.0065


def test_reduce_events_keeps_missing_usage_unknown() -> None:
    state = reduce_events(
        [
            TUIEvent.from_agent_event(
                AgentEventBuilder(step_index=0, step_id="step-1").llm_completed(
                    step=AgentStep(llm_response=LLMResponse(text="answer"))
                )
            )
        ]
    )

    assert state.usage.current_context_tokens is None
    assert state.usage.session_input_tokens is None
    assert state.usage.session_output_tokens is None
    assert state.usage.session_total_tokens is None


def test_session_notification_uses_native_toast() -> None:
    app = AceAITUI([])
    calls: list[tuple[str, dict[str, object]]] = []
    app.notify = lambda message, **kwargs: calls.append((message, kwargs))

    app.notify_session("Resumed session abc")

    assert calls == [
        (
            "Resumed session abc",
            {
                "title": "AceAI",
                "severity": "information",
                "timeout": 3.0,
            },
        )
    ]


def test_llm_retrying_uses_native_notification_without_stream_event() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    app = AceAITUI([])
    calls: list[tuple[str, dict[str, object]]] = []
    app.notify = lambda message, **kwargs: calls.append((message, kwargs))

    app.append_agent_event(
        builder.llm_retrying(
            retry_count=1,
            retry_max=2,
            retry_delay_seconds=0.5,
            error="RemoteProtocolError: peer closed",
        )
    )

    assert app._state.events == []
    assert calls == [
        (
            "Retrying LLM request 1/2 in 0.5s after RemoteProtocolError: peer closed",
            {
                "title": "Retrying LLM",
                "severity": "warning",
                "timeout": 3.0,
            },
        )
    ]


@pytest.mark.anyio
async def test_static_tui_loads_fixture_events() -> None:
    events = static_demo_events()
    app = AceAITUI(events)

    async with app.run_test():
        assert app._state.status == "completed"
        stream = app.query_one(StreamWidget)
        detail = app.query_one(DetailWidget)
        assert stream.can_focus
        assert detail.can_focus
        assert not stream.debug_mode
        assert detail.has_class("collapsed")


@pytest.mark.anyio
async def test_debug_mode_stream_selection_opens_tool_result_detail() -> None:
    events = static_demo_events()
    tool_event = _first_event(events, "tool_completed")
    app = AceAITUI(events)

    async with app.run_test() as pilot:
        stream = app.query_one(StreamWidget)
        stream.post_message(StreamWidget.EventSelected(tool_event.event_id))
        await pilot.pause()

        detail = app.query_one(DetailWidget)

        assert app._state.selected_event_id == tool_event.event_id
        assert not detail.has_class("collapsed")


@pytest.mark.anyio
async def test_debug_mode_reuses_message_panel_and_opens_selected_detail() -> None:
    app = AceAITUI(static_demo_events())

    async with app.run_test() as pilot:
        stream = app.query_one(StreamWidget)
        detail = app.query_one(DetailWidget)

        await pilot.press("d")

        assert stream.debug_mode
        assert stream.has_focus
        assert not detail.has_class("collapsed")
        assert app._state.selected_event_id is not None


@pytest.mark.anyio
async def test_debug_mode_can_move_selection_inside_message_panel() -> None:
    app = AceAITUI(static_demo_events())

    async with app.run_test() as pilot:
        stream = app.query_one(StreamWidget)

        await pilot.press("d")
        first_selected = app._state.selected_event_id
        await pilot.press("down")

        assert stream.debug_mode
        assert app._state.selected_event_id != first_selected


@pytest.mark.anyio
async def test_debug_mode_click_selects_message_in_main_stream() -> None:
    app = AceAITUI(static_demo_events())

    async with app.run_test() as pilot:
        stream = app.query_one(StreamWidget)

        await pilot.press("d")
        assert len(stream._debug_line_spans) >= 2
        target = stream._debug_line_spans[1]
        stream.on_click(
            Click(
                stream,
                x=1,
                y=target.start_line - stream.scroll_y,
                delta_x=0,
                delta_y=0,
                button=1,
                shift=False,
                meta=False,
                ctrl=False,
            )
        )
        await pilot.pause()

        assert app._state.selected_event_id == target.event_id
        assert not app.query_one(DetailWidget).has_class("collapsed")


@pytest.mark.anyio
async def test_trajectory_screen_renders_event_trajectory() -> None:
    call = LLMToolCall(
        name="write_text_file",
        arguments='{"path":"letters.txt","content":"abc"}',
        call_id="call-1",
    )
    events = [
        TUIEvent.user_message("write a file"),
        TUIEvent(
            kind="step_started",
            step_index=0,
            step_id="step-1",
            title="step started",
            raw_event=None,
        ),
        TUIEvent(
            kind="assistant_delta",
            step_index=0,
            step_id="step-1",
            title="assistant",
            content="I will write it.",
            raw_event=None,
        ),
        TUIEvent(
            kind="tool_started",
            step_index=0,
            step_id="step-1",
            title="tool write_text_file",
            tool_name="write_text_file",
            tool_call_id=call.call_id,
            tool_call=call,
            raw_event=None,
        ),
        TUIEvent(
            kind="tool_approval_requested",
            step_index=0,
            step_id="step-1",
            title="tool write_text_file approval",
            content="approval required",
            tool_name="write_text_file",
            tool_call_id=call.call_id,
            tool_call=call,
            raw_event=None,
        ),
        TUIEvent(
            kind="tool_completed",
            step_index=0,
            step_id="step-1",
            title="tool write_text_file completed",
            content='{"path":"letters.txt","bytes_written":3}',
            tool_name="write_text_file",
            tool_call_id=call.call_id,
            tool_call=call,
            tool_result=ToolExecutionResult(
                call=call,
                output='{"path":"letters.txt","bytes_written":3}',
            ),
            raw_event=None,
        ),
        TUIEvent(
            kind="step_completed",
            step_index=0,
            step_id="step-1",
            title="step completed",
            raw_event=None,
        ),
        TUIEvent(
            kind="run_completed",
            step_index=0,
            step_id="step-1",
            title="run completed",
            content="I will write it.",
            raw_event=None,
        ),
        TUIEvent.user_message("what happened?"),
        TUIEvent(
            kind="step_started",
            step_index=1,
            step_id="step-2",
            title="step started",
            raw_event=None,
        ),
        TUIEvent(
            kind="run_completed",
            step_index=1,
            step_id="step-2",
            title="run completed",
            content="The file was written.",
            raw_event=None,
        ),
    ]
    app = AceAITUI(events)

    async with app.run_test() as pilot:
        app.open_trajectory_screen()
        await pilot.pause()

        screen = app.screen
        assert isinstance(screen, TrajectoryScreen)

        rendered = _render_to_text(Group(*_trajectory_renderables(events)))

        assert "Trajectory" in rendered
        assert "turns" in rendered
        assert "steps" in rendered
        assert "events" in rendered
        assert "tool calls" in rendered
        assert "approvals" in rendered
        assert "failures" in rendered
        assert "T   2" not in rendered
        assert "E   11" not in rendered
        assert " 1  write a file" in rendered
        assert " 2  what happened?" in rendered
        assert "▌ 1" in rendered
        assert "▌ 2" in rendered
        assert "│ call" in rendered
        assert "└ result" in rendered
        assert "◆" in rendered
        assert "answer" in rendered
        assert "TURN" not in rendered
        assert "QUESTION" not in rendered
        assert "STEP" not in rendered
        assert "OUTCOME" not in rendered
        assert "write a file" in rendered
        assert "what happened?" in rendered
        assert "I will write it." in rendered
        assert rendered.count("I will write it.") == 1
        assert "approval required" in rendered
        assert "write_text_file" in rendered
        assert "wrote 3 bytes" in rendered
        assert "The file was written." in rendered
        assert "I will write it. - I will write it." not in rendered


def test_trajectory_summarizes_shell_tool_result_output() -> None:
    call = LLMToolCall(
        name="run_shell_command",
        arguments='{"command":"ls","cwd":"/tmp","timeout_seconds":10}',
        call_id="call-shell",
    )
    events = [
        TUIEvent.user_message("run ls"),
        TUIEvent(
            kind="step_started",
            step_index=0,
            step_id="step-shell",
            title="step started",
            raw_event=None,
        ),
        TUIEvent(
            kind="tool_started",
            step_index=0,
            step_id="step-shell",
            title="tool run_shell_command",
            tool_name="run_shell_command",
            tool_call_id=call.call_id,
            tool_call=call,
            raw_event=None,
        ),
        TUIEvent(
            kind="tool_completed",
            step_index=0,
            step_id="step-shell",
            title="tool run_shell_command completed",
            content='{"command":"ls","cwd":"/tmp","exit_code":0,"stdout":"a.py\\nb.py\\n","stderr":""}',
            tool_name="run_shell_command",
            tool_call_id=call.call_id,
            tool_call=call,
            tool_result=ToolExecutionResult(
                call=call,
                output='{"command":"ls","cwd":"/tmp","exit_code":0,"stdout":"a.py\\nb.py\\n","stderr":""}',
            ),
            raw_event=None,
        ),
    ]

    rendered = _render_to_text(Group(*_trajectory_renderables(events)))

    assert "$ ls" in rendered
    assert "a.py" in rendered
    assert '"stdout"' not in rendered
    assert '"exit_code"' not in rendered


def test_trajectory_renders_plain_text_tool_result_output() -> None:
    call = LLMToolCall(
        name="replace_text_in_file",
        arguments='{"path":"tests/test_ace_agent.py"}',
        call_id="call-replace",
    )
    events = [
        TUIEvent.user_message("patch file"),
        TUIEvent(
            kind="tool_completed",
            step_index=0,
            step_id="step-replace",
            title="tool replace_text_in_file completed",
            content=(
                "the tool replace_text_in_file exceeds its max calls in this run, "
                "do not call it again"
            ),
            tool_name="replace_text_in_file",
            tool_call_id=call.call_id,
            tool_call=call,
            tool_result=ToolExecutionResult(
                call=call,
                output=(
                    "the tool replace_text_in_file exceeds its max calls in this run, "
                    "do not call it again"
                ),
            ),
            raw_event=None,
        ),
    ]

    rendered = _render_to_text(Group(*_trajectory_renderables(events)))

    assert "replace_text_in_file" in rendered
    assert "exceeds its max calls" in rendered


def test_trajectory_does_not_repeat_streamed_answer_on_llm_completion() -> None:
    events = [
        TUIEvent.user_message("explain it"),
        TUIEvent(
            kind="step_started",
            step_index=0,
            step_id="step-stream",
            title="step started",
            raw_event=None,
        ),
        TUIEvent(
            kind="assistant_delta",
            step_index=0,
            step_id="step-stream",
            title="assistant",
            content="streamed answer",
            raw_event=None,
        ),
        TUIEvent(
            kind="llm_completed",
            step_index=0,
            step_id="step-stream",
            title="llm completed",
            content="streamed answer",
            raw_event=None,
        ),
        TUIEvent(
            kind="step_completed",
            step_index=0,
            step_id="step-stream",
            title="step completed",
            raw_event=None,
        ),
    ]

    rendered = _render_to_text(Group(*_trajectory_renderables(events)))

    assert rendered.count("streamed answer") == 1
    assert "llm completed" in rendered


def test_trajectory_renders_session_notices_without_running_step() -> None:
    events = [
        TUIEvent.session_notice("Resumed session abc"),
        TUIEvent.session_notice("Switched model to gpt-5.5"),
        TUIEvent.user_message("hello"),
    ]

    rendered = _render_to_text(Group(*_trajectory_renderables(events)))

    assert "session" in rendered
    assert "Resumed session abc" in rendered
    assert "Switched model to gpt-5.5" in rendered
    assert "running" not in rendered
    assert "▌ -" not in rendered


def test_trajectory_marks_multiline_preview_as_truncated() -> None:
    events = [
        TUIEvent.user_message("show result"),
        TUIEvent(
            kind="step_started",
            step_index=0,
            step_id="step-show",
            title="step started",
            raw_event=None,
        ),
        TUIEvent(
            kind="assistant_delta",
            step_index=0,
            step_id="step-show",
            title="assistant",
            content="结果如下：\n\naceai\nAGENTS.md\nREADME.md",
            raw_event=None,
        ),
    ]

    rendered = _render_to_text(Group(*_trajectory_renderables(events)))

    assert "结果如下：" in rendered
    assert "... (+4 lines)" in rendered
    assert "aceai\n" not in rendered
    assert "AGENTS.md" not in rendered
    assert "README.md" not in rendered


def test_trajectory_distinguishes_rejected_approval_from_tool_failure() -> None:
    call = LLMToolCall(
        name="write_text_file",
        arguments='{"path":"x","content":"hello"}',
        call_id="call-rejected",
    )
    events = [
        TUIEvent.user_message("write it"),
        TUIEvent(
            kind="step_started",
            step_index=0,
            step_id="step-rejected",
            title="step started",
            raw_event=None,
        ),
        TUIEvent(
            kind="tool_approval_requested",
            step_index=0,
            step_id="step-rejected",
            title="tool write_text_file approval",
            content="Tool 'write_text_file' requires approval",
            tool_name="write_text_file",
            tool_call_id=call.call_id,
            tool_call=call,
            raw_event=None,
        ),
        TUIEvent(
            kind="tool_approval_resolved",
            step_index=0,
            step_id="step-rejected",
            title="tool write_text_file approval resolved",
            content="rejected: rejected by caller",
            tool_name="write_text_file",
            tool_call_id=call.call_id,
            tool_call=call,
            raw_event=None,
        ),
        TUIEvent(
            kind="tool_failed",
            step_index=0,
            step_id="step-rejected",
            title="tool write_text_file failed",
            content="rejected by caller",
            tool_name="write_text_file",
            tool_call_id=call.call_id,
            tool_call=call,
            tool_result=ToolExecutionResult(
                call=call,
                output="Tool execution rejected: rejected by caller",
                error="rejected by caller",
            ),
            error="rejected by caller",
            raw_event=None,
        ),
    ]

    rendered = _render_to_text(Group(*_trajectory_renderables(events)))

    assert "rejected    1" in rendered
    assert "failures    0" in rendered
    assert "rejected" in rendered


def test_trajectory_marks_suspended_step_as_waiting() -> None:
    call = LLMToolCall(
        name="run_shell_command",
        arguments='{"command":"rm x","cwd":".","timeout_seconds":10}',
        call_id="call-waiting",
    )
    events = [
        TUIEvent.user_message("delete it"),
        TUIEvent(
            kind="step_started",
            step_index=0,
            step_id="step-waiting",
            title="step started",
            raw_event=None,
        ),
        TUIEvent(
            kind="assistant_delta",
            step_index=0,
            step_id="step-waiting",
            title="assistant",
            content="I need approval.",
            raw_event=None,
        ),
        TUIEvent(
            kind="run_suspended",
            step_index=0,
            step_id="step-waiting",
            title="run suspended",
            content="waiting for approval. Choose Approve or Reject.",
            tool_name="run_shell_command",
            tool_call_id=call.call_id,
            tool_call=call,
            raw_event=None,
        ),
    ]

    rendered = _render_to_text(Group(*_trajectory_renderables(events)))

    assert "waiting" in rendered
    assert "running  step-w" not in rendered


@pytest.mark.anyio
async def test_detail_renders_tool_arguments_and_output() -> None:
    call = LLMToolCall(
        name="search_docs",
        arguments='{"query":"aceai tui"}',
        call_id="call-1234567890",
    )
    tool_event = TUIEvent(
        kind="tool_completed",
        step_index=0,
        step_id="step-1",
        title="tool search_docs",
        raw_event=None,
        content='{"matches":["spec/tui.md","docs/tui.md"]}',
        tool_name="search_docs",
        tool_call_id=call.call_id,
        tool_call=call,
        tool_result=ToolExecutionResult(
            call=call,
            output='{"matches":["spec/tui.md","docs/tui.md"]}',
        ),
    )
    app = AceAITUI([tool_event])

    async with app.run_test():
        app._state = select_event(app._state, tool_event.event_id)
        detail = app.query_one(DetailWidget)
        detail.set_state(app._state)

        rendered = _render_to_text(detail.render())

        assert '"query": "aceai tui"' in rendered
        assert '{"matches":["spec/tui.md","docs/tui.md"]}' in rendered
        assert "TOOL CALL" in rendered
        assert "RESULT" in rendered
        assert "arguments{" not in rendered
        assert "output{" not in rendered


@pytest.mark.anyio
async def test_detail_omits_empty_raw_event_section() -> None:
    event = TUIEvent.user_message("show the readable part")
    app = AceAITUI([event])

    async with app.run_test():
        app._state = select_event(app._state, event.event_id)
        detail = app.query_one(DetailWidget)
        detail.set_state(app._state)

        rendered = _render_to_text(detail.render())

        assert "CONTENT" in rendered
        assert "show the readable part" in rendered
        assert "raw event" not in rendered
        assert "None" not in rendered


@pytest.mark.anyio
async def test_detail_shortens_long_ids() -> None:
    event = TUIEvent(
        kind="session_notice",
        step_index=0,
        step_id="12345678-1234-1234-1234-123456789abc",
        title="session",
        content="notice",
        raw_event=None,
    )
    app = AceAITUI([event])

    async with app.run_test():
        app._state = select_event(app._state, event.event_id)
        detail = app.query_one(DetailWidget)
        detail.set_state(app._state)

        rendered = _render_to_text(detail.render())

        assert "12345678...9abc" in rendered
        assert "12345678-1234-1234-1234-123456789abc" not in rendered


@pytest.mark.anyio
async def test_detail_renders_errors_as_separate_section() -> None:
    event = TUIEvent(
        kind="run_failed",
        step_index=0,
        step_id="step-1",
        title="run failed",
        content="the run failed",
        error="boom",
        raw_event=None,
    )
    app = AceAITUI([event])

    async with app.run_test():
        app._state = select_event(app._state, event.event_id)
        detail = app.query_one(DetailWidget)
        detail.set_state(app._state)

        rendered = _render_to_text(detail.render())

        assert "ERROR" in rendered
        assert "boom" in rendered


@pytest.mark.anyio
async def test_tui_batches_small_stream_delta_refreshes() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    app = AceAITUI([])

    async with app.run_test() as pilot:
        refreshes: list[int] = []

        def fake_refresh_widgets() -> None:
            refreshes.append(len(app._state.events))

        app._refresh_widgets = fake_refresh_widgets
        app.append_event(
            TUIEvent.from_agent_event(builder.llm_text_delta(text_delta="hello "))
        )
        app.append_event(
            TUIEvent.from_agent_event(builder.llm_text_delta(text_delta="world"))
        )

        assert refreshes == [1]
        assert len(app._state.events) == 1
        assert app._state.events[0].content == "hello "

        await _wait_until(
            pilot,
            lambda: len(refreshes) == 2,
            timeout=STREAM_DELTA_REFRESH_SECONDS * 5,
        )

        assert refreshes == [1, 1]
        assert app._state.events[0].content == "hello world"

        app.append_event(
            TUIEvent.from_agent_event(
                builder.llm_text_delta(text_delta="x" * STREAM_DELTA_REFRESH_CHARS)
            )
        )

        assert refreshes == [1, 1, 1]
        assert len(app._state.events) == 1
        assert app._state.events[0].content == "hello world" + (
            "x" * STREAM_DELTA_REFRESH_CHARS
        )


@pytest.mark.anyio
async def test_tui_refreshes_first_stream_delta_immediately() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    app = AceAITUI([])

    async with app.run_test():
        refreshes: list[int] = []

        def fake_refresh_widgets() -> None:
            refreshes.append(len(app._state.events))

        app._refresh_widgets = fake_refresh_widgets
        app.append_event(
            TUIEvent.from_agent_event(builder.llm_text_delta(text_delta="h"))
        )

        assert refreshes == [1]
        assert app._state.events[0].content == "h"


@pytest.mark.anyio
async def test_tui_flushes_pending_stream_delta_on_completion() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    step = AgentStep(llm_response=LLMResponse(text="done"))
    app = AceAITUI([])

    async with app.run_test():
        refreshes: list[int] = []

        def fake_refresh_widgets() -> None:
            refreshes.append(len(app._state.events))

        app._refresh_widgets = fake_refresh_widgets
        app.append_event(
            TUIEvent.from_agent_event(builder.llm_text_delta(text_delta="done"))
        )
        app.append_event(
            TUIEvent.from_agent_event(
                builder.run_completed(step=step, final_answer="done")
            )
        )

        assert refreshes == [1, 2]


@pytest.mark.anyio
async def test_stream_scrolls_to_latest_content() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    events = [TUIEvent.user_message("Show me many lines")]
    for line_number in range(80):
        events.append(
            TUIEvent.from_agent_event(
                builder.llm_text_delta(text_delta=f"line {line_number}\n")
            )
        )
    app = AceAITUI(events)

    async with app.run_test(size=(80, 20)) as pilot:
        stream = app.query_one(StreamWidget)
        await _wait_until(pilot, lambda: stream.scroll_y == stream.max_scroll_y)

        assert stream.max_scroll_y > 0
        assert stream.scroll_y == stream.max_scroll_y


@pytest.mark.anyio
async def test_stream_does_not_show_horizontal_scrollbar() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    events = [
        TUIEvent.from_agent_event(
            builder.llm_text_delta(
                text_delta=(
                    "This is a long assistant response that should wrap inside "
                    "the stream instead of creating a horizontal scrollbar. " * 8
                )
            )
        )
    ]
    app = AceAITUI(events)

    async with app.run_test(size=(60, 20)) as pilot:
        await _wait_until(
            pilot,
            lambda: app.query_one(StreamWidget).max_scroll_x == 0,
        )
        stream = app.query_one(StreamWidget)

        assert not stream.show_horizontal_scrollbar
        assert stream.max_scroll_x == 0


@pytest.mark.anyio
async def test_escape_exits_input_mode_and_returns_focus_to_stream() -> None:
    app = AceAITUI([])

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        stream = app.query_one(StreamWidget)
        await pilot.pause()
        app.set_focus(command_input)
        await pilot.pause()

        assert command_input.has_focus

        await pilot.press("escape")

        assert not command_input.has_focus
        assert stream.has_focus


@pytest.mark.anyio
async def test_enter_from_main_view_focuses_input() -> None:
    app = AceAITUI([])

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        stream = app.query_one(StreamWidget)
        stream.focus()
        await pilot.pause()

        assert stream.has_focus

        await pilot.press("enter")

        assert command_input.has_focus


@pytest.mark.anyio
async def test_input_sits_at_bottom_without_footer() -> None:
    app = AceAITUI([])

    async with app.run_test(size=(80, 20)) as pilot:
        await _wait_until(
            pilot,
            lambda: app.query_one(CommandInput).region.height == 4,
        )
        command_input = app.query_one(CommandInput)

        assert command_input.region.height == 4
        assert command_input.region.y + command_input.region.height <= app.size.height


@pytest.mark.anyio
async def test_tui_header_uses_session_id(tmp_path) -> None:
    store = SessionStore(tmp_path)
    metadata = store.create_session()
    app = AceAITUI(
        [],
        session_recorder=SessionRecorder(store, metadata.session_id),
        session_id=metadata.session_id,
    )

    async with app.run_test():
        assert app.title == f"AceAI {metadata.project_name} {metadata.session_id}"


@pytest.mark.anyio
async def test_empty_tui_exit_does_not_create_session_store(monkeypatch) -> None:
    class FailingSessionStore:
        def __init__(self) -> None:
            raise AssertionError("empty TUI exit should not touch the session store")

    monkeypatch.setattr(tui_app_module, "SessionStore", FailingSessionStore)
    app = AceAITUI([])

    async with app.run_test():
        pass


@pytest.mark.anyio
async def test_tui_can_show_and_switch_sessions(tmp_path) -> None:
    store = SessionStore(tmp_path)
    first = store.create_session()
    second = store.create_session()
    _record_user_message(store, first.session_id, "first question")
    _record_user_message(store, second.session_id, "second question")
    app = AceAITUI(
        event_log_to_tui_events(store.load_event_log(first.session_id)),
        session_recorder=SessionRecorder(store, first.session_id),
        session_id=first.session_id,
    )

    async with app.run_test():
        app.show_sessions()

        assert app._state.events[-1].kind == "session_notice"
        assert "Total cost: $0.000000" in app._state.events[-1].content
        assert first.session_id in app._state.events[-1].content
        assert second.session_id in app._state.events[-1].content

        app.switch_session(second.session_id)

        assert app.title == f"AceAI {second.project_name} {second.session_id}"
        assert app._state.events[0].content == "second question"


@pytest.mark.anyio
async def test_tui_replayed_incomplete_session_does_not_show_running(tmp_path) -> None:
    store = SessionStore(tmp_path)
    session = store.create_session()
    store.append_event(
        session.session_id,
        SessionEvent(
            kind="thinking_delta",
            step_id="step-1",
            step_index=0,
            payload={"content": "thinking"},
        ),
    )
    store.append_event(
        session.session_id,
        SessionEvent(
            kind="assistant_message",
            step_id="step-1",
            step_index=0,
            payload={"content": "partial answer"},
        ),
    )
    app = AceAITUI(
        event_log_to_tui_events(store.load_event_log(session.session_id)),
        session_recorder=SessionRecorder(store, session.session_id),
        session_id=session.session_id,
    )

    async with app.run_test():
        assert app._state.status == "idle"


@pytest.mark.anyio
async def test_tui_switching_to_current_session_is_noop_for_empty_session(
    tmp_path,
) -> None:
    store = SessionStore(tmp_path)
    current = store.create_session()
    app = AceAITUI(
        [],
        session_recorder=SessionRecorder(store, current.session_id),
        session_id=current.session_id,
    )

    async with app.run_test():
        app.switch_session(current.session_id)

        assert app._session_id == current.session_id
        assert store.get_session(current.session_id).session_id == current.session_id


@pytest.mark.anyio
async def test_tui_switching_to_missing_session_keeps_current_session(tmp_path) -> None:
    store = SessionStore(tmp_path)
    current = store.create_session()
    _record_user_message(store, current.session_id, "current question")
    app = AceAITUI(
        event_log_to_tui_events(store.load_event_log(current.session_id)),
        session_recorder=SessionRecorder(store, current.session_id),
        session_id=current.session_id,
    )

    async with app.run_test():
        event_count = len(app._state.events)
        app.switch_session("missing-session")

        assert app._session_id == current.session_id
        assert len(app._state.events) == event_count


@pytest.mark.anyio
async def test_session_selector_uses_panel_list(tmp_path) -> None:
    store = SessionStore(tmp_path)
    first = store.create_session()
    second = store.create_session()
    store.update_session_title(first.session_id, "first question - 2026-05-04 12:13:14")
    store.update_session_title(second.session_id, "second question")
    app = AceAITUI(
        event_log_to_tui_events(store.load_event_log(first.session_id)),
        session_recorder=SessionRecorder(store, first.session_id),
        session_id=first.session_id,
    )

    async with app.run_test() as pilot:
        app.open_session_selector()
        await pilot.pause()
        await _wait_until(
            pilot,
            lambda: (
                app.screen.query_one(
                    "#session-list",
                    SessionListWidget,
                ).selected_session_id()
                == first.session_id
            ),
        )

        session_list = app.screen.query_one("#session-list", SessionListWidget)
        rendered = _render_to_text(
            Group(
                *_session_picker_renderables(
                    session_list.sessions(),
                    current_session_id=first.session_id,
                    selected_index=session_list.selected_index,
                )
            )
        )

        assert session_list.selected_session_id() == first.session_id
        assert "first question" in rendered
        assert "second question" in rendered
        assert "aceai  2" in rendered
        assert "created" in rendered
        assert first.session_id in rendered
        status = app.screen.query_one("#session-status", Static)
        assert "Total cost: $0.000000" in str(status.render())


@pytest.mark.anyio
async def test_session_selector_deletes_highlighted_session_after_confirmation(
    tmp_path,
) -> None:
    store = SessionStore(tmp_path)
    first = store.create_session()
    second = store.create_session()
    store.update_session_title(first.session_id, "first question")
    store.update_session_title(second.session_id, "second question")
    app = AceAITUI(
        event_log_to_tui_events(store.load_event_log(first.session_id)),
        session_recorder=SessionRecorder(store, first.session_id),
        session_id=first.session_id,
    )

    async with app.run_test() as pilot:
        app.open_session_selector()
        await pilot.pause()
        await _wait_until(
            pilot,
            lambda: (
                app.screen.query_one(
                    "#session-list",
                    SessionListWidget,
                ).selected_session_id()
                == first.session_id
            ),
        )

        session_list = app.screen.query_one("#session-list", SessionListWidget)
        second_index = _session_widget_index(session_list, second.session_id)
        session_list.move_selection(second_index - session_list.selected_index)

        await pilot.press("d")
        await _wait_until(
            pilot,
            lambda: (
                second.session_id
                in str(app.screen.query_one("#delete-session-message", Static).content)
            ),
        )

        message = app.screen.query_one("#delete-session-message", Static)
        assert second.session_id in str(message.content)

        await pilot.press("enter")
        await _wait_until(
            pilot,
            lambda: (
                second.session_id
                not in [
                    session.session_id
                    for session in app.screen.query_one(
                        "#session-list",
                        SessionListWidget,
                    ).sessions()
                ]
            ),
        )

        session_list = app.screen.query_one("#session-list", SessionListWidget)
        rendered = _render_to_text(
            Group(
                *_session_picker_renderables(
                    session_list.sessions(),
                    current_session_id=first.session_id,
                    selected_index=session_list.selected_index,
                )
            )
        )
        assert second.session_id not in rendered
        assert [session.session_id for session in store.list_sessions()] == [
            first.session_id
        ]


def _session_widget_index(widget: SessionListWidget, session_id: str) -> int:
    for index, session in enumerate(widget.sessions()):
        if session.session_id == session_id:
            return index
    raise ValueError(session_id)


def _idea(content: str) -> Idea:
    return Idea(
        idea_id="idea-1",
        created_at=datetime(2026, 5, 6, 11, 13, tzinfo=timezone.utc),
        project_id="project-1",
        project_name="aceai",
        workspace="/tmp/aceai",
        content=content,
    )


def _first_event(events: list[TUIEvent], kind: str) -> TUIEvent:
    for event in events:
        if event.kind == kind:
            return event
    raise ValueError(kind)


def _render_to_text(renderable) -> str:
    console = Console(file=StringIO(), record=True, width=120)
    console.print(renderable)
    return console.export_text()


def _record_user_message(store: SessionStore, session_id: str, content: str) -> None:
    SessionRecorder(store, session_id).record(
        tui_event_to_session_event(TUIEvent.user_message(content))
    )
