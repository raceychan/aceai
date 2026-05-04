from io import StringIO

import pytest

from aceai.agent.session import EventLog, SessionEvent, SessionRecorder, SessionStore
from aceai.core.events import AgentEventBuilder
from aceai.core.models import AgentStep
from aceai.agent.tui.app import AceAITUI
from aceai.agent.tui.app import STREAM_DELTA_REFRESH_CHARS
from aceai.agent.tui.demo import static_demo_events
from aceai.agent.tui.events import TUIEvent
from aceai.agent.tui.session_adapter import tui_event_to_session_event
from aceai.agent.tui.session_replay import event_log_to_tui_events
from aceai.agent.tui.state import initial_state, reduce_events, select_event
from aceai.agent.tui.widgets import CommandInput, DetailWidget, StreamWidget, TimelineWidget
from aceai.llm.models import LLMResponse, LLMUsage
from rich.console import Console
from textual.widgets import DataTable, Footer, Static


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


def test_reduce_events_does_not_add_user_questions_to_timeline() -> None:
    state = reduce_events([TUIEvent.user_message("What changed?")])

    assert state.steps == []
    assert state.events[0].kind == "user_message"


def test_reduce_events_does_not_add_session_notices_to_timeline() -> None:
    state = reduce_events([TUIEvent.session_notice("Sessions")])

    assert state.steps == []
    assert state.events[0].kind == "session_notice"


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

    state = reduce_events([TUIEvent.from_agent_event(first), TUIEvent.from_agent_event(second)])

    assert state.usage.current_context_tokens == 150
    assert state.usage.session_input_tokens == 250
    assert state.usage.session_cached_input_tokens == 40
    assert state.usage.session_output_tokens == 50
    assert state.usage.session_total_tokens == 300


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
                usage=LLMUsage(input_tokens=1_000, output_tokens=100, total_tokens=1_100),
            )
        )
    )

    state = reduce_events([TUIEvent.from_agent_event(first), TUIEvent.from_agent_event(second)])

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


@pytest.mark.anyio
async def test_static_tui_loads_fixture_events() -> None:
    events = static_demo_events()
    app = AceAITUI(events)

    async with app.run_test():
        assert app._state.status == "completed"
        timeline = app.query_one(TimelineWidget)
        stream = app.query_one(StreamWidget)
        detail = app.query_one(DetailWidget)
        assert timeline.can_focus
        assert stream.can_focus
        assert detail.can_focus
        assert timeline.has_class("collapsed")
        assert detail.has_class("collapsed")


@pytest.mark.anyio
async def test_timeline_selection_opens_tool_result_detail() -> None:
    events = static_demo_events()
    tool_event = _first_event(events, "tool_completed")
    app = AceAITUI(events)

    async with app.run_test() as pilot:
        timeline = app.query_one(TimelineWidget)
        timeline.post_message(TimelineWidget.EventSelected(tool_event.event_id))
        await pilot.pause()

        detail = app.query_one(DetailWidget)

        assert app._state.selected_event_id == tool_event.event_id
        assert not detail.has_class("collapsed")


@pytest.mark.anyio
async def test_timeline_accepts_step_and_tool_rows_for_same_event() -> None:
    app = AceAITUI(static_demo_events())

    async with app.run_test() as pilot:
        timeline = app.query_one(TimelineWidget)
        timeline.set_state(app._state)
        await pilot.pause()

        assert timeline.option_count > 0


@pytest.mark.anyio
async def test_detail_renders_tool_arguments_and_output() -> None:
    events = static_demo_events()
    tool_event = _first_event(events, "tool_completed")
    app = AceAITUI(events)

    async with app.run_test():
        app._state = select_event(app._state, tool_event.event_id)
        detail = app.query_one(DetailWidget)
        detail.set_state(app._state)

        rendered = _render_to_text(detail.render())

        assert '{"query":"aceai tui"}' in rendered
        assert '{"matches":["spec/tui.md","docs/tui.md"]}' in rendered


@pytest.mark.anyio
async def test_tui_batches_small_stream_delta_refreshes() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    app = AceAITUI([])

    async with app.run_test():
        refreshes: list[int] = []

        def fake_refresh_widgets() -> None:
            refreshes.append(len(app._state.events))

        app._refresh_widgets = fake_refresh_widgets
        app.append_event(TUIEvent.from_agent_event(builder.llm_text_delta(text_delta="hello ")))
        app.append_event(TUIEvent.from_agent_event(builder.llm_text_delta(text_delta="world")))

        assert refreshes == []
        assert app._state.events == []

        app.append_event(
            TUIEvent.from_agent_event(
                builder.llm_text_delta(text_delta="x" * STREAM_DELTA_REFRESH_CHARS)
            )
        )

        assert refreshes == [1]
        assert len(app._state.events) == 1
        assert app._state.events[0].content == "hello world" + (
            "x" * STREAM_DELTA_REFRESH_CHARS
        )


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
        app.append_event(TUIEvent.from_agent_event(builder.llm_text_delta(text_delta="done")))
        app.append_event(TUIEvent.from_agent_event(builder.run_completed(step=step, final_answer="done")))

        assert refreshes == [2]


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
        await pilot.pause(0.1)
        stream = app.query_one(StreamWidget)

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
                    "the stream instead of creating a horizontal scrollbar. "
                    * 8
                )
            )
        )
    ]
    app = AceAITUI(events)

    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause(0.1)
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
async def test_input_sits_above_footer_without_overlap() -> None:
    app = AceAITUI([])

    async with app.run_test(size=(80, 20)) as pilot:
        await pilot.pause(0.1)
        command_input = app.query_one(CommandInput)
        footer = app.query_one(Footer)

        assert command_input.region.y + command_input.region.height <= footer.region.y


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
        assert app.title == f"AceAI {metadata.session_id}"


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

        assert app.title == f"AceAI {second.session_id}"
        assert app._state.events[0].content == "second question"


@pytest.mark.anyio
async def test_tui_switching_to_current_session_is_noop_for_empty_session(tmp_path) -> None:
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
        app.switch_session("missing-session")

        assert app._session_id == current.session_id
        assert app._state.events[-1].kind == "session_notice"
        assert app._state.events[-1].content == "Session not found: missing-session"


@pytest.mark.anyio
async def test_session_selector_uses_table_columns(tmp_path) -> None:
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
        await pilot.pause(0.1)

        table = app.screen.query_one("#session-table", DataTable)

        assert [column.label.plain for column in table.ordered_columns] == [
            "Current",
            "Title",
            "Updated",
            "Created",
            "Session ID",
        ]
        assert table.row_count == 2
        assert table.ordered_rows[table.cursor_row].key.value == first.session_id
        assert table.get_row_at(table.cursor_row)[1] == "first question"
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
        await pilot.pause(0.1)

        table = app.screen.query_one("#session-table", DataTable)
        second_row = _table_row_index(table, second.session_id)
        table.move_cursor(row=second_row)

        await pilot.press("d")
        await pilot.pause(0.1)

        message = app.screen.query_one("#delete-session-message", Static)
        assert second.session_id in str(message.content)

        await pilot.press("enter")
        await pilot.pause(0.1)

        table = app.screen.query_one("#session-table", DataTable)
        assert second.session_id not in [
            row.key.value for row in table.ordered_rows
        ]
        assert [session.session_id for session in store.list_sessions()] == [
            first.session_id
        ]


def _table_row_index(table: DataTable, session_id: str) -> int:
    for index, row in enumerate(table.ordered_rows):
        if row.key.value == session_id:
            return index
    raise ValueError(session_id)


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
