import pytest

from aceai.agent.session import SessionRecorder, SessionStore
from aceai.core.events import AgentEventBuilder
from aceai.agent.tui.app import AceAITUI
from aceai.agent.tui.demo import static_demo_events
from aceai.agent.tui.events import adapt_agent_event, user_message_event
from aceai.agent.tui.state import initial_state, reduce_events
from aceai.agent.tui.widgets import CommandInput, DetailWidget, StreamWidget, TimelineWidget


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
        adapt_agent_event(
            AgentEventBuilder(step_index=0, step_id="run-1-step-1").llm_started()
        ),
        adapt_agent_event(
            AgentEventBuilder(step_index=0, step_id="run-2-step-1").llm_started()
        ),
    ]

    state = reduce_events(events)

    assert [step.step_index for step in state.steps] == [0, 1]


def test_reduce_events_does_not_add_user_questions_to_timeline() -> None:
    state = reduce_events([user_message_event("What changed?")])

    assert state.steps == []
    assert state.events[0].kind == "user_message"


def test_reduce_events_does_not_add_session_notices_to_timeline() -> None:
    from aceai.agent.tui.events import session_notice_event

    state = reduce_events([session_notice_event("Sessions")])

    assert state.steps == []
    assert state.events[0].kind == "session_notice"


def test_initial_state_is_idle() -> None:
    state = initial_state()

    assert state.status == "idle"
    assert state.steps == []
    assert state.events == []


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
async def test_stream_scrolls_to_latest_content() -> None:
    builder = AgentEventBuilder(step_index=0, step_id="step-1")
    events = [user_message_event("Show me many lines")]
    for line_number in range(80):
        events.append(
            adapt_agent_event(
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
async def test_escape_exits_input_mode_and_returns_focus_to_stream() -> None:
    app = AceAITUI([])

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        stream = app.query_one(StreamWidget)
        command_input.focus()

        assert command_input.has_focus

        await pilot.press("escape")

        assert not command_input.has_focus
        assert stream.has_focus


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
    SessionRecorder(store, first.session_id).record(user_message_event("first question"))
    SessionRecorder(store, second.session_id).record(user_message_event("second question"))
    app = AceAITUI(
        store.load_tui_events(first.session_id),
        session_recorder=SessionRecorder(store, first.session_id),
        session_id=first.session_id,
    )

    async with app.run_test():
        app.show_sessions()

        assert app._state.events[-1].kind == "session_notice"
        assert first.session_id in app._state.events[-1].content
        assert second.session_id in app._state.events[-1].content

        app.switch_session(second.session_id)

        assert app.title == f"AceAI {second.session_id}"
        assert app._state.events[0].content == "second question"
