import pytest

from aceai.agent.events import AgentEventBuilder
from aceai.tui.app import AceAITUI
from aceai.tui.demo import static_demo_events
from aceai.tui.events import adapt_agent_event, user_message_event
from aceai.tui.state import initial_state, reduce_events
from aceai.tui.widgets import DetailWidget, StreamWidget, TimelineWidget


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
