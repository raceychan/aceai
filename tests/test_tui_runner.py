import pytest

from aceai.core.base import AgentBase
from aceai.agent.session import SessionRecorder, SessionStore
from aceai.llm import LLMResponse
from aceai.llm.models import LLMUsage
from aceai.llm.models import LLMStreamEvent
from aceai.agent.tui.events import user_message_event
from aceai.agent.tui.runner import AceAIInteractiveTUI, AceAILiveTUI
from aceai.agent.tui.widgets import CommandInput, StatusBarWidget
from textual.widgets import Input


class StubExecutor:
    def select_tools(
        self,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
    ) -> list[object]:
        if include and exclude:
            raise ValueError("Cannot specify both include and exclude")
        return []


class StubLLMService:
    def __init__(self, events: list[LLMStreamEvent]) -> None:
        self._events = list(events)
        self.calls: list[dict] = []

    async def stream(self, **request):
        self.calls.append(request)
        for event in self._events:
            yield event

    async def complete(self, **request) -> LLMResponse:
        raise AssertionError("live TUI should use streaming")


class MultiRunLLMService:
    def __init__(self, streams: list[list[LLMStreamEvent]]) -> None:
        self._streams = [list(stream) for stream in streams]
        self.calls: list[dict] = []

    async def stream(self, **request):
        self.calls.append(request)
        events = self._streams.pop(0)
        for event in events:
            yield event

    async def complete(self, **request) -> LLMResponse:
        raise AssertionError("live TUI should use streaming")


@pytest.mark.anyio
async def test_live_tui_streams_agent_events_into_state() -> None:
    llm_service = StubLLMService(
        [
            LLMStreamEvent(
                event_type="response.output_text.delta",
                text_delta="hello",
            ),
            LLMStreamEvent(
                event_type="response.completed",
                response=LLMResponse(text="hello"),
            ),
        ]
    )
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = AceAILiveTUI(agent, "Question?")

    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        assert app._state.status == "completed"
        assert app._state.final_answer == "hello"
        assert llm_service.calls[0]["messages"][-1].content[0]["data"] == "Question?"


@pytest.mark.anyio
async def test_interactive_tui_submits_question_from_input() -> None:
    llm_service = StubLLMService(
        [
            LLMStreamEvent(
                event_type="response.output_text.delta",
                text_delta="answer",
            ),
            LLMStreamEvent(
                event_type="response.completed",
                response=LLMResponse(text="answer"),
            ),
        ]
    )
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = AceAIInteractiveTUI(agent)

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        app.on_input_submitted(Input.Submitted(command_input, "What now?"))
        await pilot.pause(0.1)

        assert app._state.status == "completed"
        assert app._state.final_answer == "answer"
        assert command_input.value == ""
        assert llm_service.calls[0]["messages"][-1].content[0]["data"] == "What now?"


@pytest.mark.anyio
async def test_interactive_tui_clear_command_resets_state() -> None:
    llm_service = StubLLMService([])
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = AceAIInteractiveTUI(agent)

    async with app.run_test():
        app.load_events([])
        app._state = app._state.__class__(
            status="completed",
            final_answer="done",
        )
        command_input = app.query_one(CommandInput)
        command_input.value = "/clear"
        app.on_input_submitted(Input.Submitted(command_input, "/clear"))

        assert app._state.status == "idle"
        assert app._state.events == []
        assert command_input.value == ""


@pytest.mark.anyio
async def test_interactive_tui_keeps_history_between_questions() -> None:
    llm_service = MultiRunLLMService(
        [
            [
                LLMStreamEvent(
                    event_type="response.output_text.delta",
                    text_delta="first",
                ),
                LLMStreamEvent(
                    event_type="response.completed",
                    response=LLMResponse(text="first"),
                ),
            ],
            [
                LLMStreamEvent(
                    event_type="response.output_text.delta",
                    text_delta="second",
                ),
                LLMStreamEvent(
                    event_type="response.completed",
                    response=LLMResponse(text="second"),
                ),
            ],
        ]
    )
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = AceAIInteractiveTUI(agent)

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        app.on_input_submitted(Input.Submitted(command_input, "First?"))
        await pilot.pause(0.1)
        first_event_count = len(app._state.events)

        app.on_input_submitted(Input.Submitted(command_input, "Second?"))
        await pilot.pause(0.1)

        assert len(app._state.events) > first_event_count
        assert app._state.final_answer == "second"
        assert llm_service.calls[0]["messages"][-1].content[0]["data"] == "First?"
        assert llm_service.calls[1]["messages"][-1].content[0]["data"] == "Second?"


@pytest.mark.anyio
async def test_interactive_tui_model_command_updates_next_request_metadata() -> None:
    llm_service = StubLLMService(
        [
            LLMStreamEvent(
                event_type="response.output_text.delta",
                text_delta="answer",
            ),
            LLMStreamEvent(
                event_type="response.completed",
                response=LLMResponse(text="answer"),
            ),
        ]
    )
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = AceAIInteractiveTUI(agent)

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        app.on_input_submitted(Input.Submitted(command_input, "/model gpt-5.5"))
        app.on_input_submitted(Input.Submitted(command_input, "What now?"))
        await pilot.pause(0.1)

        assert app._selected_model == "gpt-5.5"
        assert app._status_model == "gpt-5.5"
        assert llm_service.calls[0]["metadata"]["model"] == "gpt-5.5"


@pytest.mark.anyio
async def test_interactive_tui_status_bar_shows_selected_model() -> None:
    llm_service = StubLLMService([])
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = AceAIInteractiveTUI(agent)

    async with app.run_test():
        status = app.query_one(StatusBarWidget)

        assert "model: gpt-4o" in status.current_text

        app.switch_model("gpt-5.5")

        assert "model: gpt-5.5" in status.current_text


@pytest.mark.anyio
async def test_interactive_tui_status_bar_shows_usage() -> None:
    llm_service = StubLLMService(
        [
            LLMStreamEvent(
                event_type="response.completed",
                response=LLMResponse(
                    text="answer",
                    model="gpt-5.5",
                    usage=LLMUsage(
                        input_tokens=1_200,
                        cached_input_tokens=200,
                        output_tokens=300,
                        total_tokens=1_500,
                    ),
                ),
            ),
        ]
    )
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = AceAIInteractiveTUI(agent)

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        app.on_input_submitted(Input.Submitted(command_input, "What now?"))
        await pilot.pause(0.1)

        status = app.query_one(StatusBarWidget)
        assert "ctx: 1,200" in status.current_text
        assert "session: 1,500" in status.current_text
        assert "200 cached" in status.current_text
        assert "cost: $0.0141" in status.current_text


@pytest.mark.anyio
async def test_interactive_tui_model_selection_callback_updates_model() -> None:
    llm_service = StubLLMService(
        [
            LLMStreamEvent(
                event_type="response.completed",
                response=LLMResponse(text="answer"),
            ),
        ]
    )
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = AceAIInteractiveTUI(agent)

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        app._handle_model_selection("gpt-5.5")
        app.on_input_submitted(Input.Submitted(command_input, "Use selected model"))
        await pilot.pause(0.1)

        assert app._selected_model == "gpt-5.5"
        assert llm_service.calls[0]["metadata"]["model"] == "gpt-5.5"


@pytest.mark.anyio
async def test_interactive_tui_session_selection_callback_switches_session(
    tmp_path,
) -> None:
    store = SessionStore(tmp_path)
    first = store.create_session()
    second = store.create_session()
    SessionRecorder(store, first.session_id).record(user_message_event("first"))
    SessionRecorder(store, second.session_id).record(user_message_event("second"))
    llm_service = StubLLMService([])
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = AceAIInteractiveTUI(
        agent,
        initial_events=store.load_tui_events(first.session_id),
        initial_history=[],
        session_recorder=SessionRecorder(store, first.session_id),
        session_id=first.session_id,
    )

    async with app.run_test():
        app._handle_session_selection(second.session_id)

        assert app._session_id == second.session_id
        assert app.title == f"AceAI {second.session_id}"
        assert app._state.events[0].content == "second"
        assert app._llm_history[0].content[0]["data"] == "second"
