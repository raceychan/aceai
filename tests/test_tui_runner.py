import pytest

from aceai.core.base import AgentBase
from aceai.agent.session import SessionRecorder, SessionState, SessionStore
from aceai.llm import LLMResponse
from aceai.core.run_state import ToolRunState
from aceai.llm.models import LLMToolCall, LLMUsage
from aceai.llm.models import LLMStreamEvent
from aceai.agent.tui.events import TUIEvent
from aceai.agent.tui.session_adapter import tui_event_to_session_event
from aceai.agent.tui.session_replay import event_log_to_tui_events
from aceai.agent.tui.config import AceAITUIConfig
from aceai.agent.tui.app import AceAITUI
from aceai.agent.tui.runner import AceAIConfiguredTUI, AceAIInteractiveTUI, AceAILiveTUI
from aceai.agent.tui.setup import ModelSelectScreen, ModelSelection
from aceai.agent.tui.widgets import ApprovalWidget
from aceai.agent.tui.widgets import CommandInput, StatusBarWidget
from textual.widgets import Button, Input, RichLog, Static


class StubExecutor:
    def select_tools(
        self,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
    ) -> list[object]:
        if include and exclude:
            raise ValueError("Cannot specify both include and exclude")
        return []


class ApprovalExecutor:
    def __init__(self) -> None:
        self.calls: list[LLMToolCall] = []

    def select_tools(
        self,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
    ) -> list[object]:
        if include and exclude:
            raise ValueError("Cannot specify both include and exclude")
        return []

    def resolve_invocation(self, tool_call: LLMToolCall):
        return ApprovalInvocation(tool_call)

    async def execute(
        self,
        invocation,
        *,
        tool_state: ToolRunState,
    ) -> str:
        self.calls.append(invocation.call)
        return '{"ok":true}'


class ApprovalInvocation:
    def __init__(self, call: LLMToolCall) -> None:
        self.call = call
        self.approval_required = True
        self.tool = ApprovalTool(call.name)


class ApprovalTool:
    def __init__(self, name: str) -> None:
        self.name = name
        self.metadata = ApprovalToolMetadata()


class ApprovalToolMetadata:
    require_approval = True
    approval_policy = "filesystem_write"


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
async def test_interactive_tui_approves_suspended_tool_and_continues() -> None:
    call = LLMToolCall(
        name="write_text_file",
        arguments='{"path":"x","content":"hello"}',
        call_id="call-1",
    )
    llm_service = MultiRunLLMService(
        [
            [
                LLMStreamEvent(
                    event_type="response.completed",
                    response=LLMResponse(text="use tool", tool_calls=[call]),
                ),
            ],
            [
                LLMStreamEvent(
                    event_type="response.completed",
                    response=LLMResponse(text="done"),
                ),
            ],
        ]
    )
    executor = ApprovalExecutor()
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
    )
    app = AceAIInteractiveTUI(agent)

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        app.on_input_submitted(Input.Submitted(command_input, "Write it"))
        await pilot.pause(0.1)

        assert app._state.status == "suspended"
        assert executor.calls == []
        assert app._active_runtime is not None
        assert app._active_runtime.status == "suspended"
        assert command_input.placeholder == "Choose Approve or Reject"
        status = app.query_one(StatusBarWidget)
        assert "action: choose Approve or Reject" in status.current_text
        approval = app.query_one(ApprovalWidget)
        assert not approval.has_class("collapsed")
        assert approval.query_one("#approval-approve", Button).label.plain == "A Approve"
        assert approval.query_one("#approval-reject", Button).label.plain == "R Reject"
        assert "content:" in str(approval.query_one("#approval-summary", Static).render())

        approval.post_message(ApprovalWidget.Selected(approved=True))
        await pilot.pause(0.1)

        assert app._state.status == "completed"
        assert app._state.final_answer == "done"
        assert command_input.placeholder == "Ask AceAI or type /quit"
        assert approval.has_class("collapsed")
        assert executor.calls == [call]
        assert app._llm_history[-1].role == "assistant"
        assert app._llm_history[-1].content[0]["data"] == "done"


@pytest.mark.anyio
async def test_interactive_tui_approves_suspended_tool_with_keyboard() -> None:
    call = LLMToolCall(
        name="write_text_file",
        arguments='{"path":"x","content":"hello"}',
        call_id="call-1",
    )
    llm_service = MultiRunLLMService(
        [
            [
                LLMStreamEvent(
                    event_type="response.completed",
                    response=LLMResponse(text="use tool", tool_calls=[call]),
                ),
            ],
            [
                LLMStreamEvent(
                    event_type="response.completed",
                    response=LLMResponse(text="done"),
                ),
            ],
        ]
    )
    executor = ApprovalExecutor()
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
    )
    app = AceAIInteractiveTUI(agent)

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        app.on_input_submitted(Input.Submitted(command_input, "Write it"))
        await pilot.pause(0.1)

        await pilot.press("a")
        await pilot.pause(0.1)

        assert app._state.status == "completed"
        assert app._state.final_answer == "done"
        assert executor.calls == [call]


@pytest.mark.anyio
async def test_interactive_tui_shows_next_approval_after_resume_suspends_again() -> None:
    first_call = LLMToolCall(
        name="write_text_file",
        arguments='{"path":"x","content":"hello"}',
        call_id="call-1",
    )
    second_call = LLMToolCall(
        name="run_shell_command",
        arguments='{"command":"python binary_search.py"}',
        call_id="call-2",
    )
    llm_service = MultiRunLLMService(
        [
            [
                LLMStreamEvent(
                    event_type="response.completed",
                    response=LLMResponse(text="write file", tool_calls=[first_call]),
                ),
            ],
            [
                LLMStreamEvent(
                    event_type="response.completed",
                    response=LLMResponse(text="run file", tool_calls=[second_call]),
                ),
            ],
        ]
    )
    executor = ApprovalExecutor()
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
    )
    app = AceAIInteractiveTUI(agent)

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        app.on_input_submitted(Input.Submitted(command_input, "Write and run it"))
        await pilot.pause(0.1)

        approval = app.query_one(ApprovalWidget)
        assert app._state.status == "suspended"
        assert not approval.has_class("collapsed")
        assert "write_text_file" in str(
            approval.query_one("#approval-summary", Static).render()
        )

        approval.post_message(ApprovalWidget.Selected(approved=True))
        await pilot.pause(0.1)

        assert app._state.status == "suspended"
        assert app._active_runtime is not None
        assert app._active_runtime.status == "suspended"
        assert command_input.placeholder == "Choose Approve or Reject"
        assert executor.calls == [first_call]
        assert not approval.has_class("collapsed")
        summary = str(approval.query_one("#approval-summary", Static).render())
        assert "run_shell_command" in summary
        assert "python binary_search.py" in summary


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
async def test_interactive_tui_persists_selected_model_in_session_state(tmp_path) -> None:
    store = SessionStore(tmp_path)
    metadata = store.create_session()
    llm_service = StubLLMService([])
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = AceAIInteractiveTUI(
        agent,
        session_recorder=SessionRecorder(store, metadata.session_id),
        session_id=metadata.session_id,
    )

    async with app.run_test():
        app.switch_model("gpt-5.5")
        app.append_event(TUIEvent.user_message("keep this session"))

    assert store.get_session_state(metadata.session_id) == SessionState(
        selected_provider="openai",
        selected_model="gpt-5.5",
    )


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
        assert "cost: $0.0141" in status.current_text


@pytest.mark.anyio
async def test_interactive_tui_metadata_lists_runtime_usage_and_skills(tmp_path) -> None:
    skill_dir = tmp_path / "skills" / "debugger"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: debugger\n"
        "description: Debug flaky tests.\n"
        "---\n"
        "# Debugger\n",
        encoding="utf-8",
    )
    llm_service = StubLLMService([])
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
        skill_path=tmp_path / "skills",
    )
    app = AceAIInteractiveTUI(agent)

    async with app.run_test():
        sections = app._metadata_sections()

    section_lines = {
        section.title: "\n".join(section.lines)
        for section in sections
    }
    assert "model: gpt-4o" in section_lines["Runtime"]
    assert "session cost: -" in section_lines["Usage"]
    assert "provider: openai" in section_lines["Agent"]
    assert "debugger: Debug flaky tests." in section_lines["Skills (1)"]


@pytest.mark.anyio
async def test_metadata_screen_scrolls_and_keeps_close_button(tmp_path) -> None:
    skill_root = tmp_path / "skills"
    for index in range(20):
        skill_dir = skill_root / f"skill_{index}"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            f"name: skill_{index}\n"
            f"description: Skill number {index}.\n"
            "---\n"
            "# Skill\n",
            encoding="utf-8",
        )
    llm_service = StubLLMService([])
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
        skill_path=skill_root,
    )
    app = AceAIInteractiveTUI(agent)

    async with app.run_test(size=(100, 24)) as pilot:
        app.open_metadata_screen()
        await pilot.pause(0.1)
        body = app.screen.query_one("#metadata-body", RichLog)
        close_button = app.screen.query_one("#metadata-close", Button)

        assert body.max_scroll_y > 0
        assert close_button.display

        await pilot.press("pagedown")

        assert body.scroll_y > 0


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
async def test_interactive_tui_m_key_opens_model_selector() -> None:
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([]),  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = AceAIInteractiveTUI(agent)
    calls: list[str] = []
    app.open_model_selector = lambda: calls.append("model")

    async with app.run_test() as pilot:
        await pilot.press("m")

    assert calls == ["model"]


@pytest.mark.anyio
async def test_interactive_tui_returns_focus_to_stream_after_submit() -> None:
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
        command_input.focus()
        app.on_input_submitted(Input.Submitted(command_input, "Use selected model"))
        await pilot.pause(0.1)

        assert not command_input.has_focus
        assert app.query_one("#stream").has_focus


@pytest.mark.anyio
async def test_model_selector_prefills_masked_api_key() -> None:
    screen = ModelSelectScreen(
        provider_name="openai",
        current_model="gpt-5.5",
        api_keys={"openai": "sk-test-ending"},
    )

    async with AceAITUI([]).run_test() as pilot:
        pilot.app.push_screen(screen)
        await pilot.pause()

        assert screen.query_one("#api-key", Input).value == "*****************ding"


@pytest.mark.anyio
async def test_model_selector_apply_restores_masked_api_key() -> None:
    screen = ModelSelectScreen(
        provider_name="openai",
        current_model="gpt-5.5",
        api_keys={"openai": "sk-test-ending"},
    )

    async with AceAITUI([]).run_test() as pilot:
        pilot.app.push_screen(screen)
        await pilot.pause()
        selections: list[ModelSelection | None] = []

        def dismiss(selection: ModelSelection | None) -> None:
            selections.append(selection)

        screen.dismiss = dismiss
        _press_model_apply(screen)

        assert selections == [
            ModelSelection(
                provider="openai",
                model="gpt-5.5",
                api_key="sk-test-ending",
            )
        ]


@pytest.mark.anyio
async def test_model_selector_requires_provider_model_and_api_key() -> None:
    screen = ModelSelectScreen(
        provider_name="openai",
        current_model="gpt-5.5",
        api_keys={"openai": "sk-test-ending"},
    )

    async with AceAITUI([]).run_test() as pilot:
        pilot.app.push_screen(screen)
        await pilot.pause()
        selections: list[ModelSelection | None] = []

        def dismiss(selection: ModelSelection | None) -> None:
            selections.append(selection)

        screen.dismiss = dismiss
        provider_input = screen.query_one("#provider", Input)
        model_input = screen.query_one("#model", Input)
        api_key_input = screen.query_one("#api-key", Input)
        error = screen.query_one("#model-error", Static)

        provider_input.value = ""
        _press_model_apply(screen)
        assert selections == []
        assert str(error.render()) == "Provider is required"

        provider_input.value = "openai"
        model_input.value = ""
        _press_model_apply(screen)
        assert selections == []
        assert str(error.render()) == "Model is required"

        model_input.value = "gpt-5.5"
        api_key_input.value = ""
        _press_model_apply(screen)
        assert selections == []
        assert str(error.render()) == "API key is required"


@pytest.mark.anyio
async def test_model_selector_selects_prefixed_model_candidate_with_tab() -> None:
    screen = ModelSelectScreen(
        provider_name="deepseek",
        current_model="deepseek-v4-pro",
        api_keys={},
    )

    async with AceAITUI([]).run_test() as pilot:
        pilot.app.push_screen(screen)
        await pilot.pause()
        model_input = screen.query_one("#model", Input)
        model_input.focus()

        model_input.value = "deepseek-c"
        await pilot.pause()
        await pilot.press("tab")

        assert model_input.value == "deepseek-chat"
        assert str(screen.query_one("#model-options", Static).render()) == ""


@pytest.mark.anyio
async def test_model_selector_selects_prefixed_provider_candidate_with_tab() -> None:
    screen = ModelSelectScreen(
        provider_name="openai",
        current_model="gpt-5.5",
        api_keys={"deepseek": "sk-deepseek-ending"},
    )

    async with AceAITUI([]).run_test() as pilot:
        pilot.app.push_screen(screen)
        await pilot.pause()
        provider_input = screen.query_one("#provider", Input)
        provider_input.focus()

        provider_input.value = "d"
        await pilot.pause()
        await pilot.press("tab")

        assert provider_input.value == "deepseek"
        assert screen.query_one("#model", Input).value == "deepseek-v4-pro"
        assert screen.query_one("#api-key", Input).value == "*****************ding"
        assert str(screen.query_one("#provider-options", Static).render()) == ""


@pytest.mark.anyio
async def test_model_selector_hides_candidates_for_empty_input() -> None:
    screen = ModelSelectScreen(
        provider_name="openai",
        current_model="gpt-5.5",
        api_keys={},
    )

    async with AceAITUI([]).run_test() as pilot:
        pilot.app.push_screen(screen)
        await pilot.pause()
        provider_input = screen.query_one("#provider", Input)
        provider_input.value = ""
        await pilot.pause()

        assert str(screen.query_one("#provider-options", Static).render()) == ""


@pytest.mark.anyio
async def test_configured_tui_switches_provider_without_reusing_current_key(
    monkeypatch,
) -> None:
    calls: list[AceAITUIConfig] = []

    def agent_factory(config: AceAITUIConfig) -> AgentBase:
        calls.append(config)
        return AgentBase(
            prompt="Prompt",
            default_model=config.model,
            llm_service=StubLLMService([]),  # type: ignore[arg-type]
            executor=StubExecutor(),  # type: ignore[arg-type]
        )

    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    app = AceAIConfiguredTUI(
        agent_factory,
        initial_config=AceAITUIConfig(
            provider="openai",
            api_key="openai-key",
            model="gpt-5.5",
            api_keys={"openai": "openai-key"},
        ),
        initial_question="",
        default_model="gpt-5.5",
    )

    async with app.run_test():
        app._handle_model_selection(
            ModelSelection(
                provider="deepseek",
                model="deepseek-v4-flash",
                api_key="",
            )
        )

    assert calls[-1] == AceAITUIConfig(
        provider="deepseek",
        api_key="deepseek-key",
        model="deepseek-v4-flash",
        api_keys={"openai": "openai-key", "deepseek": "deepseek-key"},
    )


@pytest.mark.anyio
async def test_interactive_tui_session_selection_callback_switches_session(
    tmp_path,
) -> None:
    store = SessionStore(tmp_path)
    first = store.create_session()
    second = store.create_session()
    SessionRecorder(store, first.session_id).record(
        tui_event_to_session_event(TUIEvent.user_message("first"))
    )
    SessionRecorder(store, second.session_id).record(
        tui_event_to_session_event(TUIEvent.user_message("second"))
    )
    store.update_session_state(
        second.session_id,
        SessionState(selected_provider="openai", selected_model="gpt-5.5"),
    )
    llm_service = StubLLMService([])
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = AceAIInteractiveTUI(
        agent,
        initial_events=event_log_to_tui_events(store.load_event_log(first.session_id)),
        initial_history=[],
        session_recorder=SessionRecorder(store, first.session_id),
        session_id=first.session_id,
    )

    async with app.run_test():
        app._handle_session_selection(second.session_id)

        assert app._session_id == second.session_id
        assert app.title == f"AceAI {second.session_id}"
        assert app._state.events[0].content == "second"
        assert app._selected_model == "gpt-5.5"
        assert app._llm_history[0].content[0]["data"] == "second"


def _press_model_apply(screen: ModelSelectScreen) -> None:
    class Pressed:
        button = screen.query_one("#apply", Button)

    screen.on_button_pressed(Pressed())
