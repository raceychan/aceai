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
from aceai.agent.config import clear_config, current_config
from aceai.agent import app as agent_app_module
from aceai.agent.app import UpdateCheckResult
from aceai.agent.tui import app as tui_app_module
from aceai.agent.tui import runner as tui_runner_module
from aceai.agent.tui.app import AceAITUI
from aceai.agent.tui.runner import (
    UPDATE_INSTRUCTIONS,
    AceAIConfiguredTUI,
    AceAIInteractiveTUI,
    AceAILiveTUI,
    UpdateCommandResult,
)
from aceai.agent.tui.setup import (
    ConfigScreen,
    ConfigSelection,
    SkillConfigItem,
    ToolPermissionItem,
)
from aceai.agent.tui.widgets import ApprovalWidget
from aceai.agent.tui.widgets import CommandInput, StatusBarWidget
from textual.widgets import Button, Checkbox, Input, RichLog, Select, Static, TabbedContent


@pytest.fixture(autouse=True)
def tui_session_store(monkeypatch, tmp_path) -> SessionStore:
    store = SessionStore(tmp_path / "sessions")
    monkeypatch.setattr(tui_app_module, "SessionStore", lambda: store)

    async def no_update() -> None:
        return None

    monkeypatch.setattr(agent_app_module, "check_for_updates", no_update)
    return store


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
async def test_interactive_tui_submits_question_from_input(
    tui_session_store: SessionStore,
) -> None:
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
    assert app._session_recorder is None
    assert app._session_id is None

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        app.on_input_submitted(Input.Submitted(command_input, "What now?"))
        await pilot.pause(0.1)

        assert app._state.status == "completed"
        assert app._state.final_answer == "answer"
        assert command_input.value == ""
        assert llm_service.calls[0]["messages"][-1].content[0]["data"] == "What now?"
        assert app._session_recorder is not None
        assert app._session_id is not None
        event_log = tui_session_store.load_event_log(app._session_id)
        assert event_log.events[0].payload["content"] == "What now?"
        run_ids = {session_event.run_id for session_event in event_log.events}
        assert len(run_ids) == 1
        assert "" not in run_ids
        assert event_log.get_run(next(iter(run_ids))).question == "What now?"


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
    SessionRecorder(store, metadata.session_id).record(
        tui_event_to_session_event(TUIEvent.user_message("keep this session"))
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
        session_recorder=SessionRecorder(store, metadata.session_id),
        session_id=metadata.session_id,
    )

    async with app.run_test():
        app.switch_model("gpt-5.5")

    assert store.get_session_state(metadata.session_id) == SessionState(
        selected_provider="openai",
        selected_model="gpt-5.5",
    )


@pytest.mark.anyio
async def test_interactive_tui_model_only_session_finalizes_as_empty(tmp_path) -> None:
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
        app.append_event(TUIEvent.session_notice(f"Resumed session {metadata.session_id}"))
        app.switch_model("gpt-5.5")

    assert store.list_sessions() == []


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
        app._handle_config_selection("gpt-5.5")
        app.on_input_submitted(Input.Submitted(command_input, "Use selected model"))
        await pilot.pause(0.1)

        assert app._selected_model == "gpt-5.5"
        assert llm_service.calls[0]["metadata"]["model"] == "gpt-5.5"


@pytest.mark.anyio
async def test_interactive_tui_c_key_opens_config_screen() -> None:
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([]),  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = AceAIInteractiveTUI(agent)
    calls: list[str] = []
    app.open_config_screen = lambda: calls.append("config")

    async with app.run_test() as pilot:
        await pilot.press("c")

    assert calls == ["config"]


@pytest.mark.anyio
async def test_interactive_tui_config_command_opens_config_screen() -> None:
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([]),  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = AceAIInteractiveTUI(agent)
    calls: list[str] = []
    app.open_config_screen = lambda: calls.append("config")

    async with app.run_test():
        command_input = app.query_one(CommandInput)
        app.on_input_submitted(Input.Submitted(command_input, "/config"))

    assert calls == ["config"]


@pytest.mark.anyio
async def test_interactive_tui_update_command_runs_upgrade_and_restarts(monkeypatch) -> None:
    async def update_command() -> UpdateCommandResult:
        return UpdateCommandResult(return_code=0, output="updated")

    restart_calls: list[str] = []
    monkeypatch.setattr(tui_runner_module, "run_update_command", update_command)
    monkeypatch.setattr(
        tui_runner_module,
        "restart_current_process",
        lambda: restart_calls.append("restart"),
    )
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([]),  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = AceAIInteractiveTUI(agent)

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        app.on_input_submitted(Input.Submitted(command_input, "/update"))
        await pilot.pause(0.1)

    assert [event.content for event in app._state.events[-2:]] == [
        "Updating AceAI with uv tool upgrade aceai...",
        "AceAI updated. Restarting...",
    ]
    assert restart_calls == ["restart"]


@pytest.mark.anyio
async def test_interactive_tui_update_command_reports_upgrade_failure(monkeypatch) -> None:
    async def update_command() -> UpdateCommandResult:
        return UpdateCommandResult(return_code=1, output="network failed")

    restart_calls: list[str] = []
    monkeypatch.setattr(tui_runner_module, "run_update_command", update_command)
    monkeypatch.setattr(
        tui_runner_module,
        "restart_current_process",
        lambda: restart_calls.append("restart"),
    )
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([]),  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = AceAIInteractiveTUI(agent)

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        app.on_input_submitted(Input.Submitted(command_input, "/update"))
        await pilot.pause(0.1)

    assert app._state.events[-1].kind == "session_notice"
    assert app._state.events[-1].content == (
        "AceAI update failed with exit code 1.\nnetwork failed"
    )
    assert restart_calls == []


@pytest.mark.anyio
async def test_interactive_tui_automatically_reports_available_update(monkeypatch) -> None:
    async def available_update() -> UpdateCheckResult:
        return UpdateCheckResult(
            current_version="0.2.7",
            latest_version="0.2.8",
        )

    monkeypatch.setattr(agent_app_module, "check_for_updates", available_update)
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([]),  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = AceAIInteractiveTUI(agent)

    async with app.run_test() as pilot:
        await pilot.pause(0.1)

    assert app._state.events[-1].kind == "session_notice"
    assert app._state.events[-1].content == (
        "AceAI 0.2.8 is available (current 0.2.7).\n"
        f"{UPDATE_INSTRUCTIONS}"
    )


@pytest.mark.anyio
async def test_interactive_tui_reports_available_update_once(monkeypatch) -> None:
    async def available_update() -> UpdateCheckResult:
        return UpdateCheckResult(
            current_version="0.2.7",
            latest_version="0.2.8",
        )

    monkeypatch.setattr(agent_app_module, "check_for_updates", available_update)
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([]),  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = AceAIInteractiveTUI(agent)

    async with app.run_test() as pilot:
        app._start_update_check()
        await pilot.pause(0.1)

    notices = [
        event
        for event in app._state.events
        if event.kind == "session_notice"
        and "is available" in event.content
    ]
    assert len(notices) == 1


@pytest.mark.anyio
async def test_interactive_tui_starts_update_check_once_when_mount_reenters(monkeypatch) -> None:
    calls = 0

    async def no_update() -> None:
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(agent_app_module, "check_for_updates", no_update)
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([]),  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = AceAIInteractiveTUI(agent)

    async with app.run_test() as pilot:
        app.on_mount()
        await pilot.pause(0.1)

    assert calls == 1


@pytest.mark.anyio
async def test_interactive_tui_info_command_uses_registered_alias() -> None:
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([]),  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = AceAIInteractiveTUI(agent)
    calls: list[str] = []
    app.open_metadata_screen = lambda: calls.append("metadata")

    async with app.run_test():
        command_input = app.query_one(CommandInput)
        app.on_input_submitted(Input.Submitted(command_input, "/info"))

    assert calls == ["metadata"]


@pytest.mark.anyio
async def test_interactive_tui_resume_command_uses_registered_arg_handler(
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
    agent = AgentBase(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([]),  # type: ignore[arg-type]
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
        command_input = app.query_one(CommandInput)
        app.on_input_submitted(Input.Submitted(command_input, f"/resume {second.session_id}"))

    assert app._session_id == second.session_id
    assert app._state.events[0].content == "second"


@pytest.mark.anyio
async def test_interactive_tui_unknown_slash_input_runs_as_question() -> None:
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
        app.on_input_submitted(Input.Submitted(command_input, "/unknown command"))
        await pilot.pause(0.1)

    assert llm_service.calls[0]["messages"][-1].content[0]["data"] == "/unknown command"


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
async def test_config_screen_prefills_masked_api_key() -> None:
    screen = ConfigScreen(
        provider_name="openai",
        current_model="gpt-5.5",
        default_model="gpt-5.5",
        skills="auto",
        api_keys={"openai": "sk-test-ending"},
    )

    async with AceAITUI([]).run_test() as pilot:
        pilot.app.push_screen(screen)
        await pilot.pause()

        assert screen.query_one("#api-key", Input).value == "*****************ding"


@pytest.mark.anyio
async def test_config_screen_apply_restores_masked_api_key() -> None:
    screen = ConfigScreen(
        provider_name="openai",
        current_model="gpt-5.5",
        default_model="gpt-5.5",
        skills="auto",
        api_keys={"openai": "sk-test-ending"},
    )

    async with AceAITUI([]).run_test() as pilot:
        pilot.app.push_screen(screen)
        await pilot.pause()
        selections: list[ConfigSelection | None] = []

        def dismiss(selection: ConfigSelection | None) -> None:
            selections.append(selection)

        screen.dismiss = dismiss
        _press_config_apply(screen)

        assert selections == [
            ConfigSelection(
                provider="openai",
                model="gpt-5.5",
                default_model="gpt-5.5",
                api_key="sk-test-ending",
                skills="auto",
                skill_selection_mode="selected",
                enabled_skills=(),
            )
        ]


@pytest.mark.anyio
async def test_config_screen_uses_model_as_default_model_and_separates_skills() -> None:
    skill_item = SkillConfigItem(
        name="developer",
        description=(
            "Practical software development workflow with careful repository "
            "inspection, implementation, and verification."
        ),
        location="/skills/developer/SKILL.md",
    )
    screen = ConfigScreen(
        provider_name="openai",
        current_model="gpt-5.5",
        default_model="gpt-4o",
        skills="auto",
        skill_items=(skill_item,),
        skill_selection_mode="all",
        enabled_skills=(),
        api_keys={"openai": "sk-test-ending"},
    )

    async with AceAITUI([]).run_test() as pilot:
        pilot.app.push_screen(screen)
        await pilot.pause()
        selections: list[ConfigSelection | None] = []

        def dismiss(selection: ConfigSelection | None) -> None:
            selections.append(selection)

        screen.dismiss = dismiss
        screen.query_one("#model", Input).value = "gpt-4o"
        screen_ids = {node.id for node in screen.query("*") if node.id is not None}
        _press_config_apply(screen)

        assert "default-model" not in screen_ids
        assert screen.query_one("#api-key", Input).region.y < screen.query_one(
            "#config-skills-list"
        ).region.y
        assert str(screen.query_one("#skill-0", Checkbox).label) == "developer"
        assert (
            "Practical software development workflow"
            in str(screen.query_one("#skill-description-0", Static).render())
        )
        assert selections == [
            ConfigSelection(
                provider="openai",
                model="gpt-4o",
                default_model="gpt-4o",
                api_key="sk-test-ending",
                skills="auto",
                skill_selection_mode="selected",
                enabled_skills=("developer",),
            )
        ]


@pytest.mark.anyio
async def test_config_screen_is_fullscreen_and_splits_system_prompt_tab() -> None:
    screen = ConfigScreen(
        provider_name="openai",
        current_model="gpt-5.5",
        default_model="gpt-5.5",
        skills="auto",
        api_keys={"openai": "sk-test-ending"},
        system_prompt="You are AceAI.\n\nSkill instructions are active.",
    )

    async with AceAITUI([]).run_test(size=(100, 30)) as pilot:
        pilot.app.push_screen(screen)
        await pilot.pause()

        panel = screen.query_one("#config-panel")
        tabs = screen.query_one("#config-tabs", TabbedContent)
        settings_tab = screen.query_one("#settings-tab")
        prompt_tab = screen.query_one("#system-prompt-tab")
        prompt = screen.query_one("#system-prompt", Static)

        assert panel.region.width == 100
        assert panel.region.height == 30
        assert tabs.active == "settings-tab"
        assert screen.query_one("#provider", Input) in settings_tab.query("*")
        assert prompt in prompt_tab.query("*")
        assert "You are AceAI" in str(prompt.render())
        assert "Skill instructions are active" in str(prompt.render())


@pytest.mark.anyio
async def test_config_screen_has_tool_permissions_tab_and_selects_policy() -> None:
    tool_items = (
        ToolPermissionItem(
            name="run_shell_command",
            description="Run a shell command.",
            permission="ask",
        ),
        ToolPermissionItem(
            name="read_text_file",
            description="Read a file.",
            permission="always",
        ),
    )
    screen = ConfigScreen(
        provider_name="openai",
        current_model="gpt-5.5",
        default_model="gpt-5.5",
        skills="auto",
        api_keys={"openai": "sk-test-ending"},
        tool_permission_items=tool_items,
    )

    async with AceAITUI([]).run_test() as pilot:
        pilot.app.push_screen(screen)
        await pilot.pause()
        selections: list[ConfigSelection | None] = []

        def dismiss(selection: ConfigSelection | None) -> None:
            selections.append(selection)

        screen.dismiss = dismiss
        tabs = screen.query_one("#config-tabs", TabbedContent)
        tool_select = screen.query_one("#tool-permission-0", Select)

        assert screen.query_one("#tool-permissions-tab") in tabs.query("*")
        assert tool_select.value == "ask"

        tool_select.value = "never"
        _press_config_apply(screen)

        assert selections[-1].tool_permissions == {
            "run_shell_command": "never",
            "read_text_file": "always",
        }


@pytest.mark.anyio
async def test_config_screen_can_allow_or_disable_all_tools() -> None:
    tool_items = (
        ToolPermissionItem(
            name="run_shell_command",
            description="Run a shell command.",
            permission="ask",
        ),
        ToolPermissionItem(
            name="read_text_file",
            description="Read a file.",
            permission="always",
        ),
    )
    screen = ConfigScreen(
        provider_name="openai",
        current_model="gpt-5.5",
        default_model="gpt-5.5",
        skills="auto",
        api_keys={"openai": "sk-test-ending"},
        tool_permission_items=tool_items,
    )

    async with AceAITUI([]).run_test() as pilot:
        pilot.app.push_screen(screen)
        await pilot.pause()
        selections: list[ConfigSelection | None] = []

        def dismiss(selection: ConfigSelection | None) -> None:
            selections.append(selection)

        screen.dismiss = dismiss

        class DisablePressed:
            button = screen.query_one("#disable-all-tools", Button)

        screen.on_button_pressed(DisablePressed())
        assert screen.query_one("#tool-permission-0", Select).value == "never"
        assert screen.query_one("#tool-permission-1", Select).value == "never"

        class AllowPressed:
            button = screen.query_one("#allow-all-tools", Button)

        screen.on_button_pressed(AllowPressed())
        _press_config_apply(screen)

        assert selections[-1].tool_permissions == {
            "run_shell_command": "always",
            "read_text_file": "always",
        }


@pytest.mark.anyio
async def test_config_screen_can_disable_current_agent_skill() -> None:
    skill_item = SkillConfigItem(
        name="developer",
        description="Development workflow.",
        location="/skills/developer/SKILL.md",
    )
    screen = ConfigScreen(
        provider_name="openai",
        current_model="gpt-5.5",
        default_model="gpt-5.5",
        skills="auto",
        api_keys={"openai": "sk-test-ending"},
        skill_items=(skill_item,),
        skill_selection_mode="all",
        enabled_skills=(),
    )

    async with AceAITUI([]).run_test() as pilot:
        pilot.app.push_screen(screen)
        await pilot.pause()
        selections: list[ConfigSelection | None] = []

        def dismiss(selection: ConfigSelection | None) -> None:
            selections.append(selection)

        screen.dismiss = dismiss
        screen.query_one("#skill-0", Checkbox).value = False
        _press_config_apply(screen)

        assert selections[-1] == ConfigSelection(
            provider="openai",
            model="gpt-5.5",
            default_model="gpt-5.5",
            api_key="sk-test-ending",
            skills="auto",
            skill_selection_mode="selected",
            enabled_skills=(),
        )


@pytest.mark.anyio
async def test_config_screen_requires_provider_model_and_api_key() -> None:
    screen = ConfigScreen(
        provider_name="openai",
        current_model="gpt-5.5",
        default_model="gpt-5.5",
        skills="auto",
        api_keys={"openai": "sk-test-ending"},
    )

    async with AceAITUI([]).run_test() as pilot:
        pilot.app.push_screen(screen)
        await pilot.pause()
        selections: list[ConfigSelection | None] = []

        def dismiss(selection: ConfigSelection | None) -> None:
            selections.append(selection)

        screen.dismiss = dismiss
        provider_input = screen.query_one("#provider", Input)
        model_input = screen.query_one("#model", Input)
        api_key_input = screen.query_one("#api-key", Input)
        error = screen.query_one("#config-error", Static)

        provider_input.value = ""
        _press_config_apply(screen)
        assert selections == []
        assert str(error.render()) == "Provider is required"

        provider_input.value = "openai"
        model_input.value = ""
        _press_config_apply(screen)
        assert selections == []
        assert str(error.render()) == "Model is required"

        model_input.value = "gpt-5.5"
        api_key_input.value = ""
        _press_config_apply(screen)
        assert selections == []
        assert str(error.render()) == "API key is required"


@pytest.mark.anyio
async def test_config_screen_selects_prefixed_model_candidate_with_tab() -> None:
    screen = ConfigScreen(
        provider_name="deepseek",
        current_model="deepseek-v4-pro",
        default_model="deepseek-v4-pro",
        skills="auto",
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
async def test_config_screen_selects_prefixed_provider_candidate_with_tab() -> None:
    screen = ConfigScreen(
        provider_name="openai",
        current_model="gpt-5.5",
        default_model="gpt-5.5",
        skills="auto",
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
async def test_config_screen_hides_candidates_for_empty_input() -> None:
    screen = ConfigScreen(
        provider_name="openai",
        current_model="gpt-5.5",
        default_model="gpt-5.5",
        skills="auto",
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
    tmp_path,
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
    monkeypatch.chdir(tmp_path)
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

    clear_config()
    async with app.run_test():
        app._handle_config_selection(
            ConfigSelection(
                provider="deepseek",
                model="deepseek-v4-flash",
                default_model="deepseek-v4-pro",
                api_key="",
                skills="auto",
                skill_selection_mode="selected",
                enabled_skills=(),
            )
        )

    assert calls[-1] == AceAITUIConfig(
        provider="deepseek",
        api_key="deepseek-key",
        model="deepseek-v4-flash",
        default_model="deepseek-v4-pro",
        skills="auto",
        skill_selection_mode="selected",
        enabled_skills=[],
        api_keys={"openai": "openai-key", "deepseek": "deepseek-key"},
    )
    assert current_config() == calls[-1]
    assert (tmp_path / ".aceai" / "config.yml").exists()


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


def _press_config_apply(screen: ConfigScreen) -> None:
    class Pressed:
        button = screen.query_one("#apply", Button)

    screen.on_button_pressed(Pressed())
