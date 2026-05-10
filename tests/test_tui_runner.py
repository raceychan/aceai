import asyncio
import subprocess
from io import StringIO
from pathlib import Path

import pytest
from ididi import Graph
from rich.console import Console
from rich.style import Style
from rich.text import Text

from aceai import __version__
from aceai.agent.citations import (
    AdHocCitationOrigin,
    ConversationCitationOrigin,
    FileCitationOrigin,
    IdeaCitationOrigin,
    TurnCitation,
)
from aceai.core.agent import Agent
from aceai.core.executor import Executor
from aceai.agent.session import (
    MAIN_THREAD_ID,
    SessionEvent,
    SessionRecorder,
    SessionState,
    SessionStore,
)
from aceai.llm import LLMResponse
from aceai.core.run_state import ToolRunState
from aceai.core.skills import SkillLoader, SkillRegistry
from aceai.llm.models import LLMToolCall, LLMUsage
from aceai.llm.models import LLMStreamEvent
from aceai.agent.tui.events import TUIEvent
from aceai.agent.tui.session_adapter import tui_event_to_session_event
from aceai.agent.tui.session_replay import event_log_to_tui_events
from aceai.agent.tui.config import AgentAppConfig
from aceai.agent.ace_agent import ACE_AGENT_BUILTIN_SKILL_PATHS
from aceai.agent.config import ConfigAuditEntry, clear_config, current_config, load_config
from aceai.agent.provider_auth import CODEX_CLI_AUTH_SENTINEL
from aceai.agent import app as agent_app_module
from aceai.agent import session_service as session_service_module
from aceai.agent.app import UpdateCheckResult
from aceai.agent.memory.ideas import IdeaStore
from aceai.agent.project import ProjectStore
from aceai.agent.tui import app as tui_app_module
from aceai.agent.tui import runner as tui_runner_module
from aceai.agent.tui.widgets import stream as stream_widget_module
from aceai.agent.tui.app import AceAITUI
from aceai.agent.tui.runner import (
    UPDATE_INSTRUCTIONS,
    AceAIConfiguredTUI,
    UpdateCommandResult,
    _iter_reference_file_candidates,
)
from aceai.agent.tui.setup import (
    ConfigScreen,
    ConfigSelection,
    IdeaListWidget,
    IdeaPickerScreen,
    SkillConfigItem,
    ToolPermissionItem,
    _skill_config_items,
)
from aceai.agent.tui.widgets import ApprovalWidget
from aceai.agent.tui.widgets import (
    CommandCompletionWidget,
    CommandInput,
    CitationPreviewWidget,
    QueuedTurnsWidget,
    ReferenceCompletionWidget,
    StatusBarWidget,
    StreamWidget,
    SubagentStatusWidget,
    TopBarWidget,
)
from aceai.agent.tui.widgets.input import (
    _citation_preview_renderable,
    _citation_preview_text,
    _queued_turns_renderable,
)
from textual.events import Click, Key
from textual.containers import VerticalScroll
from textual.widgets import (
    Button,
    Checkbox,
    Input,
    RichLog,
    Select,
    Static,
    TabbedContent,
)


def _test_citation(content: str) -> TurnCitation:
    return TurnCitation(
        content=content,
        origin=AdHocCitationOrigin(kind="ad_hoc", label="test"),
    )


async def _wait_until(pilot, predicate, timeout: float = 1.0) -> None:
    async def wait_for_match() -> None:
        while not predicate():
            await pilot.pause()

    await asyncio.wait_for(wait_for_match(), timeout=timeout)
    assert predicate()


def write_skill(root: Path, name: str, description: str, body: str) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                f"name: {name}",
                f"description: {description}",
                "---",
                body,
            ]
        ),
        encoding="utf-8",
    )
    return skill_dir


def _make_interactive_tui_from_agent(agent: Agent, **kwargs):
    config = AgentAppConfig(
        provider="openai",
        api_key="test-key",
        model=agent.default_model,
        default_model=agent.default_model,
        api_keys={"openai": "test-key"},
    )
    def agent_factory(cfg):
        return agent
    return AceAIConfiguredTUI(
        agent_factory,
        initial_config=config,
        initial_question="",
        default_model=agent.default_model,
        **kwargs,
    )


@pytest.fixture(autouse=True)
def tui_session_store(monkeypatch, tmp_path) -> SessionStore:
    monkeypatch.chdir(tmp_path)
    store = SessionStore(tmp_path / "sessions")
    monkeypatch.setattr(tui_app_module, "SessionStore", lambda **kwargs: store)
    monkeypatch.setattr(
        session_service_module,
        "SessionStore",
        lambda **kwargs: store,
    )

    async def no_update() -> None:
        return None

    monkeypatch.setattr(agent_app_module, "check_for_updates", no_update)
    return store


class StubExecutor:
    def __init__(self) -> None:
        self._skill_registry = SkillRegistry()

    @property
    def prompt_instructions(self) -> str:
        return ""

    @property
    def skill_registry(self) -> SkillRegistry:
        return self._skill_registry

    @property
    def hosted_tools(self) -> list[object]:
        return []

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
        self._skill_registry = SkillRegistry()

    @property
    def prompt_instructions(self) -> str:
        return ""

    @property
    def skill_registry(self) -> SkillRegistry:
        return self._skill_registry

    @property
    def hosted_tools(self) -> list[object]:
        return []

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


class GatedMultiRunLLMService:
    def __init__(
        self,
        streams: list[list[LLMStreamEvent]],
        *,
        gate: asyncio.Event,
    ) -> None:
        self._streams = [list(stream) for stream in streams]
        self._gate = gate
        self.calls: list[dict] = []

    async def stream(self, **request):
        self.calls.append(request)
        events = self._streams.pop(0)
        for index, event in enumerate(events):
            yield event
            if len(self.calls) == 1 and index == 0:
                await self._gate.wait()

    async def complete(self, **request) -> LLMResponse:
        raise AssertionError("live TUI should use streaming")


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
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent)
    assert app._session_recorder is None
    assert app._session_id is None

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        app.on_input_submitted(Input.Submitted(command_input, "What now?"))
        await _wait_until(pilot, lambda: app._state.status == "completed")

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
async def test_interactive_tui_start_run_displays_citations_separately(
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
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent)

    async with app.run_test() as pilot:
        app.start_run(
            "Explain it",
            citations=(
                TurnCitation(
                    content="The job is pending.",
                    origin=ConversationCitationOrigin(
                        kind="conversation",
                        event_id="event-1",
                        role="assistant",
                        span_start=0,
                        span_end=19,
                    ),
                ),
            ),
        )
        await _wait_until(pilot, lambda: len(llm_service.calls) == 1)

        visible_user_event = app._state.events[0]
        assert visible_user_event.content == "Explain it"
        assert visible_user_event.citations[0].content == "The job is pending."

        user_text = llm_service.calls[0]["messages"][-1].content[0]["data"]
        assert "<aceai_cited_context>" in user_text
        assert "<user_request>\nExplain it\n</user_request>" in user_text

        assert app._session_id is not None
        event_log = tui_session_store.load_event_log(app._session_id)
        assert event_log.events[0].payload["content"] == "Explain it"
        assert event_log.events[0].payload["citations"][0]["content"] == (
            "The job is pending."
        )


def test_citation_preview_uses_three_content_lines() -> None:
    display_text = _citation_preview_text(
        (_test_citation("first line\nsecond line\nthird line\nfourth line"),)
    )

    assert display_text.count("\n") == 3
    assert "first line" in display_text
    assert "second line" in display_text
    assert "...more" in display_text
    assert "fourth line" not in display_text


def test_citation_preview_more_marker_has_distinct_style() -> None:
    renderable = _citation_preview_renderable("cited source\nbody ...more")

    assert isinstance(renderable, Text)
    assert renderable.plain == "cited source\nbody ...more"
    marker_spans = [
        span
        for span in renderable.spans
        if renderable.plain[span.start : span.end] == "...more"
    ]
    assert len(marker_spans) == 1
    assert str(marker_spans[0].style) == "bold #ebcb8b"


def test_citation_preview_short_content_keeps_three_content_lines() -> None:
    display_text = _citation_preview_text((_test_citation("one line"),))

    assert display_text == "cited source\none line\n \n "


def test_citation_preview_truncates_across_multiple_citations() -> None:
    display_text = _citation_preview_text(
        (
            _test_citation("one\ntwo"),
            _test_citation("three\nfour"),
        )
    )

    assert display_text == "cited source\none\ntwo\nthree ...more"


def test_citation_preview_skips_blank_lines_for_compact_display() -> None:
    display_text = _citation_preview_text(
        (_test_citation("Add a Learn button\n\nBy default, save selected failures."),)
    )

    assert display_text == (
        "cited source\nAdd a Learn button\nBy default, save selected failures.\n "
    )


@pytest.mark.anyio
async def test_interactive_tui_enter_key_submits_question() -> None:
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
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent)

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        command_input.value = "What now?"
        command_input.focus()
        await pilot.press("enter")
        await _wait_until(pilot, lambda: len(llm_service.calls) == 1)

        assert command_input.value == ""
        assert llm_service.calls[0]["messages"][-1].content[0]["data"] == "What now?"


@pytest.mark.anyio
async def test_interactive_tui_enter_key_completes_then_submits_slash_command() -> None:
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([]),  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent)

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        command_input.value = "/st"
        command_input.focus()
        await pilot.press("enter")
        await pilot.pause()

        assert command_input.value == "/stats "

        await pilot.press("enter")
        await pilot.pause()

        assert command_input.value == ""
        assert app.screen.__class__.__name__ == "MetadataScreen"
        assert app.screen.query_one("#metadata-body", RichLog) is not None


@pytest.mark.anyio
async def test_interactive_tui_clear_command_resets_state() -> None:
    llm_service = StubLLMService([])
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent)

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
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent)

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        app.on_input_submitted(Input.Submitted(command_input, "First?"))
        await _wait_until(pilot, lambda: len(llm_service.calls) == 1)
        first_event_count = len(app._state.events)

        app.on_input_submitted(Input.Submitted(command_input, "Second?"))
        await _wait_until(pilot, lambda: len(llm_service.calls) == 2)

        assert len(app._state.events) > first_event_count
        assert app._state.final_answer == "second"
        assert llm_service.calls[0]["messages"][-1].content[0]["data"] == "First?"
        assert llm_service.calls[1]["messages"][-1].content[0]["data"] == "Second?"


@pytest.mark.anyio
async def test_interactive_tui_enqueues_question_while_run_is_active() -> None:
    gate = asyncio.Event()
    llm_service = GatedMultiRunLLMService(
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
                    event_type="response.completed",
                    response=LLMResponse(text="second"),
                ),
            ],
        ],
        gate=gate,
    )
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent)

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        app.on_input_submitted(Input.Submitted(command_input, "First?"))
        await _wait_until(pilot, lambda: len(llm_service.calls) == 1)

        app.on_input_submitted(Input.Submitted(command_input, "Second?"))
        await _wait_until(
            pilot,
            lambda: app._agent_app.queued_questions == ("Second?",),
        )

        assert len(llm_service.calls) == 1
        assert app._agent_app.queued_questions == ("Second?",)
        queued_turns = app.query_one(QueuedTurnsWidget)
        assert not queued_turns.has_class("hidden")
        assert queued_turns.renderable.startswith("queued messages\n1. Second?")
        assert queued_turns.renderable.endswith("[ > ] [ x ]")

        gate.set()
        await _wait_until(pilot, lambda: app._state.final_answer == "second")

        assert len(llm_service.calls) == 2
        assert app._agent_app.queued_questions == ()
        assert queued_turns.has_class("hidden")
        assert app._state.final_answer == "second"
        assert llm_service.calls[1]["messages"][-1].content[0]["data"] == "Second?"


def test_queued_turns_widget_clicks_ascii_actions(monkeypatch) -> None:
    widget = QueuedTurnsWidget()
    messages: list[object] = []
    monkeypatch.setattr(widget, "post_message", messages.append)

    widget.on_click(
        Click(
            widget,
            x=90,
            y=1,
            delta_x=90,
            delta_y=1,
            button=1,
            shift=False,
            meta=False,
            ctrl=False,
            style=Style(meta={"queued_action": "steer", "queued_index": 0}),
        )
    )
    widget.on_click(
        Click(
            widget,
            x=92,
            y=1,
            delta_x=92,
            delta_y=1,
            button=1,
            shift=False,
            meta=False,
            ctrl=False,
            style=Style(meta={"queued_action": "cancel", "queued_index": 0}),
        )
    )

    assert len(messages) == 2
    assert isinstance(messages[0], QueuedTurnsWidget.Selected)
    assert messages[0].index == 0
    assert isinstance(messages[1], QueuedTurnsWidget.Cancelled)
    assert messages[1].index == 0


def test_queued_turns_actions_render_at_right_edge() -> None:
    _, renderable = _queued_turns_renderable(("阿斯顿发送 ...",))
    output = StringIO()
    console = Console(file=output, width=80, force_terminal=False, color_system=None)

    console.print(renderable)

    lines = output.getvalue().splitlines()
    assert lines[1].startswith("1. 阿斯顿发送 ...")
    assert lines[1].endswith("[ > ] [ x ]")
    assert lines[1].index("[ > ]") > 60


@pytest.mark.anyio
async def test_interactive_tui_clicking_queued_question_steers_it() -> None:
    gate = asyncio.Event()
    llm_service = GatedMultiRunLLMService(
        [
            [
                LLMStreamEvent(
                    event_type="response.output_text.delta",
                    text_delta="active",
                ),
                LLMStreamEvent(
                    event_type="response.completed",
                    response=LLMResponse(text="active"),
                ),
            ],
            [
                LLMStreamEvent(
                    event_type="response.completed",
                    response=LLMResponse(text="first"),
                ),
            ],
            [
                LLMStreamEvent(
                    event_type="response.completed",
                    response=LLMResponse(text="second"),
                ),
            ],
        ],
        gate=gate,
    )
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent)

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        app.on_input_submitted(Input.Submitted(command_input, "Active"))
        await _wait_until(pilot, lambda: len(llm_service.calls) == 1)
        app.on_input_submitted(Input.Submitted(command_input, "First queued"))
        app.on_input_submitted(Input.Submitted(command_input, "Second queued"))
        await _wait_until(
            pilot,
            lambda: (
                app._agent_app.queued_questions
                == (
                    "First queued",
                    "Second queued",
                )
            ),
        )

        queued_turns = app.query_one(QueuedTurnsWidget)
        assert app._agent_app.queued_questions == ("First queued", "Second queued")
        queued_turns.post_message(QueuedTurnsWidget.Selected(index=0))
        await _wait_until(pilot, lambda: len(llm_service.calls) == 3)

        assert len(llm_service.calls) == 3
        assert app._agent_app.queued_questions == ()
        assert app._state.final_answer == "second"
        assert (
            llm_service.calls[1]["messages"][-1].content[0]["data"] == "First queued"
        )
        assert llm_service.calls[2]["messages"][-1].content[0]["data"] == "Second queued"


@pytest.mark.anyio
async def test_interactive_tui_clicking_queued_cancel_removes_question() -> None:
    gate = asyncio.Event()
    llm_service = GatedMultiRunLLMService(
        [
            [
                LLMStreamEvent(
                    event_type="response.output_text.delta",
                    text_delta="active",
                ),
                LLMStreamEvent(
                    event_type="response.completed",
                    response=LLMResponse(text="active"),
                ),
            ],
            [
                LLMStreamEvent(
                    event_type="response.completed",
                    response=LLMResponse(text="second"),
                ),
            ],
        ],
        gate=gate,
    )
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent)

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        app.on_input_submitted(Input.Submitted(command_input, "Active"))
        await _wait_until(pilot, lambda: len(llm_service.calls) == 1)
        app.on_input_submitted(Input.Submitted(command_input, "First queued"))
        app.on_input_submitted(Input.Submitted(command_input, "Second queued"))
        await _wait_until(
            pilot,
            lambda: (
                app._agent_app.queued_questions
                == (
                    "First queued",
                    "Second queued",
                )
            ),
        )

        queued_turns = app.query_one(QueuedTurnsWidget)
        queued_turns.post_message(QueuedTurnsWidget.Cancelled(index=0))
        await _wait_until(
            pilot,
            lambda: app._agent_app.queued_questions == ("Second queued",),
        )

        assert any(
            event.kind == "session_notice"
            and event.content == "Cancelled queued message 1: First queued"
            for event in app._state.events
        )
        gate.set()
        await _wait_until(pilot, lambda: len(llm_service.calls) == 2)

        assert app._agent_app.queued_questions == ()
        assert llm_service.calls[1]["messages"][-1].content[0]["data"] == "Second queued"


@pytest.mark.anyio
async def test_interactive_tui_steer_cancels_active_run_and_keeps_queue() -> None:
    gate = asyncio.Event()
    llm_service = GatedMultiRunLLMService(
        [
            [
                LLMStreamEvent(
                    event_type="response.output_text.delta",
                    text_delta="old",
                ),
                LLMStreamEvent(
                    event_type="response.completed",
                    response=LLMResponse(text="old"),
                ),
            ],
            [
                LLMStreamEvent(
                    event_type="response.completed",
                    response=LLMResponse(text="new"),
                ),
            ],
            [
                LLMStreamEvent(
                    event_type="response.completed",
                    response=LLMResponse(text="queued"),
                ),
            ],
        ],
        gate=gate,
    )
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent)

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        app.on_input_submitted(Input.Submitted(command_input, "Old plan"))
        await _wait_until(pilot, lambda: len(llm_service.calls) == 1)
        app.on_input_submitted(Input.Submitted(command_input, "Queued plan"))
        await _wait_until(
            pilot,
            lambda: app._agent_app.queued_questions == ("Queued plan",),
        )

        app.on_input_submitted(Input.Submitted(command_input, "/steer New plan"))
        await _wait_until(pilot, lambda: len(llm_service.calls) == 3)

        assert len(llm_service.calls) == 3
        assert app._agent_app.queued_questions == ()
        assert app._state.final_answer == "queued"
        assert llm_service.calls[1]["messages"][-1].content[0]["data"] == "New plan"
        assert llm_service.calls[2]["messages"][-1].content[0]["data"] == "Queued plan"


@pytest.mark.anyio
async def test_interactive_tui_escape_cancels_active_run_and_keeps_queue() -> None:
    gate = asyncio.Event()
    llm_service = GatedMultiRunLLMService(
        [
            [
                LLMStreamEvent(
                    event_type="response.output_text.delta",
                    text_delta="long answer",
                ),
                LLMStreamEvent(
                    event_type="response.completed",
                    response=LLMResponse(text="long answer"),
                ),
            ],
        ],
        gate=gate,
    )
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent)

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        app.on_input_submitted(Input.Submitted(command_input, "Explain slowly"))
        await _wait_until(pilot, lambda: len(llm_service.calls) == 1)
        app.on_input_submitted(Input.Submitted(command_input, "Next queued"))
        await _wait_until(
            pilot,
            lambda: app._agent_app.queued_questions == ("Next queued",),
        )

        await pilot.press("escape")
        await _wait_until(
            pilot,
            lambda: (
                app.query_one(StatusBarWidget).current_text
                == "Esc again stops response"
            ),
        )

        assert len(llm_service.calls) == 1
        assert app._active_worker is not None
        assert app._active_worker.is_running
        assert app._agent_app.queued_questions == ("Next queued",)
        assert app.query_one(StatusBarWidget).current_text == "Esc again stops response"
        assert app.query_one(StatusBarWidget).current_style == "bold #bf616a"

        await pilot.press("escape")
        await _wait_until(pilot, lambda: app._active_worker is None)

        assert len(llm_service.calls) == 1
        assert app._active_worker is None
        assert app._active_run is None
        assert app._agent_app.queued_questions == ("Next queued",)
        assert app._state.status == "failed"
        assert app._state.error == "Cancelled current response."


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
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent)

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        app.on_input_submitted(Input.Submitted(command_input, "Write it"))
        await _wait_until(pilot, lambda: app._state.status == "suspended")

        assert app._state.status == "suspended"
        assert executor.calls == []
        assert app._active_run is not None
        assert app._active_run.status == "suspended"
        assert command_input.placeholder == "Choose Approve or Reject"
        status = app.query_one(StatusBarWidget)
        assert "action: choose Approve or Reject" in status.current_text
        approval = app.query_one(ApprovalWidget)
        assert not approval.has_class("collapsed")
        assert (
            approval.query_one("#approval-approve", Button).label.plain == "A Approve"
        )
        assert approval.query_one("#approval-reject", Button).label.plain == "R Reject"
        assert "content:" in str(
            approval.query_one("#approval-summary", Static).render()
        )

        approval.post_message(ApprovalWidget.Selected(approved=True))
        await _wait_until(pilot, lambda: app._state.status == "completed")

        assert app._state.status == "completed"
        assert app._state.final_answer == "done"
        assert command_input.placeholder == "Ask AceAI or type /quit"
        assert approval.has_class("collapsed")
        assert executor.calls == [call]
        assert app._llm_history[-1].role == "assistant"
        assert app._llm_history[-1].content[0]["data"] == "done"


@pytest.mark.anyio
async def test_interactive_tui_accepts_typed_approval_shortcut() -> None:
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
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent)

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        app.on_input_submitted(Input.Submitted(command_input, "Write it"))
        await _wait_until(pilot, lambda: app._state.status == "suspended")

        command_input.value = "a"
        app.on_command_input_submitted(CommandInput.Submitted(command_input, "a"))
        await _wait_until(pilot, lambda: app._state.status == "completed")

        assert app._state.final_answer == "done"
        assert executor.calls == [call]
        assert command_input.value == ""


@pytest.mark.anyio
async def test_interactive_tui_shows_next_approval_after_resume_suspends_again() -> (
    None
):
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
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent)

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        app.on_input_submitted(Input.Submitted(command_input, "Write and run it"))
        await _wait_until(pilot, lambda: app._state.status == "suspended")

        approval = app.query_one(ApprovalWidget)
        assert app._state.status == "suspended"
        assert not approval.has_class("collapsed")
        assert "write_text_file" in str(
            approval.query_one("#approval-summary", Static).render()
        )

        approval.post_message(ApprovalWidget.Selected(approved=True))
        await _wait_until(pilot, lambda: executor.calls == [first_call])

        assert app._state.status == "suspended"
        assert app._active_run is not None
        assert app._active_run.status == "suspended"
        assert command_input.placeholder == "Choose Approve or Reject"
        assert executor.calls == [first_call]
        assert not approval.has_class("collapsed")
        summary = str(approval.query_one("#approval-summary", Static).render())
        assert "run_shell_command" in summary
        assert "python binary_search.py" in summary


@pytest.mark.anyio
async def test_interactive_tui_persists_selected_model_in_session_state(
    tmp_path,
) -> None:
    store = SessionStore(tmp_path)
    metadata = store.create_session()
    SessionRecorder(store, metadata.session_id).record(
        tui_event_to_session_event(TUIEvent.user_message("keep this session"))
    )
    llm_service = StubLLMService([])
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(
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
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(
        agent,
        session_recorder=SessionRecorder(store, metadata.session_id),
        session_id=metadata.session_id,
    )

    async with app.run_test():
        app.append_event(
            TUIEvent.session_notice(f"Resumed session {metadata.session_id}")
        )
        app.switch_model("gpt-5.5")

    assert store.list_sessions() == []


@pytest.mark.anyio
async def test_interactive_tui_status_bar_shows_selected_model() -> None:
    llm_service = StubLLMService([])
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent)

    async with app.run_test() as pilot:
        status = app.query_one(StatusBarWidget)

        assert "model: gpt-4o" in status.current_text
        assert "reasoning:" not in status.current_text

        event_count = len(app._state.events)
        app.switch_model("gpt-5.5")

        assert "model: gpt-5.5" in status.current_text
        assert "reasoning: auto" in status.current_text
        assert len(app._state.events) == event_count

        await pilot.pause()

        assert "model: gpt-5.5" in status.current_text
        assert "reasoning: auto" in status.current_text

        status.show_notice("First notice", timeout=0.01)

        assert status.current_text == "First notice"

        await _wait_until(pilot, lambda: "model: gpt-5.5" in status.current_text)

        assert "model: gpt-5.5" in status.current_text
        assert "context:" not in status.current_text
        assert "cache rate:" not in status.current_text
        assert "cost:" not in status.current_text
        assert "version:" not in status.current_text


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
                        cache_miss_input_tokens=1_000,
                        input_cache_hit_rate=200 / 1_200,
                        output_tokens=300,
                        total_tokens=1_500,
                    ),
                ),
            ),
        ]
    )
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent)

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        app.on_input_submitted(Input.Submitted(command_input, "What now?"))
        await _wait_until(
            pilot,
            lambda: "context: 1.2k" in app.query_one(StatusBarWidget).current_text,
        )

        status = app.query_one(StatusBarWidget)
        assert "context: 1.2k" in status.current_text
        assert "cache rate: 16.7%" in status.current_text
        assert "cost: $0.014" in status.current_text
        assert "time:" in status.current_text


@pytest.mark.anyio
async def test_interactive_tui_switch_model_resets_cache_rate() -> None:
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
                        cache_miss_input_tokens=1_000,
                        input_cache_hit_rate=200 / 1_200,
                        output_tokens=300,
                        total_tokens=1_500,
                    ),
                ),
            ),
        ]
    )
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent)

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        app.on_input_submitted(Input.Submitted(command_input, "What now?"))
        await _wait_until(
            pilot,
            lambda: "cache rate: 16.7%" in app.query_one(StatusBarWidget).current_text,
        )

        status = app.query_one(StatusBarWidget)
        assert "cache rate: 16.7%" in status.current_text

        app.switch_model("gpt-5.5")

        assert "model: gpt-5.5" in status.current_text
        assert "cache rate: 0.0%" in status.current_text
        assert app._state.usage.current_input_cache_hit_rate == 0.0


@pytest.mark.anyio
async def test_configured_tui_switch_model_saves_project_config(
    tmp_path,
    monkeypatch,
) -> None:
    llm_service = StubLLMService([])

    def agent_factory(config: AgentAppConfig) -> Agent:
        return Agent(
            prompt="Prompt",
            default_model=config.default_model,
            llm_service=llm_service,  # type: ignore[arg-type]
            executor=StubExecutor(),  # type: ignore[arg-type]
        )

    monkeypatch.chdir(tmp_path)
    clear_config()
    app = AceAIConfiguredTUI(
        agent_factory,
        initial_config=AgentAppConfig(
            provider="openai",
            api_key="openai-key",
            model="gpt-4o",
            default_model="gpt-4o",
            api_keys={"openai": "openai-key"},
        ),
        initial_question="",
        default_model="gpt-4o",
    )

    async with app.run_test():
        app.switch_model("gpt-5.5")

    saved_config = load_config(tmp_path / ".aceai" / "config.yml")
    assert saved_config is not None
    assert saved_config.model == "gpt-5.5"
    assert saved_config.default_model == "gpt-5.5"


@pytest.mark.anyio
async def test_configured_tui_tool_allow_all_saves_project_config(
    tmp_path,
    monkeypatch,
) -> None:
    llm_service = StubLLMService([])

    def agent_factory(config: AgentAppConfig) -> Agent:
        return Agent(
            prompt="Prompt",
            default_model=config.default_model,
            llm_service=llm_service,  # type: ignore[arg-type]
            executor=StubExecutor(),  # type: ignore[arg-type]
        )

    monkeypatch.chdir(tmp_path)
    clear_config()
    app = AceAIConfiguredTUI(
        agent_factory,
        initial_config=AgentAppConfig(
            provider="openai",
            api_key="openai-key",
            model="gpt-5.5",
            default_model="gpt-5.5",
            api_keys={"openai": "openai-key"},
        ),
        initial_question="",
        default_model="gpt-5.5",
    )

    async with app.run_test() as pilot:
        app.action_config()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, ConfigScreen)
        screen.query_one("#tool-tag-allow-all-0", Button).press()
        await pilot.pause()
        screen.query_one("#apply-tools", Button).press()
        await pilot.pause()

    saved_config = load_config(tmp_path / ".aceai" / "config.yml")
    assert saved_config is not None
    assert saved_config.tool_permissions["run_shell_command"] == "always"
    assert saved_config.tool_permissions["write_text_file"] == "always"


@pytest.mark.anyio
async def test_configured_tui_tool_permission_select_saves_project_config(
    tmp_path,
    monkeypatch,
) -> None:
    llm_service = StubLLMService([])

    def agent_factory(config: AgentAppConfig) -> Agent:
        return Agent(
            prompt="Prompt",
            default_model=config.default_model,
            llm_service=llm_service,  # type: ignore[arg-type]
            executor=StubExecutor(),  # type: ignore[arg-type]
        )

    monkeypatch.chdir(tmp_path)
    clear_config()
    app = AceAIConfiguredTUI(
        agent_factory,
        initial_config=AgentAppConfig(
            provider="openai",
            api_key="openai-key",
            model="gpt-5.5",
            default_model="gpt-5.5",
            api_keys={"openai": "openai-key"},
        ),
        initial_question="",
        default_model="gpt-5.5",
    )

    async with app.run_test() as pilot:
        app.action_config()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, ConfigScreen)
        screen.query_one("#tool-permission-8", Select).value = "always"
        await pilot.pause()
        screen.query_one("#apply-tools", Button).press()
        await pilot.pause()

    saved_config = load_config(tmp_path / ".aceai" / "config.yml")
    assert saved_config is not None
    assert saved_config.tool_permissions["run_shell_command"] == "always"


@pytest.mark.anyio
async def test_configured_tui_tool_tag_disable_saves_project_config(
    tmp_path,
    monkeypatch,
) -> None:
    llm_service = StubLLMService([])

    def agent_factory(config: AgentAppConfig) -> Agent:
        return Agent(
            prompt="Prompt",
            default_model=config.default_model,
            llm_service=llm_service,  # type: ignore[arg-type]
            executor=StubExecutor(),  # type: ignore[arg-type]
        )

    monkeypatch.chdir(tmp_path)
    clear_config()
    app = AceAIConfiguredTUI(
        agent_factory,
        initial_config=AgentAppConfig(
            provider="openai",
            api_key="openai-key",
            model="gpt-5.5",
            default_model="gpt-5.5",
            api_keys={"openai": "openai-key"},
        ),
        initial_question="",
        default_model="gpt-5.5",
    )

    async with app.run_test() as pilot:
        app.action_config()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, ConfigScreen)
        screen.query_one("#tool-tag-enabled-0", Checkbox).value = False
        await pilot.pause()
        screen.query_one("#apply-tools", Button).press()
        await pilot.pause()

    saved_config = load_config(tmp_path / ".aceai" / "config.yml")
    assert saved_config is not None
    assert saved_config.tool_enabled["run_shell_command"] is False
    assert saved_config.tool_enabled["write_text_file"] is False


@pytest.mark.anyio
async def test_configured_tui_tool_max_calls_saves_project_config(
    tmp_path,
    monkeypatch,
) -> None:
    llm_service = StubLLMService([])

    def agent_factory(config: AgentAppConfig) -> Agent:
        return Agent(
            prompt="Prompt",
            default_model=config.default_model,
            llm_service=llm_service,  # type: ignore[arg-type]
            executor=StubExecutor(),  # type: ignore[arg-type]
        )

    monkeypatch.chdir(tmp_path)
    clear_config()
    app = AceAIConfiguredTUI(
        agent_factory,
        initial_config=AgentAppConfig(
            provider="openai",
            api_key="openai-key",
            model="gpt-5.5",
            default_model="gpt-5.5",
            api_keys={"openai": "openai-key"},
        ),
        initial_question="",
        default_model="gpt-5.5",
    )

    async with app.run_test() as pilot:
        app.action_config()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, ConfigScreen)
        screen.query_one("#tool-max-calls-8", Input).value = "7"
        await pilot.pause()
        screen.query_one("#apply-tools", Button).press()
        await pilot.pause()

    saved_config = load_config(tmp_path / ".aceai" / "config.yml")
    assert saved_config is not None
    assert saved_config.tool_max_calls["run_shell_command"] == 7


@pytest.mark.anyio
async def test_configured_tui_removed_provider_saves_and_switches_active_provider(
    tmp_path,
    monkeypatch,
) -> None:
    llm_service = StubLLMService([])

    def agent_factory(config: AgentAppConfig) -> Agent:
        return Agent(
            prompt="Prompt",
            default_model=config.default_model,
            llm_service=llm_service,  # type: ignore[arg-type]
            executor=StubExecutor(),  # type: ignore[arg-type]
        )

    monkeypatch.chdir(tmp_path)
    clear_config()
    app = AceAIConfiguredTUI(
        agent_factory,
        initial_config=AgentAppConfig(
            provider="openai",
            api_key="openai-key",
            model="gpt-5.5",
            default_model="gpt-5.5",
            api_keys={"openai": "openai-key", "deepseek": "deepseek-key"},
        ),
        initial_question="",
        default_model="gpt-5.5",
    )

    async with app.run_test() as pilot:
        app.action_config()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, ConfigScreen)
        screen.query_one("#provider", Input).value = ""
        await pilot.pause()
        await pilot.click("#provider-disable-0")
        await pilot.pause()
        assert screen.query_one("#provider", Input).value == "deepseek"
        screen.query_one("#apply", Button).press()
        await pilot.pause()

    saved_config = load_config(tmp_path / ".aceai" / "config.yml")
    assert saved_config is not None
    assert saved_config.provider == "deepseek"
    assert saved_config.model == "deepseek-v4-pro"
    assert saved_config.default_model == "deepseek-v4-pro"
    assert saved_config.disabled_providers == ["openai"]


@pytest.mark.anyio
async def test_configured_tui_removed_provider_stays_removed_after_reopen(
    tmp_path,
    monkeypatch,
) -> None:
    llm_service = StubLLMService([])

    def agent_factory(config: AgentAppConfig) -> Agent:
        return Agent(
            prompt="Prompt",
            default_model=config.default_model,
            llm_service=llm_service,  # type: ignore[arg-type]
            executor=StubExecutor(),  # type: ignore[arg-type]
        )

    monkeypatch.chdir(tmp_path)
    clear_config()
    app = AceAIConfiguredTUI(
        agent_factory,
        initial_config=AgentAppConfig(
            provider="deepseek",
            api_key="deepseek-key",
            model="deepseek-v4-pro",
            default_model="deepseek-v4-pro",
            api_keys={"openai": "openai-key", "deepseek": "deepseek-key"},
            disabled_providers=["openai"],
        ),
        initial_question="",
        default_model="gpt-5.5",
    )

    async with app.run_test() as pilot:
        app.action_config()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, ConfigScreen)
        screen.query_one("#provider", Input).value = ""
        await pilot.pause()

        assert screen.query_one("#provider-candidate-row-0").has_class("hidden")
        assert not screen.query_one("#provider-disabled-chip-0").has_class("hidden")
        assert screen.query_one("#provider", Input).value == ""


@pytest.mark.anyio
async def test_configured_tui_removed_only_keyed_provider_falls_back_to_codex_and_saves(
    tmp_path,
    monkeypatch,
) -> None:
    llm_service = StubLLMService([])

    def agent_factory(config: AgentAppConfig) -> Agent:
        return Agent(
            prompt="Prompt",
            default_model=config.default_model,
            llm_service=llm_service,  # type: ignore[arg-type]
            executor=StubExecutor(),  # type: ignore[arg-type]
        )

    monkeypatch.chdir(tmp_path)
    clear_config()
    app = AceAIConfiguredTUI(
        agent_factory,
        initial_config=AgentAppConfig(
            provider="openai",
            api_key="openai-key",
            model="gpt-5.5",
            default_model="gpt-5.5",
            api_keys={"openai": "openai-key"},
        ),
        initial_question="",
        default_model="gpt-5.5",
    )

    async with app.run_test() as pilot:
        app.action_config()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, ConfigScreen)
        screen.query_one("#provider", Input).value = ""
        await pilot.pause()
        await pilot.click("#provider-disable-0")
        await pilot.pause()
        assert screen.query_one("#provider", Input).value == "codex"
        screen.query_one("#apply", Button).press()
        await pilot.pause()

    saved_config = load_config(tmp_path / ".aceai" / "config.yml")
    assert saved_config is not None
    assert saved_config.provider == "codex"
    assert saved_config.api_key == CODEX_CLI_AUTH_SENTINEL
    assert saved_config.disabled_providers == ["openai"]


@pytest.mark.anyio
async def test_configured_tui_removed_provider_persists_immediately_without_apply(
    tmp_path,
    monkeypatch,
) -> None:
    llm_service = StubLLMService([])

    def agent_factory(config: AgentAppConfig) -> Agent:
        return Agent(
            prompt="Prompt",
            default_model=config.default_model,
            llm_service=llm_service,  # type: ignore[arg-type]
            executor=StubExecutor(),  # type: ignore[arg-type]
        )

    monkeypatch.chdir(tmp_path)
    clear_config()
    app = AceAIConfiguredTUI(
        agent_factory,
        initial_config=AgentAppConfig(
            provider="openai",
            api_key="openai-key",
            model="gpt-5.5",
            default_model="gpt-5.5",
            api_keys={"openai": "openai-key"},
        ),
        initial_question="",
        default_model="gpt-5.5",
    )

    async with app.run_test() as pilot:
        app.action_config()
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, ConfigScreen)
        screen.query_one("#provider", Input).value = ""
        await pilot.pause()
        await pilot.click("#provider-disable-0")
        await pilot.pause()

        saved_config = load_config(tmp_path / ".aceai" / "config.yml")
        assert saved_config is not None
        assert saved_config.provider == "codex"
        assert saved_config.disabled_providers == ["openai"]


@pytest.mark.anyio
async def test_configured_tui_applies_reasoning_level_to_requests(
    tmp_path, monkeypatch
) -> None:
    llm_service = StubLLMService(
        [
            LLMStreamEvent(
                event_type="response.completed",
                response=LLMResponse(text="done"),
            )
        ]
    )

    def agent_factory(config: AgentAppConfig) -> Agent:
        return Agent(
            prompt="Prompt",
            default_model=config.default_model,
            llm_service=llm_service,  # type: ignore[arg-type]
            executor=StubExecutor(),  # type: ignore[arg-type]
        )

    monkeypatch.chdir(tmp_path)
    clear_config()
    app = AceAIConfiguredTUI(
        agent_factory,
        initial_config=AgentAppConfig(
            provider="openai",
            api_key="openai-key",
            model="gpt-5.5",
            default_model="gpt-5.5",
            api_keys={"openai": "openai-key"},
            reasoning_level="high",
        ),
        initial_question="",
        default_model="gpt-5.5",
    )

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        app.on_input_submitted(Input.Submitted(command_input, "Think deeply"))
        await _wait_until(pilot, lambda: len(llm_service.calls) == 1)

    assert llm_service.calls[0]["metadata"]["reasoning"] == {
        "effort": "high",
        "summary": "auto",
    }


@pytest.mark.anyio
async def test_interactive_tui_metadata_lists_runtime_usage_and_skills(
    tmp_path,
) -> None:
    skill_dir = tmp_path / "skills" / "debugger"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: debugger\ndescription: Debug flaky tests.\n---\n# Debugger\n",
        encoding="utf-8",
    )
    llm_service = StubLLMService([])
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=Executor(Graph(), [], skill_path=tmp_path / "skills"),
    )
    app = _make_interactive_tui_from_agent(agent)

    async with app.run_test():
        app._ensure_agent_app()
        sections = app._metadata_sections()

    section_lines = {section.title: "\n".join(section.lines) for section in sections}
    assert f"project: {tmp_path.name}" in section_lines["Runtime"]
    assert "project_id:" in section_lines["Runtime"]
    assert f"version: {__version__}" in section_lines["Runtime"]
    assert "model: gpt-4o" in section_lines["Runtime"]
    assert "reasoning:" not in section_lines["Runtime"]
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
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=Executor(Graph(), [], skill_path=skill_root),
    )
    app = _make_interactive_tui_from_agent(agent)

    async with app.run_test(size=(100, 24)) as pilot:
        app._ensure_agent_app()
        app.open_metadata_screen()
        await pilot.pause()
        await _wait_until(
            pilot,
            lambda: app.screen.query_one("#metadata-body", RichLog).max_scroll_y > 0,
        )
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
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent)

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        app._handle_config_selection("gpt-5.5")
        app.on_input_submitted(Input.Submitted(command_input, "Use selected model"))
        await _wait_until(pilot, lambda: len(llm_service.calls) == 1)

        assert app._selected_model == "gpt-5.5"
        assert llm_service.calls[0]["metadata"]["model"] == "gpt-5.5"


@pytest.mark.anyio
async def test_interactive_tui_shortcuts_and_commands_route_to_actions() -> None:
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([]),  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent)
    calls: list[str] = []
    app.open_config_screen = lambda: calls.append("config")
    app._show_ideas = lambda: calls.append("ideas")
    app.action_toggle_debug_mode = lambda: calls.append("debug")
    app.open_trajectory_screen = lambda: calls.append("trajectory")

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        await pilot.press("c")
        await pilot.press("i")
        app.on_input_submitted(Input.Submitted(command_input, "/config"))
        app.on_input_submitted(Input.Submitted(command_input, "/debug"))
        app.on_input_submitted(Input.Submitted(command_input, "/trajectory"))

    assert calls == ["config", "ideas", "config", "debug", "trajectory"]


@pytest.mark.anyio
async def test_tui_clickable_chrome_routes_to_simplified_entries() -> None:
    app = AceAITUI([])
    calls: list[str] = []
    app.action_config = lambda: calls.append("config")
    app.open_metadata_screen = lambda: calls.append("metadata")
    app.action_toggle_debug_mode = lambda: calls.append("debug")

    async with app.run_test():
        app.on_top_bar_widget_config_requested(TopBarWidget.ConfigRequested())
        app.on_status_bar_widget_metadata_requested(StatusBarWidget.MetadataRequested())
        app.on_top_bar_widget_debug_requested(TopBarWidget.DebugRequested())

    assert calls == ["config", "metadata", "debug"]


@pytest.mark.anyio
async def test_command_input_shift_enter_inserts_newline() -> None:
    command_input = CommandInput()
    command_input.value = "/idea first"

    await command_input._on_key(Key("shift+enter", None))

    assert command_input.value == "/idea first\n"


def test_command_input_enter_completes_only_unfinished_slash_command(
    monkeypatch,
) -> None:
    command_input = CommandInput()
    messages: list[object] = []
    monkeypatch.setattr(command_input, "post_message", messages.append)

    command_input.value = "/"
    command_input.action_submit_or_complete()

    assert isinstance(messages.pop(), CommandInput.CompletionRequested)

    command_input.value = "/config "
    command_input.action_submit_or_complete()

    submitted = messages.pop()
    assert isinstance(submitted, CommandInput.Submitted)
    assert submitted.value == "/config "


def test_command_input_tab_does_not_complete_slash_command(monkeypatch) -> None:
    command_input = CommandInput()
    messages: list[object] = []
    monkeypatch.setattr(command_input, "post_message", messages.append)
    command_input.value = "/tr"
    messages.clear()

    command_input.on_key(Key("tab", None))

    assert messages == []


@pytest.mark.anyio
async def test_interactive_tui_shows_slash_command_completions() -> None:
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([]),  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent)

    async with app.run_test():
        command_input = app.query_one(CommandInput)
        completions = app.query_one(CommandCompletionWidget)
        command_input.value = "/"
        app._refresh_command_completions(command_input.value)

        assert not completions.has_class("hidden")
        assert completions.selected_index == 0
        assert "> /clear" in completions.display_text
        assert "/clear" in completions.display_text
        assert "Clear the visible transcript" in completions.display_text
        assert "/config" in completions.display_text
        assert "/debug" in completions.display_text
        assert "Toggle the debug detail view" in completions.display_text
        assert "/idea" in completions.display_text
        assert "/quit" in completions.display_text
        assert "/sessions" in completions.display_text
        assert "/subagents" in completions.display_text
        assert "Show delegated subagent details" in completions.display_text
        assert "/trajectory" in completions.display_text
        assert "/update" in completions.display_text
        assert "/model" not in completions.display_text
        assert "/resume" not in completions.display_text
        assert "/metadata" not in completions.display_text


@pytest.mark.anyio
async def test_interactive_tui_filters_and_tabs_slash_command_completion() -> None:
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([]),  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent)

    async with app.run_test():
        command_input = app.query_one(CommandInput)
        completions = app.query_one(CommandCompletionWidget)
        command_input.value = "/tr"
        app._refresh_command_completions(command_input.value)

        assert completions.selected_index == 0
        assert "> /trajectory" in completions.display_text
        assert "/trajectory" in completions.display_text
        assert "Open the event trajectory view" in completions.display_text

        app.on_command_input_completion_requested(
            CommandInput.CompletionRequested(command_input)
        )

        assert command_input.value == "/trajectory "
        assert completions.has_class("hidden")


@pytest.mark.anyio
async def test_interactive_tui_arrow_keys_navigate_slash_command_completion() -> None:
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([]),  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent)

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        completions = app.query_one(CommandCompletionWidget)
        command_input.value = "/"
        command_input.focus()
        app._refresh_command_completions(command_input.value)

        await pilot.press("down")
        await pilot.pause()

        assert completions.selected_index == 1
        assert "> /config" in completions.display_text

        await pilot.press("up")
        await pilot.pause()

        assert completions.selected_index == 0
        assert "> /clear" in completions.display_text


@pytest.mark.anyio
async def test_interactive_tui_shows_reference_completions(
    tmp_path,
) -> None:
    (tmp_path / "README.md").write_text("readme", encoding="utf-8")
    project = ProjectStore(tmp_path / "projects").resolve_project(tmp_path)
    ideas_path = tmp_path / "ideas.sqlite3"
    idea_store = IdeaStore(ideas_path)
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([]),  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(
        agent,
        idea_store=idea_store,
        project=project,
    )
    idea_store.capture("finish inline citation autocomplete", project=project)

    async with app.run_test():
        command_input = app.query_one(CommandInput)
        completions = app.query_one(ReferenceCompletionWidget)
        command_input.value = "please use @"
        app._refresh_reference_completions(command_input.value)

        assert completions.has_class("hidden")

        command_input.value = "please use @read"
        app._refresh_reference_completions(command_input.value)

        assert not completions.has_class("hidden")
        assert "> @README.md" in completions.display_text
        command_input.value = "please use @idea"
        app._refresh_reference_completions(command_input.value)

        assert "@idea:1" in completions.display_text
        assert "finish inline citation autocomplete" in completions.display_text


@pytest.mark.anyio
async def test_interactive_tui_reference_completion_inserts_selected_item(
    tmp_path,
) -> None:
    (tmp_path / "README.md").write_text("readme", encoding="utf-8")
    project = ProjectStore(tmp_path / "projects").resolve_project(tmp_path)
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([]),  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent, project=project)

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        command_input.value = "please use @READ"
        command_input.focus()
        app._refresh_reference_completions(command_input.value)

        await pilot.press("enter")
        await pilot.pause()

        assert command_input.value == "please use @README.md "


@pytest.mark.anyio
async def test_interactive_tui_reference_completion_searches_before_filtering(
    tmp_path,
) -> None:
    for index in range(120):
        (tmp_path / f"a{index:03d}.txt").write_text("filler", encoding="utf-8")
    target = tmp_path / "spec" / "multi-agent" / "agent_inbox.md"
    target.parent.mkdir(parents=True)
    target.write_text("# Agent Inbox Tech Spec", encoding="utf-8")
    project = ProjectStore(tmp_path / "projects").resolve_project(tmp_path)
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([]),  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent, project=project)

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        completions = app.query_one(ReferenceCompletionWidget)
        command_input.value = "@agent_inbox"
        command_input.focus()
        app._refresh_reference_completions(command_input.value)

        assert not completions.has_class("hidden")
        assert "> @spec/multi-agent/agent_inbox.md" in completions.display_text


def test_reference_file_candidates_prune_large_local_dirs_and_keep_specs(
    tmp_path,
) -> None:
    (tmp_path / ".venv" / "lib").mkdir(parents=True)
    (tmp_path / ".venv" / "lib" / "ignored.py").write_text("ignored", encoding="utf-8")
    (tmp_path / ".cache" / "plugin").mkdir(parents=True)
    (tmp_path / ".cache" / "plugin" / "manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / ".aceai" / "sessions").mkdir(parents=True)
    (tmp_path / ".aceai" / "config.yml").write_text("model: gpt-5.5", encoding="utf-8")
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "ignored.whl").write_text("ignored", encoding="utf-8")
    target = tmp_path / "spec" / "multi-agent" / "agent_inbox.md"
    target.parent.mkdir(parents=True)
    target.write_text("# Agent Inbox Tech Spec", encoding="utf-8")

    candidates = {
        path.relative_to(tmp_path).as_posix()
        for path in _iter_reference_file_candidates(tmp_path)
    }

    assert "spec/multi-agent/agent_inbox.md" in candidates
    assert ".aceai/config.yml" in candidates
    assert ".venv/lib/ignored.py" not in candidates
    assert ".cache/plugin/manifest.json" not in candidates
    assert "dist/ignored.whl" not in candidates


@pytest.mark.anyio
async def test_interactive_tui_inline_file_reference_cites_text_file(
    tmp_path,
) -> None:
    source = tmp_path / "notes.md"
    source.write_text("# Notes\nShip inline citations.", encoding="utf-8")
    llm_service = StubLLMService(
        [
            LLMStreamEvent(
                event_type="response.completed",
                response=LLMResponse(text="done"),
            ),
        ]
    )
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent)

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        question = f"summarize @{source} please"
        app.on_input_submitted(Input.Submitted(command_input, question))
        await _wait_until(pilot, lambda: len(llm_service.calls) == 1)

        visible_user_event = app._state.events[0]
        assert visible_user_event.content == question
        assert visible_user_event.citations[0].content == "# Notes\nShip inline citations."
        assert visible_user_event.citations[0].origin == FileCitationOrigin(
            kind="file",
            path=str(source.resolve()),
        )
        user_text = llm_service.calls[0]["messages"][-1].content[0]["data"]
        assert "source=\"file:" in user_text
        assert "# Notes\nShip inline citations." in user_text
        assert f"<user_request>\n{question}\n</user_request>" in user_text


@pytest.mark.anyio
async def test_interactive_tui_inline_file_reference_uses_project_relative_paths(
    tmp_path,
) -> None:
    source = tmp_path / "docs" / "plan.txt"
    source.parent.mkdir()
    source.write_text("relative file content", encoding="utf-8")
    project = ProjectStore(tmp_path / "projects").resolve_project(tmp_path)
    llm_service = StubLLMService(
        [LLMStreamEvent(event_type="response.completed", response=LLMResponse(text="done"))]
    )
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent, project=project)

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        app.on_input_submitted(Input.Submitted(command_input, "use @docs/plan.txt"))
        await _wait_until(pilot, lambda: len(llm_service.calls) == 1)

        assert app._state.events[0].citations[0].content == "relative file content"
        assert app._state.events[0].citations[0].origin == FileCitationOrigin(
            kind="file",
            path=str(source.resolve()),
        )


@pytest.mark.anyio
async def test_interactive_tui_inline_file_reference_reports_missing_file(tmp_path) -> None:
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([]),  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent)

    async with app.run_test():
        command_input = app.query_one(CommandInput)
        app.on_input_submitted(Input.Submitted(command_input, "use @missing.txt"))

        assert app._state.events == []


@pytest.mark.anyio
async def test_interactive_tui_inline_idea_reference_uses_display_index(
    tmp_path,
) -> None:
    ideas_path = tmp_path / "ideas.sqlite3"
    idea_store = IdeaStore(ideas_path)
    llm_service = StubLLMService(
        [LLMStreamEvent(event_type="response.completed", response=LLMResponse(text="done"))]
    )
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent, idea_store=idea_store)
    idea = idea_store.capture("inline idea content", project=app._project)

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        app.on_input_submitted(Input.Submitted(command_input, "use @idea:1"))
        await _wait_until(pilot, lambda: len(llm_service.calls) == 1)

        visible_user_event = app._state.events[0]
        assert visible_user_event.content == "use @idea:1"
        assert visible_user_event.citations[0].content == "inline idea content"
        assert visible_user_event.citations[0].origin == IdeaCitationOrigin(
            kind="idea",
            idea_id=idea.idea_id,
        )
        user_text = llm_service.calls[0]["messages"][-1].content[0]["data"]
        assert "source=\"idea\"" in user_text
        assert "inline idea content" in user_text


@pytest.mark.anyio
async def test_interactive_tui_inline_idea_reference_uses_idea_id(
    tmp_path,
) -> None:
    ideas_path = tmp_path / "ideas.sqlite3"
    idea_store = IdeaStore(ideas_path)
    llm_service = StubLLMService(
        [LLMStreamEvent(event_type="response.completed", response=LLMResponse(text="done"))]
    )
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent, idea_store=idea_store)
    idea = idea_store.capture("idea by id", project=app._project)

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        app.on_input_submitted(
            Input.Submitted(command_input, f"remember @idea:{idea.idea_id}")
        )
        await _wait_until(pilot, lambda: len(llm_service.calls) == 1)

        assert app._state.events[0].citations[0].content == "idea by id"
        assert app._state.events[0].citations[0].origin == IdeaCitationOrigin(
            kind="idea",
            idea_id=idea.idea_id,
        )


@pytest.mark.anyio
async def test_interactive_tui_inline_idea_reference_reports_missing_idea(
    tmp_path,
) -> None:
    ideas_path = tmp_path / "ideas.sqlite3"
    idea_store = IdeaStore(ideas_path)
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([]),  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent, idea_store=idea_store)

    async with app.run_test():
        command_input = app.query_one(CommandInput)
        app.on_input_submitted(Input.Submitted(command_input, "use @idea:1"))

        assert app._state.events == []


@pytest.mark.anyio
async def test_interactive_tui_idea_command_saves_structured_idea(tmp_path) -> None:
    ideas_path = tmp_path / "ideas.sqlite3"
    idea_store = IdeaStore(ideas_path)
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([]),  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent, idea_store=idea_store)

    async with app.run_test():
        command_input = app.query_one(CommandInput)
        app.on_input_submitted(
            Input.Submitted(command_input, "/idea 修一下 resume 默认 session")
        )

    markdown = idea_store.render_markdown(project=app._project)
    assert "修一下 resume 默认 session" in markdown
    assert [idea.content for idea in idea_store.list_recent(project=app._project)] == [
        "修一下 resume 默认 session"
    ]
    assert app._state.events == []


@pytest.mark.anyio
async def test_interactive_tui_idea_command_saves_multiline_idea(tmp_path) -> None:
    ideas_path = tmp_path / "ideas.sqlite3"
    idea_store = IdeaStore(ideas_path)
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([]),  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent, idea_store=idea_store)

    async with app.run_test():
        command_input = app.query_one(CommandInput)
        app.on_command_input_submitted(
            CommandInput.Submitted(command_input, "/idea first line\nsecond line")
        )

    markdown = idea_store.render_markdown(project=app._project)
    assert "first line\nsecond line" in markdown


@pytest.mark.anyio
async def test_interactive_tui_idea_command_opens_fifo_picker(tmp_path) -> None:
    ideas_path = tmp_path / "ideas.sqlite3"
    idea_store = IdeaStore(ideas_path)
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([]),  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent, idea_store=idea_store)
    idea_store.capture("first idea", project=app._project)
    idea_store.capture("second idea", project=app._project)

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        app.on_input_submitted(Input.Submitted(command_input, "/idea"))
        await pilot.pause()

        screen = app.screen
        assert isinstance(screen, IdeaPickerScreen)
        idea_list = screen.query_one("#idea-list", IdeaListWidget)
        assert idea_list.selected_index == 0
        assert [idea.content for idea in idea_list.ideas()] == [
            "first idea",
            "second idea",
        ]

        await pilot.press("enter")
        await pilot.pause()

        preview = app.query_one(CitationPreviewWidget)
        assert command_input.value == ""
        assert "cited source" in preview.display_text
        assert "first idea" in preview.display_text
        assert "1. idea" not in preview.display_text
        assert "ideas" not in preview.display_text
        assert command_input.has_focus


@pytest.mark.anyio
async def test_interactive_tui_idea_picker_scrolls_highlighted_row_into_view(
    tmp_path,
) -> None:
    ideas_path = tmp_path / "ideas.sqlite3"
    idea_store = IdeaStore(ideas_path)
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([]),  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent, idea_store=idea_store)
    for index in range(18):
        idea_store.capture(f"idea {index}", project=app._project)

    async with app.run_test(size=(100, 20)) as pilot:
        command_input = app.query_one(CommandInput)
        app.on_input_submitted(Input.Submitted(command_input, "/idea"))
        await pilot.pause()

        picker = app.screen
        assert isinstance(picker, IdeaPickerScreen)
        idea_list = picker.query_one("#idea-list", IdeaListWidget)
        scroll = picker.query_one("#idea-list-scroll", VerticalScroll)

        for _index in range(17):
            await pilot.press("down")

        await _wait_until(pilot, lambda: idea_list.selected_index == 17)
        await _wait_until(pilot, lambda: scroll.scroll_y > 0)

        selected_top = idea_list._selected_item_top()
        assert scroll.scroll_y <= selected_top
        assert selected_top < scroll.scroll_y + scroll.scrollable_content_region.height


@pytest.mark.anyio
async def test_interactive_tui_idea_picker_shows_other_project_when_current_empty(
    tmp_path,
) -> None:
    ideas_path = tmp_path / "ideas.sqlite3"
    idea_store = IdeaStore(ideas_path)
    project_store = ProjectStore(tmp_path / "projects")
    current_project = project_store.resolve_project(tmp_path / "ioa")
    other_project = project_store.resolve_project(tmp_path / "aceai")
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([]),  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(
        agent,
        idea_store=idea_store,
        project=current_project,
    )
    idea_store.capture("aceai idea", project=other_project)

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        app.on_input_submitted(Input.Submitted(command_input, "/idea"))
        await pilot.pause()

        screen = app.screen
        assert isinstance(screen, IdeaPickerScreen)
        idea_list = screen.query_one("#idea-list", IdeaListWidget)
        assert [idea.content for idea in idea_list.ideas()] == ["aceai idea"]
        assert idea_list.ideas()[0].project_id == other_project.project_id


@pytest.mark.anyio
async def test_interactive_tui_referenced_idea_is_read_only_citation(tmp_path) -> None:
    ideas_path = tmp_path / "ideas.sqlite3"
    long_content = (
        "triggered, AceAI should save the selected failed trajectories into memory so "
        "they can be reviewed or retrieved later with additional implementation detail"
    )
    idea_store = IdeaStore(ideas_path)
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
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent, idea_store=idea_store)
    idea_store.capture(long_content, project=app._project)

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        app.on_input_submitted(Input.Submitted(command_input, "/idea"))
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        preview = app.query_one(CitationPreviewWidget)
        assert command_input.value == ""
        assert "cited source" in preview.display_text
        assert "triggered, AceAI should save" in preview.display_text
        assert "additional implementation detail" in preview.display_text
        assert long_content not in preview.display_text

        command_input.value = "what should we do?"
        app.on_input_submitted(Input.Submitted(command_input, command_input.value))
        await _wait_until(pilot, lambda: len(llm_service.calls) == 1)

        assert preview.display_text == ""
        visible_user_event = app._state.events[0]
        assert visible_user_event.content == "what should we do?"
        assert visible_user_event.citations[0].content == long_content
        user_text = llm_service.calls[0]["messages"][-1].content[0]["data"]
        assert long_content in user_text
        assert "<user_request>\nwhat should we do?\n</user_request>" in user_text


@pytest.mark.anyio
async def test_interactive_tui_idea_picker_adds_idea(tmp_path) -> None:
    ideas_path = tmp_path / "ideas.sqlite3"
    idea_store = IdeaStore(ideas_path)
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([]),  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent, idea_store=idea_store)
    idea_store.capture("first idea", project=app._project)

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        app.on_input_submitted(Input.Submitted(command_input, "/idea"))
        await pilot.pause()
        picker = app.screen
        assert isinstance(picker, IdeaPickerScreen)

        await pilot.press("a")
        await pilot.pause()
        picker = app.screen
        assert isinstance(picker, IdeaPickerScreen)
        editor = picker.query_one("#idea-add-input", Input)
        editor.value = "new idea from picker"
        await pilot.press("enter")
        await _wait_until(
            pilot,
            lambda: (
                [
                    idea.content
                    for idea in picker.query_one("#idea-list", IdeaListWidget).ideas()
                ]
                == [
                    "first idea",
                    "new idea from picker",
                ]
            ),
        )

        idea_list = picker.query_one("#idea-list", IdeaListWidget)
        assert [idea.content for idea in idea_list.ideas()] == [
            "first idea",
            "new idea from picker",
        ]
        assert [
            idea.content for idea in idea_store.list_recent(project=app._project)
        ] == [
            "first idea",
            "new idea from picker",
        ]


@pytest.mark.anyio
async def test_interactive_tui_idea_delete_command_removes_recent_idea(
    tmp_path,
) -> None:
    ideas_path = tmp_path / "ideas.sqlite3"
    idea_store = IdeaStore(ideas_path)
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([]),  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent, idea_store=idea_store)
    idea_store.capture("first idea", project=app._project)
    idea_store.capture("second idea", project=app._project)

    async with app.run_test():
        command_input = app.query_one(CommandInput)
        app.on_input_submitted(Input.Submitted(command_input, "/idea delete 1"))

    assert app._state.events[-1].kind == "idea_list"
    assert [item.title for item in app._state.events[-1].idea_items] == ["second idea"]
    assert [idea.content for idea in idea_store.list_recent(project=app._project)] == [
        "second idea"
    ]


@pytest.mark.anyio
async def test_interactive_tui_update_command_runs_upgrade_and_restarts(
    monkeypatch,
) -> None:
    async def update_command() -> UpdateCommandResult:
        return UpdateCommandResult(return_code=0, output="updated")

    restart_calls: list[str] = []
    monkeypatch.setattr(tui_runner_module, "run_update_command", update_command)
    monkeypatch.setattr(
        tui_runner_module,
        "restart_current_process",
        lambda: restart_calls.append("restart"),
    )
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([]),  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent)

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        app.on_input_submitted(Input.Submitted(command_input, "/update"))
        await _wait_until(pilot, lambda: restart_calls == ["restart"])

    assert [event.content for event in app._state.events[-2:]] == [
        "Updating AceAI with uv tool upgrade aceai...",
        "AceAI updated. Restarting...",
    ]
    assert restart_calls == ["restart"]


@pytest.mark.anyio
async def test_interactive_tui_update_command_reports_upgrade_failure(
    monkeypatch,
) -> None:
    async def update_command() -> UpdateCommandResult:
        return UpdateCommandResult(return_code=1, output="network failed")

    restart_calls: list[str] = []
    monkeypatch.setattr(tui_runner_module, "run_update_command", update_command)
    monkeypatch.setattr(
        tui_runner_module,
        "restart_current_process",
        lambda: restart_calls.append("restart"),
    )
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([]),  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent)

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        app.on_input_submitted(Input.Submitted(command_input, "/update"))
        await _wait_until(
            pilot,
            lambda: (
                app._state.events[-1].content
                == "AceAI update failed with exit code 1.\nnetwork failed"
            ),
        )

    assert app._state.events[-1].kind == "session_notice"
    assert app._state.events[-1].content == (
        "AceAI update failed with exit code 1.\nnetwork failed"
    )
    assert restart_calls == []


@pytest.mark.anyio
async def test_interactive_tui_automatically_reports_available_update(
    monkeypatch,
) -> None:
    async def available_update() -> UpdateCheckResult:
        return UpdateCheckResult(
            current_version="0.2.7",
            latest_version="0.2.8",
        )

    monkeypatch.setattr(agent_app_module, "check_for_updates", available_update)
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([]),  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent)

    async with app.run_test() as pilot:
        app._ensure_agent_app()
        await _wait_until(
            pilot,
            lambda: (
                app._state.events and app._state.events[-1].kind == "session_notice"
            ),
        )

    assert app._state.events[-1].kind == "session_notice"
    assert app._state.events[-1].content == (
        f"AceAI 0.2.8 is available (current 0.2.7).\n{UPDATE_INSTRUCTIONS}"
    )


@pytest.mark.anyio
async def test_interactive_tui_reports_available_update_once(monkeypatch) -> None:
    async def available_update() -> UpdateCheckResult:
        return UpdateCheckResult(
            current_version="0.2.7",
            latest_version="0.2.8",
        )

    monkeypatch.setattr(agent_app_module, "check_for_updates", available_update)
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([]),  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent)

    async with app.run_test() as pilot:
        app._ensure_agent_app()
        app._start_update_check()
        await _wait_until(
            pilot,
            lambda: any(
                event.kind == "session_notice" and "is available" in event.content
                for event in app._state.events
            ),
        )

    notices = [
        event
        for event in app._state.events
        if event.kind == "session_notice" and "is available" in event.content
    ]
    assert len(notices) == 1


@pytest.mark.anyio
async def test_interactive_tui_starts_update_check_once_when_mount_reenters(
    monkeypatch,
) -> None:
    calls = 0

    async def no_update() -> None:
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(agent_app_module, "check_for_updates", no_update)
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([]),  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent)

    async with app.run_test() as pilot:
        app.on_mount()
        app._ensure_agent_app()
        await _wait_until(pilot, lambda: calls == 1)

    assert calls == 1


@pytest.mark.anyio
async def test_interactive_tui_stats_command_opens_stats_screen() -> None:
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=StubLLMService([]),  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent)
    calls: list[str] = []
    app.open_stats_screen = lambda: calls.append("stats")

    async with app.run_test():
        command_input = app.query_one(CommandInput)
        app.on_input_submitted(Input.Submitted(command_input, "/stats"))

    assert calls == ["stats"]


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
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent)

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        app.on_input_submitted(Input.Submitted(command_input, "/unknown command"))
        await _wait_until(pilot, lambda: len(llm_service.calls) == 1)

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
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(agent)

    async with app.run_test() as pilot:
        command_input = app.query_one(CommandInput)
        command_input.focus()
        app.on_input_submitted(Input.Submitted(command_input, "Use selected model"))
        await _wait_until(pilot, lambda: app.query_one("#stream").has_focus)

        assert not command_input.has_focus
        assert app.query_one("#stream").has_focus


@pytest.mark.anyio
async def test_tui_replaces_empty_stream_text_with_labrador(tmp_path) -> None:
    async with AceAITUI([]).run_test() as pilot:
        await pilot.pause()

        stream = pilot.app.query_one(StreamWidget)
        console = Console(width=80, record=True, file=StringIO())
        console.print(stream._render_empty_state())
        text = console.export_text()
        assert f"AceAI v{__version__}" in text
        assert f"Project: {tmp_path.name}" in text
        assert "shortcuts" in text
        assert "enter" in text
        assert "ask" in text
        assert "config" in text
        assert "sessions" in text
        assert "debug" in text
        assert "cancel" in text
        assert "██" in text
        assert "No events yet" not in text


def test_empty_stream_text_includes_git_branch(tmp_path) -> None:
    subprocess.run(
        ["git", "init", "-b", "feature/home-branch"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    stream = StreamWidget(
        project_name="project",
        project_root_path=str(tmp_path),
    )
    console = Console(width=80, record=True, file=StringIO())

    console.print(stream._render_empty_state())

    assert "Git: feature/home-branch" in console.export_text()


def test_empty_stream_render_uses_cached_git_branch(tmp_path, monkeypatch) -> None:
    subprocess.run(
        ["git", "init", "-b", "feature/home-branch"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    stream = StreamWidget(
        project_name="project",
        project_root_path=str(tmp_path),
    )

    def fail_git_lookup(project_root_path: str) -> str | None:
        raise AssertionError("empty-state render must not spawn git")

    monkeypatch.setattr(stream_widget_module, "_git_branch_name", fail_git_lookup)
    console = Console(width=80, record=True, file=StringIO())

    console.print(stream._render_empty_state())

    assert "Git: feature/home-branch" in console.export_text()


@pytest.mark.anyio
async def test_tui_stops_empty_labrador_when_events_are_loaded() -> None:
    app = AceAITUI([TUIEvent.session_notice("Welcome back")])

    async with app.run_test() as pilot:
        await pilot.pause()

        assert app.query_one(StreamWidget)._empty_state_timer is None


@pytest.mark.anyio
async def test_config_screen_hides_reasoning_level_for_unsupported_model() -> None:
    screen = ConfigScreen(
        provider_name="openai",
        current_model="gpt-4o",
        default_model="gpt-4o",
        skills="auto",
        api_keys={"openai": "sk-test-ending"},
        reasoning_level="auto",
    )

    async with AceAITUI([]).run_test() as pilot:
        pilot.app.push_screen(screen)
        await pilot.pause()

        assert screen.query_one("#reasoning-level-row").has_class("hidden")


@pytest.mark.anyio
async def test_config_screen_supports_deepseek_reasoning_levels() -> None:
    screen = ConfigScreen(
        provider_name="deepseek",
        current_model="deepseek-v4-pro",
        default_model="deepseek-v4-pro",
        skills="auto",
        api_keys={"deepseek": "sk-test-ending"},
        reasoning_level="max",
    )

    async with AceAITUI([]).run_test() as pilot:
        pilot.app.push_screen(screen)
        await pilot.pause()
        selections: list[ConfigSelection | None] = []

        def dismiss(selection: ConfigSelection | None) -> None:
            selections.append(selection)

        screen.dismiss = dismiss
        assert not screen.query_one("#reasoning-level-row").has_class("hidden")
        assert screen.query_one("#reasoning-level", Select).value == "max"
        _press_config_apply(screen)

        assert selections == [
            ConfigSelection(
                provider="deepseek",
                model="deepseek-v4-pro",
                default_model="deepseek-v4-pro",
                api_key="sk-test-ending",
                skills="auto",
                skill_selection_mode="selected",
                enabled_skills=(),
                reasoning_level="max",
            )
        ]


@pytest.mark.anyio
async def test_config_screen_apply_restores_masked_api_key() -> None:
    screen = ConfigScreen(
        provider_name="openai",
        current_model="gpt-5.5",
        default_model="gpt-5.5",
        skills="auto",
        api_keys={"openai": "sk-test-ending"},
        reasoning_level="medium",
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
                reasoning_level="medium",
            )
        ]


@pytest.mark.anyio
async def test_config_screen_blocks_disabled_provider() -> None:
    screen = ConfigScreen(
        provider_name="openai",
        current_model="gpt-5.5",
        default_model="gpt-5.5",
        skills="auto",
        api_keys={"openai": "sk-test-ending", "deepseek": "sk-deepseek-ending"},
        disabled_providers=("deepseek",),
    )

    async with AceAITUI([]).run_test() as pilot:
        pilot.app.push_screen(screen)
        await pilot.pause()
        selections: list[ConfigSelection | None] = []

        def dismiss(selection: ConfigSelection | None) -> None:
            selections.append(selection)

        screen.dismiss = dismiss
        provider_input = screen.query_one("#provider", Input)
        provider_input.value = "deepseek"
        screen.query_one("#model", Input).value = "deepseek-v4-pro"
        _press_config_apply(screen)

        assert selections == []
        assert str(screen.query_one("#config-error", Static).render()) == (
            "Provider is disabled"
        )


@pytest.mark.anyio
async def test_config_screen_uses_model_as_default_model_and_places_skills_on_tools_tab() -> (
    None
):
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
        assert str(screen.query_one("#tool-permissions-tab Label").render()) == (
            "skills for current agent *"
        )
        assert screen.query_one("#config-skills-list") in screen.query_one(
            "#tool-permissions-tab"
        ).query("*")
        assert str(screen.query_one("#skill-0", Checkbox).label) == "developer"
        assert "Practical software development workflow" in str(
            screen.query_one("#skill-description-0", Static).render()
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
async def test_config_screen_labels_skill_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)
    write_skill(home / ".aceai" / "skills", "global", "Global skill.", "# Global")
    write_skill(
        project / ".agents" / "skills", "project", "Project skill.", "# Project"
    )
    registry = SkillLoader.load_registry(
        "auto",
        extra_skill_paths=ACE_AGENT_BUILTIN_SKILL_PATHS,
    )
    skill_items = _skill_config_items(registry)

    screen = ConfigScreen(
        provider_name="openai",
        current_model="gpt-5.5",
        default_model="gpt-5.5",
        skills="auto",
        skill_items=skill_items,
        skill_selection_mode="all",
        enabled_skills=(),
        api_keys={"openai": "sk-test-ending"},
    )

    async with AceAITUI([]).run_test() as pilot:
        pilot.app.push_screen(screen)
        await pilot.pause()

        items_by_name = {item.name: item for item in skill_items}
        source_by_name = {
            str(screen.query_one(f"#skill-{index}", Checkbox).label): str(
                screen.query_one(f"#skill-source-{index}", Static).render()
            )
            for index in range(len(skill_items))
        }

        assert items_by_name["global"].source == "global"
        assert items_by_name["project"].source == "project"
        assert items_by_name["skill-creator"].source == "aceai builtin"
        assert source_by_name["global"] == "global skill"
        assert source_by_name["project"] == "project skill"
        assert source_by_name["skill-creator"] == "aceai builtin skill"


@pytest.mark.anyio
async def test_config_screen_searches_project_skills_and_loads_new_skill_links(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    release_dir = write_skill(
        project / "vendor", "release", "Release workflow.", "# Release"
    )
    review_dir = write_skill(
        project / "vendor", "review", "Review workflow.", "# Review"
    )
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(project)
    screen = ConfigScreen(
        provider_name="openai",
        current_model="gpt-5.5",
        default_model="gpt-5.5",
        skills="auto",
        skill_items=(),
        skill_selection_mode="selected",
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
        assert screen.query_one("#search-skills", Button) in screen.query_one(
            "#tool-permissions-tab"
        ).query("*")

        assert len(screen.query("#skill-candidate-1")) == 0

        screen.query_one("#search-skills", Button).press()
        await pilot.pause()

        assert "release" in str(screen.query_one("#skill-candidate-0", Static).render())
        assert "review" in str(screen.query_one("#skill-candidate-1", Static).render())
        assert not (project / ".agents" / "skills" / "release").exists()
        assert not (project / ".agents" / "skills" / "review").exists()

        screen.query_one("#load-skill-0", Button).press()
        await pilot.pause()
        assert (project / ".agents" / "skills" / "release").resolve() == release_dir
        assert str(screen.query_one("#skill-0", Checkbox).label) == "release"
        assert "review" in str(screen.query_one("#skill-candidate-0", Static).render())

        screen.query_one("#load-skill-0", Button).press()
        await pilot.pause()
        assert (project / ".agents" / "skills" / "review").resolve() == review_dir
        assert str(screen.query_one("#skill-1", Checkbox).label) == "review"
        assert str(
            screen.query_one("#skill-candidates-empty .skill-empty-title", Static).render()
        ) == (
            "No new skills"
        )

        _press_config_apply(screen)

        assert selections[-1].skills == "auto"
        assert selections[-1].enabled_skills == (
            "release",
            "review",
            "skill-creator",
        )


@pytest.mark.anyio
async def test_config_screen_reports_skill_search_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(project)
    screen = ConfigScreen(
        provider_name="openai",
        current_model="gpt-5.5",
        default_model="gpt-5.5",
        skills="auto",
        skill_items=(),
        skill_selection_mode="selected",
        enabled_skills=(),
        api_keys={"openai": "sk-test-ending"},
    )

    async with AceAITUI([]).run_test() as pilot:
        pilot.app.push_screen(screen)
        await pilot.pause()

        monkeypatch.setattr(
            "aceai.agent.tui.setup._find_project_skill_dirs",
            lambda: (_ for _ in ()).throw(OSError("invalid skill path")),
        )
        screen.query_one("#search-skills", Button).press()
        await pilot.pause()

        assert (
            str(screen.query_one("#skill-search-error", Static).render())
            == "Skill search failed: invalid skill path"
        )


@pytest.mark.anyio
async def test_config_screen_checks_builtin_skills_by_default_in_selected_mode() -> (
    None
):
    project_skill = SkillConfigItem(
        name="aceai-release",
        description="Release workflow.",
        location="/project/.agents/skills/aceai-release/SKILL.md",
    )
    builtin_skill = SkillConfigItem(
        name="skill-creator",
        description="Create and improve skills.",
        location="/aceai/agent/builtin_skills/skill-creator/SKILL.md",
        builtin=True,
    )
    screen = ConfigScreen(
        provider_name="openai",
        current_model="gpt-5.5",
        default_model="gpt-5.5",
        skills="auto",
        skill_items=(project_skill, builtin_skill),
        skill_selection_mode="selected",
        enabled_skills=("aceai-release",),
        api_keys={"openai": "sk-test-ending"},
    )

    async with AceAITUI([]).run_test() as pilot:
        pilot.app.push_screen(screen)
        await pilot.pause()
        selections: list[ConfigSelection | None] = []

        def dismiss(selection: ConfigSelection | None) -> None:
            selections.append(selection)

        screen.dismiss = dismiss

        assert screen.query_one("#skill-0", Checkbox).value is True
        assert screen.query_one("#skill-1", Checkbox).value is True

        _press_config_apply(screen)

        assert selections[-1].enabled_skills == ("aceai-release", "skill-creator")


@pytest.mark.anyio
async def test_config_screen_is_fullscreen_without_system_prompt_tab() -> None:
    screen = ConfigScreen(
        provider_name="openai",
        current_model="gpt-5.5",
        default_model="gpt-5.5",
        skills="auto",
        api_keys={"openai": "sk-test-ending"},
    )

    async with AceAITUI([]).run_test(size=(100, 30)) as pilot:
        pilot.app.push_screen(screen)
        await pilot.pause()

        panel = screen.query_one("#config-panel")
        tabs = screen.query_one("#config-tabs", TabbedContent)
        settings_tab = screen.query_one("#settings-tab")
        screen_ids = {node.id for node in screen.query("*") if node.id is not None}

        assert panel.region.width == 100
        assert panel.region.height == 30
        assert tabs.active == "settings-tab"
        assert screen.query_one("#provider", Input) in settings_tab.query("*")
        assert "stats-tab" not in screen_ids
        assert "system-prompt-tab" not in screen_ids
        assert "system-prompt" not in screen_ids


@pytest.mark.anyio
async def test_config_screen_has_audit_tab_with_recent_changes() -> None:
    audit_entry = ConfigAuditEntry(
        timestamp="2026-05-10T19:16:04+00:00",
        actor="raceychan",
        pid=123,
        cwd="/Users/raceychan/mylab/aceai",
        target="/Users/raceychan/mylab/aceai/.aceai/config.yml",
        caller=("runner.py:1226:switch_model",),
        changed_fields=("provider", "disabled_providers"),
        before={
            "provider": "openai",
            "model": "gpt-5.5",
            "default_model": "gpt-5.5",
            "skills": "auto",
            "skill_selection_mode": "selected",
            "enabled_skills": [],
            "disabled_providers": [],
            "api_key_providers": ["openai"],
            "tool_permissions": {},
            "tool_enabled": {},
            "tool_max_calls": {},
            "compress_threshold": "100%",
            "reasoning_level": "auto",
        },
        after={
            "provider": "codex",
            "model": "gpt-5.5",
            "default_model": "gpt-5.5",
            "skills": "auto",
            "skill_selection_mode": "selected",
            "enabled_skills": [],
            "disabled_providers": ["openai"],
            "api_key_providers": ["codex", "openai"],
            "tool_permissions": {},
            "tool_enabled": {},
            "tool_max_calls": {},
            "compress_threshold": "100%",
            "reasoning_level": "auto",
        },
    )
    screen = ConfigScreen(
        provider_name="codex",
        current_model="gpt-5.5",
        default_model="gpt-5.5",
        skills="auto",
        api_keys={"codex": CODEX_CLI_AUTH_SENTINEL},
        audit_entries=(audit_entry,),
    )

    async with AceAITUI([]).run_test(size=(120, 30)) as pilot:
        pilot.app.push_screen(screen)
        await pilot.pause()

        tabs = screen.query_one("#config-tabs", TabbedContent)
        meta_text = "\n".join(
            str(node.render()) for node in screen.query(".config-audit-meta")
        )
        field_text = str(screen.query_one(".config-audit-fields", Static).render())
        change_cells = [
            str(node.render())
            for node in screen.query("#config-audit-change-0-0 Static")
        ]
        disabled_change_cells = [
            str(node.render())
            for node in screen.query("#config-audit-change-0-1 Static")
        ]

        assert screen.query_one("#config-audit-tab") in tabs.query("*")
        assert str(screen.query_one(".config-audit-time", Static).render()) == (
            "2026-05-10T19:16:04+00:00"
        )
        assert str(screen.query_one(".config-audit-actor", Static).render()) == (
            "raceychan"
        )
        assert str(screen.query_one(".config-audit-pid", Static).render()) == "pid 123"
        assert "caller  runner.py:1226:switch_model" in meta_text
        assert field_text == "changed provider, disabled_providers"
        assert change_cells == ["provider", "openai", "->", "codex"]
        assert disabled_change_cells == [
            "disabled_providers",
            "[]",
            "->",
            '["openai"]',
        ]


@pytest.mark.anyio
async def test_config_screen_audit_tab_empty_state() -> None:
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

        assert str(screen.query_one("#config-audit-empty", Static).render()) == (
            "No config changes recorded"
        )


@pytest.mark.anyio
async def test_config_screen_has_tool_permissions_tab_and_selects_policy() -> None:
    tool_items = (
        ToolPermissionItem(
            name="run_shell_command",
            description="Run a shell command.",
            permission="ask",
            max_calls_per_run=5,
            tags=("dev",),
        ),
        ToolPermissionItem(
            name="read_text_file",
            description="Read a file.",
            permission="always",
            enabled=False,
            tags=("dev",),
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
        enabled_toggle = screen.query_one("#tool-enabled-1", Checkbox)
        max_calls_input = screen.query_one("#tool-max-calls-0", Input)

        assert screen.query_one("#tool-permissions-tab") in tabs.query("*")
        assert list(screen.query("#enable-all-tools")) == []
        assert list(screen.query("#allow-all-tools")) == []
        assert list(screen.query("#disable-all-tools")) == []
        tool_tag_tabs = screen.query_one("#tool-tag-tabs", TabbedContent)
        assert tool_tag_tabs.active == "tool-tag-tab-0"
        assert str(screen.query_one(".tool-tag-status", Static).render()) == "1/2 enabled"
        assert str(screen.query_one("#tool-tag-allow-all-0", Button).label) == (
            "Allow all"
        )
        header_cells = list(screen.query(".tool-permission-header Static"))
        assert [str(cell.render()) for cell in header_cells] == [
            "On",
            "Tool",
            "Permission",
            "Max calls",
            "Description",
        ]
        assert screen.query_one("#tool-enabled-0", Checkbox).has_class(
            "tool-enabled-toggle"
        )
        assert str(
            screen.query_one("#tool-permission-description-1", Static).render()
        ) == ("Read a file.")
        assert screen.query_one("#tool-permission-description-1", Static).has_class(
            "tool-disabled"
        )
        assert tool_select.value == "ask"
        assert not enabled_toggle.value
        assert max_calls_input.value == "5"

        enabled_toggle.value = True
        await pilot.pause()
        assert str(
            screen.query_one("#tool-permission-description-0", Static).render()
        ) == ("Run a shell command.")
        assert str(
            screen.query_one("#tool-permission-description-1", Static).render()
        ) == ("Read a file.")
        tool_select = screen.query_one("#tool-permission-0", Select)
        max_calls_input = screen.query_one("#tool-max-calls-0", Input)
        tool_select.value = "always"
        max_calls_input.value = "2"
        _press_config_apply(screen)

        assert selections[-1].tool_permissions == {
            "run_shell_command": "always",
            "read_text_file": "always",
        }
        assert selections[-1].tool_enabled == {
            "run_shell_command": True,
            "read_text_file": True,
        }
        assert selections[-1].tool_max_calls == {
            "run_shell_command": 2,
        }


@pytest.mark.anyio
async def test_config_screen_groups_tools_by_tag_tabs() -> None:
    tool_items = (
        ToolPermissionItem(
            name="run_shell_command",
            description="Run a shell command.",
            permission="ask",
            tags=("dev",),
        ),
        ToolPermissionItem(
            name="read_text_file",
            description="Read a file.",
            permission="always",
            tags=("dev",),
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

        assert screen.query_one("#tool-tag-tabs", TabbedContent).active == "tool-tag-tab-0"
        assert screen.query_one("#tool-tag-enabled-0", Checkbox).value
        assert screen.query_one("#tool-enabled-0", Checkbox).value
        assert screen.query_one("#tool-enabled-1", Checkbox).value

        screen.query_one("#tool-tag-enabled-0", Checkbox).value = False
        await pilot.pause()

        assert not screen.query_one("#tool-enabled-0", Checkbox).value
        assert not screen.query_one("#tool-enabled-1", Checkbox).value
        _press_config_apply(screen)

        assert selections[-1].tool_enabled == {
            "run_shell_command": False,
            "read_text_file": False,
        }


@pytest.mark.anyio
async def test_config_screen_rejects_invalid_tool_max_calls() -> None:
    screen = ConfigScreen(
        provider_name="openai",
        current_model="gpt-5.5",
        default_model="gpt-5.5",
        skills="auto",
        api_keys={"openai": "sk-test-ending"},
        tool_permission_items=(
            ToolPermissionItem(
                name="run_shell_command",
                description="Run a shell command.",
                permission="ask",
            ),
        ),
    )

    async with AceAITUI([]).run_test() as pilot:
        pilot.app.push_screen(screen)
        await pilot.pause()
        selections: list[ConfigSelection | None] = []

        def dismiss(selection: ConfigSelection | None) -> None:
            selections.append(selection)

        screen.dismiss = dismiss
        screen.query_one("#tool-max-calls-0", Input).value = "0"
        _press_config_apply(screen)

        assert selections == []
        assert (
            str(screen.query_one("#config-error", Static).render())
            == "Max calls must be empty or a positive integer"
        )


@pytest.mark.anyio
async def test_config_screen_can_allow_all_tools_for_current_tag() -> None:
    tool_items = (
        ToolPermissionItem(
            name="run_shell_command",
            description="Run a shell command.",
            permission="ask",
            tags=("dev",),
        ),
        ToolPermissionItem(
            name="read_text_file",
            description="Read a file.",
            permission="ask",
            tags=("dev",),
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

        allow_all = screen.query_one("#tool-tag-allow-all-0", Button)
        assert str(allow_all.label) == "Allow all"
        allow_all.press()
        await pilot.pause()
        assert str(screen.query_one("#tool-tag-allow-all-0", Button).label) == (
            "All allowed"
        )
        _press_config_apply(screen)

        assert selections[-1].tool_enabled == {
            "run_shell_command": True,
            "read_text_file": True,
        }
        assert selections[-1].tool_permissions == {
            "run_shell_command": "always",
            "read_text_file": "always",
        }
        assert selections[-1].tool_max_calls == {}


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
        compress_threshold="80%",
    )

    async with AceAITUI([]).run_test() as pilot:
        pilot.app.push_screen(screen)
        await pilot.pause()
        selections: list[ConfigSelection | None] = []

        def dismiss(selection: ConfigSelection | None) -> None:
            selections.append(selection)

        screen.dismiss = dismiss
        screen.query_one("#skill-0", Checkbox).value = False
        screen.query_one("#compress-threshold", Input).value = "75%"
        _press_config_apply(screen)

        assert selections[-1] == ConfigSelection(
            provider="openai",
            model="gpt-5.5",
            default_model="gpt-5.5",
            api_key="sk-test-ending",
            skills="auto",
            skill_selection_mode="selected",
            enabled_skills=(),
            compress_threshold="75%",
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
async def test_config_screen_uses_codex_cli_auth_when_key_is_blank() -> None:
    screen = ConfigScreen(
        provider_name="codex",
        current_model="gpt-5.5",
        default_model="gpt-5.5",
        skills="auto",
        api_keys={},
    )

    async with AceAITUI([]).run_test() as pilot:
        pilot.app.push_screen(screen)
        await pilot.pause()
        selections: list[ConfigSelection | None] = []

        def dismiss(selection: ConfigSelection | None) -> None:
            selections.append(selection)

        screen.dismiss = dismiss
        assert screen.query_one("#api-key-row").has_class("hidden")
        screen.query_one("#api-key", Input).value = ""
        _press_config_apply(screen)

        assert selections[-1] == ConfigSelection(
            provider="codex",
            model="gpt-5.5",
            default_model="gpt-5.5",
            api_key=CODEX_CLI_AUTH_SENTINEL,
            skills="auto",
            skill_selection_mode="selected",
            enabled_skills=(),
            compress_threshold="100%",
        )


@pytest.mark.anyio
async def test_config_screen_hides_api_key_when_switching_to_subscription_provider() -> None:
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

        assert not screen.query_one("#api-key-row").has_class("hidden")
        screen.query_one("#provider", Input).value = "codex"
        await pilot.pause()

        assert screen.query_one("#api-key-row").has_class("hidden")


@pytest.mark.anyio
async def test_config_screen_candidate_completion() -> None:
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
        assert screen.query_one("#provider-options").has_class("hidden")
        assert screen.query_one("#provider-disabled-list").has_class("hidden")
        assert screen.query_one("#provider-candidate-row-1").has_class("hidden")

        model_input = screen.query_one("#model", Input)
        model_input.focus()
        model_input.value = "deepseek-c"
        await pilot.pause()
        await pilot.press("tab")

        assert model_input.value == "deepseek-chat"
        assert str(screen.query_one("#model-options", Static).render()) == ""

        provider_input.value = ""
        await pilot.pause()

        assert not screen.query_one("#provider-options").has_class("hidden")
        assert not screen.query_one("#provider-candidate-row-0").has_class("hidden")
        assert not screen.query_one("#provider-candidate-row-1").has_class("hidden")
        assert not screen.query_one("#provider-candidate-row-2").has_class("hidden")


@pytest.mark.anyio
async def test_config_screen_empty_provider_candidates_exclude_disabled_provider() -> None:
    screen = ConfigScreen(
        provider_name="openai",
        current_model="gpt-5.5",
        default_model="gpt-5.5",
        skills="auto",
        api_keys={"openai": "sk-test-ending", "deepseek": "sk-deepseek-ending"},
        disabled_providers=("deepseek",),
    )

    async with AceAITUI([]).run_test() as pilot:
        pilot.app.push_screen(screen)
        await pilot.pause()
        provider_input = screen.query_one("#provider", Input)
        provider_input.focus()
        provider_input.value = ""
        await pilot.pause()

        assert not screen.query_one("#provider-options").has_class("hidden")
        assert not screen.query_one("#provider-candidate-row-0").has_class("hidden")
        assert screen.query_one("#provider-candidate-row-1").has_class("hidden")
        assert not screen.query_one("#provider-candidate-row-2").has_class("hidden")


@pytest.mark.anyio
async def test_config_screen_provider_candidate_remove_and_add() -> None:
    screen = ConfigScreen(
        provider_name="openai",
        current_model="gpt-5.5",
        default_model="gpt-5.5",
        skills="auto",
        api_keys={"openai": "sk-test-ending", "deepseek": "sk-deepseek-ending"},
    )

    async with AceAITUI([]).run_test() as pilot:
        pilot.app.push_screen(screen)
        await pilot.pause()
        provider_input = screen.query_one("#provider", Input)
        provider_input.value = ""
        await pilot.pause()

        await pilot.click("#provider-disable-1")
        await pilot.pause()

        assert screen.query_one("#provider-candidate-row-1").has_class("hidden")
        assert not screen.query_one("#provider-disabled-list").has_class("hidden")
        assert not screen.query_one("#provider-disabled-chip-1").has_class("hidden")

        await pilot.click("#provider-disabled-chip-1")
        await pilot.pause()

        assert not screen.query_one("#provider-candidate-row-1").has_class("hidden")
        assert screen.query_one("#provider-disabled-list").has_class("hidden")
        assert screen.query_one("#provider-disabled-chip-1").has_class("hidden")


@pytest.mark.anyio
async def test_config_screen_removing_current_provider_switches_to_remaining_provider() -> None:
    screen = ConfigScreen(
        provider_name="openai",
        current_model="gpt-5.5",
        default_model="gpt-5.5",
        skills="auto",
        api_keys={"openai": "sk-openai-ending", "deepseek": "sk-deepseek-ending"},
    )

    async with AceAITUI([]).run_test(size=(150, 28)) as pilot:
        pilot.app.push_screen(screen)
        await pilot.pause()
        provider_input = screen.query_one("#provider", Input)
        provider_input.value = ""
        await pilot.pause()

        await pilot.click("#provider-disable-0")
        await pilot.pause()

        assert provider_input.value == "deepseek"
        assert screen.query_one("#model", Input).value == "deepseek-v4-pro"
        assert screen.query_one("#provider-candidate-row-0").has_class("hidden")
        assert not screen.query_one("#provider-disabled-chip-0").has_class("hidden")


@pytest.mark.anyio
async def test_configured_tui_switches_provider_without_reusing_current_key(
    tmp_path,
    monkeypatch,
) -> None:
    calls: list[AgentAppConfig] = []
    llm_service = StubLLMService(
        [
            LLMStreamEvent(
                event_type="response.completed",
                response=LLMResponse(text="done"),
            )
        ]
    )

    def agent_factory(config: AgentAppConfig) -> Agent:
        calls.append(config)
        return Agent(
            prompt="Prompt",
            default_model=config.model,
            llm_service=llm_service,  # type: ignore[arg-type]
            executor=StubExecutor(),  # type: ignore[arg-type]
        )

    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.chdir(tmp_path)
    app = AceAIConfiguredTUI(
        agent_factory,
        initial_config=AgentAppConfig(
            provider="openai",
            api_key="openai-key",
            model="gpt-5.5",
            api_keys={"openai": "openai-key"},
        ),
        initial_question="",
        default_model="gpt-5.5",
    )

    clear_config()
    async with app.run_test() as pilot:
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
        expected_config = AgentAppConfig(
            provider="deepseek",
            api_key="deepseek-key",
            model="deepseek-v4-flash",
            default_model="deepseek-v4-pro",
            skills="auto",
            skill_selection_mode="selected",
            enabled_skills=[],
            api_keys={"openai": "openai-key", "deepseek": "deepseek-key"},
        )

        assert calls == []
        assert current_config() == expected_config

        app.start_run("hello")
        await _wait_until(pilot, lambda: len(calls) == 1)

    assert calls == [expected_config]
    assert (tmp_path / ".aceai" / "config.yml").exists()


@pytest.mark.anyio
async def test_configured_tui_switches_to_codex_with_cli_auth_default(
    tmp_path,
    monkeypatch,
) -> None:
    calls: list[AgentAppConfig] = []
    llm_service = StubLLMService(
        [
            LLMStreamEvent(
                event_type="response.completed",
                response=LLMResponse(text="done"),
            )
        ]
    )

    def agent_factory(config: AgentAppConfig) -> Agent:
        calls.append(config)
        return Agent(
            prompt="Prompt",
            default_model=config.model,
            llm_service=llm_service,  # type: ignore[arg-type]
            executor=StubExecutor(),  # type: ignore[arg-type]
        )

    monkeypatch.chdir(tmp_path)
    app = AceAIConfiguredTUI(
        agent_factory,
        initial_config=AgentAppConfig(
            provider="openai",
            api_key="openai-key",
            model="gpt-5.5",
            api_keys={"openai": "openai-key"},
        ),
        initial_question="",
        default_model="gpt-5.5",
    )

    clear_config()
    async with app.run_test() as pilot:
        app._handle_config_selection(
            ConfigSelection(
                provider="codex",
                model="gpt-5.5",
                default_model="gpt-5.5",
                api_key="",
                skills="auto",
                skill_selection_mode="selected",
                enabled_skills=(),
            )
        )
        expected_config = AgentAppConfig(
            provider="codex",
            api_key=CODEX_CLI_AUTH_SENTINEL,
            model="gpt-5.5",
            default_model="gpt-5.5",
            skills="auto",
            skill_selection_mode="selected",
            enabled_skills=[],
            api_keys={
                "openai": "openai-key",
                "codex": CODEX_CLI_AUTH_SENTINEL,
            },
        )

        assert calls == []
        assert current_config() == expected_config

        app.start_run("hello")
        await _wait_until(pilot, lambda: len(calls) == 1)

    assert calls == [expected_config]


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
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(
        agent,
        initial_events=event_log_to_tui_events(store.load_event_log(first.session_id)),
        initial_history=[],
        session_recorder=SessionRecorder(store, first.session_id),
        session_id=first.session_id,
    )

    async with app.run_test():
        app._ensure_agent_app()
        app._handle_session_selection(second.session_id)

        assert app._session_id == second.session_id
        assert app.title == f"AceAI {second.project_name} {second.session_id}"
        assert app._state.events[0].content == "second"
        assert app._selected_model == "gpt-5.5"
        assert app._llm_history[0].content[0]["data"] == "second"


@pytest.mark.anyio
async def test_interactive_tui_subagents_command_switches_active_thread(
    tmp_path,
) -> None:
    store = SessionStore(tmp_path)
    metadata = store.create_session()
    store.create_thread(
        session_id=metadata.session_id,
        thread_id="child-thread-1",
        agent_id="child-agent-1",
        role="subagent",
        title="Inspect",
        status="completed",
        parent_thread_id=MAIN_THREAD_ID,
        metadata={
            "instructions": "Report evidence.",
            "context_brief": "repo",
            "allowed_tools": [],
        },
    )
    store.append_event(
        metadata.session_id,
        SessionEvent(
            thread_id="child-thread-1",
            agent_id="child-agent-1",
            run_id="child-run-1",
            kind="user_message",
            payload={"content": "child question"},
        ),
    )
    call = LLMToolCall(
        name="delegate_to_subagent",
        arguments=(
            '{"task":"Inspect","instructions":"Report evidence.",'
            '"context_brief":"repo","allowed_tools":[]}'
        ),
        call_id="call-delegate-1",
    )
    audit_output = (
        '{"type":"subagent_audit","thread_id":"child-thread-1",'
        '"agent_id":"child-agent-1","run_id":"child-run-1",'
        '"status":"completed","summary":"done","step_count":1}'
    )
    store.append_event(
        metadata.session_id,
        SessionEvent(
            run_id="parent-run-1",
            kind="tool_started",
            payload={
                "content": "",
                "tool_name": "delegate_to_subagent",
                "tool_call_id": call.call_id,
                "tool_call": call.asdict(),
            },
        ),
    )
    store.append_event(
        metadata.session_id,
        SessionEvent(
            run_id="parent-run-1",
            kind="tool_result",
            payload={
                "content": "completed",
                "tool_name": "delegate_to_subagent",
                "tool_call_id": call.call_id,
                "tool_arguments": call.arguments,
                "output": audit_output,
                "model_output": audit_output,
                "status": "completed",
            },
        ),
    )
    llm_service = StubLLMService([])
    agent = Agent(
        prompt="Prompt",
        default_model="gpt-4o",
        llm_service=llm_service,  # type: ignore[arg-type]
        executor=StubExecutor(),  # type: ignore[arg-type]
    )
    app = _make_interactive_tui_from_agent(
        agent,
        initial_events=event_log_to_tui_events(
            store.load_thread_event_log(metadata.session_id, MAIN_THREAD_ID)
        ),
        initial_history=[],
        session_recorder=SessionRecorder(store, metadata.session_id),
        session_id=metadata.session_id,
    )

    async with app.run_test() as pilot:
        app._command_subagents("")
        subagents = app.query_one(SubagentStatusWidget)
        activate = subagents.query_one("#subagent-activate", Button)
        activate.press()
        await pilot.pause()

        assert app._agent_app.active_thread_id == "child-thread-1"
        assert app._state.events[0].content == "child question"
        assert subagents.has_class("hidden")

        app._command_subagents("")
        assert not subagents.has_class("hidden")
        assert "main agent" in subagents.renderable
        assert "parent conversation" in subagents.renderable
        assert "activate returns to main" in subagents.renderable
        assert "Inspect" not in subagents.renderable
        assert "< [1] >" in subagents.renderable
        assert activate.label == "activate"
        activate.press()
        await pilot.pause()

        assert app._agent_app.active_thread_id == MAIN_THREAD_ID
        assert app._state.events[0].tool_call_id == "call-delegate-1"
        assert subagents.has_class("hidden")

        app._command_subagents("child-thread-1")

        assert app._agent_app.active_thread_id == "child-thread-1"
        assert app._state.events[0].content == "child question"

        app._command_subagents("main")

        assert app._agent_app.active_thread_id == MAIN_THREAD_ID


@pytest.mark.anyio
async def test_configured_tui_activate_subagent_initializes_runtime_and_switches_thread(
    tmp_path,
) -> None:
    store = SessionStore(tmp_path)
    metadata = store.create_session()
    store.create_thread(
        session_id=metadata.session_id,
        thread_id="child-thread-1",
        agent_id="child-agent-1",
        role="subagent",
        title="Inspect",
        status="completed",
        parent_thread_id=MAIN_THREAD_ID,
        metadata={
            "instructions": "Report evidence.",
            "context_brief": "repo",
            "allowed_tools": [],
        },
    )
    store.append_event(
        metadata.session_id,
        SessionEvent(
            thread_id="child-thread-1",
            agent_id="child-agent-1",
            run_id="child-run-1",
            kind="user_message",
            payload={"content": "child question"},
        ),
    )
    call = LLMToolCall(
        name="delegate_to_subagent",
        arguments=(
            '{"task":"Inspect","instructions":"Report evidence.",'
            '"context_brief":"repo","allowed_tools":[]}'
        ),
        call_id="call-delegate-1",
    )
    audit_output = (
        '{"type":"subagent_audit","thread_id":"child-thread-1",'
        '"agent_id":"child-agent-1","run_id":"child-run-1",'
        '"status":"completed","summary":"done","step_count":1}'
    )
    store.append_event(
        metadata.session_id,
        SessionEvent(
            run_id="parent-run-1",
            kind="tool_started",
            payload={
                "content": "",
                "tool_name": "delegate_to_subagent",
                "tool_call_id": call.call_id,
                "tool_call": call.asdict(),
            },
        ),
    )
    store.append_event(
        metadata.session_id,
        SessionEvent(
            run_id="parent-run-1",
            kind="tool_result",
            payload={
                "content": "completed",
                "tool_name": "delegate_to_subagent",
                "tool_call_id": call.call_id,
                "tool_arguments": call.arguments,
                "output": audit_output,
                "model_output": audit_output,
                "status": "completed",
            },
        ),
    )
    llm_service = StubLLMService([])

    def agent_factory(config: AgentAppConfig) -> Agent:
        return Agent(
            prompt="Prompt",
            default_model=config.default_model,
            llm_service=llm_service,  # type: ignore[arg-type]
            executor=StubExecutor(),  # type: ignore[arg-type]
        )

    app = AceAIConfiguredTUI(
        agent_factory,
        initial_config=AgentAppConfig(
            provider="openai",
            api_key="openai-key",
            model="gpt-4o",
            default_model="gpt-4o",
            api_keys={"openai": "openai-key"},
        ),
        initial_question="",
        default_model="gpt-4o",
        initial_events=event_log_to_tui_events(
            store.load_thread_event_log(metadata.session_id, MAIN_THREAD_ID)
        ),
        initial_history=[],
        session_recorder=SessionRecorder(store, metadata.session_id),
        session_id=metadata.session_id,
    )

    async with app.run_test() as pilot:
        assert app._agent_app is None

        app._command_subagents("")
        app.query_one("#subagent-activate", Button).press()
        await pilot.pause()

        assert app._agent_app is not None
        assert app._agent_app.active_thread_id == "child-thread-1"
        assert app._state.events[0].content == "child question"
        assert "Configure AceAI before switching threads." not in [
            event.content for event in app._state.events
        ]


def _press_config_apply(screen: ConfigScreen) -> None:
    class Pressed:
        button = screen.query_one("#apply", Button)

    screen.on_button_pressed(Pressed())
