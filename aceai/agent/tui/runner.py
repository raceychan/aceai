"""Live runner that bridges the AceAI app facade into the Textual app."""

import asyncio
import os
import sys
from pathlib import Path
from typing import AsyncGenerator, Callable, cast

from msgspec import Struct
from opentelemetry.context import Context
from textual.timer import Timer
from textual.widgets import Input
from textual.worker import Worker

from aceai.agent.app import AceAgentApp, UpdateCheckResult
from aceai.agent.citations import (
    IdeaCitationOrigin,
    TurnCitation,
)
from aceai.agent.ideas import Idea, IdeaStore
from aceai.agent.project import ProjectMetadata
from aceai.agent.features import default_agent_tools
from aceai.agent.provider_catalog import (
    api_key_env,
    model_options,
    supported_models,
    supports_reasoning_effort,
)
from aceai.agent.provider_auth import default_api_key_for_provider
from aceai.agent.session import SessionRecorder, SessionState
from aceai.agent.ace_agent import ACE_AGENT_BUILTIN_SKILL_PATHS
from aceai.agent.config import (
    AgentAppConfig,
    ReasoningLevel,
    replace_config,
    save_config,
)
from aceai.core import Agent
from aceai.core.events import AgentEvent, RunSuspendedEvent
from aceai.core.executor import Executor
from aceai.core.skills import SkillLoader, SkillLoadingError, SkillRegistry
from aceai.llm.models import LLMMessage
from aceai.llm.openai import OpenAIModel
from aceai.llm.models import LLMRequestMeta

from .app import AceAITUI
from .events import TUIEvent, TUIIdeaItem
from .metadata import MetadataSection
from .session_replay import event_log_to_tui_events
from .setup import (
    ConfigSelection,
    ConfigScreen,
    IdeaPickerScreen,
    ProviderSetupScreen,
    SkillConfigItem,
    ToolPermissionItem,
    _skill_source,
)
from .widgets import ApprovalWidget
from .widgets import CitationPreviewWidget
from .widgets import CommandCompletionItem
from .widgets import CommandInput
from .widgets import QueuedTurnsWidget
from .widgets import StatusBarWidget
from .widgets import TopBarWidget

AgentFactory = Callable[[AgentAppConfig], Agent]
CommandHandler = Callable[[str], None]
COMMAND_NAMES_ATTR = "_aceai_tui_command_names"
UPDATE_INSTRUCTIONS = "Run /update to upgrade AceAI and restart."
UPDATE_COMMAND: tuple[str, ...] = ("uv", "tool", "upgrade", "aceai")
COMMAND_DESCRIPTIONS: dict[str, str] = {
    "clear": "Clear the visible transcript",
    "config": "Open provider, model, skill, and tool settings",
    "debug": "Toggle the debug detail view",
    "idea": "Show ideas, save an idea, or delete one",
    "quit": "Exit AceAI",
    "sessions": "Open the session picker",
    "stats": "Open runtime and usage details in config",
    "steer": "Interrupt or redirect the current run",
    "trajectory": "Open the event trajectory view",
    "update": "Upgrade AceAI and restart",
}


def tui_command(*names: str):
    if not names:
        raise ValueError("TUI command must declare at least one name")
    for name in names:
        if name == "":
            raise ValueError("TUI command name cannot be empty")
        if name.startswith("/"):
            raise ValueError("TUI command name must not include slash")

    def decorate(handler):
        setattr(handler, COMMAND_NAMES_ATTR, names)
        return handler

    return decorate


def _as_model(provider_name: str, model: str) -> OpenAIModel:
    if model not in supported_models(provider_name):
        raise ValueError("Unsupported model")
    return cast(OpenAIModel, model)


def _model_options_text(provider_name: str) -> str:
    model_names = ", ".join(option[1] for option in model_options(provider_name))
    return f"Available models: {model_names}"


def _model_from_request_meta(
    request_meta: LLMRequestMeta,
    default_model: str,
    provider_name: str,
) -> OpenAIModel:
    if "model" in request_meta:
        return _as_model(provider_name, request_meta["model"])
    return _as_model(provider_name, default_model)


def _request_meta_with_reasoning_level(
    request_meta: LLMRequestMeta,
    reasoning_level: ReasoningLevel,
) -> LLMRequestMeta:
    next_meta = dict(request_meta)
    if reasoning_level == "auto":
        next_meta.pop("reasoning", None)
        return next_meta
    next_meta["reasoning"] = {"effort": reasoning_level, "summary": "auto"}
    return next_meta


def _reasoning_level_from_request_meta(request_meta: LLMRequestMeta) -> ReasoningLevel:
    if "reasoning" not in request_meta:
        return "auto"
    effort = request_meta["reasoning"].get("effort")
    if effort not in ("low", "medium", "high", "max"):
        return "auto"
    return effort


def _skill_config_items(registry: SkillRegistry) -> tuple[SkillConfigItem, ...]:
    return tuple(
        SkillConfigItem(
            name=skill.name,
            description=skill.description,
            location=str(skill.skill_file),
            builtin=_is_builtin_skill_location(skill.skill_file),
            source=_skill_source(skill.skill_file),
        )
        for skill in registry.get_skills()
    )


def _is_builtin_skill_location(skill_file: Path) -> bool:
    return _skill_source(skill_file) == "aceai builtin"


def _parse_command(text: str) -> tuple[str, str] | None:
    if not text.startswith("/"):
        return None
    body = text.removeprefix("/")
    if body == "":
        return None
    name, separator, arg = body.partition(" ")
    if separator == "":
        return name, ""
    return name, arg


class _RuntimeStreamMixin:
    _agent_app: AceAgentApp | None
    _active_worker: Worker[None] | None
    _idea_store: IdeaStore
    _pending_turn_citations: list[TurnCitation]
    _cancel_armed: bool
    _cancel_arm_timer: Timer | None

    def on_mount(self) -> None:
        super().on_mount()
        self._start_update_check()

    def _start_update_check(self) -> None:
        if self._agent_app is None:
            return
        self.run_worker(
            self._check_for_updates(),
            name="aceai-update-check",
            description="Check whether a newer AceAI release is available",
            exit_on_error=False,
        )

    async def _check_for_updates(self) -> None:
        result = await self._agent_app.check_for_updates()
        if result is None:
            return
        self.append_event(TUIEvent.session_notice(_update_available_notice(result)))

    async def _stream_agent_turn(
        self,
        question: str,
        citations: tuple[TurnCitation, ...] = (),
    ) -> None:
        if self._agent_app is None:
            raise RuntimeError("AceAI app is not configured")
        await self._consume_agent_stream(
            self._agent_app.start_turn(question, citations=citations)
        )
        self._start_next_queued_run()

    async def _stream_approval_decision(
        self, *, approved: bool, reason: str = ""
    ) -> None:
        if approved:
            stream = self._agent_app.approve_tool()
        else:
            stream = self._agent_app.reject_tool(reason)
        await self._consume_agent_stream(stream)
        self._start_next_queued_run()

    async def _consume_agent_stream(
        self,
        stream: AsyncGenerator[AgentEvent, None],
    ) -> None:
        try:
            async for event in stream:
                self.append_agent_event(event)
                if isinstance(event, RunSuspendedEvent):
                    self.show_pending_approval()
        finally:
            await stream.aclose()
            self._sync_app_state()

    def _sync_app_state(self) -> None:
        if self._agent_app is None:
            return
        self._session_recorder = self._agent_app.session_recorder
        self._session_id = self._agent_app.session_id
        self._llm_history = self._agent_app.llm_history
        self._active_run = self._agent_app.active_run
        self._refresh_queued_turns()
        if self._session_id is not None:
            self.title = self._window_title()
            if self.is_mounted:
                self.query_one(TopBarWidget).set_title(self.title)

    def on_unmount(self) -> None:
        agent_app = self._agent_app
        if agent_app is not None:
            agent_app.session_service.finalize()
        super().on_unmount()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if not isinstance(event.input, CommandInput):
            return
        self._handle_command_input_submitted(event.input, event.value)

    def on_command_input_submitted(self, event: CommandInput.Submitted) -> None:
        self._handle_command_input_submitted(event.input, event.value)

    def _handle_command_input_submitted(
        self,
        command_input: CommandInput,
        value: str,
    ) -> None:
        question = value
        if question == "":
            return
        if self._dispatch_command(question):
            self.exit_command_input(command_input)
            return
        self.start_run(question)
        self.exit_command_input(command_input)

    def cancel_active_run(self) -> bool:
        if self._active_worker is None or not self._active_worker.is_running:
            self._clear_cancel_arm()
            return False
        if not self._cancel_armed:
            self._arm_cancel()
            return True
        self._clear_cancel_arm()
        self._active_worker.cancel()
        self._active_worker = None
        if self._agent_app is not None:
            self._agent_app.cancel_active_turn()
        self.clear_approval_request()
        self._sync_app_state()
        self.append_event(TUIEvent.run_cancelled("Cancelled current response."))
        return True

    def _arm_cancel(self) -> None:
        self._cancel_armed = True
        if self._cancel_arm_timer is not None:
            self._cancel_arm_timer.stop()
        self.query_one(StatusBarWidget).show_notice(
            "Esc again stops response",
            timeout=1.4,
            style="bold #bf616a",
        )
        self._cancel_arm_timer = self.set_timer(1.4, self._clear_cancel_arm)

    def _clear_cancel_arm(self) -> None:
        self._cancel_armed = False
        timer = self._cancel_arm_timer
        self._cancel_arm_timer = None
        if timer is not None:
            timer.stop()

    def _dispatch_command(self, text: str) -> bool:
        parsed = _parse_command(text)
        if parsed is None:
            return False
        name, arg = parsed
        command = self._command_handlers().get(name)
        if command is None:
            return False
        command(arg)
        return True

    def _command_handlers(self) -> dict[str, CommandHandler]:
        handlers: dict[str, CommandHandler] = {}
        for cls in reversed(type(self).mro()):
            for value in cls.__dict__.values():
                names = getattr(value, COMMAND_NAMES_ATTR, None)
                if names is None:
                    continue
                bound = value.__get__(self, type(self))
                for name in names:
                    handlers[name] = bound
        return handlers

    def command_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._command_handlers()))

    def command_completion_items(self) -> tuple[CommandCompletionItem, ...]:
        return tuple(
            CommandCompletionItem(
                command=name,
                description=COMMAND_DESCRIPTIONS[name],
            )
            for name in self.command_names()
        )

    @tui_command("quit")
    def _command_quit(self, arg: str) -> None:
        self.exit()

    @tui_command("clear")
    def _command_clear(self, arg: str) -> None:
        self.load_events([])

    @tui_command("sessions")
    def _command_sessions(self, arg: str) -> None:
        self.open_session_selector()

    @tui_command("stats")
    def _command_stats(self, arg: str) -> None:
        self.open_config_screen(initial_tab="stats-tab")

    @tui_command("config")
    def _command_config(self, arg: str) -> None:
        self.open_config_screen()

    @tui_command("debug")
    def _command_debug(self, arg: str) -> None:
        self.action_toggle_debug_mode()

    @tui_command("trajectory")
    def _command_trajectory(self, arg: str) -> None:
        self.open_trajectory_screen()

    @tui_command("update")
    def _command_update(self, arg: str) -> None:
        if self._active_worker is not None and self._active_worker.is_running:
            self._active_worker.cancel()
        self.append_event(
            TUIEvent.session_notice("Updating AceAI with uv tool upgrade aceai...")
        )
        self.run_worker(
            self._run_update_and_restart(),
            name="aceai-self-update",
            description="Upgrade AceAI and restart this process",
            exit_on_error=False,
        )

    @tui_command("steer")
    def _command_steer(self, arg: str) -> None:
        if arg == "":
            self.append_event(TUIEvent.session_notice("Usage: /steer <message>"))
            return
        self.steer_run(arg)

    @tui_command("idea")
    def _command_idea(self, arg: str) -> None:
        if arg == "":
            self._show_ideas()
            return
        try:
            delete_index = _idea_delete_index(arg)
        except ValueError:
            self.notify_session("Usage: /idea delete <number>")
            return
        if delete_index is not None:
            self._delete_idea(delete_index)
            return
        idea = self._capture_idea(arg)
        self.notify_session(
            f"Saved idea from {idea.created_at.strftime('%Y-%m-%d %H:%M')}."
        )

    def enqueue_run(self, question: str) -> None:
        self._clear_cancel_arm()
        agent_app = self._agent_app
        if agent_app is None:
            self.append_event(
                TUIEvent.session_notice("Configure AceAI before enqueueing a run.")
            )
            return
        if (
            self._active_worker is None or not self._active_worker.is_running
        ) and not agent_app.is_running_suspended:
            self._start_run_now(question)
            return
        agent_app.enqueue_turn(question)
        self._refresh_queued_turns()

    def steer_run(self, question: str) -> None:
        self._clear_cancel_arm()
        agent_app = self._agent_app
        if agent_app is None:
            self.append_event(
                TUIEvent.session_notice("Configure AceAI before steering a run.")
            )
            return
        if agent_app.is_running_suspended:
            self.append_event(
                TUIEvent.session_notice(
                    "Choose Approve or Reject before steering this run."
                )
            )
            return
        if self._active_worker is not None and self._active_worker.is_running:
            self._active_worker.cancel()
        self._start_run_now(question)

    def steer_queued_run(self, index: int) -> None:
        self._clear_cancel_arm()
        agent_app = self._agent_app
        if agent_app is None:
            self.append_event(
                TUIEvent.session_notice("Configure AceAI before steering a run.")
            )
            return
        if agent_app.is_running_suspended:
            self.append_event(
                TUIEvent.session_notice(
                    "Choose Approve or Reject before steering this run."
                )
            )
            return
        try:
            question = agent_app.take_queued_turn(index)
        except IndexError:
            self._refresh_queued_turns()
            return
        self._refresh_queued_turns()
        if self._active_worker is not None and self._active_worker.is_running:
            self._active_worker.cancel()
        self._start_run_now(question)

    def _start_run_now(
        self,
        question: str,
        *,
        citations: tuple[TurnCitation, ...] = (),
    ) -> None:
        self._clear_cancel_arm()
        if self._agent_app is None:
            raise RuntimeError("AceAI app is not configured")
        self._agent_app.ensure_session()
        self._sync_app_state()
        self._persist_session_state()
        if not citations:
            citations = self._consume_pending_citations()
        self.append_event(TUIEvent.user_message(question, citations=citations))
        self._active_worker = self.run_worker(
            self._stream_agent_turn(question, citations),
            name="aceai-agent",
            description="Run AceAI agent and stream events into the TUI",
            exit_on_error=True,
        )
        self._refresh_citation_preview()

    def _start_next_queued_run(self) -> None:
        agent_app = self._agent_app
        if agent_app is None:
            return
        if agent_app.is_running_suspended:
            return
        question = agent_app.pop_queued_turn()
        if question is None:
            return
        self._refresh_queued_turns()
        self.call_after_refresh(lambda: self._start_run_now(question))

    def _refresh_queued_turns(self) -> None:
        if not self.is_mounted:
            return
        agent_app = self._agent_app
        questions = () if agent_app is None else agent_app.queued_questions
        widgets = list(self.query(QueuedTurnsWidget))
        if not widgets:
            return
        widgets[0].set_questions(questions)

    def on_queued_turns_widget_selected(
        self,
        event: QueuedTurnsWidget.Selected,
    ) -> None:
        event.stop()
        self.steer_queued_run(event.index)

    def _capture_idea(self, content: str) -> Idea:
        agent_app = self._agent_app
        if agent_app is None:
            return self._idea_store.capture(
                content,
                project=self._project,
                source_session_id=self._session_id,
            )
        return agent_app.capture_idea(content)

    def _list_ideas(self) -> list[Idea]:
        agent_app = self._agent_app
        if agent_app is None:
            return self._idea_store.list_for_display(current_project=self._project)
        return agent_app.list_ideas()

    def _show_ideas(self) -> None:
        self.push_screen(
            IdeaPickerScreen(
                ideas=self._list_ideas(),
                capture_idea=self._capture_idea_and_list,
                save_idea=self._update_idea_and_list,
                delete_idea=self._delete_idea_and_list,
            ),
            self._reference_idea,
        )

    def action_ideas(self) -> None:
        self._show_ideas()

    def _reference_idea(self, idea: Idea | None) -> None:
        if idea is None:
            self.query_one(CommandInput).focus()
            return
        self._pending_turn_citations = [
            TurnCitation(
                content=idea.content,
                origin=IdeaCitationOrigin(kind="idea", idea_id=idea.idea_id),
            )
        ]
        self._refresh_citation_preview()
        self.query_one(CommandInput).focus()

    def _pending_citations(self) -> list[TurnCitation]:
        if not hasattr(self, "_pending_turn_citations"):
            self._pending_turn_citations = []
        return self._pending_turn_citations

    def _consume_pending_citations(self) -> tuple[TurnCitation, ...]:
        citations = tuple(self._pending_citations())
        self._pending_turn_citations = []
        return citations

    def _refresh_citation_preview(self) -> None:
        if not self.is_mounted:
            return
        widgets = list(self.query(CitationPreviewWidget))
        if not widgets:
            return
        widgets[0].set_citations(tuple(self._pending_citations()))

    def _capture_idea_and_list(self, content: str) -> list[Idea]:
        self._capture_idea(content)
        return self._list_ideas()

    def _update_idea_and_list(self, index: int, content: str) -> list[Idea]:
        agent_app = self._agent_app
        if agent_app is None:
            self._idea_store.update_displayed(
                index,
                content,
                current_project=self._project,
            )
            return self._idea_store.list_for_display(current_project=self._project)
        agent_app.update_idea(index, content)
        return agent_app.list_ideas()

    def _delete_idea_and_list(self, index: int) -> list[Idea]:
        agent_app = self._agent_app
        if agent_app is None:
            self._idea_store.delete_displayed(
                index,
                current_project=self._project,
            )
            return self._idea_store.list_for_display(current_project=self._project)
        agent_app.delete_idea(index)
        return agent_app.list_ideas()

    def _delete_idea(self, index: int) -> None:
        try:
            agent_app = self._agent_app
            if agent_app is None:
                idea = self._idea_store.delete_displayed(
                    index,
                    current_project=self._project,
                )
            else:
                idea = agent_app.delete_idea(index)
        except IndexError:
            self.notify_session(f"No idea found at {index}.")
            return
        self.append_event(TUIEvent.idea_list(_idea_items(self._list_ideas())))
        self.notify_session(f"Deleted idea {index}: {idea.content}")

    async def _run_update_and_restart(self) -> None:
        result = await run_update_command()
        if result.return_code != 0:
            self.append_event(
                TUIEvent.session_notice(
                    f"AceAI update failed with exit code {result.return_code}.\n"
                    f"{result.output}"
                )
            )
            return
        self.append_event(TUIEvent.session_notice("AceAI updated. Restarting..."))
        restart_current_process()


def _update_available_notice(result: UpdateCheckResult) -> str:
    return (
        f"AceAI {result.latest_version} is available "
        f"(current {result.current_version}).\n"
        f"{UPDATE_INSTRUCTIONS}"
    )


def _idea_items(ideas: list[Idea]) -> list[TUIIdeaItem]:
    return [_idea_item(index, idea) for index, idea in enumerate(ideas, start=1)]


def _idea_item(index: int, idea: Idea) -> TUIIdeaItem:
    lines = idea.content.splitlines()
    title = ""
    body_lines: list[str] = []
    for line in lines:
        if title == "" and line != "":
            title = line
            continue
        if title != "":
            body_lines.append(line)
    return TUIIdeaItem(
        index=index,
        project_name=idea.project_name,
        created_at=idea.created_at.strftime("%Y-%m-%d %H:%M"),
        title=title,
        body="\n".join(body_lines),
    )


def _idea_delete_index(arg: str) -> int | None:
    prefix = "delete "
    if not arg.startswith(prefix):
        return None
    return int(arg.removeprefix(prefix))


class UpdateCommandResult(Struct, frozen=True, kw_only=True):
    return_code: int
    output: str


async def run_update_command() -> UpdateCommandResult:
    try:
        process = await asyncio.create_subprocess_exec(
            *UPDATE_COMMAND,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except OSError as err:
        return UpdateCommandResult(return_code=127, output=str(err))
    output, _ = await process.communicate()
    if process.returncode is None:
        raise RuntimeError("uv update process did not report an exit code")
    return UpdateCommandResult(
        return_code=process.returncode,
        output=output.decode(errors="replace"),
    )


def restart_current_process() -> None:
    if not sys.argv:
        os.execv(sys.executable, [sys.executable])
    executable = sys.argv[0]
    if executable == "":
        os.execv(sys.executable, [sys.executable])
    os.execvp(executable, sys.argv)


class AceAIInteractiveTUI(_RuntimeStreamMixin, AceAITUI):
    """Textual app that runs an Agent from submitted questions."""

    def __init__(
        self,
        agent: Agent,
        *,
        initial_events: list[TUIEvent] | None = None,
        initial_history: list[LLMMessage] | None = None,
        session_recorder: SessionRecorder | None = None,
        session_id: str | None = None,
        project: ProjectMetadata | None = None,
        idea_store: IdeaStore | None = None,
        trace_ctx: Context | None = None,
        request_meta: LLMRequestMeta | None = None,
    ) -> None:
        self._request_meta: LLMRequestMeta = dict(request_meta or {})
        self._provider_name = "openai"
        self._reasoning_level = _reasoning_level_from_request_meta(self._request_meta)
        self._selected_model = _model_from_request_meta(
            self._request_meta,
            agent.default_model,
            self._provider_name,
        )
        if self._reasoning_level != "auto" and not supports_reasoning_effort(
            self._provider_name,
            self._selected_model,
        ):
            self._reasoning_level = "auto"
        self._request_meta = _request_meta_with_reasoning_level(
            self._request_meta,
            self._reasoning_level,
        )
        self._request_meta["model"] = self._selected_model
        super().__init__(
            events=initial_events or [],
            model=self._selected_model,
            reasoning_level=self._reasoning_level,
            session_recorder=session_recorder,
            session_id=session_id,
            project=project,
            idea_store=idea_store,
            record_events=False,
        )
        self._agent = agent
        self._trace_ctx = trace_ctx
        self._agent_app = AceAgentApp(
            agent,
            provider_name=self._provider_name,
            selected_model=self._selected_model,
            initial_history=initial_history,
            session_store=self._session_store(),
            session_recorder=session_recorder,
            session_id=session_id,
            project=self._project,
            idea_store=self._idea_store,
            trace_ctx=trace_ctx,
            request_meta=self._request_meta,
        )
        self._persist_session_state()
        self._llm_history = self._agent_app.llm_history
        self._active_worker: Worker[None] | None = None
        self._active_run = self._agent_app.active_run
        self._cancel_armed = False
        self._cancel_arm_timer: Timer | None = None

    def start_run(
        self,
        question: str,
        *,
        citations: tuple[TurnCitation, ...] = (),
    ) -> None:
        if self._active_worker is not None and self._active_worker.is_running:
            self.enqueue_run(question)
            return
        if self._agent_app.is_running_suspended:
            self.append_event(
                TUIEvent.session_notice(
                    "Choose Approve or Reject before starting another run."
                )
            )
            return
        self._start_run_now(question, citations=citations)

    def approve_pending_tool(self) -> None:
        request = self._pending_approval_request()
        if request is None:
            self.append_event(TUIEvent.session_notice("No pending tool approval."))
            return
        self.clear_approval_request()
        self._active_worker = self.run_worker(
            self._stream_approval_decision(approved=True),
            name="aceai-approval",
            description="Resume AceAI agent after tool approval",
            exit_on_error=True,
        )

    def show_pending_approval(self) -> None:
        request = self._pending_approval_request()
        if request is None:
            self.append_event(TUIEvent.session_notice("No pending tool approval."))
            return
        self.show_approval_request(request)

    def on_approval_widget_selected(self, event: ApprovalWidget.Selected) -> None:
        event.stop()
        if event.approved:
            self.approve_pending_tool()
            return
        self.reject_pending_tool("rejected by caller")

    def reject_pending_tool(self, reason: str) -> None:
        request = self._pending_approval_request()
        if request is None:
            self.append_event(TUIEvent.session_notice("No pending tool approval."))
            return
        if reason == "":
            reason = "rejected by caller"
        self.clear_approval_request()
        self._active_worker = self.run_worker(
            self._stream_approval_decision(approved=False, reason=reason),
            name="aceai-approval",
            description="Resume AceAI agent after tool rejection",
            exit_on_error=True,
        )

    def _pending_approval_request(self):
        return self._agent_app.pending_approval_request()

    def switch_session(self, session_id: str) -> None:
        if session_id == self._session_id:
            return
        try:
            snapshot = self._agent_app.switch_session(session_id)
        except KeyError:
            self.notify_session(f"Session not found: {session_id}")
            return
        self._sync_app_state()
        self.load_events(event_log_to_tui_events(snapshot.event_log))
        self.notify_session(f"Resumed session {snapshot.metadata.session_id}")
        self._restore_session_state()

    def show_model(self) -> None:
        self.append_event(
            TUIEvent.session_notice(
                f"Current model: {self._selected_model}\n{_model_options_text(self._provider_name)}"
            )
        )

    def action_config(self) -> None:
        self.open_config_screen()

    def open_config_screen(self, initial_tab: str = "settings-tab") -> None:
        self.push_screen(
            ConfigScreen(
                provider_name=self._provider_name,
                current_model=self._selected_model,
                default_model=cast(OpenAIModel, self._agent.default_model),
                reasoning_level=self._reasoning_level,
                skills="auto",
                skill_items=_skill_config_items(self._agent.skill_registry),
                skill_selection_mode="all",
                enabled_skills=(),
                api_keys={},
                stats_sections=self._metadata_sections(),
                initial_tab=initial_tab,
            ),
            self._handle_config_selection,
        )

    def _handle_config_selection(self, selection: ConfigSelection | None) -> None:
        if selection is None:
            return
        if type(selection) is str:
            self.switch_model(selection)
            return
        if selection.provider != self._provider_name:
            self.append_event(
                TUIEvent.session_notice(
                    "Provider changes are only available in the configured AceAI app."
                )
            )
            return
        self.switch_model(selection.model)

    def switch_model(self, model: str) -> None:
        if self._active_worker is not None and self._active_worker.is_running:
            self.append_event(
                TUIEvent.session_notice(
                    "Model changes apply after the current run finishes."
                )
            )
            return
        if model not in supported_models(self._provider_name):
            self.append_event(
                TUIEvent.session_notice(_model_options_text(self._provider_name))
            )
            return
        model_changed = model != self._selected_model
        if self._reasoning_level != "auto" and not supports_reasoning_effort(
            self._provider_name,
            model,
        ):
            self._reasoning_level = "auto"
        self._selected_model = cast(OpenAIModel, model)
        self._request_meta = _request_meta_with_reasoning_level(
            self._request_meta,
            self._reasoning_level,
        )
        self._request_meta["model"] = self._selected_model
        self._agent_app.switch_model(self._selected_model)
        self._sync_app_state()
        if model_changed:
            self.reset_status_cache_rate()
        self.set_status_model(
            self._selected_model,
            reasoning_level=self._reasoning_level,
        )
        self.notify_session(f"Switched model to {self._selected_model}")

    def _request_meta_for_run(self) -> LLMRequestMeta:
        request_meta: LLMRequestMeta = dict(self._request_meta)
        request_meta["model"] = self._selected_model
        return request_meta

    def _reload_llm_history(self) -> None:
        if self._session_recorder is None or self._session_id is None:
            self._llm_history = []
            return
        self._agent_app.restore_history_from_active_session()
        self._sync_app_state()

    def _persist_session_state(self) -> None:
        if self._session_recorder is None or self._session_id is None:
            return
        self._agent_app.persist_session_state()

    def _restore_session_state(self) -> None:
        if self._session_recorder is None or self._session_id is None:
            return
        state = self._session_recorder.store.get_session_state(self._session_id)
        if state.selected_model == "":
            return
        if (
            state.selected_provider != ""
            and state.selected_provider != self._provider_name
        ):
            return
        if state.selected_model not in supported_models(self._provider_name):
            return
        self._selected_model = cast(OpenAIModel, state.selected_model)
        self._request_meta["model"] = self._selected_model
        self._agent_app.switch_model(self._selected_model)
        self._sync_app_state()
        self.set_status_model(
            self._selected_model,
            reasoning_level=self._reasoning_level,
        )

    def _metadata_sections(self) -> list[MetadataSection]:
        return [
            *super()._metadata_sections(),
            *_agent_metadata_sections(
                self._agent,
                provider_name=self._provider_name,
                selected_model=self._selected_model,
            ),
        ]


class AceAIConfiguredTUI(_RuntimeStreamMixin, AceAITUI):
    """TUI that asks for provider settings before creating the agent."""

    def __init__(
        self,
        agent_factory: AgentFactory,
        *,
        initial_config: AgentAppConfig | None,
        initial_question: str,
        default_model: OpenAIModel,
        initial_events: list[TUIEvent] | None = None,
        initial_history: list[LLMMessage] | None = None,
        session_recorder: SessionRecorder | None = None,
        session_id: str | None = None,
        project: ProjectMetadata | None = None,
        idea_store: IdeaStore | None = None,
        trace_ctx: Context | None = None,
        request_meta: LLMRequestMeta | None = None,
    ) -> None:
        self._request_meta: LLMRequestMeta = dict(request_meta or {})
        self._provider_name = (
            initial_config.provider if initial_config is not None else "openai"
        )
        self._current_config = initial_config
        initial_model = (
            initial_config.model
            if initial_config is not None
            else _model_from_request_meta(
                self._request_meta,
                default_model,
                self._provider_name,
            )
        )
        self._reasoning_level = (
            initial_config.reasoning_level
            if initial_config is not None
            else _reasoning_level_from_request_meta(self._request_meta)
        )
        if self._reasoning_level != "auto" and not supports_reasoning_effort(
            self._provider_name,
            initial_model,
        ):
            self._reasoning_level = "auto"
        self._request_meta = _request_meta_with_reasoning_level(
            self._request_meta,
            self._reasoning_level,
        )
        self._request_meta["model"] = initial_model
        super().__init__(
            events=initial_events or [],
            model=initial_model,
            reasoning_level=self._reasoning_level,
            session_recorder=session_recorder,
            session_id=session_id,
            project=project,
            idea_store=idea_store,
            record_events=False,
        )
        self._agent_factory = agent_factory
        self._initial_config = initial_config
        self._initial_question = initial_question
        self._default_model: OpenAIModel = default_model
        self._trace_ctx = trace_ctx
        self._selected_model: OpenAIModel = initial_model
        self._llm_history = list(initial_history or [])
        self._agent: Agent | None = None
        self._agent_app: AceAgentApp | None = None
        self._active_worker: Worker[None] | None = None
        self._active_run = None
        self._cancel_armed = False
        self._cancel_arm_timer: Timer | None = None

    def on_mount(self) -> None:
        super().on_mount()
        if self._initial_config is not None:
            self.apply_config(self._initial_config)
            return
        self.push_screen(
            ProviderSetupScreen(default_model=self._default_model),
            self._handle_setup_config,
        )

    def _handle_setup_config(self, config: AgentAppConfig | None) -> None:
        if config is None:
            return
        self.apply_config(config)

    def apply_config(self, config: AgentAppConfig) -> None:
        next_config = replace_config(config)
        model_changed = next_config.model != self._selected_model
        self._provider_name = next_config.provider
        self._current_config = next_config
        self._selected_model = next_config.model
        self._reasoning_level = next_config.reasoning_level
        self._request_meta = _request_meta_with_reasoning_level(
            self._request_meta,
            self._reasoning_level,
        )
        self._request_meta["model"] = self._selected_model
        self._agent = None
        self._agent_app = None
        self._active_run = None
        self._persist_session_state()
        if model_changed:
            self.reset_status_cache_rate()
        self.set_status_model(
            self._selected_model,
            reasoning_level=self._reasoning_level,
        )
        if self._initial_question != "":
            self.start_run(self._initial_question)

    def start_run(
        self,
        question: str,
        *,
        citations: tuple[TurnCitation, ...] = (),
    ) -> None:
        if self._current_config is None:
            self.query_one(CommandInput).value = question
            return
        if self._active_worker is not None and self._active_worker.is_running:
            self.enqueue_run(question)
            return
        self._ensure_agent_app()
        if self._agent_app.is_running_suspended:
            self.append_event(
                TUIEvent.session_notice(
                    "Choose Approve or Reject before starting another run."
                )
            )
            return
        self._start_run_now(question, citations=citations)

    def _ensure_agent_app(self) -> None:
        if self._agent_app is not None:
            return
        if self._current_config is None:
            raise RuntimeError("AceAI app is not configured")
        self._agent = self._agent_factory(self._current_config)
        self._agent_app = AceAgentApp(
            self._agent,
            provider_name=self._provider_name,
            selected_model=self._selected_model,
            initial_history=self._llm_history,
            session_store=self._session_store(),
            session_recorder=self._session_recorder,
            session_id=self._session_id,
            project=self._project,
            idea_store=self._idea_store,
            trace_ctx=self._trace_ctx,
            request_meta=self._request_meta,
        )
        self._sync_app_state()
        self._start_update_check()

    def switch_session(self, session_id: str) -> None:
        if self._agent_app is None:
            super().switch_session(session_id)
            self._reload_llm_history()
            return
        if session_id == self._session_id:
            return
        try:
            snapshot = self._agent_app.switch_session(session_id)
        except KeyError:
            self.notify_session(f"Session not found: {session_id}")
            return
        self._sync_app_state()
        self.load_events(event_log_to_tui_events(snapshot.event_log))
        self.notify_session(f"Resumed session {snapshot.metadata.session_id}")
        self._restore_session_state()

    def approve_pending_tool(self) -> None:
        request = self._pending_approval_request()
        if request is None:
            self.append_event(TUIEvent.session_notice("No pending tool approval."))
            return
        self.clear_approval_request()
        self._active_worker = self.run_worker(
            self._stream_approval_decision(approved=True),
            name="aceai-approval",
            description="Resume AceAI agent after tool approval",
            exit_on_error=True,
        )

    def show_pending_approval(self) -> None:
        request = self._pending_approval_request()
        if request is None:
            self.append_event(TUIEvent.session_notice("No pending tool approval."))
            return
        self.show_approval_request(request)

    def on_approval_widget_selected(self, event: ApprovalWidget.Selected) -> None:
        event.stop()
        if event.approved:
            self.approve_pending_tool()
            return
        self.reject_pending_tool("rejected by caller")

    def reject_pending_tool(self, reason: str) -> None:
        request = self._pending_approval_request()
        if request is None:
            self.append_event(TUIEvent.session_notice("No pending tool approval."))
            return
        if reason == "":
            reason = "rejected by caller"
        self.clear_approval_request()
        self._active_worker = self.run_worker(
            self._stream_approval_decision(approved=False, reason=reason),
            name="aceai-approval",
            description="Resume AceAI agent after tool rejection",
            exit_on_error=True,
        )

    def _pending_approval_request(self):
        if self._agent_app is None:
            return None
        return self._agent_app.pending_approval_request()

    def show_model(self) -> None:
        self.append_event(
            TUIEvent.session_notice(
                f"Current model: {self._selected_model}\n{_model_options_text(self._provider_name)}"
            )
        )

    def action_config(self) -> None:
        self.open_config_screen()

    def open_config_screen(self, initial_tab: str = "settings-tab") -> None:
        api_keys: dict[str, str] = {}
        if self._current_config is not None:
            api_keys.update(self._current_config.api_keys)
            api_keys[self._current_config.provider] = self._current_config.api_key
        self.push_screen(
            ConfigScreen(
                provider_name=self._provider_name,
                current_model=self._selected_model,
                default_model=self._current_config.default_model
                if self._current_config is not None
                else self._selected_model,
                skills=self._current_config.skills
                if self._current_config is not None
                else "",
                skill_items=self._available_skill_items(),
                skill_selection_mode=self._current_config.skill_selection_mode
                if self._current_config is not None
                else "all",
                enabled_skills=tuple(self._current_config.enabled_skills)
                if self._current_config is not None
                else (),
                tool_permission_items=self._available_tool_permission_items(),
                api_keys=api_keys,
                compress_threshold=self._current_config.compress_threshold
                if self._current_config is not None
                else "100%",
                reasoning_level=self._reasoning_level,
                stats_sections=self._metadata_sections(),
                initial_tab=initial_tab,
            ),
            self._handle_config_selection,
        )

    def _handle_config_selection(self, selection: ConfigSelection | None) -> None:
        if selection is None:
            return
        if type(selection) is str:
            self.switch_model(selection)
            return
        if selection.provider == self._provider_name:
            if selection.api_key != "" or self._current_config is not None:
                api_keys = {}
                if self._current_config is not None:
                    api_keys.update(self._current_config.api_keys)
                api_key = (
                    selection.api_key
                    if selection.api_key != ""
                    else self._current_config.api_key
                )
                api_keys[selection.provider] = api_key
                self.apply_user_config(
                    AgentAppConfig(
                        provider=selection.provider,
                        api_key=api_key,
                        model=selection.model,
                        default_model=selection.default_model,
                        skills=selection.skills,
                        skill_selection_mode=selection.skill_selection_mode,
                        enabled_skills=list(selection.enabled_skills),
                        api_keys=api_keys,
                        tool_permissions=selection.tool_permissions,
                        tool_enabled=selection.tool_enabled,
                        tool_max_calls=selection.tool_max_calls,
                        compress_threshold=selection.compress_threshold,
                        reasoning_level=selection.reasoning_level,
                    )
                )
                self.notify_session(
                    f"Updated provider credentials and switched model to {selection.model}"
                )
                return
            self.switch_model(selection.model)
            return
        api_key = selection.api_key
        if api_key == "":
            env_name = api_key_env(selection.provider)
            if env_name in os.environ:
                api_key = os.environ[env_name]
        if api_key == "":
            api_key = default_api_key_for_provider(selection.provider)
        if api_key == "":
            self.append_event(
                TUIEvent.session_notice(
                    f"API key required for provider {selection.provider}."
                )
            )
            return
        api_keys = {}
        if self._current_config is not None:
            api_keys.update(self._current_config.api_keys)
        api_keys[selection.provider] = api_key
        self.apply_user_config(
            AgentAppConfig(
                provider=selection.provider,
                api_key=api_key,
                model=selection.model,
                default_model=selection.default_model,
                skills=selection.skills,
                skill_selection_mode=selection.skill_selection_mode,
                enabled_skills=list(selection.enabled_skills),
                api_keys=api_keys,
                tool_permissions=selection.tool_permissions,
                tool_enabled=selection.tool_enabled,
                tool_max_calls=selection.tool_max_calls,
                compress_threshold=selection.compress_threshold,
                reasoning_level=selection.reasoning_level,
            )
        )
        self.notify_session(
            f"Switched provider to {selection.provider} and model to {selection.model}"
        )

    def apply_user_config(self, config: AgentAppConfig) -> None:
        self.apply_config(config)
        save_config(config)

    def switch_model(self, model: str) -> None:
        if self._active_worker is not None and self._active_worker.is_running:
            self.append_event(
                TUIEvent.session_notice(
                    "Model changes apply after the current run finishes."
                )
            )
            return
        if model not in supported_models(self._provider_name):
            self.append_event(
                TUIEvent.session_notice(_model_options_text(self._provider_name))
            )
            return
        model_changed = model != self._selected_model
        reasoning_level = self._reasoning_level
        if self._current_config is not None:
            reasoning_level = self._current_config.reasoning_level
        if reasoning_level != "auto" and not supports_reasoning_effort(
            self._provider_name,
            model,
        ):
            reasoning_level = "auto"
        if self._current_config is not None:
            next_config = AgentAppConfig(
                provider=self._current_config.provider,
                api_key=self._current_config.api_key,
                model=cast(OpenAIModel, model),
                default_model=cast(OpenAIModel, model),
                skills=self._current_config.skills,
                skill_selection_mode=self._current_config.skill_selection_mode,
                enabled_skills=self._current_config.enabled_skills,
                api_keys=self._current_config.api_keys,
                tool_permissions=self._current_config.tool_permissions,
                tool_enabled=self._current_config.tool_enabled,
                tool_max_calls=self._current_config.tool_max_calls,
                compress_threshold=self._current_config.compress_threshold,
                reasoning_level=reasoning_level,
            )
            save_config(next_config)
            self._current_config = next_config
        self._selected_model = cast(OpenAIModel, model)
        self._reasoning_level = reasoning_level
        self._request_meta = _request_meta_with_reasoning_level(
            self._request_meta,
            self._reasoning_level,
        )
        self._request_meta["model"] = self._selected_model
        if self._agent_app is not None:
            self._agent_app.switch_model(self._selected_model)
            self._sync_app_state()
        self._persist_session_state()
        if model_changed:
            self.reset_status_cache_rate()
        self.set_status_model(
            self._selected_model,
            reasoning_level=self._reasoning_level,
        )
        self.notify_session(f"Switched model to {self._selected_model}")

    def _available_skill_items(self) -> tuple[SkillConfigItem, ...]:
        if self._current_config is None:
            if self._agent is None:
                return ()
            return _skill_config_items(self._agent.skill_registry)
        try:
            registry = SkillLoader.load_registry(
                self._current_config.skills,
                extra_skill_paths=ACE_AGENT_BUILTIN_SKILL_PATHS,
            )
        except (SkillLoadingError, OSError) as exc:
            self.notify_session(f"Skill search failed: {exc}")
            return ()
        return _skill_config_items(registry)

    def _available_tool_permission_items(self) -> tuple[ToolPermissionItem, ...]:
        configured_permissions = (
            self._current_config.tool_permissions
            if self._current_config is not None
            else {}
        )
        configured_enabled = (
            self._current_config.tool_enabled
            if self._current_config is not None
            else {}
        )
        configured_max_calls = (
            self._current_config.tool_max_calls
            if self._current_config is not None
            else {}
        )
        items: list[ToolPermissionItem] = []
        for configured_tool in default_agent_tools():
            permission = configured_permissions.get(configured_tool.name)
            if permission is None:
                permission = (
                    "ask" if configured_tool.metadata.require_approval else "always"
                )
            items.append(
                ToolPermissionItem(
                    name=configured_tool.name,
                    description=configured_tool.description,
                    permission=permission,
                    enabled=configured_enabled.get(configured_tool.name, True),
                    max_calls_per_run=configured_max_calls.get(configured_tool.name),
                )
            )
        return tuple(items)

    def _request_meta_for_run(self) -> LLMRequestMeta:
        request_meta: LLMRequestMeta = dict(self._request_meta)
        request_meta["model"] = self._selected_model
        return request_meta

    def _reload_llm_history(self) -> None:
        if self._session_recorder is None or self._session_id is None:
            self._llm_history = []
            return
        if self._agent_app is None:
            event_log = self._session_recorder.store.load_event_log(self._session_id)
            self._llm_history = event_log.replay_llm_history()
            self._active_run = None
            return
        self._agent_app.restore_history_from_active_session()
        self._sync_app_state()

    def _persist_session_state(self) -> None:
        if self._session_recorder is None or self._session_id is None:
            return
        if self._agent_app is None:
            self._session_recorder.store.update_session_state(
                self._session_id,
                SessionState(
                    selected_provider=self._provider_name,
                    selected_model=self._selected_model,
                ),
            )
            return
        self._agent_app.persist_session_state()

    def _restore_session_state(self) -> None:
        if self._session_recorder is None or self._session_id is None:
            return
        state = self._session_recorder.store.get_session_state(self._session_id)
        if state.selected_model == "":
            return
        if state.selected_provider == self._provider_name:
            self.switch_model(state.selected_model)

    def _metadata_sections(self) -> list[MetadataSection]:
        sections = super()._metadata_sections()
        if self._agent is None:
            return [
                *sections,
                MetadataSection(
                    title="Agent",
                    lines=[
                        f"provider: {self._provider_name}",
                        f"model: {self._selected_model}",
                        "configured: no",
                    ],
                ),
            ]
        return [
            *sections,
            *_agent_metadata_sections(
                self._agent,
                provider_name=self._provider_name,
                selected_model=self._selected_model,
            ),
        ]


class AceAILiveTUI(AceAIInteractiveTUI):
    """Textual app that starts a single run on mount."""

    def __init__(
        self,
        agent: Agent,
        question: str,
        *,
        initial_events: list[TUIEvent] | None = None,
        initial_history: list[LLMMessage] | None = None,
        session_recorder: SessionRecorder | None = None,
        session_id: str | None = None,
        project: ProjectMetadata | None = None,
        trace_ctx: Context | None = None,
        request_meta: LLMRequestMeta | None = None,
    ) -> None:
        super().__init__(
            agent,
            initial_events=initial_events,
            initial_history=initial_history,
            session_recorder=session_recorder,
            session_id=session_id,
            project=project,
            trace_ctx=trace_ctx,
            request_meta=request_meta,
        )
        self._initial_question = question

    def on_mount(self) -> None:
        super().on_mount()
        self.start_run(self._initial_question)


def run_interactive_tui(
    agent: Agent,
    *,
    initial_events: list[TUIEvent] | None = None,
    initial_history: list[LLMMessage] | None = None,
    session_recorder: SessionRecorder | None = None,
    session_id: str | None = None,
    project: ProjectMetadata | None = None,
    trace_ctx: Context | None = None,
    request_meta: LLMRequestMeta | None = None,
) -> None:
    AceAIInteractiveTUI(
        agent,
        initial_events=initial_events,
        initial_history=initial_history,
        session_recorder=session_recorder,
        session_id=session_id,
        project=project,
        trace_ctx=trace_ctx,
        request_meta=request_meta,
    ).run()


def run_configured_tui(
    agent_factory: AgentFactory,
    *,
    initial_config: AgentAppConfig | None,
    initial_question: str,
    default_model: OpenAIModel,
    initial_events: list[TUIEvent] | None = None,
    initial_history: list[LLMMessage] | None = None,
    session_recorder: SessionRecorder | None = None,
    session_id: str | None = None,
    project: ProjectMetadata | None = None,
    trace_ctx: Context | None = None,
    request_meta: LLMRequestMeta | None = None,
) -> None:
    AceAIConfiguredTUI(
        agent_factory,
        initial_config=initial_config,
        initial_question=initial_question,
        default_model=default_model,
        initial_events=initial_events,
        initial_history=initial_history,
        session_recorder=session_recorder,
        session_id=session_id,
        project=project,
        trace_ctx=trace_ctx,
        request_meta=request_meta,
    ).run()


def run_agent_tui(
    agent: Agent,
    question: str,
    *,
    initial_events: list[TUIEvent] | None = None,
    initial_history: list[LLMMessage] | None = None,
    session_recorder: SessionRecorder | None = None,
    session_id: str | None = None,
    project: ProjectMetadata | None = None,
    trace_ctx: Context | None = None,
    request_meta: LLMRequestMeta | None = None,
) -> None:
    AceAILiveTUI(
        agent,
        question,
        initial_events=initial_events,
        initial_history=initial_history,
        session_recorder=session_recorder,
        session_id=session_id,
        project=project,
        trace_ctx=trace_ctx,
        request_meta=request_meta,
    ).run()


def _agent_metadata_sections(
    agent: Agent,
    *,
    provider_name: str,
    selected_model: str,
) -> list[MetadataSection]:
    skills = agent.skill_registry.get_skills()
    skill_lines = [
        f"{skill.name}: {skill.description} ({skill.skill_file})" for skill in skills
    ]
    executor = agent.executor
    tool_lines: list[str] = []
    if isinstance(executor, Executor):
        for tool in executor.tools.values():
            tags = ", ".join(tool.metadata.tags)
            tag_text = f" [{tags}]" if tags else ""
            tool_lines.append(f"{tool.name}{tag_text}: {tool.description}")
    hosted_lines = [
        f"{tool.provider_name}:{tool.native_name}" for tool in agent.hosted_tools
    ]
    return [
        MetadataSection(
            title="Agent",
            lines=[
                f"provider: {provider_name}",
                f"selected model: {selected_model}",
                f"default model: {agent.default_model}",
                f"max steps: {agent.max_steps}",
            ],
        ),
        MetadataSection(title=f"Skills ({len(skill_lines)})", lines=skill_lines),
        MetadataSection(title=f"Tools ({len(tool_lines)})", lines=tool_lines),
        MetadataSection(
            title=f"Hosted Tools ({len(hosted_lines)})",
            lines=hosted_lines,
        ),
    ]
