"""Live runner that bridges the AceAI app facade into the Textual app."""

import asyncio
import os
import sys
from pathlib import Path
from typing import AsyncGenerator, Callable, cast

from msgspec import Struct
from opentelemetry.context import Context
from rapidfuzz import fuzz
from textual.timer import Timer
from textual.widgets import Input
from textual.worker import Worker

from aceai.agent.app import (
    AceAgentApp,
    AgentAppEvent,
    effective_reasoning_level,
    is_model_supported,
    model_options_text_for,
    normalize_user_config,
    resolve_provider_api_key,
)
from aceai.agent.citations import (
    FileCitationOrigin,
    IdeaCitationOrigin,
    TurnCitation,
)
from aceai.agent.memory.ideas import Idea, IdeaStore
from aceai.agent.project import ProjectMetadata
from aceai.agent.features import default_agent_tools
from aceai.agent.session import MAIN_THREAD_ID, SessionEvent, SessionRecorder, SessionState
from aceai.agent.ace_agent import ACE_AGENT_BUILTIN_SKILL_PATHS
from aceai.agent.config import (
    AgentAppConfig,
    ReasoningLevel,
    effective_config_path,
    load_config_audit,
)
from aceai.agent.update_check import UpdateCheckResult, check_for_updates
from aceai.core import Agent
from aceai.core.events import AgentEvent, RunSuspendedEvent
from aceai.core.skills import SkillLoader, SkillLoadingError, SkillRegistry
from aceai.llm.models import LLMMessage
from aceai.llm.openai import OpenAIModel

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
from .widgets import ReferenceCompletionItem
from .widgets import ReferenceCompletionWidget
from .widgets import StatusBarWidget
from .widgets import TopBarWidget

AgentFactory = Callable[[AgentAppConfig], Agent]
CommandHandler = Callable[[str], None]
COMMAND_NAMES_ATTR = "_aceai_tui_command_names"
UPDATE_INSTRUCTIONS = "Run /update to upgrade AceAI and restart."
UPDATE_COMMAND: tuple[str, ...] = ("uv", "tool", "upgrade", "aceai")
REFERENCE_IGNORED_DIRS = {
    ".cache",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "wheels",
}
COMMAND_DESCRIPTIONS: dict[str, str] = {
    "clear": "Clear the visible transcript",
    "config": "Open provider, model, skill, and tool settings",
    "debug": "Toggle the debug detail view",
    "idea": "Show ideas, save an idea, or delete one",
    "quit": "Exit AceAI",
    "sessions": "Open the session picker",
    "stats": "Open runtime and usage stats",
    "steer": "Interrupt or redirect the current run",
    "subagents": "Show delegated subagent details",
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


class AceAIConfiguredTUI(AceAITUI):
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
        reasoning_level: ReasoningLevel = "auto",
        inline_viewport_height: int | None = None,
    ) -> None:
        self._provider_name = (
            initial_config.provider if initial_config is not None else "openai"
        )
        self._current_config = initial_config
        initial_model = (
            initial_config.model if initial_config is not None else default_model
        )
        requested_level: ReasoningLevel = (
            initial_config.reasoning_level
            if initial_config is not None
            else reasoning_level
        )
        self._reasoning_level: ReasoningLevel = effective_reasoning_level(
            self._provider_name,
            initial_model,
            requested_level,
        )
        super().__init__(
            events=initial_events or [],
            model=initial_model,
            reasoning_level=self._reasoning_level,
            session_recorder=session_recorder,
            session_id=session_id,
            project=project,
            idea_store=idea_store,
            record_events=False,
            inline_viewport_height=inline_viewport_height,
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
        self._reference_completion_selected_index = 0
        self._reference_file_items_cache_root: Path | None = None
        self._reference_file_items_cache: tuple[ReferenceCompletionItem, ...] = ()
        self._update_check_completed = False
        self._update_check_lock = asyncio.Lock()

    def _start_update_check(self) -> None:
        self.run_worker(
            self._check_for_updates(),
            name="aceai-update-check",
            description="Check whether a newer AceAI release is available",
            exit_on_error=False,
        )

    async def _check_for_updates(self) -> None:
        async with self._update_check_lock:
            if self._update_check_completed:
                return
            self._update_check_completed = True
        result = await check_for_updates()
        if result is None or not result.has_update:
            return
        self.query_one(StatusBarWidget).show_notice(
            _update_available_notice(result),
            timeout=6.0,
            style="bold #ebcb8b",
        )

    async def _stream_agent_turn(
        self,
        question: str,
        citations: tuple[TurnCitation, ...] = (),
    ) -> None:
        if self._agent_app is None:
            raise RuntimeError("AceAI app is not configured")
        await self._consume_agent_app_stream(
            self._agent_app.start_turn_events(question, citations=citations)
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
        provider_name = (
            self._agent_app.provider_name if self._agent_app is not None else None
        )
        try:
            async for event in stream:
                self.append_agent_event(event, provider_name=provider_name)
                if isinstance(event, RunSuspendedEvent):
                    self.show_pending_approval()
        finally:
            await stream.aclose()
            self._sync_app_state()

    async def _consume_agent_app_stream(
        self,
        stream: AsyncGenerator[AgentAppEvent, None],
    ) -> None:
        provider_name = (
            self._agent_app.provider_name if self._agent_app is not None else None
        )
        try:
            async for app_event in stream:
                event = app_event.event
                if isinstance(event, SessionEvent):
                    tui_event = TUIEvent.from_session_event(event)
                    if tui_event is not None:
                        self.append_persisted_event(tui_event)
                    continue
                if app_event.thread_id != self._active_thread_id:
                    continue
                self.append_agent_event(event, provider_name=provider_name)
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
        self._active_thread_id = self._agent_app.active_thread_id
        self._refresh_queued_turns()
        if self._session_id is not None:
            self.title = self._window_title()
            if self.is_mounted:
                self.query_one(TopBarWidget).set_title(self.title)

    def on_unmount(self) -> None:
        agent_app = self._agent_app
        if agent_app is not None:
            agent_app.session_service.finalize()
        elif self._session_recorder is not None:
            self._session_recorder.finalize()
        super().on_unmount()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if not isinstance(event.input, CommandInput):
            return
        self._handle_command_input_submitted(event.input, event.value)

    def on_command_input_submitted(self, event: CommandInput.Submitted) -> None:
        self._handle_command_input_submitted(event.input, event.value)

    def on_text_area_changed(self, event) -> None:
        super().on_text_area_changed(event)
        if not isinstance(event.text_area, CommandInput):
            return
        self._reference_completion_selected_index = 0
        self._refresh_reference_completions(event.text_area.value)

    def on_command_input_reference_completion_requested(
        self,
        event: CommandInput.ReferenceCompletionRequested,
    ) -> None:
        matches = self._matching_reference_items(event.input.value)
        if not matches:
            return
        index = min(self._reference_completion_selected_index, len(matches) - 1)
        event.input.value = _replace_active_reference(
            event.input.value,
            matches[index].value,
        )
        self._reference_completion_selected_index = 0
        widget = self._reference_completion_widget()
        if widget is not None:
            widget.hide()
        event.stop()

    def on_command_input_reference_completion_navigation_requested(
        self,
        event: CommandInput.ReferenceCompletionNavigationRequested,
    ) -> None:
        command_input = self.query_one(CommandInput)
        matches = self._matching_reference_items(command_input.value)
        if not matches:
            return
        self._reference_completion_selected_index = (
            self._reference_completion_selected_index + event.direction
        ) % len(matches)
        self._refresh_reference_completions(command_input.value)
        event.stop()

    def _handle_command_input_submitted(
        self,
        command_input: CommandInput,
        value: str,
    ) -> None:
        question = value
        if question == "":
            return
        if self._dispatch_approval_input(question):
            self.exit_command_input(command_input)
            return
        if self._dispatch_command(question):
            self.exit_command_input(command_input)
            return
        try:
            inline_citations = self._inline_citations(question)
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            self.notify_session(str(exc))
            return
        citations = (*self._consume_pending_citations(), *inline_citations)
        self.start_run(question, citations=citations)
        self.exit_command_input(command_input)

    def _dispatch_approval_input(self, text: str) -> bool:
        agent_app = self._agent_app
        if agent_app is None or not agent_app.is_running_suspended:
            return False
        if text in ("a", "approve"):
            self.approve_pending_tool()
            return True
        if text in ("r", "reject"):
            self.reject_pending_tool("rejected by caller")
            return True
        return False

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

    def _refresh_reference_completions(self, value: str) -> None:
        widget = self._reference_completion_widget()
        if widget is None:
            return
        matches = self._matching_reference_items(value)
        if not matches:
            widget.hide()
            return
        if self._reference_completion_selected_index >= len(matches):
            self._reference_completion_selected_index = 0
        widget.show_references(
            list(matches),
            selected_index=self._reference_completion_selected_index,
        )

    def _matching_reference_items(
        self,
        value: str,
    ) -> tuple[ReferenceCompletionItem, ...]:
        prefix = _active_reference_prefix(value)
        if prefix is None:
            return ()
        if prefix == "":
            return ()
        search_items = self._idea_reference_items()
        if not prefix.startswith("idea:"):
            search_items = (*self._file_reference_items(), *search_items)
        ranked_items = _rank_reference_items(prefix, search_items)
        return _filter_ranked_reference_items(ranked_items)

    def _idea_reference_items(self) -> tuple[ReferenceCompletionItem, ...]:
        return tuple(
            ReferenceCompletionItem(
                value=f"@idea:{index}",
                description=_reference_idea_description(idea),
            )
            for index, idea in enumerate(self._list_ideas(), start=1)
        )

    def _file_reference_items(self) -> tuple[ReferenceCompletionItem, ...]:
        root = Path(self._project.root_path)
        if not root.is_dir():
            return ()
        if self._reference_file_items_cache_root == root:
            return self._reference_file_items_cache
        items: list[ReferenceCompletionItem] = []
        for path in _iter_reference_file_candidates(root):
            items.append(
                ReferenceCompletionItem(
                    value="@" + path.relative_to(root).as_posix(),
                    description="file",
                )
            )
        self._reference_file_items_cache_root = root
        self._reference_file_items_cache = tuple(items)
        return self._reference_file_items_cache

    def _reference_completion_widget(self) -> ReferenceCompletionWidget | None:
        matches = list(self.query(ReferenceCompletionWidget))
        if not matches:
            return None
        return matches[0]

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
        self.open_stats_screen()

    @tui_command("config")
    def _command_config(self, arg: str) -> None:
        self.open_config_screen()

    @tui_command("debug")
    def _command_debug(self, arg: str) -> None:
        self.action_toggle_debug_mode()

    @tui_command("subagents")
    def _command_subagents(self, arg: str) -> None:
        if arg != "":
            if arg == "main":
                self.switch_thread(MAIN_THREAD_ID)
                return
            self.switch_thread(arg)
            return
        self.action_show_subagents()

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

    def enqueue_run(
        self,
        question: str,
        *,
        citations: tuple[TurnCitation, ...] = (),
    ) -> None:
        self._clear_cancel_arm()
        agent_app = self._agent_app
        if agent_app is None:
            self.append_event(
                TUIEvent.session_notice("Configure AceAI before enqueueing a run.")
            )
            return
        if not agent_app.active_thread_accepts_user_turn:
            if agent_app.steer_active_child_thread(question):
                self.append_event(TUIEvent.user_message(question))
                self.append_event(TUIEvent.session_notice("Steered running subagent."))
            return
        if (
            self._active_worker is None or not self._active_worker.is_running
        ) and not agent_app.is_running_suspended:
            self._start_run_now(question, citations=citations)
            return
        agent_app.enqueue_turn(question)
        if citations:
            self._queued_turn_citations()[question].append(citations)
        self._refresh_queued_turns()

    def steer_run(self, question: str) -> None:
        self._clear_cancel_arm()
        agent_app = self._agent_app
        if agent_app is None:
            self.append_event(
                TUIEvent.session_notice("Configure AceAI before steering a run.")
            )
            return
        if not agent_app.active_thread_accepts_user_turn:
            if agent_app.steer_active_child_thread(question):
                self.append_event(TUIEvent.user_message(question))
                self.append_event(TUIEvent.session_notice("Steered running subagent."))
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
        citations = self._pop_queued_turn_citations(question)
        self._refresh_queued_turns()
        if self._active_worker is not None and self._active_worker.is_running:
            self._active_worker.cancel()
        self._start_run_now(question, citations=citations)

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
        citations = self._pop_queued_turn_citations(question)
        self._refresh_queued_turns()
        self.call_after_refresh(
            lambda: self._start_run_now(question, citations=citations)
        )

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

    def on_queued_turns_widget_cancelled(
        self,
        event: QueuedTurnsWidget.Cancelled,
    ) -> None:
        event.stop()
        self.cancel_queued_run(event.index)

    def cancel_queued_run(self, index: int) -> None:
        self._clear_cancel_arm()
        agent_app = self._agent_app
        if agent_app is None:
            self.append_event(
                TUIEvent.session_notice("Configure AceAI before cancelling a queued run.")
            )
            return
        try:
            question = agent_app.cancel_queued_turn(index)
        except IndexError:
            self._refresh_queued_turns()
            return
        self._pop_queued_turn_citations(question)
        self._refresh_queued_turns()
        self.append_event(
            TUIEvent.session_notice(
                f"Cancelled queued message {index + 1}: {question}"
            )
        )

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
                quote=idea.content,
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

    def _queued_turn_citations(self) -> dict[str, list[tuple[TurnCitation, ...]]]:
        if not hasattr(self, "_queued_turn_citation_map"):
            self._queued_turn_citation_map = {}
        return self._queued_turn_citation_map

    def _pop_queued_turn_citations(self, question: str) -> tuple[TurnCitation, ...]:
        queued = self._queued_turn_citations()
        values = queued.get(question)
        if not values:
            return ()
        citations = values.pop(0)
        if not values:
            queued.pop(question, None)
        return citations

    def _inline_citations(self, question: str) -> tuple[TurnCitation, ...]:
        citations: list[TurnCitation] = []
        for reference in _inline_references(question):
            if reference.kind == "file":
                citations.append(self._file_citation(reference.value))
            elif reference.kind == "idea":
                citations.append(self._idea_citation(reference.value))
        return tuple(citations)

    def _file_citation(self, path_text: str) -> TurnCitation:
        path = Path(path_text).expanduser()
        if not path.is_absolute():
            path = Path(self._project.root_path) / path
        path = path.resolve()
        if not path.is_file():
            raise ValueError(f"File not found: {path}")
        return TurnCitation(
            quote=str(path),
            origin=FileCitationOrigin(kind="file", path=str(path)),
        )

    def _idea_citation(self, value: str) -> TurnCitation:
        try:
            index = int(value)
        except ValueError:
            idea = self._idea_by_id(value)
        else:
            idea = self._idea_by_display_index(index)
        return TurnCitation(
            quote=idea.content,
            origin=IdeaCitationOrigin(kind="idea", idea_id=idea.idea_id),
        )

    def _idea_by_display_index(self, index: int) -> Idea:
        if index < 1:
            raise ValueError("Idea reference index must be one-based")
        ideas = self._list_ideas()
        if index > len(ideas):
            raise ValueError(f"Idea not found: {index}")
        return ideas[index - 1]

    def _idea_by_id(self, idea_id: str) -> Idea:
        for idea in self._list_ideas():
            if idea.idea_id == idea_id:
                return idea
        raise ValueError(f"Idea not found: {idea_id}")

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


    def on_mount(self) -> None:
        super().on_mount()
        self._start_update_check()
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
        next_config = normalize_user_config(config)
        model_changed = next_config.model != self._selected_model
        self._provider_name = next_config.provider
        self._current_config = next_config
        self._selected_model = next_config.model
        self._reasoning_level = next_config.reasoning_level
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
            self.enqueue_run(question, citations=citations)
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
            reasoning_level=self._reasoning_level,
            initial_history=self._llm_history,
            session_store=self._session_store(),
            session_recorder=self._session_recorder,
            session_id=self._session_id,
            project=self._project,
            idea_store=self._idea_store,
            trace_ctx=self._trace_ctx,
        )
        new_level = self._agent_app.reasoning_level
        if new_level != self._reasoning_level:
            self._reasoning_level = new_level
            self.set_status_model(
                self._selected_model,
                reasoning_level=self._reasoning_level,
            )
        self._sync_app_state()

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

    def switch_thread(self, thread_id: str) -> None:
        if self._current_config is None:
            AceAITUI.switch_thread(self, thread_id)
            return
        self._ensure_agent_app()
        agent_app = self._agent_app
        if agent_app is None:
            self.append_event(
                TUIEvent.session_notice("Configure AceAI before switching threads.")
            )
            return
        try:
            snapshot = agent_app.switch_thread(thread_id)
        except (KeyError, RuntimeError) as exc:
            self.append_event(TUIEvent.session_notice(str(exc)))
            return
        self._sync_app_state()
        self.load_events(event_log_to_tui_events(snapshot.event_log))
        self.notify_session(f"Switched thread {thread_id}")

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
                f"Current model: {self._selected_model}\n{model_options_text_for(self._provider_name)}"
            )
        )

    def action_config(self) -> None:
        self.open_config_screen()

    def open_config_screen(self) -> None:
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
                audit_entries=load_config_audit(
                    limit=50,
                    target=effective_config_path(),
                ),
                api_keys=api_keys,
                compress_threshold=self._current_config.compress_threshold
                if self._current_config is not None
                else "100%",
                reasoning_level=self._reasoning_level,
                disabled_providers=tuple(self._current_config.disabled_providers)
                if self._current_config is not None
                else (),
            ),
            self._handle_config_selection,
        )

    def _handle_config_selection(self, selection: ConfigSelection | None) -> None:
        if selection is None:
            return
        if type(selection) is str:
            self.switch_model(selection)
            return
        same_provider = selection.provider == self._provider_name
        if same_provider and selection.api_key == "" and self._current_config is None:
            self.switch_model(selection.model)
            return
        api_key = self._resolve_selection_api_key(selection, same_provider=same_provider)
        if api_key == "":
            self.append_event(
                TUIEvent.session_notice(
                    f"API key required for provider {selection.provider}."
                )
            )
            return
        api_keys = (
            dict(self._current_config.api_keys)
            if self._current_config is not None
            else {}
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
                disabled_providers=list(selection.disabled_providers),
            )
        )
        if same_provider:
            self.notify_session(
                f"Updated provider credentials and switched model to {selection.model}"
            )
        else:
            self.notify_session(
                f"Switched provider to {selection.provider} and model to {selection.model}"
            )

    def on_config_screen_persist_requested(
        self,
        event: ConfigScreen.PersistRequested,
    ) -> None:
        event.stop()
        self._handle_config_selection(event.selection)

    def _resolve_selection_api_key(
        self,
        selection: ConfigSelection,
        *,
        same_provider: bool,
    ) -> str:
        if selection.api_key != "":
            return selection.api_key
        if same_provider and self._current_config is not None:
            return self._current_config.api_key
        return resolve_provider_api_key(selection.provider)

    def apply_user_config(self, config: AgentAppConfig) -> None:
        normalized = normalize_user_config(config, persist=True)
        self.apply_config(normalized)

    def switch_model(self, model: str) -> None:
        if self._active_worker is not None and self._active_worker.is_running:
            self.append_event(
                TUIEvent.session_notice(
                    "Model changes apply after the current run finishes."
                )
            )
            return
        if not is_model_supported(self._provider_name, model):
            self.append_event(
                TUIEvent.session_notice(model_options_text_for(self._provider_name))
            )
            return
        model_changed = model != self._selected_model
        requested_level = self._reasoning_level
        if self._current_config is not None:
            requested_level = self._current_config.reasoning_level
        next_level = effective_reasoning_level(
            self._provider_name,
            model,
            requested_level,
        )
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
                reasoning_level=next_level,
                disabled_providers=self._current_config.disabled_providers,
            )
            self._current_config = normalize_user_config(next_config, persist=True)
        self._selected_model = cast(OpenAIModel, model)
        self._reasoning_level = next_level
        if self._agent_app is not None:
            self._agent_app.switch_model(
                self._selected_model,
                reasoning_level=self._reasoning_level,
            )
            self._reasoning_level = self._agent_app.reasoning_level
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
                    tags=tuple(configured_tool.metadata.tags),
                )
            )
        return tuple(items)

    def _reload_llm_history(self) -> None:
        if self._session_recorder is None or self._session_id is None:
            self._llm_history = []
            return
        if self._agent_app is None:
            event_log = self._session_recorder.store.load_thread_event_log(
                self._session_id,
                self._active_thread_id,
            )
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
        if self._agent_app is None:
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
            *_agent_metadata_sections(self._agent_app),
        ]


def _update_available_notice(result: UpdateCheckResult) -> str:
    return (
        f"AceAI {result.latest_version} is available "
        f"(current {result.current_version}). "
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


class InlineReference(Struct, frozen=True, kw_only=True):
    kind: str
    value: str


def _inline_references(text: str) -> tuple[InlineReference, ...]:
    references: list[InlineReference] = []
    seen: set[tuple[str, str]] = set()
    for token in text.split():
        reference = _inline_reference(token)
        if reference is None:
            continue
        key = (reference.kind, reference.value)
        if key in seen:
            continue
        seen.add(key)
        references.append(reference)
    return tuple(references)


def _inline_reference(token: str) -> InlineReference | None:
    token = token.strip()
    if not token.startswith("@") or len(token) == 1:
        return None
    value = token[1:].strip(".,;:!?)]}\"'")
    if value == "" or "://" in value:
        return None
    if value.startswith("idea:"):
        idea = value.removeprefix("idea:")
        if idea == "":
            return None
        return InlineReference(kind="idea", value=idea)
    return InlineReference(kind="file", value=value)


def _active_reference_prefix(value: str) -> str | None:
    if value.endswith((" ", "\n", "\t")):
        return None
    tail = value.rsplit(maxsplit=1)[-1] if value.split() else value
    if tail.startswith("@"):
        return tail[1:]
    return None


def _replace_active_reference(value: str, replacement: str) -> str:
    if not value.split():
        return replacement + " "
    head, separator, tail = value.rpartition(" ")
    if not tail.startswith("@"):
        return value
    if separator == "":
        return replacement + " "
    return f"{head} {replacement} "


def _reference_idea_description(idea: Idea) -> str:
    first_line = idea.content.splitlines()[0] if idea.content.splitlines() else "idea"
    if len(first_line) > 64:
        first_line = first_line[:61] + "..."
    return first_line


def _rank_reference_items(
    prefix: str,
    items: tuple[ReferenceCompletionItem, ...],
) -> tuple[tuple[ReferenceCompletionItem, float, int], ...]:
    ranked_items = []
    query = prefix.casefold()
    for index, item in enumerate(items):
        candidate = item.value.removeprefix("@").casefold()
        score = fuzz.WRatio(query, candidate)
        ranked_items.append((item, score, index))
    return tuple(
        sorted(
            ranked_items,
            key=lambda match: (-match[1], match[2]),
        )
    )


def _filter_ranked_reference_items(
    ranked_items: tuple[tuple[ReferenceCompletionItem, float, int], ...],
    *,
    score_cutoff: float = 50,
    limit: int = 12,
) -> tuple[ReferenceCompletionItem, ...]:
    return tuple(
        item
        for item, score, _index in ranked_items
        if score >= score_cutoff
    )[:limit]


def _iter_reference_file_candidates(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            dirname for dirname in dirnames if dirname not in REFERENCE_IGNORED_DIRS
        )
        for filename in sorted(filenames):
            if filename.startswith("."):
                continue
            path = Path(dirpath) / filename
            if not path.is_file():
                continue
            yield path


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


def _agent_metadata_sections(agent_app: AceAgentApp) -> list[MetadataSection]:
    info = agent_app.runtime_info()
    skill_lines = agent_app.skill_summary_lines()
    tool_lines = agent_app.tool_summary_lines()
    hosted_lines = agent_app.hosted_tool_summary_lines()
    return [
        MetadataSection(
            title="Agent",
            lines=[
                f"provider: {info.provider_name}",
                f"selected model: {info.selected_model}",
                f"default model: {info.default_model}",
                f"max steps: {info.max_steps}",
            ],
        ),
        MetadataSection(title=f"Skills ({len(skill_lines)})", lines=skill_lines),
        MetadataSection(title=f"Tools ({len(tool_lines)})", lines=tool_lines),
        MetadataSection(
            title=f"Hosted Tools ({len(hosted_lines)})",
            lines=hosted_lines,
        ),
    ]
