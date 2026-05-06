"""Live runner that bridges the AceAI app facade into the Textual app."""

import asyncio
import os
import sys
from typing import AsyncGenerator, Callable, cast

from msgspec import Struct
from opentelemetry.context import Context
from textual.widgets import Input
from textual.worker import Worker

from aceai.agent.app import AceAgentApp, UpdateCheckResult
from aceai.agent.ideas import Idea, IdeaStore
from aceai.agent.features import default_agent_tools
from aceai.agent.provider_catalog import api_key_env, model_options, supported_models
from aceai.agent.session import SessionRecorder, SessionState
from aceai.agent.config import AceAITUIConfig, replace_config, save_config
from aceai.core import AgentBase
from aceai.core.events import AgentEvent, RunSuspendedEvent
from aceai.core.executor import ToolExecutor
from aceai.core.skills import SkillLoader, SkillRegistry
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
    ProviderSetupScreen,
    SkillConfigItem,
    ToolPermissionItem,
)
from .widgets import ApprovalWidget
from .widgets import CommandInput

AgentFactory = Callable[[AceAITUIConfig], AgentBase]
CommandHandler = Callable[[str], None]
COMMAND_NAMES_ATTR = "_aceai_tui_command_names"
UPDATE_INSTRUCTIONS = (
    "Run /update to upgrade AceAI and restart."
)
UPDATE_COMMAND: tuple[str, ...] = ("uv", "tool", "upgrade", "aceai")


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


def _skill_config_items(registry: SkillRegistry) -> tuple[SkillConfigItem, ...]:
    return tuple(
        SkillConfigItem(
            name=skill.name,
            description=skill.description,
            location=str(skill.skill_file),
        )
        for skill in registry.get_skills()
    )


def _system_prompt_text(agent: AgentBase) -> str:
    parts: list[str] = []
    for part in agent.system_message.content:
        if part["type"] == "text":
            parts.append(part["data"])
    return "".join(parts)


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

    async def _stream_agent_turn(self, question: str) -> None:
        await self._consume_agent_stream(self._agent_app.start_turn(question))

    async def _stream_approval_decision(self, *, approved: bool, reason: str = "") -> None:
        if approved:
            stream = self._agent_app.approve_tool()
        else:
            stream = self._agent_app.reject_tool(reason)
        await self._consume_agent_stream(stream)

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
        self._active_runtime = self._agent_app.active_runtime
        if self._session_id is not None:
            self.title = f"AceAI {self._session_id}"

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

    @tui_command("quit")
    def _command_quit(self, arg: str) -> None:
        self.exit()

    @tui_command("clear")
    def _command_clear(self, arg: str) -> None:
        self.load_events([])

    @tui_command("sessions")
    def _command_sessions(self, arg: str) -> None:
        self.open_session_selector()

    @tui_command("metadata", "info")
    def _command_metadata(self, arg: str) -> None:
        self.open_metadata_screen()

    @tui_command("config")
    def _command_config(self, arg: str) -> None:
        self.open_config_screen()

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

    @tui_command("resume")
    def _command_resume(self, arg: str) -> None:
        if arg == "":
            self.append_event(TUIEvent.session_notice("Usage: /resume <session_id>"))
            return
        self.switch_session(arg)

    def _capture_idea(self, content: str) -> Idea:
        agent_app = self._agent_app
        if agent_app is None:
            return self._idea_store.capture(
                content,
                source_session_id=self._session_id,
            )
        return agent_app.capture_idea(content)

    def _list_ideas(self) -> list[Idea]:
        agent_app = self._agent_app
        if agent_app is None:
            return self._idea_store.list_recent()
        return agent_app.list_ideas()

    def _show_ideas(self) -> None:
        self.append_event(TUIEvent.idea_list(_idea_items(self._list_ideas())))
        self.notify_session("Delete an idea with /idea delete <number>.")

    def _delete_idea(self, index: int) -> None:
        try:
            agent_app = self._agent_app
            if agent_app is None:
                idea = self._idea_store.delete_recent(index)
            else:
                idea = agent_app.delete_idea(index)
        except IndexError:
            self.notify_session(f"No idea found at {index}.")
            return
        self.append_event(TUIEvent.idea_list(_idea_items(self._list_ideas())))
        self.notify_session(f"Deleted idea {index}: {idea.content}")

    @tui_command("model")
    def _command_model(self, arg: str) -> None:
        if arg == "":
            self.open_config_screen()
            return
        self.switch_model(arg)

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
    """Textual app that runs an AgentBase from submitted questions."""

    def __init__(
        self,
        agent: AgentBase,
        *,
        initial_events: list[TUIEvent] | None = None,
        initial_history: list[LLMMessage] | None = None,
        session_recorder: SessionRecorder | None = None,
        session_id: str | None = None,
        idea_store: IdeaStore | None = None,
        trace_ctx: Context | None = None,
        request_meta: LLMRequestMeta | None = None,
    ) -> None:
        self._request_meta: LLMRequestMeta = dict(request_meta or {})
        self._provider_name = "openai"
        self._selected_model = _model_from_request_meta(
            self._request_meta,
            agent.default_model,
            self._provider_name,
        )
        self._request_meta["model"] = self._selected_model
        super().__init__(
            events=initial_events or [],
            model=self._selected_model,
            session_recorder=session_recorder,
            session_id=session_id,
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
            idea_store=self._idea_store,
            trace_ctx=trace_ctx,
            request_meta=self._request_meta,
        )
        self._persist_session_state()
        self._llm_history = self._agent_app.llm_history
        self._active_worker: Worker[None] | None = None
        self._active_runtime = self._agent_app.active_runtime

    def start_run(self, question: str) -> None:
        if self._active_worker is not None and self._active_worker.is_running:
            self._active_worker.cancel()
        if self._agent_app.is_running_suspended:
            self.append_event(
                TUIEvent.session_notice("Choose Approve or Reject before starting another run.")
            )
            return
        self._agent_app.ensure_session()
        self._sync_app_state()
        self._persist_session_state()
        self.append_event(TUIEvent.user_message(question))
        self._active_worker = self.run_worker(
            self._stream_agent_turn(question),
            name="aceai-agent",
            description="Run AceAI agent and stream events into the TUI",
            exit_on_error=True,
        )

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

    def open_config_screen(self) -> None:
        self.push_screen(
            ConfigScreen(
                provider_name=self._provider_name,
                current_model=self._selected_model,
                default_model=cast(OpenAIModel, self._agent.default_model),
                skills="auto",
                skill_items=_skill_config_items(self._agent.skill_registry),
                skill_selection_mode="all",
                enabled_skills=(),
                api_keys={},
                system_prompt=_system_prompt_text(self._agent),
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
                TUIEvent.session_notice("Model changes apply after the current run finishes.")
            )
            return
        if model not in supported_models(self._provider_name):
            self.append_event(
                TUIEvent.session_notice(_model_options_text(self._provider_name))
            )
            return
        self._selected_model = cast(OpenAIModel, model)
        self._request_meta["model"] = self._selected_model
        self._agent_app.switch_model(self._selected_model)
        self._sync_app_state()
        self.set_status_model(self._selected_model)
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
        if state.selected_provider != "" and state.selected_provider != self._provider_name:
            return
        if state.selected_model not in supported_models(self._provider_name):
            return
        self._selected_model = cast(OpenAIModel, state.selected_model)
        self._request_meta["model"] = self._selected_model
        self._agent_app.switch_model(self._selected_model)
        self._sync_app_state()
        self.set_status_model(self._selected_model)

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
        initial_config: AceAITUIConfig | None,
        initial_question: str,
        default_model: OpenAIModel,
        initial_events: list[TUIEvent] | None = None,
        initial_history: list[LLMMessage] | None = None,
        session_recorder: SessionRecorder | None = None,
        session_id: str | None = None,
        idea_store: IdeaStore | None = None,
        trace_ctx: Context | None = None,
        request_meta: LLMRequestMeta | None = None,
    ) -> None:
        self._request_meta: LLMRequestMeta = dict(request_meta or {})
        self._provider_name = initial_config.provider if initial_config is not None else "openai"
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
        self._request_meta["model"] = initial_model
        super().__init__(
            events=initial_events or [],
            model=initial_model,
            session_recorder=session_recorder,
            session_id=session_id,
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
        self._agent: AgentBase | None = None
        self._agent_app: AceAgentApp | None = None
        self._active_worker: Worker[None] | None = None
        self._active_runtime = None

    def on_mount(self) -> None:
        super().on_mount()
        if self._initial_config is not None:
            self.apply_config(self._initial_config)
            return
        self.push_screen(
            ProviderSetupScreen(default_model=self._default_model),
            self._handle_setup_config,
        )

    def _handle_setup_config(self, config: AceAITUIConfig | None) -> None:
        if config is None:
            return
        self.apply_config(config)

    def apply_config(self, config: AceAITUIConfig) -> None:
        next_config = replace_config(config)
        self._provider_name = next_config.provider
        self._current_config = next_config
        self._agent = self._agent_factory(next_config)
        self._selected_model = next_config.model
        self._request_meta["model"] = self._selected_model
        self._agent_app = AceAgentApp(
            self._agent,
            provider_name=self._provider_name,
            selected_model=self._selected_model,
            initial_history=self._llm_history,
            session_store=self._session_store(),
            session_recorder=self._session_recorder,
            session_id=self._session_id,
            idea_store=self._idea_store,
            trace_ctx=self._trace_ctx,
            request_meta=self._request_meta,
        )
        self._persist_session_state()
        self._sync_app_state()
        self.set_status_model(self._selected_model)
        self._start_update_check()
        if self._initial_question != "":
            self.start_run(self._initial_question)

    def start_run(self, question: str) -> None:
        if self._agent is None:
            self.query_one(CommandInput).value = question
            return
        if self._active_worker is not None and self._active_worker.is_running:
            self._active_worker.cancel()
        if self._agent_app is None:
            raise RuntimeError("AceAI app is not configured")
        if self._agent_app.is_running_suspended:
            self.append_event(
                TUIEvent.session_notice("Choose Approve or Reject before starting another run.")
            )
            return
        self._agent_app.ensure_session()
        self._sync_app_state()
        self._persist_session_state()
        self.append_event(TUIEvent.user_message(question))
        self._active_worker = self.run_worker(
            self._stream_agent_turn(question),
            name="aceai-agent",
            description="Run AceAI agent and stream events into the TUI",
            exit_on_error=True,
        )

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
                skills=self._current_config.skills if self._current_config is not None else "",
                skill_items=self._available_skill_items(),
                skill_selection_mode=self._current_config.skill_selection_mode
                if self._current_config is not None
                else "all",
                enabled_skills=tuple(self._current_config.enabled_skills)
                if self._current_config is not None
                else (),
                tool_permission_items=self._available_tool_permission_items(),
                api_keys=api_keys,
                system_prompt=_system_prompt_text(self._agent),
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
                    AceAITUIConfig(
                        provider=selection.provider,
                        api_key=api_key,
                        model=selection.model,
                        default_model=selection.default_model,
                        skills=selection.skills,
                        skill_selection_mode=selection.skill_selection_mode,
                        enabled_skills=list(selection.enabled_skills),
                        api_keys=api_keys,
                        tool_permissions=selection.tool_permissions,
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
            AceAITUIConfig(
                provider=selection.provider,
                api_key=api_key,
                model=selection.model,
                default_model=selection.default_model,
                skills=selection.skills,
                skill_selection_mode=selection.skill_selection_mode,
                enabled_skills=list(selection.enabled_skills),
                api_keys=api_keys,
                tool_permissions=selection.tool_permissions,
            )
        )
        self.notify_session(
            f"Switched provider to {selection.provider} and model to {selection.model}"
        )

    def apply_user_config(self, config: AceAITUIConfig) -> None:
        self.apply_config(config)
        save_config(config)

    def switch_model(self, model: str) -> None:
        if self._active_worker is not None and self._active_worker.is_running:
            self.append_event(
                TUIEvent.session_notice("Model changes apply after the current run finishes.")
            )
            return
        if model not in supported_models(self._provider_name):
            self.append_event(
                TUIEvent.session_notice(_model_options_text(self._provider_name))
            )
            return
        if self._current_config is not None:
            self._current_config = replace_config(
                AceAITUIConfig(
                    provider=self._current_config.provider,
                    api_key=self._current_config.api_key,
                    model=cast(OpenAIModel, model),
                    default_model=self._current_config.default_model,
                    skills=self._current_config.skills,
                    skill_selection_mode=self._current_config.skill_selection_mode,
                    enabled_skills=self._current_config.enabled_skills,
                    api_keys=self._current_config.api_keys,
                    tool_permissions=self._current_config.tool_permissions,
                )
            )
        self._selected_model = cast(OpenAIModel, model)
        self._request_meta["model"] = self._selected_model
        if self._agent_app is not None:
            self._agent_app.switch_model(self._selected_model)
            self._sync_app_state()
        self._persist_session_state()
        self.set_status_model(self._selected_model)
        self.notify_session(f"Switched model to {self._selected_model}")

    def _available_skill_items(self) -> tuple[SkillConfigItem, ...]:
        if self._current_config is None:
            return _skill_config_items(self._agent.skill_registry)
        registry = SkillLoader.load_registry(self._current_config.skills)
        return _skill_config_items(registry)

    def _available_tool_permission_items(self) -> tuple[ToolPermissionItem, ...]:
        configured_permissions = (
            self._current_config.tool_permissions
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
            self._active_runtime = None
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
        agent: AgentBase,
        question: str,
        *,
        initial_events: list[TUIEvent] | None = None,
        initial_history: list[LLMMessage] | None = None,
        session_recorder: SessionRecorder | None = None,
        session_id: str | None = None,
        trace_ctx: Context | None = None,
        request_meta: LLMRequestMeta | None = None,
    ) -> None:
        super().__init__(
            agent,
            initial_events=initial_events,
            initial_history=initial_history,
            session_recorder=session_recorder,
            session_id=session_id,
            trace_ctx=trace_ctx,
            request_meta=request_meta,
        )
        self._initial_question = question

    def on_mount(self) -> None:
        super().on_mount()
        self.start_run(self._initial_question)


def run_interactive_tui(
    agent: AgentBase,
    *,
    initial_events: list[TUIEvent] | None = None,
    initial_history: list[LLMMessage] | None = None,
    session_recorder: SessionRecorder | None = None,
    session_id: str | None = None,
    trace_ctx: Context | None = None,
    request_meta: LLMRequestMeta | None = None,
) -> None:
    AceAIInteractiveTUI(
        agent,
        initial_events=initial_events,
        initial_history=initial_history,
        session_recorder=session_recorder,
        session_id=session_id,
        trace_ctx=trace_ctx,
        request_meta=request_meta,
    ).run()


def run_configured_tui(
    agent_factory: AgentFactory,
    *,
    initial_config: AceAITUIConfig | None,
    initial_question: str,
    default_model: OpenAIModel,
    initial_events: list[TUIEvent] | None = None,
    initial_history: list[LLMMessage] | None = None,
    session_recorder: SessionRecorder | None = None,
    session_id: str | None = None,
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
        trace_ctx=trace_ctx,
        request_meta=request_meta,
    ).run()


def run_agent_tui(
    agent: AgentBase,
    question: str,
    *,
    initial_events: list[TUIEvent] | None = None,
    initial_history: list[LLMMessage] | None = None,
    session_recorder: SessionRecorder | None = None,
    session_id: str | None = None,
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
        trace_ctx=trace_ctx,
        request_meta=request_meta,
    ).run()


def _agent_metadata_sections(
    agent: AgentBase,
    *,
    provider_name: str,
    selected_model: str,
) -> list[MetadataSection]:
    skills = agent.skill_registry.get_skills()
    skill_lines = [
        f"{skill.name}: {skill.description} ({skill.skill_file})"
        for skill in skills
    ]
    executor = agent.executor
    tool_lines: list[str] = []
    if isinstance(executor, ToolExecutor):
        for tool in executor.tools.values():
            tags = ", ".join(tool.metadata.tags)
            tag_text = f" [{tags}]" if tags else ""
            tool_lines.append(f"{tool.name}{tag_text}: {tool.description}")
    hosted_lines = [
        f"{tool.provider_name}:{tool.native_name}"
        for tool in agent.hosted_tools
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
