"""Console entry point for the AceAI TUI."""

import argparse
import importlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, Sequence

from aceai.agent.ace_agent import ACE_AGENT_SKILL_PATH, build_ace_agent
from aceai.agent.permissions import ToolPermission
from aceai.agent.provider_catalog import (
    all_supported_models,
    api_key_env,
    default_model,
    supported_models,
    supported_provider_names,
)
from aceai.core import AgentBase
from aceai.llm.interface import UNSET
from aceai.llm.models import LLMMessage
from aceai.llm.openai import OpenAIModel

from aceai.agent.config import (
    AceAITUIConfig,
    load_config,
    replace_config,
)
from aceai.agent.cost import format_usd

CLI_MODELS: tuple[OpenAIModel, ...] = all_supported_models()
TUI_EXTRA_MODULES = frozenset(("rich", "sqlalchemy", "textual"))
TUI_EXTRA_INSTALL_HINT = (
    "AceAI TUI dependencies are not installed.\n"
    "Install them with one of:\n"
    "  uv add 'aceai[tui]'\n"
    "  pip install 'aceai[tui]'"
)

SessionStore = None
SessionRecorder = None
event_log_to_tui_events = None
run_configured_tui = None
run_interactive_tui = None


class SessionMetadataLike(Protocol):
    session_id: str


class SessionStoreLike(Protocol):
    def list_sessions(self) -> list[SessionMetadataLike]: ...


def require_tui_extra() -> None:
    global SessionStore
    global SessionRecorder
    global event_log_to_tui_events
    global run_configured_tui
    global run_interactive_tui
    if SessionStore is not None:
        return
    try:
        session_module = importlib.import_module("aceai.agent.session")
        replay_module = importlib.import_module("aceai.agent.tui.session_replay")
        runner_module = importlib.import_module("aceai.agent.tui.runner")
    except ModuleNotFoundError as exc:
        if exc.name in TUI_EXTRA_MODULES:
            raise SystemExit(TUI_EXTRA_INSTALL_HINT) from None
        raise
    SessionStore = session_module.SessionStore
    SessionRecorder = session_module.SessionRecorder
    event_log_to_tui_events = replay_module.event_log_to_tui_events
    run_configured_tui = runner_module.run_configured_tui
    run_interactive_tui = runner_module.run_interactive_tui


def build_default_agent(
    *,
    api_key: str,
    model: OpenAIModel,
    default_model: OpenAIModel | None = None,
    provider: str = "openai",
    skills: str | None = None,
    skill_selection_mode: str = "all",
    enabled_skills: list[str] | None = None,
    tool_permissions: dict[str, ToolPermission] | None = None,
    tool_enabled: dict[str, bool] | None = None,
    tool_max_calls: dict[str, int] | None = None,
) -> AgentBase:
    agent_model = model if default_model is None else default_model
    skill_path = skills if skills is not None else None
    enabled_skill_names = (
        tuple(enabled_skills or [])
        if skill_selection_mode == "selected"
        else UNSET
    )
    if provider == "openai":
        if skill_path is None:
            return build_ace_agent(
                api_key=api_key,
                model=agent_model,
                enabled_skill_names=enabled_skill_names,
                tool_permissions=tool_permissions,
                tool_enabled=tool_enabled,
                tool_max_calls=tool_max_calls,
            )
        return build_ace_agent(
            api_key=api_key,
            model=agent_model,
            skill_path=skill_path,
            enabled_skill_names=enabled_skill_names,
            tool_permissions=tool_permissions,
            tool_enabled=tool_enabled,
            tool_max_calls=tool_max_calls,
        )
    if skill_path is None:
        return build_ace_agent(
            api_key=api_key,
            model=agent_model,
            provider_name=provider,
            enabled_skill_names=enabled_skill_names,
            tool_permissions=tool_permissions,
            tool_enabled=tool_enabled,
            tool_max_calls=tool_max_calls,
        )
    return build_ace_agent(
        api_key=api_key,
        model=agent_model,
        provider_name=provider,
        skill_path=skill_path,
        enabled_skill_names=enabled_skill_names,
        tool_permissions=tool_permissions,
        tool_enabled=tool_enabled,
        tool_max_calls=tool_max_calls,
    )


def build_agent_from_config(config: AceAITUIConfig) -> AgentBase:
    if config.provider not in supported_provider_names():
        raise ValueError("Unsupported provider")
    if config.provider == "openai":
        return build_default_agent(
            api_key=config.api_key,
            model=config.model,
            default_model=config.default_model,
            skills=config.skills,
            skill_selection_mode=config.skill_selection_mode,
            enabled_skills=config.enabled_skills,
            tool_permissions=config.tool_permissions,
            tool_enabled=config.tool_enabled,
            tool_max_calls=config.tool_max_calls,
        )
    return build_default_agent(
        api_key=config.api_key,
        model=config.model,
        default_model=config.default_model,
        provider=config.provider,
        skills=config.skills,
        skill_selection_mode=config.skill_selection_mode,
        enabled_skills=config.enabled_skills,
        tool_permissions=config.tool_permissions,
        tool_enabled=config.tool_enabled,
        tool_max_calls=config.tool_max_calls,
    )


def resolve_model(provider: str, value: str) -> OpenAIModel:
    if value not in supported_models(provider):
        raise ValueError("Unsupported model")
    return value


def resolve_env_provider() -> str:
    provider = os.environ.get("ACEAI_PROVIDER", "openai")
    if provider not in supported_provider_names():
        raise ValueError("Unsupported provider")
    return provider


def resolve_initial_config(
    *,
    provider: str,
    model: OpenAIModel,
    model_from_env: bool,
) -> AceAITUIConfig | None:
    stored = load_config()
    if stored is not None:
        if model_from_env:
            selected_model = resolve_model(stored.provider, model)
            return replace_config(
                AceAITUIConfig(
                    provider=stored.provider,
                    api_key=stored.api_key,
                    model=selected_model,
                    default_model=stored.default_model,
                    skills=stored.skills,
                    skill_selection_mode=stored.skill_selection_mode,
                    enabled_skills=stored.enabled_skills,
                    api_keys=stored.api_keys,
                    tool_permissions=stored.tool_permissions,
                    tool_enabled=stored.tool_enabled,
                    tool_max_calls=stored.tool_max_calls,
                )
            )
        return stored
    env_name = api_key_env(provider)
    if env_name in os.environ:
        selected_model = model
        if not model_from_env:
            selected_model = default_model(provider)
        return replace_config(
            AceAITUIConfig(
                provider=provider,
                api_key=os.environ[env_name],
                model=selected_model,
                default_model=default_model(provider),
                skills=ACE_AGENT_SKILL_PATH,
                skill_selection_mode="all",
                enabled_skills=[],
                api_keys={provider: os.environ[env_name]},
                tool_permissions={},
                tool_enabled={},
                tool_max_calls={},
            )
        )
    return None


def apply_session_state_to_initial_config(
    config: AceAITUIConfig | None,
    state,
) -> AceAITUIConfig | None:
    if state.selected_model == "":
        return config
    provider = state.selected_provider
    if provider == "":
        if config is None:
            return config
        provider = config.provider
    model = resolve_model(provider, state.selected_model)
    if config is not None and config.provider == provider:
        return replace_config(
            AceAITUIConfig(
                provider=config.provider,
                api_key=config.api_key,
                model=model,
                default_model=config.default_model,
                skills=config.skills,
                skill_selection_mode=config.skill_selection_mode,
                enabled_skills=config.enabled_skills,
                api_keys=config.api_keys,
                tool_permissions=config.tool_permissions,
                tool_enabled=config.tool_enabled,
                tool_max_calls=config.tool_max_calls,
            )
        )
    if config is not None and provider in config.api_keys:
        return replace_config(
            AceAITUIConfig(
                provider=provider,
                api_key=config.api_keys[provider],
                model=model,
                default_model=default_model(provider),
                skills=config.skills,
                skill_selection_mode=config.skill_selection_mode,
                enabled_skills=config.enabled_skills,
                api_keys=config.api_keys,
                tool_permissions=config.tool_permissions,
                tool_enabled=config.tool_enabled,
                tool_max_calls=config.tool_max_calls,
            )
        )
    env_name = api_key_env(provider)
    if env_name in os.environ:
        api_keys: dict[str, str] = {}
        if config is not None:
            api_keys.update(config.api_keys)
        api_keys[provider] = os.environ[env_name]
        return replace_config(
            AceAITUIConfig(
                provider=provider,
                api_key=os.environ[env_name],
                model=model,
                default_model=default_model(provider),
                skills=config.skills if config is not None else ACE_AGENT_SKILL_PATH,
                skill_selection_mode=config.skill_selection_mode
                if config is not None
                else "all",
                enabled_skills=config.enabled_skills if config is not None else [],
                api_keys=api_keys,
                tool_permissions=config.tool_permissions if config is not None else {},
                tool_enabled=config.tool_enabled if config is not None else {},
                tool_max_calls=config.tool_max_calls if config is not None else {},
            )
        )
    return config


def load_session_context(
    *,
    session_id: str,
) -> tuple[object, object, list[object], list[LLMMessage], object]:
    require_tui_extra()
    store = SessionStore()
    metadata = store.get_session(session_id)
    event_log = store.load_event_log(session_id)
    return (
        store,
        metadata,
        event_log_to_tui_events(event_log),
        event_log.replay_llm_history(),
        store.get_session_state(session_id),
    )


def latest_session_id(store: SessionStoreLike) -> str:
    sessions = store.list_sessions()
    if not sessions:
        raise ValueError("aceai resume found no sessions")
    return sessions[0].session_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aceai",
        description="Launch the AceAI terminal UI.",
    )
    parser.add_argument(
        "command",
        nargs="*",
        help="Optional command: resume <session_id>, export <session_id>, or cost.",
    )
    parser.add_argument(
        "--model",
        default=None,
        choices=CLI_MODELS,
        help="Model for the default AceAI CLI agent.",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Write aceai export output to a new file instead of stdout.",
    )
    return parser


def run_with_tui_extra(action: Callable[[], None]) -> None:
    try:
        action()
    except ModuleNotFoundError as exc:
        if exc.name in TUI_EXTRA_MODULES:
            raise SystemExit(TUI_EXTRA_INSTALL_HINT) from None
        raise


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_with_tui_extra(lambda: run_main(args))


def run_main(args: argparse.Namespace) -> None:
    command_parts = list(args.command)
    if command_parts and command_parts[0] == "export":
        if len(command_parts) != 2:
            raise ValueError("aceai export requires a session_id")
        require_tui_extra()
        export_text = SessionStore().export_text(command_parts[1])
        if args.file is not None:
            with Path(args.file).open("x", encoding="utf-8") as stream:
                stream.write(export_text)
            return
        print(export_text, end="")
        return
    if command_parts and command_parts[0] == "cost":
        if len(command_parts) != 1:
            raise ValueError("aceai cost does not accept arguments")
        require_tui_extra()
        print(format_usd(SessionStore().total_cost_usd()))
        return
    resume_session_id: str | None = None
    if command_parts and command_parts[0] == "resume":
        if len(command_parts) > 2:
            raise ValueError("aceai resume requires a session_id")
        if len(command_parts) == 2:
            resume_session_id = command_parts[1]
        else:
            require_tui_extra()
            resume_session_id = latest_session_id(SessionStore())
    elif command_parts:
        raise ValueError(
            "aceai only accepts no arguments, resume <session_id>, "
            "export <session_id>, or cost"
        )
    provider = resolve_env_provider()
    model_value = args.model or os.environ.get("ACEAI_MODEL")
    model_from_env = model_value is not None
    if model_value is None:
        model_value = default_model(provider)
    selected_model = resolve_model(provider, model_value)
    config = resolve_initial_config(
        provider=provider,
        model=selected_model,
        model_from_env=model_from_env,
    )
    require_tui_extra()
    if resume_session_id is None:
        initial_events = []
        initial_history = []
        recorder = None
        session_id = None
    else:
        store, metadata, initial_events, initial_history, session_state = load_session_context(
            session_id=resume_session_id,
        )
        config = apply_session_state_to_initial_config(config, session_state)
        recorder = SessionRecorder(store, metadata.session_id)
        session_id = metadata.session_id
    if config is None:
        run_configured_tui(
            build_agent_from_config,
            initial_config=None,
            initial_question="",
            default_model=selected_model,
            initial_events=initial_events,
            initial_history=initial_history,
            session_recorder=recorder,
            session_id=session_id,
        )
        if recorder is not None and recorder.saved:
            print(f"Session saved: {session_id}")
        return
    if run_configured_tui is None:
        agent = build_agent_from_config(config)
        run_interactive_tui(
            agent,
            initial_events=initial_events,
            initial_history=initial_history,
            session_recorder=recorder,
            session_id=session_id,
        )
    else:
        run_configured_tui(
            build_agent_from_config,
            initial_config=config,
            initial_question="",
            default_model=selected_model,
            initial_events=initial_events,
            initial_history=initial_history,
            session_recorder=recorder,
            session_id=session_id,
        )
    if recorder is not None and recorder.saved:
        print(f"Session saved: {session_id}")
