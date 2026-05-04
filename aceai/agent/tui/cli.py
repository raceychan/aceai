"""Console entry point for the AceAI TUI."""

import argparse
import importlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, Sequence

from aceai.agent.ace_agent import build_ace_agent
from aceai.agent.provider_catalog import (
    all_supported_models,
    api_key_env,
    default_model,
    supported_models,
    supported_provider_names,
)
from aceai.core import AgentBase
from aceai.llm.models import LLMMessage
from aceai.llm.openai import OpenAIModel

from .config import (
    AceAITUIConfig,
    load_config,
)
from .session_adapter import session_messages_to_tui_events
from aceai.agent.cost import format_usd

CLI_MODELS: tuple[OpenAIModel, ...] = all_supported_models()
TUI_EXTRA_MODULES = frozenset(("rich", "sqlalchemy", "textual"))
TUI_EXTRA_INSTALL_HINT = (
    "AceAI TUI dependencies are not installed.\n"
    "Install them with one of:\n"
    "  uv add 'aceai[tui]'\n"
    "  pip install 'aceai[tui]'"
)

SessionRecorder = None
SessionStore = None
messages_to_llm_history = None
run_configured_tui = None
run_interactive_tui = None


class SessionMetadataLike(Protocol):
    session_id: str


class SessionStoreLike(Protocol):
    def list_sessions(self) -> list[SessionMetadataLike]: ...


def require_tui_extra() -> None:
    global SessionRecorder
    global SessionStore
    global messages_to_llm_history
    global run_configured_tui
    global run_interactive_tui
    if SessionStore is not None:
        return
    try:
        session_module = importlib.import_module("aceai.agent.session")
        runner_module = importlib.import_module("aceai.agent.tui.runner")
    except ModuleNotFoundError as exc:
        if exc.name in TUI_EXTRA_MODULES:
            raise SystemExit(TUI_EXTRA_INSTALL_HINT) from None
        raise
    SessionRecorder = session_module.SessionRecorder
    SessionStore = session_module.SessionStore
    messages_to_llm_history = session_module.messages_to_llm_history
    run_configured_tui = runner_module.run_configured_tui
    run_interactive_tui = runner_module.run_interactive_tui


def build_default_agent(
    *,
    api_key: str,
    model: OpenAIModel,
    provider: str = "openai",
) -> AgentBase:
    if provider == "openai":
        return build_ace_agent(api_key=api_key, model=model)
    return build_ace_agent(api_key=api_key, model=model, provider_name=provider)


def build_agent_from_config(config: AceAITUIConfig) -> AgentBase:
    if config.provider not in supported_provider_names():
        raise ValueError("Unsupported provider")
    if config.provider == "openai":
        return build_default_agent(api_key=config.api_key, model=config.model)
    return build_default_agent(
        api_key=config.api_key,
        model=config.model,
        provider=config.provider,
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
    env_name = api_key_env(provider)
    if env_name in os.environ:
        selected_model = model
        if not model_from_env:
            selected_model = default_model(provider)
        return AceAITUIConfig(
            provider=provider,
            api_key=os.environ[env_name],
            model=selected_model,
            api_keys={provider: os.environ[env_name]},
        )
    stored = load_config()
    if stored is None:
        return None
    if model_from_env:
        selected_model = resolve_model(stored.provider, model)
        return AceAITUIConfig(
            provider=stored.provider,
            api_key=stored.api_key,
            model=selected_model,
            api_keys=stored.api_keys,
        )
    return stored


def create_session_context(
    *,
    resume_session_id: str | None,
) -> tuple[object, object, list[object], list[LLMMessage]]:
    require_tui_extra()
    store = SessionStore()
    if resume_session_id is None:
        metadata = store.create_session()
        return store, metadata, [], []
    metadata = store.get_session(resume_session_id)
    messages = store.load_messages(resume_session_id)
    return (
        store,
        metadata,
        session_messages_to_tui_events(messages),
        messages_to_llm_history(messages),
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
    store, metadata, initial_events, initial_history = create_session_context(
        resume_session_id=resume_session_id,
    )
    recorder = SessionRecorder(store, metadata.session_id)
    if config is None:
        run_configured_tui(
            build_agent_from_config,
            initial_config=None,
            initial_question="",
            default_model=selected_model,
            initial_events=initial_events,
            initial_history=initial_history,
            session_recorder=recorder,
            session_id=metadata.session_id,
        )
        if recorder.saved:
            print(f"Session saved: {metadata.session_id}")
        return
    if run_configured_tui is None:
        agent = build_agent_from_config(config)
        run_interactive_tui(
            agent,
            initial_events=initial_events,
            initial_history=initial_history,
            session_recorder=recorder,
            session_id=metadata.session_id,
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
            session_id=metadata.session_id,
        )
    if recorder.saved:
        print(f"Session saved: {metadata.session_id}")
