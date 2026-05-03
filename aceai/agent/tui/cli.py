"""Console entry point for the AceAI TUI."""

import argparse
import os
from typing import Sequence

from aceai.agent.ace_agent import build_ace_agent
from aceai.core import AgentBase
from aceai.llm.openai import OpenAIModel

from .config import AceAITUIConfig, load_config
from .runner import run_agent_tui, run_configured_tui, run_interactive_tui

OPENAI_MODELS: tuple[OpenAIModel, ...] = (
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-5o",
    "gpt-5o-mini",
    "gpt-5.1",
    "o3-large",
    "o4-mini",
)


def build_default_agent(*, api_key: str, model: OpenAIModel) -> AgentBase:
    return build_ace_agent(api_key=api_key, model=model)


def build_agent_from_config(config: AceAITUIConfig) -> AgentBase:
    if config.provider != "openai":
        raise ValueError("Unsupported provider")
    return build_default_agent(api_key=config.api_key, model=config.model)


def resolve_model(value: str) -> OpenAIModel:
    if value not in OPENAI_MODELS:
        raise ValueError("Unsupported OpenAI model")
    return value


def resolve_initial_config(model: OpenAIModel) -> AceAITUIConfig | None:
    if "OPENAI_API_KEY" in os.environ:
        return AceAITUIConfig(
            provider="openai",
            api_key=os.environ["OPENAI_API_KEY"],
            model=model,
        )
    stored = load_config()
    if stored is None:
        return None
    if "ACEAI_MODEL" in os.environ:
        return AceAITUIConfig(
            provider=stored.provider,
            api_key=stored.api_key,
            model=model,
        )
    return stored


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="aceai",
        description="Launch the AceAI terminal UI.",
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="Optional question to run immediately. Omit for interactive mode.",
    )
    parser.add_argument(
        "--model",
        default=None,
        choices=OPENAI_MODELS,
        help="OpenAI model for the default AceAI CLI agent.",
    )
    args = parser.parse_args(argv)
    selected_model = resolve_model(args.model or os.environ.get("ACEAI_MODEL", "gpt-5.1"))
    config = resolve_initial_config(selected_model)
    question = " ".join(args.question)
    if config is None:
        run_configured_tui(
            build_agent_from_config,
            initial_config=None,
            initial_question=question,
            default_model=selected_model,
        )
        return
    agent = build_agent_from_config(config)
    if question == "":
        run_interactive_tui(agent)
    else:
        run_agent_tui(agent, question)
