"""Minimal helper showing how to launch a live agent in the AceAI TUI."""

from aceai import AgentBase
from aceai.tui.runner import run_agent_tui, run_interactive_tui


def run_live_agent_tui(agent: AgentBase, question: str) -> None:
    run_agent_tui(agent, question)


def run_interactive_agent_tui(agent: AgentBase) -> None:
    run_interactive_tui(agent)
