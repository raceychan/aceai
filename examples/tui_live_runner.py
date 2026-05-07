"""Minimal helper showing how to launch a live agent in the AceAI TUI."""

from aceai import Agent
from aceai.agent.tui.runner import run_agent_tui, run_interactive_tui


def run_live_agent_tui(agent: Agent, question: str) -> None:
    run_agent_tui(agent, question)


def run_interactive_agent_tui(agent: Agent) -> None:
    run_interactive_tui(agent)
