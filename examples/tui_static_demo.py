"""Launch the read-only AceAI TUI with static fixture events."""

from agent_core.tui.app import run_static_tui
from agent_core.tui.demo import static_demo_events


def main() -> None:
    run_static_tui(static_demo_events())


if __name__ == "__main__":
    main()
