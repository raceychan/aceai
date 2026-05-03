"""Run the static AceAI TUI demo."""

from .app import run_static_tui
from .demo import static_demo_events


def main() -> None:
    run_static_tui(static_demo_events())


if __name__ == "__main__":
    main()
