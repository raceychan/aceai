"""Shared display labels for the read-only TUI prototype."""

from .events import TUIEventKind

NORD_POLAR_NIGHT_0 = "#2e3440"
NORD_POLAR_NIGHT_1 = "#3b4252"
NORD_POLAR_NIGHT_2 = "#434c5e"
NORD_POLAR_NIGHT_3 = "#4c566a"
NORD_SNOW_STORM_0 = "#d8dee9"
NORD_SNOW_STORM_1 = "#e5e9f0"
NORD_SNOW_STORM_2 = "#eceff4"
NORD_FROST_0 = "#8fbcbb"
NORD_FROST_1 = "#88c0d0"
NORD_FROST_2 = "#81a1c1"
NORD_FROST_3 = "#5e81ac"
NORD_RED = "#bf616a"
NORD_ORANGE = "#d08770"
NORD_YELLOW = "#ebcb8b"
NORD_GREEN = "#a3be8c"
NORD_PURPLE = "#b48ead"

EVENT_LABELS: dict[TUIEventKind, str] = {
    "user_message": "you",
    "session_notice": "session",
    "run_completed": "completed",
    "run_failed": "failed",
    "step_completed": "step done",
    "step_failed": "step failed",
    "step_started": "step",
    "llm_completed": "llm",
    "assistant_delta": "assistant",
    "thinking_delta": "thinking",
    "reasoning_summary": "reasoning",
    "tool_call_delta": "arguments",
    "tool_started": "tool",
    "tool_output": "tool output",
    "tool_completed": "result",
    "tool_failed": "failed",
    "media": "media",
}

EVENT_STYLES: dict[TUIEventKind, str] = {
    "user_message": f"bold {NORD_FROST_1}",
    "session_notice": f"bold {NORD_FROST_0}",
    "run_completed": f"bold {NORD_GREEN}",
    "run_failed": f"bold {NORD_RED}",
    "step_completed": NORD_GREEN,
    "step_failed": NORD_RED,
    "step_started": f"bold {NORD_FROST_1}",
    "llm_completed": NORD_SNOW_STORM_1,
    "assistant_delta": NORD_SNOW_STORM_2,
    "thinking_delta": f"italic {NORD_FROST_1}",
    "reasoning_summary": f"italic {NORD_PURPLE}",
    "tool_call_delta": NORD_YELLOW,
    "tool_started": f"bold {NORD_YELLOW}",
    "tool_output": NORD_YELLOW,
    "tool_completed": NORD_GREEN,
    "tool_failed": NORD_RED,
    "media": NORD_FROST_0,
}
