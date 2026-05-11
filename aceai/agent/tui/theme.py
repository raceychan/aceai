"""Shared display labels for the read-only TUI prototype."""

from collections.abc import Mapping
from typing import get_args

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
    "agent_inbox_delivered": "inbox",
    "agent_inbox_item": "inbox",
    "session_notice": "session",
    "idea_list": "ideas",
    "run_completed": "completed",
    "run_failed": "failed",
    "run_suspended": "approval",
    "step_completed": "step done",
    "step_failed": "step failed",
    "step_started": "step",
    "llm_completed": "llm",
    "assistant_delta": "aceai",
    "thinking_delta": "thinking",
    "reasoning_summary": "thought",
    "llm_retrying": "retry",
    "context_compaction_started": "compact",
    "context_compaction_failed": "compact failed",
    "context_compressed": "compact",
    "tool_call_delta": "arguments",
    "tool_started": "tool",
    "tool_output": "tool output",
    "tool_approval_requested": "approval",
    "tool_approval_resolved": "approval",
    "tool_completed": "result",
    "tool_failed": "failed",
    "media": "media",
}

EVENT_STYLES: dict[TUIEventKind, str] = {
    "user_message": f"bold {NORD_FROST_1}",
    "agent_inbox_delivered": f"bold {NORD_YELLOW}",
    "agent_inbox_item": f"bold {NORD_YELLOW}",
    "session_notice": f"bold {NORD_FROST_0}",
    "idea_list": f"bold {NORD_FROST_0}",
    "run_completed": f"bold {NORD_GREEN}",
    "run_failed": f"bold {NORD_RED}",
    "run_suspended": f"bold {NORD_YELLOW}",
    "step_completed": NORD_GREEN,
    "step_failed": NORD_RED,
    "step_started": f"bold {NORD_FROST_1}",
    "llm_completed": NORD_SNOW_STORM_1,
    "assistant_delta": NORD_SNOW_STORM_2,
    "thinking_delta": f"italic {NORD_FROST_1}",
    "reasoning_summary": f"italic {NORD_FROST_0}",
    "llm_retrying": f"bold {NORD_YELLOW}",
    "context_compaction_started": f"bold {NORD_YELLOW}",
    "context_compaction_failed": f"bold {NORD_RED}",
    "context_compressed": f"bold {NORD_GREEN}",
    "tool_call_delta": NORD_FROST_0,
    "tool_started": f"bold {NORD_FROST_0}",
    "tool_output": NORD_FROST_0,
    "tool_approval_requested": f"bold {NORD_FROST_0}",
    "tool_approval_resolved": f"bold {NORD_FROST_0}",
    "tool_completed": NORD_FROST_0,
    "tool_failed": NORD_RED,
    "media": NORD_FROST_0,
}


def validate_event_theme_registry(
    labels: Mapping[str, str],
    styles: Mapping[str, str],
) -> None:
    event_kinds = set(get_args(TUIEventKind))
    label_keys = set(labels)
    style_keys = set(styles)
    missing_labels = sorted(event_kinds - label_keys)
    missing_styles = sorted(event_kinds - style_keys)
    extra_labels = sorted(label_keys - event_kinds)
    extra_styles = sorted(style_keys - event_kinds)
    if (
        not missing_labels
        and not missing_styles
        and not extra_labels
        and not extra_styles
    ):
        return
    details = []
    if missing_labels:
        details.append("missing labels: " + ", ".join(missing_labels))
    if missing_styles:
        details.append("missing styles: " + ", ".join(missing_styles))
    if extra_labels:
        details.append("extra labels: " + ", ".join(extra_labels))
    if extra_styles:
        details.append("extra styles: " + ", ".join(extra_styles))
    raise RuntimeError("TUI event theme registry is incomplete: " + "; ".join(details))


validate_event_theme_registry(EVENT_LABELS, EVENT_STYLES)
