import pytest

from aceai.agent.tui.theme import (
    EVENT_LABELS,
    EVENT_STYLES,
    validate_event_theme_registry,
)


def test_tui_event_theme_registry_is_complete() -> None:
    validate_event_theme_registry(EVENT_LABELS, EVENT_STYLES)


def test_tui_event_theme_registry_reports_missing_label() -> None:
    labels = dict(EVENT_LABELS)
    del labels["agent_inbox_item"]

    with pytest.raises(RuntimeError, match="missing labels: agent_inbox_item"):
        validate_event_theme_registry(labels, EVENT_STYLES)


def test_tui_event_theme_registry_reports_missing_style() -> None:
    styles = dict(EVENT_STYLES)
    del styles["agent_inbox_item"]

    with pytest.raises(RuntimeError, match="missing styles: agent_inbox_item"):
        validate_event_theme_registry(EVENT_LABELS, styles)


def test_tui_event_theme_registry_reports_extra_key() -> None:
    labels = dict(EVENT_LABELS)
    labels["not_a_real_event"] = "bad"

    with pytest.raises(RuntimeError, match="extra labels: not_a_real_event"):
        validate_event_theme_registry(labels, EVENT_STYLES)
