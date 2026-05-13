"""Session display helpers for the AceAI TUI."""

import re


_LEGACY_TITLE_TIMESTAMP = re.compile(r" - \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")


def session_display_title(title: str) -> str:
    return _LEGACY_TITLE_TIMESTAMP.sub("", title)
