---
name: aceai-tui-screenshot
description: "Use when you need to drive an already-open AceAI TUI in Ghostty to a specific screen or state and return a cropped screenshot of that Ghostty window for visual verification."
---

# AceAI TUI Screenshot

Use this skill after AceAI is open in Ghostty, especially for TUI layout checks, visual regressions, screenshots for the user, or verifying a specific AceAI screen.

## Preconditions

- AceAI should already be running in Ghostty. If not, use the `aceai-ghostty-launch` skill first.
- Do not close existing Ghostty windows or tabs.
- Prefer operating the foreground Ghostty tab that is already running AceAI.

## Drive The TUI

Use the bundled script to focus Ghostty, optionally paste text into AceAI, optionally press Enter, wait, and capture the foreground Ghostty window:

```sh
.agent/skills/aceai-tui-screenshot/scripts/drive_and_capture_aceai.sh \
  --out /tmp/aceai_tui.png \
  --text "/config" \
  --enter \
  --wait 1
```

Examples:

```sh
# Capture the current AceAI screen without changing it.
.agent/skills/aceai-tui-screenshot/scripts/drive_and_capture_aceai.sh --out /tmp/aceai_tui.png

# Open config, wait, and capture.
.agent/skills/aceai-tui-screenshot/scripts/drive_and_capture_aceai.sh --out /tmp/aceai_config.png --text "/config" --enter --wait 1

# Press a Textual binding and capture.
.agent/skills/aceai-tui-screenshot/scripts/drive_and_capture_aceai.sh --out /tmp/aceai_debug.png --key d --wait 1
```

## Screenshot Requirements

- Return the screenshot to the user with Markdown image syntax using the absolute file path:

```md
![AceAI TUI](/tmp/aceai_tui.png)
```

- Inspect the screenshot before claiming success.
- If the screenshot is gray, check whether AceAI was launched with `NO_COLOR=1` or `TERM=dumb`; relaunch with `aceai-ghostty-launch`.
- If the screenshot includes other apps, recapture after bringing Ghostty to the foreground.

## If Automation Is Blocked

If macOS denies Accessibility access for focusing Ghostty, typing, or reading the window bounds, say exactly which step failed. Do not invent a screenshot from terminal text output.
