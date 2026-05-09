---
name: aceai-ghostty-launch
description: "Use when you need to open the AceAI TUI in Ghostty from Codex, preserving existing Ghostty windows by opening a new tab when Ghostty is already running, and launching AceAI with a clean truecolor terminal environment."
---

# AceAI Ghostty Launch

Use this skill when the user asks to open, start, launch, or prepare AceAI in Ghostty for visual TUI validation.

## Rules

- Never close or kill an existing Ghostty window or process.
- If Ghostty is already running, open a new Ghostty tab and launch AceAI there.
- If Ghostty is not running, open Ghostty and launch AceAI in the new window.
- Do not inherit Codex's terminal color environment. Codex may set `NO_COLOR=1`, `TERM=dumb`, or an empty `COLORTERM`, which makes AceAI render gray.
- Launch from the AceAI repository root unless the user explicitly asks for another project path.

## Command

Run the bundled launcher from the AceAI repo root:

```sh
.agent/skills/aceai-ghostty-launch/scripts/open_aceai_ghostty.sh /Users/raceychan/mylab/aceai
```

The launcher runs AceAI with:

```sh
env -u NO_COLOR TERM=xterm-ghostty COLORTERM=truecolor UV_CACHE_DIR=/tmp/uv-cache uv run aceai
```

## Verification

After launch, verify with `ps aux | rg -i 'ghostty|uv run aceai|/aceai$' | rg -v rg`.

For visual confirmation, use the `aceai-tui-screenshot` skill to capture the Ghostty window. A correct idle AceAI screen should show color, including the blue stream border and yellow empty-state mascot.

## If Automation Is Blocked

If macOS denies Accessibility automation for opening a tab or pasting the launch command, say that explicitly. Do not fall back to killing Ghostty or opening an uncolored `open --args -e` session.
