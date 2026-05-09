#!/usr/bin/env bash
set -euo pipefail

workdir="${1:-/Users/raceychan/mylab/aceai}"

if [[ ! -d "$workdir" ]]; then
  echo "AceAI workdir does not exist: $workdir" >&2
  exit 2
fi

command_text="cd \"$workdir\" && env -u NO_COLOR TERM=xterm-ghostty COLORTERM=truecolor UV_CACHE_DIR=/tmp/uv-cache uv run aceai"

if pgrep -x "ghostty" >/dev/null 2>&1; then
  mode="tab"
else
  mode="window"
  open -a Ghostty
  sleep 1
fi

osascript - "$command_text" "$mode" <<'OSA'
on run argv
  set commandText to item 1 of argv
  set launchMode to item 2 of argv

  tell application "Ghostty" to activate
  delay 0.5

  tell application "System Events"
    tell process "ghostty"
      set frontmost to true
      delay 0.2
      if launchMode is "tab" then
        keystroke "t" using command down
        delay 0.4
      end if
    end tell

    set oldClipboard to the clipboard
    set the clipboard to commandText
    keystroke "v" using command down
    key code 36
    delay 0.2
    set the clipboard to oldClipboard
  end tell
end run
OSA

echo "Launched AceAI in Ghostty ${mode}: $workdir"
