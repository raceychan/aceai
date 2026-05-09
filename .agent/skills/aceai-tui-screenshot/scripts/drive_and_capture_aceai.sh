#!/usr/bin/env bash
set -euo pipefail

out="/tmp/aceai_tui.png"
text=""
key_name=""
press_enter=0
wait_seconds="0.5"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)
      out="$2"
      shift 2
      ;;
    --text)
      text="$2"
      shift 2
      ;;
    --key)
      key_name="$2"
      shift 2
      ;;
    --enter)
      press_enter=1
      shift
      ;;
    --wait)
      wait_seconds="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if ! pgrep -x "ghostty" >/dev/null 2>&1; then
  echo "Ghostty is not running; launch AceAI first." >&2
  exit 2
fi

mkdir -p "$(dirname "$out")"

bounds="$(
  osascript - "$text" "$key_name" "$press_enter" "$wait_seconds" <<'OSA'
on run argv
  set textToType to item 1 of argv
  set keyName to item 2 of argv
  set shouldEnter to item 3 of argv
  set waitSeconds to item 4 of argv as real

  tell application "Ghostty" to activate
  delay 0.3

  tell application "System Events"
    tell process "ghostty"
      set frontmost to true
      delay 0.2
      if textToType is not "" then
        set oldClipboard to the clipboard
        set the clipboard to textToType
        keystroke "v" using command down
        delay 0.1
        set the clipboard to oldClipboard
      end if
      if keyName is not "" then
        keystroke keyName
      end if
      if shouldEnter is "1" then
        key code 36
      end if
      delay waitSeconds
      set targetWindow to front window
      set windowPosition to position of targetWindow
      set windowSize to size of targetWindow
      return (item 1 of windowPosition as string) & "," & (item 2 of windowPosition as string) & "," & (item 1 of windowSize as string) & "," & (item 2 of windowSize as string)
    end tell
  end tell
end run
OSA
)"

screencapture -x -R"$bounds" "$out"
echo "$out"
