# Terminal UI

AceAI's terminal UI should make an agent run inspectable while it is still in
progress. It should render assistant output, thinking output, reasoning
summaries, tool calls, tool results, media, step boundaries, and failures from
the structured `AgentEvent` stream.

The complete implementation spec is kept in `spec/tui.md`.

## Goals

- Render every run as a step-by-step timeline.
- Distinguish assistant text, thinking, reasoning summaries, tool calls, tool
  outputs, media, and errors using layout, labels, and color.
- Keep raw event details available for debugging.
- Support keyboard-first operation in a real terminal.
- Keep the core package installable without TUI dependencies.
- Stream assistant output into one updating response block instead of rendering
  one visual frame per token.

## Technical Choice

Use Textual as the full-screen terminal UI framework and Rich as the renderable
layer.

Textual matches AceAI's async event stream and provides layouts, widgets,
keyboard bindings, scrolling panes, and reactive updates. Rich remains useful
for styled text, panels, Markdown, and syntax-highlighted JSON blocks.

The installed package exposes an `aceai` command:

```toml
[project.scripts]
aceai = "aceai.agent.tui.cli:main"
```

`aceai` launches the terminal UI after installation. It uses `OPENAI_API_KEY`
for the default OpenAI-backed agent and `ACEAI_MODEL` to override the default
model.

If no API key is available from the environment or saved config, the TUI opens a
provider setup screen. The user can choose the provider, enter the API key,
choose the model, and decide whether to persist the config. AceAI reads
`.aceai/config.yml` in the current project first, then falls back to
`~/.aceai/config.yaml`. Persisted config is stored at `.aceai/config.yml` with
file mode `0600`.

The runtime config page has separate tabs for provider settings, local tool
permissions, and runtime stats. Each app tool can be set to `always`, `ask`, or
`never`: `always` exposes the tool without approval, `ask` routes execution
through the approval strip, and `never` removes the tool from the model-visible
tool list.

## Event Model

The TUI should consume AceAI agent events through a small adapter. The adapter
converts agent events into UI events such as:

```python
TUIEventKind = Literal[
    "run_started",
    "step_started",
    "llm_completed",
    "thinking_delta",
    "assistant_delta",
    "reasoning_summary",
    "tool_call_delta",
    "tool_started",
    "tool_output",
    "tool_completed",
    "tool_failed",
    "media",
    "step_completed",
    "run_completed",
    "run_failed",
]
```

Before the UI ships, the agent event model should expose two missing event
types:

- `agent.llm.tool_call.delta` for streamed tool-call argument construction.
- `agent.llm.reasoning` for provider-approved reasoning segments.

The UI must not expose private chain-of-thought. It should show visible
streamed output, provider-approved reasoning summaries, and operational state.

## Layout

The first version should use four regions:

- Header: model, status, step count, active tools, elapsed time.
- Timeline pane: step and tool execution structure.
- Main stream pane: assistant text, thinking blocks, reasoning summaries, tool
  calls, tool results, media, and errors.
- Detail pane: selected event metadata, raw arguments, raw output, provider
  metadata, and token usage.
- Input bar: user question and commands such as `/clear`, `/raw`, `/save`,
  `/events`, and `/quit`.

## Visual Language

Use redundant signals so the UI does not rely on color alone.

| Event class | Shape | Label | Color role |
| --- | --- | --- | --- |
| Thinking delta | Left rule, dim text | `thinking` | blue-gray |
| Reasoning summary | Collapsible block | `reasoning` | muted violet |
| Assistant text | Updating response block | `assistant` | default |
| Tool started | Bordered block with spinner | `tool` | yellow |
| Tool arguments | JSON block | `arguments` | yellow |
| Tool completed | Collapsible result block | `result` | green |
| Tool failed | Expanded error block | `failed` | red |
| Media | Metadata block | `media` | cyan |
| Run completed | Footer state | `completed` | green |
| Run failed | Footer state | `failed` | red |

## Implementation Path

1. Event completeness:
   Add `LLMToolCallDeltaEvent`, emit it from `Agent._call_llm()`, and test
   streamed tool-call arguments. This is implemented.

2. Reasoning exposure:
   Add `LLMReasoningEvent`, emit it for `LLMSegment(type="reasoning")`, and test
   final reasoning segment visibility. This is implemented.

3. TUI event adapter:
   Add `aceai/agent/tui/events.py` to convert `AgentEvent` records into stable TUI
   records with raw event details attached. This is implemented.

4. Read-only prototype:
   Add Textual dependencies, build static event rendering, and implement header,
   timeline, stream, and detail panes. A static prototype is available through:

   ```bash
   python -m aceai.agent.tui
   ```

5. Live agent runner:
   Add `aceai/agent/tui/runner.py` to bridge `Agent.run()` into the Textual app,
   then wire cancellation and command handling. The live runner bridge is
   available as `aceai.agent.tui.runner.run_agent_tui`.

6. Interactive input:
   Use the bottom input bar to submit a question with Enter. The interactive
   runner is available as `aceai.agent.tui.runner.run_interactive_tui`.

   The installed command starts the interactive TUI:

   ```bash
   aceai
   ```

   If no `OPENAI_API_KEY` is set, the TUI asks for provider settings before the
   first run. Persisting those settings is opt-in.

7. Debugging features:
   Add raw event toggle, JSONL trace export, event filtering, collapsed long
   outputs, and search within the current run.

All test runs should use:

```bash
uv run pytest
```

## Acceptance Criteria

- A user can run an AceAI agent in a full-screen terminal UI.
- The UI visibly distinguishes assistant output, thinking, reasoning summaries,
  tool calls, tool results, media, and failures.
- Tool-call argument deltas are visible when the provider streams them.
- Reasoning summaries are visible when the provider returns them.
- Raw event details are available for debugging.
- The TUI is available through the installed `aceai` command.
- The default test suite passes with `uv run pytest`.
