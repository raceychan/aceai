# AceAI TUI Technical Plan

## Purpose

AceAI needs a terminal UI that makes an agent run readable while it is still in
progress. The TUI should show not only final assistant text, but also the
structure of the run: reasoning summaries, visible thinking output, tool calls,
tool results, media events, step boundaries, and failures.

The TUI must be driven by AceAI's structured event stream. It should not parse
logs or infer state from printed text.

## Product Goals

- Render each agent run as a visible timeline of steps and events.
- Distinguish assistant text, thinking output, reasoning summaries, tool calls,
  tool outputs, media, and errors using shape, labels, color, and layout.
- Keep long tool arguments and outputs inspectable without making the main
  conversation unreadable.
- Preserve raw event details for debugging.
- Support keyboard-first operation in a real terminal.
- Keep the core agent API strict and explicit. Add new event types when the UI
  needs new observable behavior.

## Non-Goals

- Do not expose private chain-of-thought. The UI may show provider-approved
  reasoning summaries, visible output deltas, and operational state.
- Do not build a web UI in this phase.
- Do not make the TUI depend on OpenAI-specific SDK objects.
- Do not maintain compatibility with consumers that assume the current event set
  is complete.

## Current State

AceAI already has the right foundation:

- `AgentBase.run()` returns an async stream of `AgentEvent` records.
- `aceai/core/events.py` defines explicit lifecycle events for LLM, tool, step,
  and run boundaries.
- `aceai/llm/models.py` defines provider-agnostic `LLMStreamEvent` and
  structured `LLMSegment` records.
- `OpenAIProvider._map_stream_event()` maps streamed text, tool-call argument
  deltas, media, and errors into `LLMStreamEvent`.
- `examples/terminal_ui.py` already renders a basic event stream with Rich.

There are also important gaps:

- `AgentBase._call_llm()` currently ignores
  `response.function_call_arguments.delta`, so the agent event stream cannot
  show tool-call argument construction in real time.
- Reasoning content is represented as final `LLMSegment(type="reasoning")`
  records, not as first-class agent events.
- `examples/terminal_ui.py` is a printer, not a stateful TUI. It cannot support
  panes, selection, collapsible details, or keyboard navigation.
- Rich is currently a development dependency, not a runtime or optional TUI
  dependency.

## Technical Choice

Use Textual as the TUI framework and Rich as the rendering layer.

Textual fits AceAI because:

- It is an async Python TUI framework, so it matches `AgentBase.run()` naturally.
- It uses Rich renderables, which preserves the existing rendering investment.
- It provides layouts, widgets, keyboard bindings, styling, scroll views, input
  widgets, and reactive updates.
- It avoids turning Rich `Live` updates into a custom UI framework.

Rich remains useful for:

- Markdown and syntax-highlighted JSON blocks.
- Panels and styled text inside Textual widgets.
- Reusing the lightweight terminal renderer for non-interactive examples.

Do not use prompt_toolkit as the primary framework for this UI. It can build
full-screen terminal apps, but the layout and state model would be more manual
than Textual for this use case.

## Packaging

The installed package should expose an `aceai` command:

```toml
[project.scripts]
aceai = "aceai.agent.tui.cli:main"
```

The CLI uses the default OpenAI-backed agent:

```bash
OPENAI_API_KEY=... aceai
ACEAI_MODEL=gpt-4o-mini OPENAI_API_KEY=... aceai
```

Textual, Rich, and OpenAI are runtime dependencies because the `aceai` command
must work after a normal package install.

If `OPENAI_API_KEY` and saved config are both absent, the TUI opens a provider
setup screen before creating the agent. The setup screen currently exposes:

- provider selection, starting with OpenAI
- API key entry
- model selection
- opt-in persistence

Persisted settings are written to `~/.aceai/config.yaml` with mode `0600`.

## Event Model

The TUI should consume a normalized TUI event model, not raw provider events.
That model should be derived from `AgentEvent` records.

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

The agent layer should grow new first-class events before the TUI ships:

```python
class LLMToolCallDeltaEvent(AgentLifecycleEvent):
    EVENT_TYPE = "agent.llm.tool_call.delta"
    tool_call_delta: LLMToolCallDelta
    text_delta: str


class LLMReasoningEvent(AgentLifecycleEvent):
    EVENT_TYPE = "agent.llm.reasoning"
    segment: LLMSegment
```

These events keep the API explicit. They also avoid overloading
`LLMOutputDeltaEvent`, which should continue to mean user-visible assistant text
or visible model output.

### Provider to Agent Mapping

| Provider event | Current agent behavior | Target agent behavior |
| --- | --- | --- |
| `response.output_text.delta` | Emits `agent.llm.output_text.delta` | Render as `assistant_delta` and coalesce consecutive deltas into one updating assistant block |
| `response.function_call_arguments.delta` | Ignored | Emit `agent.llm.tool_call.delta` |
| `response.media` | Emits `agent.llm.media` | Keep and render as media blocks |
| `response.completed` with reasoning segments | Emits `agent.llm.completed` | Also emit `agent.llm.reasoning` for each reasoning segment before completion or while finalizing |
| `response.error` | Raises runtime error | Keep failure propagation and render as run failure |

### Thinking vs Reasoning

The TUI should make this distinction clear:

- **Assistant delta**: visible streamed answer text emitted during the current
  LLM step. In the current code this is `LLMOutputDeltaEvent`.
- **Thinking delta**: reserved for future provider-approved thinking/status
  events that are not normal answer text.
- **Reasoning summary**: provider-approved reasoning content represented by
  `LLMSegment(type="reasoning")`.
- **Tool activity**: structured operational state around tool arguments,
  execution, output, and errors.

The UI must not label hidden chain-of-thought as visible content.

## UI Layout

The first version should be a full-screen Textual app with four regions.

### Header

Displays the current run status:

- model
- step count
- active tool count
- elapsed time
- final success or failure state

### Timeline Pane

A narrow left pane shows the run structure:

```text
● Step 1
  ◌ Thinking
  ✓ Tool search_docs
  ✓ Tool read_file
● Step 2
  ✓ Final
```

The timeline is the fastest way to understand what the agent did.

### Main Stream Pane

The center pane shows the readable run transcript:

- Assistant text as one updating response block, so streamed tokens look like
  printer output instead of one panel per delta.
- Thinking deltas as dim, italic text with a left rule.
- Reasoning summaries as collapsible blocks.
- Tool calls as titled blocks with the tool name and call id.
- Tool-call argument deltas as live-updating JSON blocks.
- Tool streaming output as live-updating tool-output blocks.
- Tool results as collapsed blocks when long.
- Media as media placeholders with metadata.
- Errors as expanded red blocks.

### Detail Pane

The right pane shows the selected event:

- event type
- step id
- tool call id
- tool name
- raw arguments
- raw output
- provider metadata
- token usage and reasoning tokens when available

### Input Bar

The bottom bar accepts a user question and command shortcuts:

- `/clear`
- `/raw`
- `/save`
- `/events`
- `/quit`

## Visual Language

Use redundant visual signals so the UI does not rely on color alone.

| Event class | Shape | Label | Color role |
| --- | --- | --- | --- |
| Thinking delta | Left rule, dim text | `thinking` | blue-gray |
| Reasoning summary | Collapsible bordered block | `reasoning` | muted violet |
| Assistant text | Updating response block | `assistant` | Snow Storm |
| Tool started | Bordered block with spinner | `tool` | yellow |
| Tool arguments | Syntax-highlighted JSON block | `arguments` | yellow |
| Tool completed | Collapsible result block | `result` | green |
| Tool failed | Expanded error block | `failed` | red |
| Media | Metadata block | `media` | cyan |
| Run completed | Footer state | `completed` | green |
| Run failed | Footer state | `failed` | red |

Long values should be collapsed by default with a clear expansion affordance.

## Proposed Package Structure

```text
aceai/agent/tui/
  __init__.py
  __main__.py
  app.py
  events.py
  runner.py
  theme.py
  widgets/
    __init__.py
    detail.py
    input.py
    stream.py
    timeline.py
```

Responsibilities:

- `events.py`: Convert `AgentEvent` into TUI-specific records.
- `runner.py`: Bridge an `AgentBase.run()` async iterator into Textual messages.
- `app.py`: Own the Textual `App`, layout, key bindings, and command handling.
- `widgets/stream.py`: Render transcript and live event blocks.
- `widgets/timeline.py`: Render step/tool timeline state.
- `widgets/detail.py`: Render selected raw event details.
- `theme.py`: Keep visual styles centralized.

## Implementation Path

### Phase 1: Event Completeness

1. Add `LLMToolCallDeltaEvent` to `aceai/core/events.py`.
2. Add `AgentEventBuilder.tool_call_delta()`.
3. Change `AgentBase._call_llm()` so
   `response.function_call_arguments.delta` emits the new agent event.
4. Add tests proving tool-call argument deltas are visible from
   `AgentBase.run()`.
5. Run `uv run pytest`.

This phase is intentionally breaking for consumers that assumed tool-call
deltas are invisible.

### Phase 2: Reasoning Event Exposure

1. Add `LLMReasoningEvent` to `aceai/core/events.py`.
2. During `response.completed`, inspect `LLMResponse.segments` and emit one
   reasoning event for each `LLMSegment(type="reasoning")`.
3. Keep `LLMCompletedEvent` as the final step snapshot event.
4. Add tests for final reasoning segment emission.
5. Run `uv run pytest`.

If providers later expose streaming reasoning summaries, map them to the same
event type without changing the TUI.

### Phase 3: TUI Event Adapter

1. Add `aceai/agent/tui/events.py`.
2. Convert every supported `AgentEvent` into a single TUI record shape with
   concrete fields.
3. Keep raw `AgentEvent` attached for detail-pane inspection.
4. Add unit tests for event conversion.
5. Run `uv run pytest`.

The adapter is the contract between agent internals and rendering.

### Phase 4: Read-Only TUI Prototype

1. Add Textual, Rich, and OpenAI as runtime dependencies for the installed
   `aceai` command.
2. Build a Textual app that consumes a supplied async event iterator.
3. Implement header, timeline, stream, and detail panes.
4. Render static test events before wiring live agents.
5. Add snapshot-style tests around event state reducers where practical.
6. Run `uv run pytest`.

This phase does not need a real LLM.

Current prototype entry point:

```bash
python -m aceai.agent.tui
```

### Phase 5: Live Agent Runner

1. Add `aceai/agent/tui/runner.py` to run an `AgentBase` and post events into the app.
2. Add cancellation handling for `Ctrl+C` and `/quit`.
3. Add an example that starts a real agent with the TUI.
4. Preserve `examples/terminal_ui.py` as the lightweight non-interactive Rich
   renderer.
5. Run `uv run pytest`.

Current live runner API:

```python
from aceai.agent.tui.runner import run_agent_tui, run_interactive_tui

run_agent_tui(agent, "Question?")
run_interactive_tui(agent)
```

The interactive runner uses the bottom input bar to submit questions. `/clear`
resets the current run state and `/quit` exits the app.

When launched through the `aceai` console command, missing provider credentials
are collected inside the TUI before the first run.

### Phase 6: Debugging Features

1. Add `/raw` to toggle raw event inspection.
2. Add `/save` to export a JSONL event trace.
3. Add filtering by event type.
4. Add collapsed-by-default long outputs.
5. Add search within the current run.

## State Management

Use a small reducer-style state model.

```python
class TUIRunState(Record, kw_only=True):
    status: Literal["idle", "running", "completed", "failed"]
    steps: list[TUIStepState]
    selected_event_id: str | None


class TUIStepState(Record, kw_only=True):
    step_index: int
    step_id: str
    events: list[TUIEvent]
    tools: list[TUIToolState]
```

Widgets should render from state. They should not mutate each other directly.

## Testing Strategy

All tests must run through:

```bash
uv run pytest
```

Recommended test layers:

- Agent event tests for tool-call delta and reasoning events.
- Adapter tests for `AgentEvent` to `TUIEvent` conversion.
- Reducer tests for timeline and selected-event state.
- Lightweight Textual app tests for key bindings and command parsing.
- One example smoke test that imports the TUI package when `textual` is
  installed.

Do not require live LLM credentials for normal tests.

## Risks

### Event Volume

Streaming can produce many deltas. The TUI adapter should be able to coalesce
small text chunks for rendering while preserving raw events in the trace when
debug mode is enabled.

### Terminal Width

Tool arguments and outputs can be wide. JSON blocks should wrap or scroll inside
their pane, not push layout boundaries.

### Reasoning Semantics

Providers differ in what they expose. The UI must label content according to
AceAI event types, not provider marketing terms.

### Dependency Weight

Textual should stay optional. Core AceAI imports must not import Textual.

### Event Ordering

Future concurrent tool execution may make completion order non-deterministic.
The TUI should order by emission time and show stable tool call ids.

## Acceptance Criteria

- A user can run an AceAI agent in a full-screen terminal UI.
- The UI visibly distinguishes assistant output, thinking, reasoning summaries,
  tool calls, tool results, media, and failures.
- Tool-call argument deltas are visible when the provider streams them.
- Reasoning summaries are visible when the provider returns them.
- Raw event details are available for debugging.
- The TUI is available through the installed `aceai` command.
- The default test suite passes with `uv run pytest`.

## First Pull Request Scope

The first implementation PR should be narrow:

1. Add missing agent events for tool-call deltas and reasoning segments.
2. Add tests for the new event emissions.
3. Add the `aceai/agent/tui/events.py` adapter and tests.
4. Add this document to the public docs navigation.

The Textual UI itself can land in a second PR after the event contract is stable.
