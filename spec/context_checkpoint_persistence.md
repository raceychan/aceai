# Persistent Context Checkpoints

Status: implemented.

## Problem

AceAI currently has runtime context compression, but the compressed context is
not durable model-input state.

The important boundary:

- A session is a durable transcript/event log for user-visible conversation,
  tools, audit, export, and resume identity.
- A context is the model input assembled for a run. It can include session
  transcript, compressed summaries, citations, skills, retrieved context,
  memories, tool results, and the current user message.

Session can be one source for context, and it can physically store context
checkpoints, but session replay is not the same thing as context replay.

Current behavior:

- `ContextManager.prepare_for_llm()` compresses the in-memory run context when
  the local threshold is exceeded.
- Reactive context-window recovery also compresses the in-memory run context and
  retries the current LLM step once.
- After a successful turn, `AceAgentApp._finish_run_turn()` copies
  `run.context.context[1:]` back into `_llm_history`, so the live process keeps
  using the compressed history.
- Session storage persists transcript events, not the compressed LLM history.
- On session resume, the app currently rebuilds LLM history from transcript
  replay and loses the compressed checkpoint.

This means:

- A live TUI can continue with `compressed history + new messages`.
- A restarted or re-attached session may reconstruct the original large history.
- A failed turn that compressed context but never completed may not update
  `_llm_history`.

## Goals

- Persist compressed model-input history as a durable context checkpoint.
- Keep full transcript events intact for export, debugging, and audit.
- Make context construction use the latest checkpoint plus later transcript
  events when a session is one of the context sources.
- Avoid adding session semantics to `aceai/core`.
- Keep TUI rendering user-facing and concise; checkpoints are not normal chat
  messages.

## Non-Goals

- Do not delete or rewrite old session events.
- Do not change the human-readable transcript/export format.
- Do not solve oversized recent tool results in this change. That remains a
  separate subagent-result payload issue.
- Do not introduce a database schema migration for sessions. Checkpoints use a
  separate append-only store.

## Architecture

### Source of Truth

There are three different truths:

- Transcript truth: all user messages, assistant messages, tool calls, and tool
  results stay in the event log.
- Context truth: the model-input history for a run is assembled by a context
  builder from transcript, checkpoints, citations, skills, memories, and current
  turn input.
- Checkpoint truth: the latest context checkpoint can replace earlier transcript
  material inside the context builder.

The checkpoint is therefore a context-construction optimization, not a transcript
mutation.

### Layer Boundaries

- `aceai/core/context_manager.py`
  - Owns in-memory context compression.
  - Does not know about sessions or persistence.
- `aceai/core/run_loop.py`
  - Detects that compression happened during an agent step.
  - Emits a structured agent event describing the compressed context snapshot.
- `aceai/agent/context_history.py` or equivalent new app-layer boundary
  - Builds model-input history for an app run.
  - Can use session transcript, latest context checkpoint, and future context
    sources.
- `aceai/agent/context_checkpoint_store.py` or equivalent app-layer boundary
  - Persists and loads context checkpoints.
  - Stores checkpoints outside the session transcript event log.
- `aceai/agent/session_service.py`
  - Records transcript events and can expose transcript events to the context
    builder.
  - Should not make `EventLog.replay_llm_history()` mean "effective context".
- `aceai/agent/session.py`
  - Owns transcript event serialization and replay.
  - Does not store context checkpoints in the transcript event log.
- `aceai/agent/tui`
  - May show a small debug/detail signal later, but should not render checkpoint
    payloads as normal assistant content.

## Proposed Runtime Event Model

Add a core event:

```python
class ContextCompressedEvent(AgentLifecycleEvent):
    EVENT_TYPE = "agent.context.compressed"
    reason: Literal["threshold", "context_window_retry"]
    compression_count: int
    history: list[LLMMessage]
```

Notes:

- `history` should be the LLM history excluding the root system prompt, matching
  the shape stored in `AceAgentApp._llm_history`.
- The event should be emitted only after compression actually changes the
  context.
- `reason="threshold"` is for proactive compression before a normal LLM request.
- `reason="context_window_retry"` is for reactive compression after provider
  context-window rejection.

## Proposed Persistent Checkpoint Model

Add an app-layer checkpoint record:

```python
class ContextCheckpoint(Struct, frozen=True, kw_only=True):
    checkpoint_id: str
    session_id: str
    run_id: str
    step_id: str
    reason: Literal["threshold", "context_window_retry"]
    compression_count: int
    included_event_id: str
    message_count: int
    estimated_tokens: int
    history: list[LLMMessage]
```

`included_event_id` is the last transcript event already represented inside
`history`. For a normal turn this is usually the current `user_message` event.
When rebuilding context, the context builder starts from `history` and then
replays transcript events after `included_event_id`.

Physical storage for the first implementation:

- Store checkpoints under the session root in a separate directory, for example:

```text
~/.aceai/sessions/context_checkpoints/{session_id}.checkpoints.jsonl
```

- Each line is one `ContextCheckpoint` payload.
- The latest valid checkpoint for a session is the last line for that session.
- Session JSONL remains transcript-only.

Payload shape:

```json
{
  "version": 1,
  "checkpoint_id": "checkpoint-id",
  "session_id": "session-id",
  "run_id": "run-id",
  "step_id": "step-id",
  "reason": "threshold",
  "compression_count": 1,
  "included_event_id": "event-id",
  "message_count": 12,
  "estimated_tokens": 100000,
  "history": [
    {
      "message_type": "message",
      "role": "user",
      "content": [{"type": "text", "data": "..."}]
    }
  ]
}
```

The payload should include enough structured data to rebuild:

- `LLMMessage`
- `LLMToolCallMessage`
- `LLMToolUseMessage`

Do not store opaque provider SDK objects.

## Message Serialization

Add explicit helpers in the session layer:

```python
def llm_message_to_payload(message: LLMMessage) -> dict[str, Any]: ...
def llm_message_from_payload(payload: dict[str, Any]) -> LLMMessage: ...
```

Supported message payload variants:

- plain message:
  - `message_type="message"`
  - `role`
  - `content`
- assistant tool call message:
  - `message_type="tool_call"`
  - `role`
  - `content`
  - `tool_calls`
  - `reasoning_content`
- tool result message:
  - `message_type="tool_use"`
  - `role`
  - `content`
  - `name`
  - `call_id`

Use direct field access and strict validation. If a checkpoint payload is
malformed, context construction should fail loudly.

## Runtime Flow

### Proactive Compression

1. `run_loop._call_llm()` records `compression_count` before
   `prepare_for_llm()`.
2. `prepare_for_llm()` may call `ContextManager.compress()`.
3. If `compression_count` increased, `_call_llm()` yields
   `ContextCompressedEvent(reason="threshold", history=context[1:])`.
4. The app layer records a context checkpoint associated with the active session
   and the latest transcript event included in the compressed history.
5. The LLM request uses the compressed context.

### Reactive Context-Window Compression

1. `LLMService.stream()` raises `LLMContextWindowExceededError`.
2. `run_loop._call_llm()` catches it.
3. It calls `ContextManager.compress()`.
4. If compression returns `False`, the error is re-raised and the run fails.
5. If compression returns `True`, `_call_llm()` yields
   `ContextCompressedEvent(reason="context_window_retry", history=context[1:])`.
6. `_call_llm()` retries the current LLM step once with the compressed context.
7. If the second attempt still exceeds context, the run fails.

### Successful Turn Completion

`AceAgentApp._finish_run_turn()` should continue to update live `_llm_history`
from `run.context.context[1:]`.

The checkpoint event is for durability and resume; the in-memory update remains
the source for the current process.

## Context Construction Flow

Do not make `EventLog.replay_llm_history()` the effective context builder.
Transcript replay and context construction should be separate APIs.

Add an app-layer function such as:

```python
def build_context_history(session_id: str) -> list[LLMMessage]: ...
```

Flow:

1. Load the latest context checkpoint for the session.
2. If none exists, use transcript replay as the session-derived context source.
3. If one exists:
   - Start from checkpoint `history`.
   - Replay transcript events after `included_event_id`.
   - Append user, assistant, assistant-tool-call, and tool-result messages using
     the existing transcript-to-LLM-message rules.
4. Add non-session context sources separately when needed: citations, skills,
   memory, retrieval, current user turn, etc.
5. Return the merged history.

This makes resume behavior:

```text
latest context checkpoint history + transcript events after checkpoint + other context sources
```

instead of:

```text
all transcript events from session start
```

`SessionService.snapshot(...).history` may continue to exist temporarily, but it
should be understood as app context history, not pure session history. A later
cleanup should rename it to avoid ambiguity.

## Event Ordering

The checkpoint record should be persisted after the user message event is
persisted and before the LLM response/tool events produced from the compressed
context.

Transcript event order remains:

```text
user_message
llm_started
assistant_message | assistant_tool_call ...
tool_result ...
run_completed | run_failed | run_suspended
```

Checkpoint store order is independent:

```text
context_checkpoint(session_id, included_event_id=user_message.event_id)
```

When context construction starts from the checkpoint, the user message is already
included in the checkpoint history, and later assistant/tool events are appended
normally.

## Export and UI

- `aceai export` is unchanged because checkpoints are not in the transcript log.
- TUI main stream should not render checkpoint payloads as assistant content.
- A later debug/detail view may show:

```text
context compressed: threshold, 12 messages, ~100k tokens
```

but that is not required for the first implementation.

## Tests

Add focused tests:

- Transcript replay still returns full transcript history and does not treat
  checkpoints as normal messages.
- Context construction uses the latest checkpoint when present.
- Context construction appends transcript events after the checkpoint and does
  not replay earlier transcript events.
- Multiple checkpoints use only the latest checkpoint for context construction.
- Malformed checkpoint message payload raises a clear error.
- `ContextCheckpointStore` records `ContextCompressedEvent` as a checkpoint.
- `AceAgentApp` can attach/resume a session and receive compressed history from
  the checkpoint.
- Proactive compression emits a checkpoint event.
- Reactive context-window compression emits a checkpoint event and retries.
- Export remains transcript-only.

## Implementation Steps

1. Add `ContextCompressedEvent` to `aceai/core/events.py`.
2. Add `AgentEventBuilder.context_compressed(...)`.
3. In `run_loop._call_llm()`, emit the event after proactive compression and
   after reactive context-window compression.
4. Add `aceai/agent/context_checkpoint_store.py`.
5. Add LLM message serialization helpers in the context checkpoint boundary.
6. Teach the app runtime to persist checkpoints with `included_event_id`.
7. Add `build_context_history(...)` and switch session attach/resume to use it.
8. Keep transcript export unchanged.
9. Add tests listed above.
10. Run focused tests with:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_agent_session.py tests/test_agent_behavior.py tests/test_tui_runner.py -q
```

## Risks

- Checkpoint events can still be large. They are smaller than the full session
  history after compression, but repeated checkpoints add storage overhead.
- If recent messages contain huge tool results, a checkpoint may still be too
  large. This plan does not replace the subagent result slimming work.
- A checkpoint inside a malformed tool-call sequence could break replay. The
  implementation should only checkpoint at LLM-call boundaries, never in the
  middle of tool execution.

## Future Work

- Add optional event-log compaction so old superseded checkpoints can be removed
  or ignored more efficiently.
- Add a small debug event in the TUI that shows when context was compressed.
- Slim subagent tool-result payloads before they enter parent context.
- Align `context_window_tokens` with provider/model catalog limits so proactive
  compression triggers closer to real model limits.
