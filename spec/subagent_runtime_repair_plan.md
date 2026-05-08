# Subagent Runtime Repair Plan

This note collects the recent TUI, delegation, retry, and context-window issues
so they can be fixed one by one without relying on chat history.

## Goals

- Main-agent and child-agent behavior must be explicit and inspectable.
- The TUI should show enough detail for debugging without making the main stream
  unreadable.
- Parallel subagents must actually run in parallel when the model emits multiple
  independent delegation tool calls.
- Child-agent results must not explode the main-agent context.
- Context-window errors must trigger compression or fail fast with a useful
  message, not retry the same oversized request.

## Issue List

### 1. Child agents do not get hosted tools

Status: implemented, keep regression coverage.

Symptom:

- The main AceAI agent can see hosted tools such as `openai:web_search`, but a
  delegated child agent may be created with only local tools.
- The child then reports that it cannot browse or search even when the parent
  provider supports hosted search.

Expected behavior:

- `delegate_to_subagent` exposes the available hosted tool identifiers in its
  schema.
- If `allowed_tools` includes a hosted tool identifier, the child executor
  receives that hosted tool.
- Child hosted tools are selected explicitly, not implicitly granted.

Regression target:

- A delegation call with `allowed_tools=["openai:web_search"]` creates a child
  executor whose hosted tools include `openai:web_search`.

### 2. Multiple delegated subagents do not run concurrently

Status: implemented, keep regression coverage.

Symptom:

- A prompt asks for three independent subagents to run in parallel.
- The TUI shows only one running subagent because approval-free tool calls are
  executed sequentially inside one LLM turn.

Expected behavior:

- Consecutive approval-free tool calls from one model response are started as a
  batch.
- All `tool_started` events are emitted before awaiting the batch.
- Results are appended back to context in the original tool-call order.
- Approval-required tool calls still suspend the run at the approval boundary.

Regression target:

- Two approval-free tool calls block until both have started; the run must not
  deadlock or time out.

### 3. Subagent panel is too dense and hard to read

Status: implemented, needs live TUI validation during future changes.

Symptom:

- A single compact list entry tries to show task, context, instructions, run id,
  steps, summary, evidence, and tool results.
- With multiple subagents this becomes unreadable.

Expected behavior:

- The panel shows one subagent per page.
- Full detail is shown for that subagent without artificial truncation.
- The panel header shows total, running, done, failed, and current page.
- Keyboard pagination uses left/right or `h`/`l`.

Regression target:

- Two subagents render as two pages, and the second page includes full run and
  detail fields.

### 4. Subagent panel cannot scroll with the mouse wheel

Status: implemented, keep regression coverage.

Symptom:

- The panel content is longer than the visible area, but wheel scrolling does
  not move the content.
- Earlier deferred scroll reset logic can accidentally jump back to the top.

Expected behavior:

- The panel uses a native scrollable widget.
- Mouse wheel events move the panel content.
- Re-rendering a page resets to the top once, but normal wheel scrolling is not
  undone on the next refresh.

Regression target:

- A long subagent detail page responds to mouse scroll events by increasing
  `scroll_y`.

### 5. Tool-call summaries are noisy or misleading

Status: implemented, keep regression coverage.

Symptom:

- Successful shell/search calls display low-value summaries such as
  `command exited 0`.
- Search result payloads that include `exit_code` but use `matches` instead of
  `stdout` can crash summary rendering with `KeyError: 'stdout'`.

Expected behavior:

- Success is summarized as `succeeded`.
- Nonzero shell exits are summarized as `exit N`.
- Search payloads are recognized before shell payloads.
- Full stdout, stderr, and matches remain available in detail/raw views instead
  of being squeezed into the main stream line.

Regression target:

- A search payload containing `matches` and `exit_code` renders without trying
  to read `stdout`.

### 6. Run failures can look like the assistant stopped responding

Status: implemented, keep regression coverage.

Symptom:

- The status bar timer stops after a failed request.
- The input box returns, but the main stream does not show the `run_failed` or
  `step_failed` reason clearly.

Expected behavior:

- `step_failed` and `run_failed` events are rendered into the main stream.
- The error should be visible without opening raw debug state.

Regression target:

- A `run_failed` event renders a visible failure block in the stream.

### 7. Retry UI is not tied to the current message/run

Status: implemented, keep regression coverage.

Symptom:

- Retry status may appear only as a transient notification.
- Different messages/runs are hard to distinguish.
- After the notification disappears, the stream no longer explains why nothing
  is happening.

Expected behavior:

- Each retry event includes `run_id`, `retry_count`, `retry_max`, and
  `retry_delay_seconds`.
- Retry status is both notified and appended to the stream.
- The title/content says which message retry is happening.

Regression target:

- `llm_retrying` appears in the stream with the run id and retry count.

### 8. Context compression does not react to provider context-window errors

Status: implemented, keep regression coverage.

Symptom:

- The provider returns an error like `Your input exceeds the context window of
  this model`.
- AceAI retries the identical oversized request five times.
- The normal proactive compression path does not trigger because it only runs
  before the request and uses local estimates.

Expected behavior:

- Context-window errors are classified as non-transient.
- The request is not retried unchanged. Done.
- The run loop should either compress and retry the current step once, or fail
  with a clear context-window message if compression cannot reduce enough.
  Done.

Likely changes:

- Add an explicit context-window error classifier in the LLM service/provider
  boundary. Done.
- Exclude context-window errors from generic retry. Done.
- Add a run-loop recovery path that performs compression and retries the current
  LLM step once. Done.

Regression target:

- A simulated context-window error triggers compression before the next LLM
  attempt and does not consume all generic retries unchanged.

### 9. Agent context-window settings can disagree with provider limits

Status: implemented, keep regression coverage.

Symptom:

- The status bar may show a tiny context percentage from provider/catalog usage.
- The agent compression threshold can still be based on a default context window
  instead of the selected provider/model limit.

Expected behavior:

- `build_ace_agent()` passes the selected provider/model context window into the
  `Agent` context manager.
- Child agents inherit the same effective context-window and compression policy.
- UI context percentage and compression decisions are based on the same limit.

Regression target:

- Building AceAI for a known model sets `Agent.context.context_window_tokens` to
  the provider catalog value.
- Delegated child agents inherit the same effective context-window token limit.

### 10. Child-agent results can bloat the parent context

Status: pending, high priority. Tracked separately in
`spec/subagent_result_artifact_plan.md`.

Summary:

- Child-agent audit data must stay complete for UI/export/debug.
- Parent-model context must receive only a bounded handoff summary plus stable
  artifact ids.
- Full child final answers, tool calls, tool arguments, tool results, errors,
  and hosted-tool metadata belong behind artifact references and bounded
  inspection tools.

### 11. Session ordering should follow session creation time

Status: implemented, keep regression coverage.

Symptom:

- Session metadata `updated_at` can change for model switches, read-only
  events, or notifications.
- Sorting sessions by metadata update time makes the resume list move in ways
  that do not match user intent.
- Sorting by last user message can also move older sessions ahead of newer
  sessions after late edits or replayed messages.

Expected behavior:

- Session list ordering uses session `created_at`.
- Session metadata remains metadata and does not need to be repurposed as the
  conversation ordering source.

Regression target:

- A session with later user or metadata events does not sort above a newer
  session.

### 12. Codex provider complete path must stream

Status: implemented, keep regression coverage.

Symptom:

- The Codex-compatible provider can reject non-streaming complete requests with
  `Stream must be set to true`.

Expected behavior:

- `OpenAICodex.complete()` consumes the streaming API internally and returns a
  complete response.

Regression target:

- The Codex complete path calls the client with streaming enabled.

### 13. Compressed context is not durable session state

Status: implemented. See `spec/context_checkpoint_persistence.md`.

Symptom:

- Runtime compression updates the active run context and the live app history
  after a successful turn.
- Session persistence stores transcript events, not compressed LLM history.
- Re-attaching or restarting a session rebuilds history from the full event log
  and can lose the compressed context.

Expected behavior:

- Compression emits a durable context checkpoint event.
- Session replay starts from the latest checkpoint and appends later events.
- Full transcript/export remains intact.

Regression target:

- A resumed session with a checkpoint uses `checkpoint history + later events`,
  not all transcript events from the beginning.

## Repair Order

1. Implement child-agent handoff, audit envelope, artifact workspace, and
   bounded inspection tools. See `spec/subagent_result_artifact_plan.md`.
2. Re-run the parallel subagent scenario and validate TUI pagination, scrolling,
   hosted tools, retry rendering, and bounded parent context together.
3. Update older specs that still describe sequential tool execution as the
   current behavior.
