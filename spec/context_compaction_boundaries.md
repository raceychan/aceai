# Context Compaction Boundaries

Status: partially implemented.

Implementation status as of 2026-05-09:

- Implemented semantic context units in `aceai/core/context_manager.py`:
  `PriorRunSummary`, `CurrentRunSummary`, `RunUnit`, `StepUnit`,
  `OpenStepUnit`, and `StructuredContext`.
- Implemented run-level compaction for completed prior runs into a scoped
  `prior_runs` summary.
- Implemented current-run completed-step compaction into a scoped
  `current_run` summary while retaining recent completed steps as whole units.
- Implemented tool-call step validation so an assistant tool-call message and
  its matching tool outputs are treated as one atomic step.
- Implemented context-window retry compaction in `aceai/core/run_loop.py` so a
  long active run can retry after summarizing older completed steps.
- Implemented preflight compaction before sending provider requests when AceAI's
  local budget estimate says the next request is too large.
- Implemented chunked summary generation so compaction does not send all
  compressible history to the summarizer in one oversized request.
- Implemented `ContextCompactionStartedEvent`,
  `ContextCompactionFailedEvent`, and `ContextCompressedEvent` as structured
  lifecycle events. The TUI now shows compaction in progress, successful
  summary output, and compaction failure reasons.
- Implemented checkpoint schema version `2` in
  `aceai/agent/memory/context_checkpoint_store.py`, storing structured `units` rather
  than a flat `history` payload. Old flat v1 checkpoints are ignored.
- Implemented model-facing `model_output` usage for tool results, including the
  subagent artifact handoff path, so full audit output does not have to enter
  future model context.
- Verified targeted compaction/TUI/app regression coverage with
  `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_agent_behavior.py tests/test_context_checkpoint.py tests/test_tui_state.py tests/test_tui_stream_rendering.py tests/test_agent_app.py -q`:
  `161 passed`.

Remaining work:

- Provider-aware token estimation is not implemented yet. The current budgeting
  path still uses AceAI's existing conservative estimator plus a safety margin.
- Oversized required-context detection is clear but still coarse. If required
  context such as the current user message, open step, tool schemas, or attached
  context cannot be reduced, AceAI reports that compaction has no eligible
  completed run/step units. A future pass should point to the exact oversized
  source.
- Checkpoints are structured at the storage boundary, but context construction
  still rebuilds effective `LLMMessage` history before `ContextManager` parses
  it into units. A future cleanup can carry structured units end-to-end through
  the app-layer context builder.

## Background

AceAI already has context summarization and persistent context checkpoints.
That mechanism solves one important problem: after a conversation grows large,
older model-input history can be summarized and persisted so resumed sessions do
not have to replay the entire transcript.

The remaining problem is boundary correctness.

OpenAI Responses API tool history is structured. A function/tool output is valid
only when the same request context also contains the corresponding assistant
tool call. If compaction keeps or drops individual messages without respecting
tool-call boundaries, AceAI can create invalid provider input such as a
`function_call_output` whose `call_id` no longer has a matching
`function_call`.

The recent failure mode was:

- subagent tool calls completed and entered model history;
- the main run exceeded the context window;
- context compression kept a recent message tail;
- the tail could cut through a tool exchange;
- the next provider request failed with "No tool call found for function call
  output".

That failure shows that "summarize old context plus keep the last N messages" is
the wrong abstraction. The issue is not that summarization or checkpoints are
missing. The issue is that compaction must use domain boundaries, not raw message
counts.

## Problem

AceAI currently has three related but different units:

- transcript events: user-visible session history and audit trail;
- LLM messages: provider-facing input records;
- agent execution structure: runs, steps, tool calls, and tool results.

Compaction currently operates too close to the LLM message layer. LLM messages
are a serialization detail, not the right semantic unit for retention.

Two distinct context-window scenarios must be handled:

1. Cross-run history is too large.
   Older completed user turns should be summarized. This is naturally a
   run-level problem.
2. The current run becomes too large before it finishes.
   A single run may contain many steps, large tool outputs, or delegated
   subagent results. Keeping the whole active run raw can still exceed the
   context window. This is naturally a step-level problem.

Run-level compaction alone is therefore insufficient. It protects old turn
boundaries, but it cannot save a long active run.

Message-level compaction is also insufficient. It can cut a tool exchange in
half and produce invalid provider input.

## Goals

- Compact context at semantic boundaries.
- Keep provider-facing tool history valid.
- Allow a long active run to continue by summarizing older completed steps.
- Preserve enough recent raw context for the model to continue the current task.
- Keep transcript/export/audit history complete and separate from model-input
  compaction.
- Fail fast when context history is structurally invalid instead of silently
  repairing it.
- Make configuration names describe semantic units, not implementation details
  such as "messages".

## Non-Goals

- Do not mutate or delete transcript events.
- Do not make TUI rendering responsible for context compaction.
- Do not replay child-agent internals directly into the parent context.
- Do not solve oversized single tool outputs only with compaction. If one tool
  result is too large by itself, it must be reduced before entering model-facing
  context, for example through artifact storage plus a concise `model_output`.
- Do not preserve backwards compatibility for bad internal APIs or misleading
  configuration names.

## Target Model

Context compaction should operate over structured units.

### Run Unit

A run begins with one user message and includes all assistant steps and tool
exchanges needed to answer that user message.

Completed prior runs can be summarized as run-level units.

### Step Unit

A step is one model response and the tool work caused by that response.

Examples:

```text
assistant text answer
```

```text
assistant tool_call(s)
tool output(s)
```

For provider validity, an assistant tool-call message and all matching tool
outputs are one atomic step unit. A compaction boundary must not split that unit.

Completed steps inside the current run can be summarized when the current run is
too large.

### Open Step Unit

The open step is the step currently being constructed or resumed.

It may include:

- a pending assistant tool call;
- tool outputs required for the next model request;
- the latest model-visible work that has not yet reached a stable completed
  step boundary.

The open step must be kept raw. It cannot be summarized or split during the
request that needs it.

### Summary Unit

A summary unit is a model-facing system message produced by compaction.

There may be separate summary scopes:

- prior-run summary;
- current-run completed-step summary.

Both summaries are context truth, not transcript truth.

## Proposed Context Shape

For a normal resumed run:

```text
system instructions
prior-run summary
current user message
current-run completed-step summary, if needed
recent completed steps in raw form
open step in raw form
```

For a new run after a checkpoint:

```text
system instructions
checkpoint summary/history
events after checkpoint replayed into structured units
new user message
```

For an active long run:

```text
system instructions
prior-run summary
current user message
summary of older completed steps in this run
recent completed steps in raw form
open step in raw form
```

## Compaction Rules

### Cross-Run Compaction

Completed prior runs are always represented through summary/checkpoint context
in normal model input. Raw prior runs are not retained by default and are not a
normal context-retention knob.

Rationale:

- checkpoint summaries already preserve cross-run continuity;
- raw context is usually more valuable inside the current active run;
- retaining old raw runs can consume context on stale detail;
- if the model needs exact old details, that should come from explicit
  retrieval/citation or session inspection, not accidental raw-history
  retention.

### Current-Run Step Compaction

The active run is not automatically protected as one raw unit. If it grows too
large, older completed steps inside the current run are eligible for
step-level summarization.

Candidate policy:

```text
recent_step_retention = adaptive token budget
```

Rationale:

- the available raw-step budget changes with model context window, current user
  input, summaries, skills, citations, and open-step size;
- older completed steps can be summarized into current-run progress;
- keeping steps, rather than messages, preserves tool-call validity.

Retention order:

1. Reserve space for system instructions, prior-run summary, current user
   message, current-run summary, and the open step.
2. Use the remaining raw-step budget for completed steps in reverse
   chronological order.
3. Keep only whole completed steps. Never include a partial step to fit a token
   budget.
4. Summarize completed steps that do not fit.

### Open-Step Protection

The open step is always kept raw.

If the open step alone exceeds the context window, compaction cannot solve the
problem. AceAI should fail with a clear error that points to oversized
model-facing tool output or an oversized pending exchange.

### Tool Output Size

Large tool outputs should be reduced before they enter LLM history.

For delegated subagents and other bulky tools, the tool result should separate:

- audit output: full result stored in artifacts/session data;
- model output: concise provider-facing summary and artifact references.

Context compaction should not be asked to summarize a huge tool output that has
already overwhelmed the active step.

## Configuration

Keep these concepts separate:

```text
compress_threshold
```

Controls when compaction is attempted.

```text
recent_step_budget
```

Controls how much of the remaining context window may be used for raw completed
steps inside the current run. The policy is adaptive: it keeps as many recent
whole steps as fit, rather than a fixed count.

Do not expose or retain `keep_recent_messages` as a core policy. Message count is
not a semantic boundary and should not guide compaction.

## Token Budgeting

Adaptive raw-step retention should be calculated from the actual provider
context window and the token cost of the structured context units being rendered.

AceAI should use a provider-aware token estimator:

- OpenAI-compatible providers should use `tiktoken` where available.
- Providers with official tokenizers should use their provider tokenizer.
- Providers without a tokenizer may use a conservative fallback estimator.

The fallback estimator is only a compatibility path. It should not be the
primary budget mechanism for OpenAI-compatible models.

Budgeting order:

1. Compute the model context window from the provider catalog.
2. Estimate required tokens for system instructions, tool schemas/provider tool
   configuration, prior-run summary, current user message, current-run summary,
   and the open step.
3. Subtract a safety margin for provider serialization overhead, reasoning
   configuration, multimodal metadata, and tokenizer mismatch.
4. Use the remaining budget for raw completed steps, starting from the newest
   completed step and moving backward.
5. Keep only whole steps. If a completed step does not fit, summarize it and
   all older completed steps.

If the required units plus safety margin exceed the context window, compaction
cannot recover. AceAI should fail with a clear error pointing to the oversized
open step, tool output, tool schema, or attached context source.

## Summary Scopes

Prior-run summaries and current-run summaries should be distinct context units.

Prior-run summary answers:

- What durable context from earlier user turns matters now?
- What decisions, constraints, and unresolved goals should carry forward?

Current-run summary answers:

- What has already happened in this active run?
- Which completed steps, tool results, and intermediate conclusions have been
  summarized?
- What remains relevant for the next step?

These scopes should not be merged. Keeping them separate lets AceAI update the
current-run working summary without rewriting or polluting the cross-run
checkpoint summary.

Suggested rendering shape:

```text
<aceai_context_summary scope="prior_runs">
...
</aceai_context_summary>

<aceai_context_summary scope="current_run">
...
</aceai_context_summary>
```

## Structural Validation

Context compaction should fail fast if model history cannot be parsed into valid
semantic units.

Invalid examples:

- assistant/tool messages before any user-message run boundary;
- tool output with no matching assistant tool call in the same step;
- tool output whose matching tool call appears in a different run;
- system messages inserted inside a run as ordinary history;
- an open step with partial provider state that cannot be represented as a valid
  step unit.

AceAI should not silently repair these cases. They indicate a bug in replay,
checkpoint construction, or provider adaptation.

## Checkpoint Interaction

Persistent checkpoints should store the effective model-input history after
compaction, but the compaction algorithm should know the unit boundaries that
produced that history.

Newly written checkpoints should store structured context units, not only a flat
list of `LLMMessage` records.

Checkpoint payloads should include enough structure to distinguish:

- prior-run summary;
- current-run summary;
- current user message;
- raw step units;
- open step unit when a context-window retry needs to preserve unfinished
  provider-facing state.

The `version` field is a schema version, not a checkpoint sequence number. Every
new checkpoint should use the structured checkpoint schema from the first write.

Suggested payload direction:

```text
version: 2
units:
  - type: prior_run_summary
  - type: current_run_summary
  - type: current_user_message
  - type: step
  - type: open_step
```

Do not implement compatibility for old flat checkpoint payloads. A checkpoint
without structured units is not a valid context checkpoint for this design. If
such a payload exists from an earlier development build, AceAI should ignore or
reject it and rebuild context from transcript/checkpointable sources instead of
normalizing it into the new model.

This keeps checkpoint restore from guessing run and step boundaries from
provider messages and avoids carrying the old flat-message abstraction forward.

`included_event_id` should point to an event that is fully represented by the
checkpoint. Prefer run or step boundaries where possible. If it points into the
middle of a run, the checkpoint must still preserve enough structured state to
rebuild the remaining current-run units without replay ambiguity.

## Session Replay Interaction

Session replay remains transcript-oriented.

`EventLog.replay_llm_history()` can produce provider-facing messages, but context
construction should not treat the resulting flat list as the compaction model.
The app-layer context builder should parse or build structured units before
compression.

This keeps the boundary clear:

- transcript replay answers "what happened?";
- context construction answers "what should the model see next?";
- compaction answers "which semantic units can be summarized now?".

## Implementation Direction

1. Introduce internal context units. Done in
   `aceai/core/context_manager.py`.

```text
ContextUnit
  PriorRunSummary
  CurrentRunSummary
  RunUnit
  StepUnit
  OpenStepUnit
```

2. Build units from session/checkpoint/current run state. Partially done:
   `ContextManager` parses effective model history into units, and v2
   checkpoints store units. Session replay still produces `LLMMessage` history
   before unit parsing.

3. Compact completed prior runs into a prior-run summary/checkpoint. Done.

4. Compact old completed steps in the current run into a current-run summary.
   Done.

5. Render units back to `LLMMessage` only at the provider boundary. Partially
   done inside `ContextManager`; app-layer session restore still uses
   `LLMMessage` history as the transfer representation.

6. Add provider-aware token estimation, using `tiktoken` for OpenAI-compatible
   providers where available. Not done.

7. Remove message-count retention from `ContextCompressionPolicy`. Done.
   `recent_step_budget` replaced message-count retention.

8. Add validation tests for malformed histories and provider-invalid tool
   sequences. Partially done for missing run boundaries, orphan tool outputs,
   current-run step compaction, structured checkpoint payloads, summary
   chunking, and compaction failure events.

9. Surface compaction progress and results in the TUI. Done. The stream shows
   `compact` while summarization is running, `compact` with the resulting
   summary after success, and `compact failed` when the summary request or
   required-context budget cannot be reduced.

## Test Plan

Required regression coverage:

- Cross-run compaction represents old completed runs through summary/checkpoint
  context, not raw prior-run retention. Covered.
- Current-run compaction summarizes older completed steps while preserving recent
  completed steps. Covered.
- The open step remains raw. Implemented by the unit parser, but needs a
  dedicated regression test.
- A tool-call step is never split between assistant tool call and tool output.
  Covered by current-run step compaction tests.
- Tool output without a same-step tool call fails fast. Covered.
- Assistant/tool history before a user boundary fails fast. Covered.
- Context-window retry on a long active run compacts completed steps and retries.
  Partially covered by retry and active-run compaction tests; add a combined
  long-active-run retry regression before treating this as complete.
- A single oversized required context fails with a clear compaction failure
  event when summarization itself exceeds the provider window. Covered at the
  event/TUI level; exact-source attribution is not done.
- Summary compaction requests are chunked so the summarizer is not called with
  the entire old context at once. Covered.
- TUI shows compaction start, compacted summary, and compaction failure.
  Covered.
- Checkpoint restore preserves enough structure to avoid orphan tool outputs.
  Partially covered by v2 structured-unit restore tests.
- New checkpoints are written as structured units rather than flat message
  history. Covered.
- OpenAI-compatible token budgeting uses a tokenizer-backed estimator. Not done.
- Subagent tool results use concise `model_output` in context while full audit
  output remains available through artifacts/session data. Covered by the
  subagent artifact implementation and session/model-output tests.
