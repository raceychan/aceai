# Subagent Result Artifact Tech Plan

Status: pending.

This plan splits the child-agent result bloat repair out of
`spec/subagent_runtime_repair_plan.md` so the remaining runtime fixes can be
tracked without carrying the full artifact design inline.

## Problem

A delegated child agent currently returns `final_answer`, `summary`,
`important_evidence`, and `tool_results` as one tool result payload. That single
payload serves three different consumers:

- parent LLM continuation;
- TUI/debug/export/audit;
- durable session replay.

Those consumers need different size and lifecycle rules. Today they are mixed,
so audit data is replayed into model context.

Confirmed current behavior:

- `delegate_to_subagent` sets `summary=final_answer`.
- `important_evidence` stores complete child
  `ToolCompletedEvent.tool_result.output` strings.
- `tool_results[].output` stores the same complete child tool outputs again.
- `ContextManager.add_tool_use()` stores
  `ToolCompletedEvent.tool_result.output` as the next parent
  `LLMToolUseMessage` content.
- When multiple children finish, the parent context receives all of their large
  payloads together.
- Compression may preserve recent tool messages, so compression alone cannot
  guarantee the next request fits the model context window.

## Decision

Keep child-agent audit data complete, but keep parent-model context small by
default.

The only child-agent material that should automatically enter parent context is
a bounded handoff summary plus stable artifact ids. Full child final answers,
tool calls, tool arguments, tool results, errors, and hosted-tool metadata are
returned to the app/UI layer through an audit envelope and stored in a
session-scoped artifact workspace.

The parent model can inspect full details only through bounded app-layer
inspection tools.

## Design Principles

- Recording is not the problem. Treating recorded audit data as parent-model
  input is the problem.
- Do not shrink or hide TUI details to solve model-context pressure.
- Do not rely on prompt wording alone. The runtime must enforce a bounded
  handoff.
- Prefer references over payloads in model context.
- Do not expose filesystem paths in `model_output` by default.

## Data Surfaces

### Parent Model Handoff: `model_output`

Enters the next parent LLM request.

Contains:

- bounded child-to-parent handoff summary;
- stable artifact ids;
- tool names, statuses, counts, and short evidence excerpts when useful.

Must not contain:

- complete tool outputs;
- full child transcript;
- large final answers;
- local filesystem paths;
- fields needed only for UI/audit.

### App/UI Audit Envelope: `output`

Returned to the app layer in the normal tool result event.

Drives:

- subagent panel;
- detail/raw views;
- exports;
- diagnostics.

Contains metadata plus artifact references. It must let the UI show complete
child final answer, tool calls, tool arguments, tool outputs, errors,
hosted-tool metadata, and run metadata by resolving artifact references.

### Artifact Workspace

Stores large child-agent artifacts outside parent LLM context. This lives in
the app/session layer, not `aceai/core`.

Default layout:

```text
~/.aceai/sessions/
  {session_id}/
    artifacts/
      {parent_run_id}/
        {child_agent_id}/
          manifest.json
          handoff.json
          final_answer.md
          transcript.jsonl
          tool-results/
            {artifact_id}/
              metadata.json
              arguments.json
              output.txt
              error.txt
```

Path rules:

- `~/.aceai/sessions/{session_id}/artifacts` is the app-layer default.
- Tests can inject a temporary artifact root.
- Production uses `~/.aceai/sessions/{session_id}/artifacts`.
- Artifact cleanup is session-owned: deleting a session must remove or mark
  `~/.aceai/sessions/{session_id}/artifacts` for cleanup.
- Do not use provider-controlled ids as raw path segments. Use AceAI-generated
  artifact ids for directories/files and store provider ids such as
  `tool_call_id` inside metadata.
- "Temporary workspace" means run/session-scoped workspace, not disposable OS
  temp.

File roles:

- `manifest.json` indexes artifacts for TUI, exports, diagnostics, and
  inspection tools.
- `handoff.json` stores the compact parent-facing result.
- `final_answer.md` and `tool-results/*/output.txt` may be large and must not
  be automatically loaded into parent context.
- `transcript.jsonl` is optional in the first slice if event replay already has
  enough data.

### Inspection Tools

App-layer tools let the parent model inspect artifacts on demand. Their outputs
must also be bounded because inspection tool outputs enter future model context.

Initial tools:

- `list_subagent_artifacts(run_id)` returns ids, labels, status, sizes, and
  short descriptions only.
- `read_subagent_handoff(artifact_id)` returns the compact handoff.
- `read_subagent_tool_result(artifact_id, tool_call_id, offset, limit)` returns
  a bounded slice of one tool result.
- `search_subagent_artifacts(run_id, query, limit)` searches artifact text and
  returns matching ids plus short snippets.

## Tool Result Model

Add `model_output` to the core tool result record:

```python
class ToolExecutionResult(Record, kw_only=True):
    call: LLMToolCall
    output: str
    model_output: str
    error: str | None = None
```

For ordinary tools, `model_output` can initially equal `output`.

For `delegate_to_subagent`, `model_output` is compact handoff JSON:

```json
{
  "type": "subagent_handoff",
  "agent_id": "child-id",
  "run_id": "run-id",
  "status": "completed",
  "task": "delegated task",
  "handoff": "bounded child summary for the parent model",
  "artifact_id": "subagent-artifact-id",
  "evidence": ["small curated evidence item"],
  "step_count": 2,
  "tool_result_count": 1,
  "tool_names": ["read_text_file"]
}
```

For `delegate_to_subagent`, `output` is an audit envelope:

```json
{
  "type": "subagent_audit",
  "agent_id": "child-id",
  "run_id": "run-id",
  "status": "completed",
  "task": "delegated task",
  "artifact_id": "subagent-artifact-id",
  "manifest_path": "artifacts/session-id/run-id/child-id/manifest.json",
  "handoff_path": "artifacts/session-id/run-id/child-id/handoff.json",
  "final_answer_path": "artifacts/session-id/run-id/child-id/final_answer.md",
  "tool_result_count": 1,
  "tool_names": ["read_text_file"]
}
```

The audit envelope may include relative artifact paths because it is consumed by
the app/UI layer. `model_output` should include artifact ids only.

## Child-Agent Result Shape

- Keep `final_answer` as the full child answer for humans and raw inspection.
- Stop making `summary` a blind alias of `final_answer`.
- Add `handoff` as the context-safe child-to-parent summary.
- Keep child tool calls, tool arguments, `important_evidence`, and
  `tool_results` available through the audit surface.
- Persist large bodies through artifact references.
- Do not copy complete tool outputs into `model_output`.

## Handoff Generation

First ask the child to produce a concise `Summary / Evidence / Risks / Next`
result as part of the child output contract.

Then build the parent handoff deterministically from the child result:

- enforce a maximum handoff size;
- cap evidence count;
- include tool names, statuses, and counts;
- include small excerpts only when needed;
- never include complete raw tool output by default.

A later implementation may add an LLM summarizer step for very large child
answers, but the first repair should not require another model call.

## Runtime Flow

1. `ContextManager.add_tool_use()` uses `event.tool_result.model_output` when
   creating the parent `LLMToolUseMessage`.
2. Session persistence stores both `output` and `model_output`.
3. Large subagent artifacts are stored in the artifact workspace and referenced
   from both `output` and `model_output`.
4. TUI state is built from the audit `output` envelope and artifact manifest,
   not from `model_output`.
5. TUI lazily loads large artifact bodies when rendering selected subagent
   detail.
6. Loading artifact bodies for UI display must not mutate effective LLM context.
7. Export/replay/debug views may expand artifact references when they need full
   details.
8. Session replay and checkpoint construction rebuild effective model context
   from `model_output`.
9. Deleting a session removes or marks its artifact tree for cleanup.

## Implementation Phases

1. Add `ToolExecutionResult.model_output` and make context construction use it.
2. Update session serialization, session replay, and context checkpoint
   rebuilding so effective LLM history uses `model_output`.
3. Add an app-layer subagent artifact store rooted at
   `~/.aceai/sessions/{session_id}/artifacts`, with tests able to inject a
   temporary root.
4. Change `delegate_to_subagent` to write full child artifacts, return an audit
   envelope as `output`, and return compact handoff JSON as `model_output`.
5. Update the subagent TUI panel to render from the audit envelope and lazily
   load artifact bodies.
6. Add bounded subagent inspection tools.
7. Add cleanup behavior when deleting sessions or pruning session artifacts.

## Regression Targets

- Three child agents with large internal outputs produce bounded parent-context
  tool results.
- The TUI can still display full child details.
- A replayed session uses bounded subagent handoff messages in rebuilt LLM
  history.
- Checkpoint histories do not embed raw subagent artifact bodies.
- Inspection tools return bounded slices/snippets.
- Deleting a session removes or marks its artifact tree for cleanup.
