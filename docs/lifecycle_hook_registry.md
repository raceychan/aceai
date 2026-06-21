# Lifecycle Hook Registry

## Status

Design, implementation, and migration notes for the typed lifecycle hook system
that replaced the old `before_llm_hooks` callable list.

## Problem

AceAI used to expose one narrow request-time extension point:

```python
before_llm_hooks: list[Callable[[str, int], Awaitable[list[LLMMessage]]]]
```

The run loop calls those hooks immediately before context preparation and
extends the live context with the returned messages. This is enough for simple
message injection, but it does not define ownership of the model request
boundary.

The missing contract matters for downstream runtimes such as Digpaw:

- run-specific product context such as session id, thread id, project id, and
  agent identity must be available to hooks without AceAI depending on those
  concepts;
- preview requests must produce the same model-visible input as real requests
  without committing side effects;
- request-time additions need trace metadata so a UI can explain what the model
  saw;
- multiple extensions must have deterministic ordering and merge rules;
- provider retry and context-window retry must not duplicate request additions;
- tool policy and approval state should not require downstream code to mutate
  private run state directly.

## Design Goals

- Keep the user-facing API light: users create a registry and decorate ordinary
  async functions.
- Keep runtime semantics explicit: hooks are awaited decision points, while
  events remain immutable observations.
- Preserve type safety for downstream context with generics.
- Avoid global hook discovery and import-time side effects.
- Normalize hook registrations once into an immutable execution plan.
- Make preview and execute modes first-class.
- Replace `before_llm_hooks` without forcing a broad middleware model.

## Non-Goals

- No generic middleware `wrap(next)` model.
- No global hook scanning.
- No direct mutation of `ContextManager` from hooks.
- No provider-specific hook behavior.
- No stream-transform API.
- No hook-controlled provider retry, provider switching, or control-flow rewrite
  in this lifecycle layer.
- No permanent backward-compatibility layer for `before_llm_hooks`.

## Migration Compatibility Policy

The `before_llm_hooks` bridge was allowed only as a short feasibility and
migration aid. It is not part of the target architecture and has been removed
from the current implementation.

Final-state requirements:

- AceAI does not expose `AgentRunContext.before_llm_hooks`.
- AceAI does not keep a legacy bridge that converts old hook returns into
  `ModelRequestPatch(append_messages=...)`.
- Digpaw registers `HookRegistry` hooks directly through `ContextBuilder`.
- Downstream code must not depend on the old callable signature
  `Callable[[str, int], Awaitable[list[LLMMessage]]]`.
- No feature flag, adapter, or compatibility wrapper should reintroduce the old
  behavior.

The removal phase is intentionally strict because this mechanism defines the
model request ownership boundary. Keeping both paths would make trace, preview,
retry, and side-effect semantics ambiguous again.

## Public API

Users construct a registry and pass it to `Agent` or a specific run.

```python
from dataclasses import dataclass

from aceai.core.hooks import (
    HookEffect,
    HookContext,
    HookRegistry,
    HookTraceSlot,
    ModelRequestDraft,
    ModelRequestPatch,
)


@dataclass(frozen=True)
class DigpawRunContext:
    session_id: str
    thread_id: str
    agent_id: str
    source: str


hooks = HookRegistry[DigpawRunContext]()


@hooks.before_model_request(name="digpaw.inbox", order=20)
async def add_inbox_context(
    ctx: HookContext[DigpawRunContext],
    draft: ModelRequestDraft,
) -> ModelRequestPatch | None:
    if ctx.data is None:
        return None
    items = await inbox.pending_items(thread_id=ctx.data.thread_id)
    selected = tuple(items[:8])
    if not selected:
        return None
    return ModelRequestPatch(
        append_messages=(render_agent_inbox_message(selected),),
        trace=(
            HookTraceSlot(
                slot="inbox",
                source="agent inbox",
                detail=f"{len(selected)} pending item(s)",
            ),
        ),
        effects=tuple(
            HookEffect(
                kind="digpaw.inbox.delivered",
                payload={"event_id": item.event_id},
            )
            for item in selected
        ),
    )
```

Agent-level hooks apply to all runs created by the agent:

```python
agent = Agent(..., hook_registry=hooks)
```

Run-level hooks and run context can specialize a single run:

```python
run = agent.create_resume_run(
    question,
    history,
    hook_context=DigpawRunContext(
        session_id=session_id,
        thread_id=thread_id,
        agent_id=agent_id,
        source="current user turn",
    ),
    hook_registry=runtime_hooks,
)
```

## Hook Registry

The registry is mutable during application composition and compiled into an
immutable `HookPlan` before execution. `Agent` accepts only `HookRegistry`; it
does not accept precompiled plans, raw callables, or a broad union type.

```python
class HookRegistry(Generic[TContext]):
    def before_model_request(
        self,
        fn: BeforeModelRequestHook[TContext] | None = None,
        *,
        name: str | None = None,
        order: int = 0,
        timeout_seconds: float | None = None,
    ) -> (
        BeforeModelRequestHook[TContext]
        | Callable[[BeforeModelRequestHook[TContext]], BeforeModelRequestHook[TContext]]
    ):
        ...

    def build_plan(self) -> HookPlan[TContext]:
        ...
```

Supported decorator forms:

```python
@hooks.before_model_request
async def simple(ctx, draft): ...


@hooks.before_model_request(name="memory", order=10)
async def named(ctx, draft): ...
```

The decorator must return the original function object, not a wrapper. The
registry records point-specific specs:

```python
BeforeModelRequestHookSpec(
    name="digpaw.inbox",
    order=20,
    registration_index=0,
    fn=add_inbox_context,
    timeout_seconds=None,
)
```

## Protocols

Each hook point has its own Protocol so type checkers can catch wrong
signatures.

```python
class BeforeModelRequestHook(Protocol[TContext]):
    async def __call__(
        self,
        ctx: HookContext[TContext],
        draft: ModelRequestDraft,
    ) -> ModelRequestPatch | None: ...


class BeforeModelCallHook(Protocol[TContext]):
    async def __call__(
        self,
        ctx: HookContext[TContext],
        request: PreparedModelRequest,
    ) -> ModelRequestPatch | None: ...
```

Use `before_model_request` when a hook needs the base draft before context
preparation. Use `before_model_call` when a hook must inspect the final prepared
request after context preparation and compression, immediately before the
provider call.

Implemented protocols:

- `BeforeModelRequestHook`
- `BeforeModelCallHook`
- `ModelRequestCommittedHook`
- `AfterModelResponseHook`
- `ModelErrorHook`
- `BeforeToolExecuteHook`
- `AfterToolExecuteHook`

Run-level start/end hooks are intentionally not part of the current system. The
first stable surface focuses on boundaries where downstream code needs to affect
or observe model-visible behavior: model request assembly, final model request
inspection, model request commit, tool invocation, completed model response, and
model-call failure.

## Code Ownership

The lifecycle system is split across four explicit owners:

- `aceai.core.hooks` owns public contracts and registration only:
  `HookRegistry`, point-specific hook specs, `HookPlan`, hook protocols,
  request patch data, prepared request data, and hook error types.
- `aceai.core.hook_execution` owns internal hook invocation:
  awaiting hook functions, applying hook timeouts, validating hook return values,
  and wrapping hook failures with hook-point context.
- `aceai.core.model_request` owns model request assembly:
  tool selection, request hook execution, patch merge rules, model-visible
  context mutation, preview preparation, and commit hook invocation.
- `aceai.core.run_loop` owns run/step orchestration and provider streaming. It
  asks `model_request` to assemble one logical request before provider attempts,
  invokes tool/response/error hooks at their lifecycle boundaries, and does not
  know how request patches are merged.

This prevents the registry from becoming a dynamic dispatcher and keeps the
public entry point unambiguous: downstream code registers hooks in
`HookRegistry`, while AceAI compiles that registry into an internal `HookPlan`.

## Boundary Diagnosis

Caller capability matrix:

| Caller | Capabilities needed | Boundary decision |
| --- | --- | --- |
| Hook authors | Register typed async functions, receive run/request facts, return typed patches | Use `HookRegistry`; do not expose raw `HookPlan` mutation or run-loop internals. |
| `Agent.create_run()` / `create_resume_run()` | Combine agent-level and run-level hooks, attach opaque downstream context | Compile registries into one immutable plan at run creation. |
| `Agent.prepare_model_request()` | Produce preview input without committing side effects | Call model request assembly in `preview` mode on a copied context. |
| Run loop provider path | Assemble one logical model request, avoid duplicate request hooks on context-window retry, commit only real requests | Delegate request assembly and commit hooks to `aceai.core.model_request`. |
| Run loop tool path | Refresh approval state and observe tool outcomes at the exact tool boundary | Execute tool hooks inside `aceai.core.run_loop`, after tool resolution and before/after execution. |
| Digpaw runtime orchestrators | Compose context hooks and tool-access hooks for one thread/run | Build a run-level `HookRegistry` explicitly and pass it to AceAI; no default registry fallback. |
| Digpaw `ContextBuilder` | Render late context, describe trace slots, commit inbox delivery effects | Own Digpaw-specific hook effects and strict effect payload decoding. |

Diagnosis:

- `hooks.py` is allowed to be repetitive because the repetition preserves
  point-specific Protocol and decorator types. Collapsing it into a generic
  register helper would hide the public API boundary again.
- `hook_execution.py` is a separate owner because it is the user-code failure
  boundary: it awaits hooks, applies timeouts, validates return values, and wraps
  failures with hook-point context.
- `model_request.py` is the request assembly owner because it is the only layer
  that should merge messages, metadata, tool filters, trace, effects, preview,
  and committed request data.
- Downstream integrations should assemble the registry before run creation. A
  callee that silently creates missing hooks is a fallback path, not a lifecycle
  contract.

## Runtime Data Model

### HookContext

```python
@dataclass(frozen=True)
class HookContext(Generic[TContext]):
    mode: Literal["preview", "execute"]
    run_id: str
    step_id: str
    step_index: int
    agent_id: str
    data: TContext | None = None
```

`data` is downstream-owned and opaque to AceAI. Digpaw can use it for session
and thread identity without leaking Digpaw concepts into AceAI.

### ModelRequestDraft

The draft is the planning input visible to hooks. It is immutable and should be
treated as read-only by hook authors.

```python
@dataclass(frozen=True)
class ModelRequestDraft:
    messages: tuple[LLMMessage, ...]
    tools: tuple[LLMToolSpec, ...] = ()
    metadata: LLMRequestMeta = field(default_factory=empty_request_meta)
```

### ModelRequestPatch

Hooks return patches instead of mutating runtime state.

```python
@dataclass(frozen=True)
class ModelRequestPatch:
    prepend_messages: tuple[LLMMessage, ...] = ()
    append_messages: tuple[LLMMessage, ...] = ()
    tool_allowlist: frozenset[str] | None = None
    tool_denylist: frozenset[str] = field(default_factory=frozenset)
    metadata: Mapping[str, object] = field(default_factory=dict)
    trace: tuple[HookTraceSlot, ...] = ()
    effects: tuple[HookEffect, ...] = ()
```

### PreparedModelRequest

The prepared request is the frozen request AceAI will send to the provider.

```python
@dataclass(frozen=True)
class PreparedModelRequest:
    request_id: str
    attempt_id: str
    run_id: str
    step_id: str
    step_index: int
    messages: tuple[LLMMessage, ...]
    tools: tuple[LLMToolSpec, ...]
    metadata: LLMRequestMeta
    patches: tuple[AppliedHookPatch, ...] = ()
```

### ToolExecutionRequest And ToolExecutionPatch

`before_tool_execute` hooks receive the resolved tool invocation and may return a
small patch for tool execution state. The current patch surface is intentionally
narrow:

```python
@dataclass(frozen=True)
class ToolExecutionRequest:
    call: LLMToolCall
    tool_name: str
    approval_required: bool
    approved_tool_names: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class ToolExecutionPatch:
    approved_tool_names: frozenset[str] | None = None
```

This lets downstream runtimes supply remembered approvals without mutating
private run state before the tool boundary is reached.

`after_tool_execute` hooks receive:

```python
@dataclass(frozen=True)
class ToolExecutionOutcome:
    call: LLMToolCall
    tool_name: str
    result: ToolExecutionResult
```

### ModelError

`on_model_error` hooks receive a structured failure object:

```python
@dataclass(frozen=True)
class ModelError:
    error: BaseException
    request: PreparedModelRequest | None = None
    request_committed: bool = False
```

`request` is `None` when request preparation failed before a provider attempt was
created. `request_committed` tells downstream code whether request commit hooks
already ran.

## Execution Semantics

The model request boundary should be assembled once per logical step before
provider attempts:

```text
build base draft from ContextManager and Executor
run before_model_request hooks
apply patches to a temporary request assembly
prepare/compress context
freeze PreparedModelRequest
run before_model_call hooks
apply final request patches
call provider
when provider stream yields its first event, commit request effects
on completed response, run after_model_response hooks
on provider/preparation failure, run on_model_error hooks
```

The important distinction is:

- `request_id`: one logical model request for a step;
- `attempt_id`: one provider attempt or context-window retry attempt.

`before_model_request` runs once per `request_id` by default and builds the
logical model request. `before_model_call` runs for each prepared provider
attempt, after compression has produced the final message list. Context-window
retry may create a new `attempt_id`; it must not duplicate logical request
additions from `before_model_request`, but it may rerun `before_model_call`
against the newly prepared final request.

Tool hooks run per resolved tool invocation:

```text
resolve LLM tool call to an executor invocation
run before_tool_execute hooks
perform approval check
execute tool or suspend for approval
append the tool result to the current step
run after_tool_execute hooks
emit tool completion/failure event
```

`before_tool_execute` is allowed to patch the approved tool-name set because that
state is part of the tool execution boundary. It cannot replace the resolved tool
call, change tool arguments, or skip execution.

`after_model_response` and `on_model_error` are observation/validation hook
points. They are awaited and may fail the run by raising, but they do not return
retry or provider-switch decisions.

## Preview Semantics

AceAI exposes a preview API:

```python
prepared = await agent.prepare_model_request(run)
```

Preview runs the same model request hooks with `HookContext.mode == "preview"`
and returns a `PreparedModelRequest`, but it does not commit hook effects or
alter the live run context.

The public API intentionally exposes only preview mode. Execute mode is an
internal run-loop boundary because executing request hooks may mutate the live
context and must stay tied to the real provider request lifecycle.

This replaces downstream hand-built preview paths that must currently duplicate
before-LLM logic.

## Effect Commit Boundary

Hooks may return `HookEffect` records to describe side effects that should happen
only after AceAI commits to the real request.

Effect payloads remain opaque:

```python
@dataclass(frozen=True)
class HookEffect:
    kind: str
    payload: Mapping[str, object] = field(default_factory=dict)
```

AceAI does not execute arbitrary effects. It passes the prepared request to
`on_model_request_committed` hooks. Downstream hooks inspect effects they own and
commit them there. This keeps side effects out of preview and out of failed
preparation paths.

## Merge Rules

- Hooks run in ascending `(order, registration_index)`.
- Duplicate names in one frozen plan are rejected.
- `append_messages` are appended in hook order.
- `prepend_messages` are prepended in hook order before base messages.
- Tool denylist wins over allowlist.
- Multiple allowlists intersect.
- Metadata is shallow-merged in hook order; duplicate keys are recorded in trace.
- Returning `None` is equivalent to an empty patch.
- Hook exceptions fail the run.

## Error Model

Registration-time errors:

- non-async hook function;
- duplicate hook name;
- invalid hook point;
- invalid timeout value.

Execution-time errors:

- hook timeout;
- exception raised by hook;
- invalid patch result.

Errors should be surfaced as AceAI runtime errors with the hook point and hook
name included.

## Events and Observability

Hooks are decision points. Agent events remain immutable observations.

AceAI exposes request observability through `PreparedModelRequest`: preview mode
returns it directly, execute mode passes it to `on_model_request_committed`, and
every applied hook patch carries trace/effect summaries. Downstream runtimes can
render those summaries without subscribing to private hook state or copying full
prompt payloads into ordinary transcript events.

## Downstream Mapping: Digpaw

Previous Digpaw integration points:

- `ContextBuilder.configure_run()` appends a callable to
  `run.before_llm_hooks`.
- `ContextBuilder.before_llm_messages()` builds inbox context, commits delivered
  writes, records traces, and returns messages.
- `MessageResponder.context_snapshot()` manually builds a preview by calling
  `build_before_llm()`.
- `ToolAccess.configure_run()` mutates `run.run_state.tools.approved_tool_names`.

Current HookRegistry integration:

- `ContextBuilder.model_request_hooks()` creates a run-level `HookRegistry`.
- Its `before_model_request` hook calls `build_before_llm()` and returns inbox
  messages, trace slots, and opaque delivery effects.
- Its `before_model_call` hook may add final, transient model-request hints after
  AceAI prepares the provider-facing request.
- Its `on_model_request_committed` hook records delivered inbox items and records
  the `before_llm` trace only after AceAI has committed to a real provider
  request.
- `MessageResponder.context_snapshot()` uses `Agent.prepare_model_request()` and
  `ContextBuilder.model_request_trace_context()` so preview runs the same
  `before_model_request` hooks in preview mode without committing effects.
- `ToolAccess.register_hooks()` registers `before_tool_execute` and supplies the
  remembered approved-tool set as `ToolExecutionPatch`, so Digpaw no longer
  mutates a live run's tool approval state directly.

## Refactor Yield Assessment

This migration was useful because it replaced a real downstream extension point,
not only a framework-local abstraction.

What was consolidated:

- AceAI no longer stores a mutable `before_llm_hooks` list on
  `AgentRunContext`.
- AceAI no longer has a legacy callable bridge in `model_request.py`.
- Digpaw no longer configures context injection by mutating a run after creation.
  The hook registry is attached while `ContextBuilder.build_resume_run()` creates
  the run.
- Digpaw inbox delivery side effects moved from "rendering context" to
  `on_model_request_committed`, so preview does not consume inbox items and
  provider-preparation failures do not falsely mark items delivered.
- Digpaw tests now assert `hook_plan` instead of the old list and model the
  committed boundary explicitly.
- Digpaw context snapshot now uses AceAI preview preparation instead of a
  hand-built duplicate before-LLM path.
- Digpaw remembered tool approvals now enter through `before_tool_execute`
  instead of `ToolAccess` reaching into active run state.

Positive signals:

- The downstream API shape is small: Digpaw only needed a registry factory and
  two decorated async functions.
- The model request boundary now owns message patching, trace summaries, metadata
  merge, tool filtering, preview mode, and committed effects in one place.
- The tool execution boundary now owns approval-state refresh, so approval memory
  is applied at the same point as the real tool policy check.
- Removing the old list made ambiguous ordering and duplicate injection paths
  disappear immediately.
- The event order became more honest: `step_started` is emitted before the model
  request commits, and `agent_inbox_delivered` is recorded only after commit.

Costs and remaining edges:

- Fake agents in downstream tests need a small hook-plan runner. That is a real
  cost, but it also makes tests assert the same lifecycle shape as AceAI.
- `ContextBuilder.model_request_hooks()` keeps commit state out of closures:
  `on_model_request_committed` reconstructs Digpaw writes and trace from
  `PreparedModelRequest.patches`, which makes the committed boundary explicit.
- Digpaw still pins AceAI through a git dependency, so cross-repo verification
  must run with local AceAI on `PYTHONPATH` until the AceAI branch is published or
  Digpaw updates its dependency.
- `after_model_response` and `on_model_error` deliberately stop at lifecycle
  observation/validation. Hook-controlled retry and provider switching are not
  part of this system.

Verdict:

The hook system has positive value once a real downstream integration migrates to
it. If AceAI had kept both the new registry and the old callable list, the result
would have been negative because there would be two model-request ownership
paths. After strict migration, the extra types pay for themselves by separating
preview, patching, trace, and durable side effects at the correct boundary.

## Feasibility Review

### AceAI

- `Agent.create_run()` and `Agent.create_resume_run()` are the correct place to
  attach per-run hook context and optional run-level hook registries.
- `AgentRunContext` already carries mutable run state, request metadata, and
  context manager, so it can carry a frozen hook plan without provider changes.
- `aceai.core.model_request` is the correct model request assembly boundary; it
  owns tool selection, hook execution, patch merging, preview preparation,
  prepared request construction, and final request patching.
- `_call_llm()` is the correct provider-attempt boundary; it owns context
  compression, provider stream consumption, and retry behavior.
- The previous placement of legacy `before_llm_hooks` execution inside the retry
  loop was a correctness risk because context-window retry could repeat hook
  output. New model request assembly happens outside repeated provider attempts.
- `LLMService.stream()` is provider passthrough and should not receive hook
  logic.

### Digpaw

- Digpaw already centralizes run configuration in `MessageResponder` and
  `SubagentDelegator`, so migrating to run-level hook context is feasible.
- Digpaw preview now uses AceAI's preview API, so context snapshot and real
  request assembly share the same hook path.
- Digpaw needs opaque downstream context and opaque effects because AceAI cannot
  own session event semantics.

## Phased Goal

### Phase 1: Replace Request-Time Hooks

- Done: add `aceai.core.hooks` with `HookRegistry`, `HookPlan`, protocols,
  `HookContext`, `ModelRequestDraft`, `ModelRequestPatch`,
  `PreparedModelRequest`, trace/effect types, and hook errors.
- Done: add `hook_registry` and `hook_context` support to `Agent` and
  `AgentRunContext`.
- Done: replace `before_llm_hooks` execution with the new hook plan and remove
  the old field.
- Done: add `Agent.prepare_model_request(run)`.
- Done: ensure request hooks run once per logical request and do not duplicate on
  context-window retry.
- Done: add focused tests for decorator registration, ordering, duplicate names,
  preview no-mutation, execute mutation, commit-boundary behavior, and retry
  non-duplication.

### Phase 2: Strict Downstream Migration and Legacy Removal

- Done: migrate Digpaw `ContextBuilder.configure_run()` to
  `ContextBuilder.model_request_hooks()` returning `HookRegistry`.
- Done: remove all Digpaw usage of `run.before_llm_hooks`.
- Done: remove `AgentRunContext.before_llm_hooks`, `LegacyBeforeLLMHook`,
  `apply_legacy_before_llm_hooks()`, and related legacy bridge code from AceAI.
- Done: add a regression test that rejects `before_llm_hooks` on
  `AgentRunContext`.
- Done: change Digpaw context snapshot to use AceAI preview and convert
  `PreparedModelRequest.patches` back into `ContextTrace` for UI/debug preview.
- Done: add source-search regression guards so AceAI core and Digpaw runtime do
  not reintroduce `before_llm_hooks`.

### Phase 3: Tool Boundary Hooks

- Done: add `before_tool_execute` and `after_tool_execute` to
  `HookRegistry`, `HookPlan`, and run-loop execution.
- Done: add typed `ToolExecutionRequest`, `ToolExecutionPatch`, and
  `ToolExecutionOutcome` contracts.
- Done: move Digpaw remembered tool approvals out of direct run-state mutation
  and into `ToolAccess.register_hooks()`.
- Done: add AceAI tests covering approval-state patching and after-tool outcome
  delivery.

### Phase 4: Response/Error Hooks

- Done: add `after_model_response` and `on_model_error` to `HookRegistry`,
  `HookPlan`, and run-loop execution.
- Done: add typed `ModelError` with optional prepared request and committed
  status.
- Done: add AceAI tests covering completed-response observation and provider
  error observation.
- Explicit non-goal: retry decisions, provider switching, and stream
  transformation are not part of this hook system.

### Phase 5: Final Model Request Hooks

- Done: add `before_model_call` to `HookRegistry`, `HookPlan`, and model request
  finalization.
- Done: run final model-call hooks after context preparation and before provider
  streaming.
- Done: preserve preview semantics by applying final request hooks in preview
  mode without committing effects.
