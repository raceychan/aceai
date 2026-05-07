# Agent Run Architecture Review

## Purpose

AceAI's core agent API should separate an agent definition from a single run of
that agent. The current implementation moved the former `AgentRuntime` behavior
back into `Agent`, which removed a redundant public class but made
`Agent` responsible for too many things.

This document records the target split before implementation. The goal is not
to recreate the old `AgentRuntime` class under another name. The goal is to make
run state explicit, keep execution algorithms outside the agent definition, and
prepare the core model for multiple agents.

## Current Problem

`Agent` currently owns all of these responsibilities:

- agent definition: prompt, default model, LLM service, executor, compression
  policy;
- run construction: building a context from a new question or prior history;
- mutable run state: `run_id`, `question`, `context`, `steps`, `run_state`,
  trace context, and request metadata;
- run orchestration: step loop, max-step behavior, run tracing, run completion,
  and run failure;
- LLM stream handling: provider stream event mapping into agent events and
  `AgentStep` records;
- tool flow: tool selection, invocation resolution, approval suspension,
  approval resume, execution, rejection, and writing tool results back into
  context.

This is workable for a single interactive agent, but it creates two concrete
problems:

- A single `Agent` instance can only safely represent one active run because
  every call to `create_run()` or `create_resume_run()` overwrites its mutable
  run fields.
- Future multi-agent work has no stable place to store per-agent and per-run
  state independently from the shared agent definition.

## Target Boundary

### `Agent`

`Agent` should represent the definition of an agent, not a live run.

It should own:

- prompt and instruction management;
- default model metadata;
- provider/service references;
- executor and hosted-tool references;
- skill registry and skill tool registration;
- compression policy;
- public convenience APIs such as `run()`, `resume()`, and `ask()`.

It should not own mutable fields that describe one active run, such as
`run_id`, `steps`, pending approval, or the current `ContextManager`.

### `AgentRunContext`

Add a core run-state object, tentatively named `AgentRunContext`.

It should own:

- `agent_id`;
- `run_id`;
- user question or run input;
- `ContextManager`;
- completed `AgentStep` records;
- `AgentRunState`;
- trace context;
- request metadata;
- max-step label if the runner needs a stable display value.

This object should be state only. It should not contain the execution loop.

### Run Loop Module

Move orchestration into a core module such as `aceai/core/run_loop.py`.

It should provide functions or a small internal service that operate on
`Agent` plus `AgentRunContext`:

- create an event builder for each step;
- run the LLM/tool loop;
- handle max-step failure;
- emit run completion, suspension, and failure events;
- resume a suspended tool approval.

This module may be internal. Public users should still prefer
`Agent.run()` and `Agent.ask()`.

### LLM Flow

Move provider-stream handling into a focused helper, such as
`aceai/core/llm_flow.py`.

It should:

- select local and hosted tools;
- call `ILLMService.stream()`;
- convert `LLMStreamEvent` records into `AgentEvent` records;
- return or yield the completed `AgentStep`.

It should not know about app sessions, TUI state, or product workflows.

### Tool Flow

Move tool execution and approval mechanics into a focused helper, such as
`aceai/core/tool_flow.py`.

It should:

- resolve tool calls through `IExecutor`;
- emit tool start/completion/failure events;
- suspend the `AgentRunContext` for approval;
- resume an approved or rejected invocation;
- write completed or failed tool results back into the run context.

It should not own permission policy. Permission policy remains on the app/tool
metadata side; core only enforces the explicit approval state it is given.

## Public API Direction

Keep these APIs:

```python
async for event in agent.run("question"):
    ...

answer = await agent.ask("question")
```

Add or reshape these APIs:

```python
run_context = agent.create_run("question")
async for event in agent.execute(run_context):
    ...

async for event in agent.resume_approval(run_context, decision):
    ...
```

`create_run()` should return `AgentRunContext`, not `Agent`.

`execute()` and `resume_approval()` may live as `Agent` methods for
ergonomics, but they should delegate to the run-loop module and must require an
explicit `AgentRunContext`.

## Multi-Agent Requirements

AceAI currently assumes one agent in several layers. Multi-agent support should
not start with a large orchestration framework. It should start by removing
single-agent state assumptions.

### Agent Identity

Every agent should have a stable identity:

- `agent_id`: stable internal identifier;
- `agent_name`: optional display name;
- `agent_role`: optional app-layer description for manager/worker style setups.

`AgentEvent` should carry `agent_id`. `agent_name` can be resolved by the UI or
session layer when needed.

### Run Identity

`run_id` should remain the identifier for one user-input-to-final-answer run or
one delegated sub-run.

For multi-agent delegation, add parent linkage:

- `parent_run_id`;
- `parent_step_id` or `parent_tool_call_id` when a sub-agent was invoked from a
  specific manager step;
- optional `task_id` for grouping several sub-agent runs under one user turn.

### Session History

The current session replay model is a linear single-assistant transcript. That
is correct for the visible chat, but insufficient for multiple agents.

Split session state into:

- visible transcript: user-visible conversation and final assistant responses;
- per-agent run logs: full event streams grouped by `agent_id` and `run_id`;
- delegation summaries: worker-agent outputs recorded as structured results for
  the manager agent;
- optional per-agent scratch context, which should not automatically enter the
  visible transcript.

Do not blindly replay every worker event into every agent context.

### Active Runs

`AceAgentApp` currently has one `_agent` and one `_active_run`.

For multi-agent readiness, reshape this toward:

- `agents: dict[str, Agent]`;
- `active_runs: dict[str, AgentRunContext]`;
- approval lookup by `run_id` and `tool_call_id`;
- selected or focused run for the current TUI pane.

The first implementation can still enforce one active foreground run. The data
model should not require that forever.

### Tool Approval and Permissions

Approved tool names are currently tracked at the app/session level. Multi-agent
support needs explicit approval scope:

- per run;
- per agent;
- per session;
- per capability or tool name.

The default should be narrow. A tool approved for one sub-agent run should not
silently become approved for every agent.

### TUI Rendering

The TUI currently renders one active stream. Multi-agent support needs an event
model that can render at least:

- a manager run with nested delegated runs;
- multiple suspended approvals;
- per-agent labels in the timeline;
- a focused active run while background runs continue.

The first UI slice can keep a single visible main lane and show sub-agent runs
as collapsible child sections.

## Migration Plan

### Phase 1: Extract Run State

- Add `AgentRunContext`.
- Move `run_id`, `question`, `context`, `steps`, `run_state`, `trace_ctx`, and
  request metadata out of `Agent`.
- Change `create_run()` and `create_resume_run()` to return `AgentRunContext`.
- Keep `Agent.run()` and `Agent.ask()` working as convenience wrappers.

Acceptance criteria:

- A single `Agent` can create two run contexts without overwriting either.
- Existing single-agent tests still pass.
- Approval resume receives an explicit run context.

### Phase 2: Extract Run Loop

- Move the step loop, max-step behavior, run tracing, and resume flow into
  `aceai/core/run_loop.py`.
- Keep the module independent of app sessions and TUI rendering.
- Preserve the existing `AgentEvent` stream behavior.

Acceptance criteria:

- `Agent` no longer contains the main execution loop.
- Trace parenting tests still pass.
- Tool approval tests still pass.

### Phase 3: Extract LLM and Tool Flow

- Move LLM streaming conversion into `llm_flow.py`.
- Move tool invocation and approval mechanics into `tool_flow.py`.
- Keep core permission behavior strict and explicit.

Acceptance criteria:

- LLM stream mapping can be tested without constructing a full app.
- Tool approval/resume can be tested around `AgentRunContext`.
- `aceai/llm` remains provider-only and does not learn about app features.

### Phase 4: Add Agent Identity to Events

- Add `agent_id` to `AgentLifecycleEvent`.
- Stamp it from `AgentRunContext`.
- Update session recording and TUI state to preserve it.

Acceptance criteria:

- Single-agent runs emit a default `agent_id`.
- Session export remains readable.
- Event log replay remains deterministic.

### Phase 5: App-Layer Multi-Agent Shell

- Change `AceAgentApp` to hold an agent registry and active run registry.
- Keep first behavior serial if needed.
- Add manager-to-worker delegation as an app-layer tool or capability, not as
  provider behavior and not as TUI parsing.

Acceptance criteria:

- The app can run two different `Agent` definitions in one session.
- Approval requests identify the owning run and agent.
- TUI can show which agent produced each event.

## Non-Goals

- Do not reintroduce the old public `AgentRuntime` class.
- Do not add multi-agent scheduling before run state is separated.
- Do not put app-specific tools, filesystem behavior, browser behavior, or
  product workflow policy in `aceai/core`.
- Do not make the TUI infer agent structure by parsing assistant text.
- Do not replay every sub-agent token into the manager context by default.

## Open Questions

- Should `agent_id` be required at `Agent` construction time, or generated
  when an agent is registered with an app?
- Should `AgentRunContext` be a `msgspec.Struct` or a regular class? It holds
  mutable objects such as `ContextManager`, so a regular class may be simpler.
- Should the core expose `execute(agent, run_context)` as a function only, or
  keep `agent.execute(run_context)` as a public ergonomic wrapper?
- Should delegated sub-agent output appear to the manager as a tool result, a
  structured event, or both?
- How much background concurrency should the first TUI support?

## Implemented Slice

The first implementation slice covers Phase 1 and Phase 2.

It removes the current design pressure without committing AceAI to a large
multi-agent runtime:

- `Agent` is now the agent definition and convenience API.
- `AgentRunContext` is now the explicit mutable state for one run.
- `aceai/core/run_loop.py` now owns the run execution and approval-resume flow.

The remaining splits, especially dedicated `llm_flow.py` and `tool_flow.py`,
should be reviewed separately because they are mechanical but still affect
tracing, approval, and event-order guarantees.
