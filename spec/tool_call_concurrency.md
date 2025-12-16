@spec Tool Call Concurrency

# Tool Call Concurrency

## Overview
- Enable the agent loop to execute all tool calls returned within a single LLM turn concurrently instead of serially awaiting each call.
- Preserve existing observability semantics (events, messages) while accepting that completion order becomes non-deterministic once concurrency is enabled.

## Goals
- Dispatch every `LLMToolCall` discovered in `aceai/agent.py:67-112` as an independent async task and allow overlap across I/O-bound tools.
- Emit `agent.tool.*` events that reflect actual execution timing without blocking later calls behind earlier ones.
- Preserve `LLMToolUseMessage` plumbing so each completed tool feeds its output back to the language model regardless of finish order.
- Surface all tool failures that occur in a batch and ensure a `ToolExecutionFailure` is raised after event emission.

## Non-Goals
- No retries, throttling, or serialized fallback paths.
- No modifications to `aceai.llm` provider APIs beyond passing along tool specs.
- No compatibility guarantees for downstream consumers that depended on sequential completion ordering.

## Current State
- `AgentBase._run_step` iterates `response.tool_calls` one-by-one, awaiting `ToolExecutor.execute_tool` before proceeding. This blocks long-running calls and prevents independent tools from overlapping.
- Failures raised by the first tool immediately abort execution of the remaining calls, leaving later tools unobserved.
- `ToolExecutor` (`aceai/executor.py:11-84`) is a simple adapter around `Tool.decode_params` → dependency resolution via `ididi.Graph` → tool invocation → `tool.encode_return`. It performs no scheduling and is oblivious to concurrency.
- Event emission order today is strictly deterministic: start event, run tool, completion event, step completion. This will change once completions race.

## Design
1. **Task Fan-out**
   - After building `assistant_msg` for the LLM response, collect `response.tool_calls` into a list of `ToolCallTask` records storing the call, an asyncio `Task`, and bookkeeping handles.
   - Emit `event_builder.tool_started` synchronously for each call (still in original order) before any concurrent work begins, so logs remain deterministic at the start boundary.

2. **Concurrent Execution**
   - Spawn every tool task with `asyncio.create_task(self.executor.execute_tool(call))`.
   - Await them via `asyncio.gather(*tasks, return_exceptions=True)` to accumulate completions without cancelling the batch prematurely.
   - (Optional) Provide a helper such as `ToolExecutor.execute_many(tool_calls)` that wraps the same gather logic but returns a structured container described below, letting the agent remain the sole event emitter.

   **Batch Result Container**
   - Introduce records purely for data transfer, not for event emission:
     ```python
     class ToolResultRecord:
         call: LLMToolCall
         output: str | None
         error: Exception | None
         finished_at: float

     class ToolBatchResult:
         results: list[ToolResultRecord]
         duration: float
     ```
   - `execute_many` would decode inputs, launch concurrent `execute_tool` tasks, and build a `ToolBatchResult` using `asyncio.gather(..., return_exceptions=True)` plus per-task timestamps.
   - `_run_step` would iterate the returned `results`, emitting `tool_completed`/`tool_failed` events and appending `LLMToolUseMessage`s according to each record. This keeps agent responsibilities explicit while still giving other call sites (telemetry, diagnostics) access to the batch outcome.

3. **Result Plumbing**
   - Iterate over task outcomes in completion order (e.g., by awaiting `asyncio.as_completed`) so we can emit `tool_completed`/`tool_failed` immediately as each tool resolves.
   - For successful calls:
     - Build `ToolExecutionResult(call=call, output=encoded_output)`.
     - Append to `step.tool_results`, emit `tool_completed`, and push an `LLMToolUseMessage` into `messages`.
     - If the tool name is `"final_answer"`, emit `run_completed` and decide whether to cancel remaining tasks or allow them to finish (see Risks). The spec leans toward continuing execution and letting the agent terminate after all tasks settle to simplify cleanup.
   - For exceptions:
     - Wrap the original error inside a `ToolExecutionFailure`.
     - Emit `tool_failed` with a `ToolExecutionResult(call=call, error=str(exc))`.
     - Record the failure so the agent can raise once the batch finishes.

4. **Failure Propagation**
   - After all tasks settle, inspect the recorded failures:
     - If none failed, emit `event_builder.step_completed(step=step)` and continue the loop.
     - If one or more failed, raise the first captured `ToolExecutionFailure` (or aggregate them if we introduce a new exception type). The run-level failure handling in `AgentBase.run` already emits `step_failed`/`run_failed`.

5. **Message Ordering**
   - Accept that tool-use messages will now reach the LLM in completion order, not request order.
   - Document this behavioral change in the README/design notes so downstream consumers update expectations.

6. **Executor Safety**
   - Audit tools and shared dependencies accessed through `ididi.Graph` for concurrency awareness. If mutable globals exist, add per-tool locks or require tool authors to manage their own synchronization.

## Testing Strategy
- Create deterministic async test doubles for tools (e.g., tasks with configurable delays) to assert that concurrent execution shortens wall-clock time versus sequential behavior.
- Add tests where multiple tools return and at least one fails; ensure events are emitted for every tool, `ToolExecutionResult` entries include both successes and failures, and the agent raises after the batch completes.
- Test that a `final_answer` tool short-circuits the run completion while still flushing any in-flight events/results as specified.
- Run the suite via `uv run pytest` per repo policy.

## Risks & Mitigations
- **Shared Mutable State**: Concurrent tools may step on each other if they touch global caches. Mitigate by documenting the new contract and, if needed, adding optional serialization flags on unsafe tools.
- **Event Ordering Expectations**: Consumers relying on sequential completion order may break. Communicate the change in the spec + changelog and suggest ordering by timestamps or call IDs instead.
- **Final Answer Races**: If one tool produces the final answer while others are mid-flight, we must ensure downstream consumers know additional tool outputs may follow. Consider cancelling remaining tasks once `final_answer` resolves to keep semantics clear.
- **Exception Storms**: Multiple failing tools will now raise together; ensure stack traces remain debuggable by storing the original exception on each `ToolExecutionFailure`.

## Open Questions
- Should we cancel remaining tool tasks after the first failure or final-answer success to save resources?
- Do we need a concurrency limiter to avoid overwhelming hosts when the LLM emits dozens of tool calls?
- Is it worthwhile to aggregate multiple failures into a dedicated exception type for richer error reporting?

## Milestones
1. Implement `_run_step` fan-out/fan-in logic with concurrent tasks and adapt event/message handling.
2. Add regression tests covering multi-tool success, mixed success/failure, and final-answer races.
3. Update docs (`spec/agent_layer.md`, README) to describe the new concurrent behavior and its implications for downstream consumers.
