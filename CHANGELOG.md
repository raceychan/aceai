# Changelog

## AceAI v0.1.9

### Features

1. support "self"  arg in tool

```python

@tool
def off_load_context(self: AgentBase, )
```

2. support tool tag for grouping tools
3. support max tool usage per run 
4. tool registry & dynamic tool description

## AceAI v0.1.8

### Highlights
- Release automation now infers the target version from the current `version/<x.y.z>` branch when `--version` is omitted, and `make release` no longer forces a placeholder `VERSION`.
- Agent surface simplified: `BufferedStreamingAgent` export removed, streaming deltas remain unbuffered and no longer hydrate `reasoning_log` by default.
- Agent instruction handling now deduplicates added instructions while keeping `system_message` in sync.

### Tooling
- `scripts/release.py` adds branch-based version parsing with validation and keeps `--version` optional; `make release` conditionally forwards the flag and avoids fake defaults.
- Added regression coverage for release version inference from branch names.

### Breaking
- `aceai.agent.BufferedStreamingAgent` is gone; importers must use `AgentBase`.
- `AgentBase.instructions` attribute was removed; consumers should rely on `add_instruction` + `system_message`.

## AceAI v0.1.7

### Highlights
- `Tool` description precedence now prefers `tool(description=...)` over `func.__doc__` (also reflected in `tool_schema["description"]`).
- `AgentBase` prompt handling refactor: `prompt` seeds an `instructions` list, with `system_message` derived from accumulated instructions.
- Tracing overhaul: `agent.run` → `agent.step` → `llm.*` / `tool.*` spans are now parented via explicit `trace_ctx`, avoiding async-generator context detach errors and root-span fragmentation.
- OpenAI Responses support now surfaces structured output `segments` (reasoning/tool/media) and richer `LLMStreamEvent` mapping (tool deltas + image generation).

### Tracing / Observability
- `trace_ctx` is now threaded through `AgentBase`, `LLMService`, provider adapters, and `ToolExecutor` so downstream spans consistently inherit the same trace.
- Adds Langfuse-friendly span attributes: `langfuse.trace.name`, `langfuse.trace.input`, `langfuse.trace.output`.

### Docs / DX
- Added regression coverage for trace parenting and async-generator cancellation edge cases.
- Refactored agent tests for the `AgentBase` API changes and expanded tool description/schema coverage.

### Notes
- Breaking: `AgentBase.__init__` renamed `sys_prompt` -> `prompt`; max-step runtime error message changed to “Agent exceeded maximum steps: N without answering”.
- Breaking: provider implementers and tool executors must accept/pass through `trace_ctx` to keep spans correctly chained.
- Tracing: `ToolExecutor` span attribute changed from `tool.arguments.raw_len` to `tool.arguments` (larger attribute payloads).
