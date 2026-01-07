# Changelog

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
