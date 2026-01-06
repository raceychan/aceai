# Changelog

## AceAI v0.1.7

### Highlights
- `Tool` description precedence now prefers `tool(description=...)` over `func.__doc__` (also reflected in `tool_schema["description"]`).
- `AgentBase` prompt handling refactor: `prompt` seeds an `instructions` list, with `system_message` derived from accumulated instructions.

### Docs / DX
- Refactored agent tests for the `AgentBase` API changes and expanded tool description/schema coverage.

### Notes
- Breaking: `AgentBase.__init__` renamed `sys_prompt` -> `prompt`; max-step runtime error message changed to “Agent exceeded maximum steps: N without answering”.
- Tracing: `ToolExecutor` span attribute changed from `tool.arguments.raw_len` to `tool.arguments` (larger attribute payloads).
