# Changelog

## AceAI v0.2.3

### Features

- `tui`: Add token usage and estimated cost tracking to live runs, restored sessions, the status bar, and session browsing.
- `cli`: Add `aceai cost` for the total estimated cost across saved sessions, and allow `aceai resume` to reopen the latest updated session when no session id is provided.

### Improvements

- `sessions`: Persist assistant usage and cost metadata with compact session messages so restored transcripts keep their accounting context.
- `openai`: Capture cached input token usage from OpenAI Responses usage details.
- `tui`: Keep restored transcript events out of the execution timeline and improve stream rendering coverage for restored sessions and long content.

### Breaking Changes

- `cli`: Remove direct one-shot question execution from `aceai <question>`; launch `aceai` and submit the question inside the TUI instead.

## AceAI v0.2.0

### Features

- `skills`: Add filesystem skill discovery and loading through `Skill`, `SkillLoader`, and `SkillRegistry`, including support for `SKILL.md` frontmatter and bundled `references/`, `scripts/`, and `assets/` resources.
- `agent`: Register `skills_list` and `skill_view` tools on `ToolExecutor` so agents can discover matching skills and progressively load full instructions or supporting files during a run.
- `tools`: Expand built-in tooling and schemas used by skill-aware agent workflows.

### Improvements

- `skills-e2e`: Add deep progressive-disclosure e2e coverage based on a mature skill pattern, including multi-step loading of entry instructions, references, scripts, and assets.
- `docs-ci`: Restrict GitHub Pages deployment to `master`, allowing `version/**` branches to build docs without attempting protected Pages deployments.
- `release`: Add an `aceai-release` project skill documenting the release sequence, PR message format, category rules, tag timing, and PyPI publishing checks.
- `repo hygiene`: Ignore local plugin cache files and remove `.cache/plugin/social` from git tracking.

### Fixes

- `skills`: Validate malformed skill frontmatter and duplicate skill names with explicit configuration errors.
- `agent`: Keep skill tools scoped to tool-enabled agents and preserve existing executor behavior when no skills are configured.

### Breaking Changes

- None.

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
