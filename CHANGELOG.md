# Changelog

## AceAI v0.2.5

### Features

- `cli`: Add `aceai export <session_id> --file=<path>` to write session exports to a new file without overwriting existing files.
- `agent`: Let default Ace agents run without a fixed step cap unless callers set `max_steps`.

### Improvements

- `sessions`: Decouple durable session storage from TUI display events with explicit session/TUI adapters.
- `cost`: Move usage-cost estimates out of the TUI layer so session recording and display can share the same app-layer cost model.
- `tui`: Batch small streaming text deltas before rendering to reduce full-transcript redraw pressure for long answers.
- `tools`: Report app tool filesystem, shell, timeout, and text-replacement failures as tool results the model can observe and recover from.

### Fixes

- `tui`: Preserve masked API keys when applying model/provider selection so switching no longer reports a missing key after showing one.
- `tui`: Require provider, model, and API key before applying model-selection changes.

### Breaking Changes

- `sessions`: Remove the storage-layer `SessionStore.load_tui_events()` and `messages_to_tui_events()` TUI helpers; TUI callers must use `aceai.agent.tui.session_adapter`.
- `core`: `AgentBase.max_steps` now defaults to unset/unlimited instead of `5`, and `build_ace_agent()` no longer sets an app-specific step cap.

## AceAI v0.2.4

### Features

- `deepseek`: Add DeepSeek as a supported provider using the OpenAI-compatible chat completions API, including streaming text, reasoning content, tool calls, and token usage.
- `models`: Move supported providers, default models, model lists, and token pricing into a provider catalog so app defaults can be maintained outside provider code.
- `tui`: Replace model selection with provider and model text inputs that support progressive autocomplete, keyboard selection, and provider-specific API key lookup.

### Improvements

- `tui`: Show masked API keys as `*****************xxxx` and automatically reuse stored provider keys without exposing secrets in the selector.
- `tui`: Render streamed reasoning before the answer content it belongs to, and keep reasoning visible for models that expose thinking output.
- `cost`: Restore live cost estimates for DeepSeek by consuming streamed usage chunks and centralizing model pricing through the provider catalog.
- `ci`: Add wheel build and install smoke tests, including the base install hint for users who launch the TUI without the optional `tui` dependencies.

### Fixes

- `cli`: Keep TUI dependencies optional and show the correct `aceai[tui]` installation hint instead of failing with a raw missing-module traceback.
- `tui`: Restore command-input cleanup after submitting a prompt and preserve API keys when switching between providers.

### Breaking Changes

- `config`: Replace the legacy top-level `api_key` config field with provider-scoped `api_keys`; users should store keys under `api_keys.<provider>`.

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
