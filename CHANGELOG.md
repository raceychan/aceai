# Changelog

## AceAI v0.2.15

### Features

- `tui`: Replace `/idea` list output with an interactive idea picker that supports keyboard selection, Enter-to-reference, `e` editing, and `d` deletion from the highlighted idea.

### Improvements

- `ideas`: List, search, edit, and delete ideas in FIFO order so the oldest saved idea remains first and picker numbering stays stable.
- `tui`: Render ideas as bordered panels in the picker while keeping creation timestamps aligned with terminal-cell-aware title truncation.

### Fixes

- `tui`: Intercept TextArea Enter handling before it inserts a newline so normal prompts submit, slash-command candidates complete, and completed slash commands execute.
- `tui`: Intercept slash-command up/down keys before TextArea cursor movement so command candidates can be selected with the keyboard.

### Breaking Changes

- `core`: Rename `AgentBase` to `Agent`; import from `aceai` or `aceai.core`.

## AceAI v0.2.14

### Features

- `tui`: Add a compact custom top bar with dedicated quit, config, debug, and clock controls while removing the default footer shortcut strip.
- `tui`: Add a Labrador pixel-art empty state in the main stream pane so new sessions start with a centered visual prompt instead of `No events yet`.
- `tui`: Add an interactive slash-command palette with one command per row, descriptions, keyboard navigation, and Enter/Tab selection.
- `tools`: Add per-tool enabled-state and maximum-call configuration so app tools can be disabled or capped independently from approval policy.

### Improvements

- `tui`: Simplify command names by keeping `/config`, `/sessions`, and `/stats` as the primary configuration, session, and usage commands.
- `tui`: Move runtime statistics behind `/stats` and clickable usage/cost chrome while keeping model display read-only in the status bar.
- `config`: Remove the System Prompt tab from the TUI config screen and focus configuration on provider, model, skills, and app tool controls.
- `tools`: Remove built-in max-call defaults from app tools so limits are explicit user configuration instead of hidden tool metadata.

### Fixes

- `tui`: Avoid startup layout artifacts where empty-state text or top-bar controls could collapse into unreadable vertical or ellipsized fragments.
- `tui`: Ensure Enter in slash-command mode completes the selected command instead of inserting a newline.

### Breaking Changes

- `config`: `tool_permissions` now accepts only `always` or `ask`; disable tools with `tool_enabled: {tool_name: false}` instead of `tool_permissions: {tool_name: never}`.

## AceAI v0.2.13

### Features

- `tui`: Add queued-turn workflow — type a new question while the agent is running and it automatically runs after the current turn completes, with a clickable queue widget above the input bar.
- `tui`: Add escape-to-cancel with a two-press arming pattern — first Escape arms the cancel, second Escape cancels the running turn, with a status-bar notice during the arm window.
- `tui`: Add `/steer <message>` command to cancel the active run in-place and replace it with a new prompt while preserving the queued turn backlog.

### Improvements

- `tui`: Render saved ideas as a formatted idea list with title, creation date, and body panel instead of a plain-text session notice.
- `tui`: Show `/idea` output as a dedicated `idea_list` transcript entry with styled group rendering instead of a single `session_notice` line.
- `agent`: Expose `cancel_active_turn`, `enqueue_turn`, `pop_queued_turn`, and `take_queued_turn` on `AceAgentApp` so TUI layer can manage active-run lifecycle with a typed contract.

### Fixes

- `tui`: Keep terminal run-control events (`run_completed`, `run_failed`, `run_suspended`) from being treated as restored-transcript events so they correctly update the run status in state transitions.

### Breaking Changes

- None.

## AceAI v0.2.12

### Features

- `ideas`: Add persistent idea capture with `IdeaStore` and `/idea` commands — agents and users can save, list, and delete ideas scoped to the current workspace, with ideas surfaced in the TUI transcript.
- `release`: Add project-local `aceai-release` skill with the release sequence, PR message template, category rules, tag timing, and PyPI publishing verification.

### Improvements

- `tui`: Add `/idea` command with `save`, `list`, and `delete` sub-commands, including idea capture from agent runs.
- `ci`: Restrict GitHub Pages deployment to `master` so version branches can build docs without attempting protected Pages deployments.
- `ci`: Add `v*` tag-push trigger for the PyPI publish job.

### Fixes

- `repo`: Remove `.cache/plugin/social` from git tracking and ignore local plugin cache files.

### Breaking Changes

- None.

## AceAI v0.2.11

### Improvements

- `llm`: Standardize provider cache accounting around cache hits divided by cache-accounted input tokens, with OpenAI and DeepSeek adapting their provider-specific usage fields into one AceAI-owned `LLMUsage` contract.
- `tui`: Move run status and model to the left side of the status bar, rename `ctx` to `context`, add cache-rate display, and render running/completed states with compact symbols.
- `tui`: Show session resume, model-switch, and provider-switch feedback as short-lived queued status-bar notices instead of persistent transcript events.
- `dev`: Add Ruff to the managed development dependency set.

### Fixes

- `sessions`: Migrate legacy usage payloads when loading saved events so older assistant messages derive cache miss and hit-rate fields from existing token data.
- `tui`: Keep replayed historical sessions from displaying a running status unless replay includes a terminal run-control event.
- `tui`: Queue multiple short-lived notices so consecutive resume/model-switch notifications display in order instead of overwriting each other.

### Breaking Changes

- None.

## AceAI v0.2.10

### Fixes

- `tui`: Keep collapsed work history directly below the user input and above the assistant answer after a run completes, including replay orders where assistant text is seen before the tool activity.

### Breaking Changes

- None.

## AceAI v0.2.9

### Fixes

- `tui`: Remove the `packaging` runtime import from the update checker so `aceai` can start in the published `uv tool` environment.
- `packaging`: Move `httpx` into runtime dependencies because `aceai.llm.service` imports it directly for provider transport retry handling.

### Breaking Changes

- None.

## AceAI v0.2.8

### Features

- `tui`: Add automatic update checks against PyPI on startup so users are notified when a newer AceAI release is available.
- `tui`: Add `/update` self-upgrade flow that runs `uv tool upgrade aceai` and restarts the current AceAI process after a successful upgrade.

### Improvements

- `llm`: Add exponential retry for retryable provider transport, timeout, rate-limit, and 5xx failures, with structured retry progress events for streaming runs.
- `tui`: Render LLM retry progress in the transcript so interrupted provider streams show visible recovery attempts instead of appearing stuck.
- `tui`: Improve session replay and tool work history rendering so restored sessions preserve more of the original execution timeline.
- `sessions`: Persist and replay LLM retry events so transient provider failures remain visible in saved transcripts.

### Fixes

- `llm`: Recover from incomplete streamed HTTP responses such as `RemoteProtocolError: peer closed connection without sending complete message body` by retrying the stream request.
- `llm`: Treat stalled streaming reads as retryable after a 3-second per-event timeout so network disconnects do not wait indefinitely for the next provider chunk.
- `tui`: Show a final "please try again later" assistant message after retry exhaustion instead of letting provider transport failures crash the TUI with a traceback.
- `tui`: Report self-update command failures in the transcript instead of silently failing when `uv` or the upgrade command cannot run.

### Breaking Changes

- `llm`: `LLMStreamEvent.event_type` now includes `response.retrying`; stream consumers with exhaustive event handling must handle or ignore this new event type.
- `core`: Agent event consumers may now receive `agent.llm.retrying` events during streaming runs.

## AceAI v0.2.7

### Features

- `config`: Add project-level `.aceai/config.yml` support with startup precedence over the global `~/.aceai/config.yaml` fallback.
- `permissions`: Add per-tool app permissions with `always`, `ask`, and `never` policies that control whether tools execute directly, require approval, or stay hidden from the model.
- `tui`: Add a dedicated Tools tab in the config screen with per-tool permission dropdowns plus `Allow all` and `Disable all` bulk actions.

### Improvements

- `config`: Persist config changes from the TUI back into the current project while preserving explicit path overrides for tests and tooling.
- `agent`: Apply configured tool permissions when building the default Ace agent without mutating the module-level tool objects.
- `docs`: Document project config precedence and the user-facing semantics of tool permission values.

### Fixes

- `tests`: Isolate CLI config tests from a developer machine's real saved AceAI config so local `~/.aceai/config.yaml` cannot change expected behavior.
- `packaging`: Keep the `aceai.agent` package import light enough for the base `aceai` command to show the optional TUI dependency hint instead of importing session storage dependencies too early.

### Breaking Changes

- `config`: `save_config()` without an explicit path now writes `.aceai/config.yml` in the current project instead of `~/.aceai/config.yaml`; callers that need global config writes must pass `default_config_path()` explicitly.
- `agent`: Remove session and app facade re-exports from `aceai.agent`; callers must import `AceAgentApp`, session models, event stores, and `SessionService` from their concrete modules.

## AceAI v0.2.6

### Features

- `core`: Add `AgentRuntime` with explicit runtime state, pending approval tracking, and resumable human-in-the-loop tool approval flow.
- `tools`: Add approval policy metadata for app tools, including filesystem write and shell command approval requirements.
- `tui`: Add inline approval controls for suspended tool calls with clickable `A Approve` / `R Reject` actions and keyboard shortcuts.
- `sessions`: Persist and export tool approval requested/resolved events so saved runs show the human-in-the-loop decision chain.

### Improvements

- `runtime`: Collapse executor-owned pending approval fields into core runtime state so tool approval, resume, and run status share one source of truth.
- `tui`: Route initial runtime execution and approval resume streams through one shared event consumer so suspended/completed handling cannot drift between paths.
- `tui`: Keep approval controls compact and near the input area while preserving readable transcript rendering.
- `sessions`: Mark unresolved restored approvals as expired in the transcript instead of showing non-actionable approval buttons.

### Fixes

- `sessions`: Omit unresolved assistant tool calls from replayed LLM history so resumed conversations no longer trigger provider errors about missing tool messages.
- `tui`: Show the next approval prompt when a resumed approval flow suspends again for a subsequent tool call.
- `tui`: Keep restored pending approval history visible without pretending old approvals can still be clicked.

### Breaking Changes

- `core`: Rename `AgentRun` to `AgentRuntime` and `AgentRunState` to `AgentRuntimeState`; callers must use the new runtime naming.
- `executor`: Replace executor-owned run state with `ToolRunState`; custom executors must accept the updated `tool_state` contract.

## AceAI v0.2.5

### Features

- `cli`: Add `aceai export <session_id> --file=<path>` to write session exports to a new file without overwriting existing files.
- `agent`: Let default Ace agents run without a fixed step cap unless callers set `max_steps`.
- `tui`: Add a metadata/config screen for runtime state, usage, loaded skills, local tools, and hosted tools via `i`, `/config`, `/metadata`, and `/info`.
- `sessions`: Persist per-session app state such as the selected provider and model so resumed sessions restore their runtime choice.

### Improvements

- `sessions`: Decouple durable session storage from TUI display events with explicit session/TUI adapters.
- `cost`: Move usage-cost estimates out of the TUI layer so session recording and display can share the same app-layer cost model.
- `tui`: Batch small streaming text deltas before rendering to reduce full-transcript redraw pressure for long answers.
- `tui`: Restyle the main transcript with full-width prompt bands, lighter assistant text, compact tool summaries, and a shorter right-aligned status bar.
- `tools`: Report app tool filesystem, shell, timeout, and text-replacement failures as tool results the model can observe and recover from.

### Fixes

- `tui`: Preserve masked API keys when applying model/provider selection so switching no longer reports a missing key after showing one.
- `tui`: Require provider, model, and API key before applying model-selection changes.
- `tui`: Keep metadata/config content scrollable while leaving the close action visible.

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
