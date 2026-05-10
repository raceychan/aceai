# AceAI Layer Boundary Review - 2026-05-09 (revalidated 2026-07-11)

## Purpose

This document records a read-only review of AceAI's current layer boundaries.
It is meant to be a repair backlog, not an implementation plan. The review
uses the layer contract in `AGENTS.md` and `spec/agent_layer.md` as the source
of truth.

Layer contract summary:

- `aceai/llm` adapts model providers into AceAI-owned request, response,
  streaming, and tool-spec contracts. It must not know about product UI, app
  presets, filesystem tools, shell tools, TUI, or end-user workflows.
- `aceai/core` owns framework primitives: `Agent`, event types, context,
  tool protocol, registry/executor, skills loading, tracing hooks, and strict
  schema behavior. It must not contain opinionated app tools, UI behavior, or
  provider SDK details.
- `aceai/agent` is the app layer for the user-facing AceAI app. It may contain
  default tools, capability bundles, built-in app skills, and `build_ace_agent()`.
- `aceai/agent/tui` is the presentation layer. It should consume structured
  `AgentEvent` / `TUIEvent` state and should not implement framework semantics,
  provider behavior, or tool execution semantics.

## Executive Summary

The main structural risk is not that app tools have moved into core. They mostly
have not. The real boundary debt is concentrated in three places:

1. `aceai/llm/openai_codex.py` contains app identity and user credential
   workflow. **(fixed 2026-05-09)**
2. `aceai/core` still exposes provider-specific tool-schema defaults and app
   skill discovery policy. **(partially fixed: tool spec moved; skill auto-discovery remains)**
3. `aceai/agent/tui` understands too much about provider request metadata,
   context-compression internals, and concrete app tool result schemas.
   **(none of the six TUI findings from 2026-05-09 have been resolved; four new violations found 2026-07-11)**

The first repair pass should remove provider/app leaks from `aceai/llm` and
`aceai/core`. The second pass should introduce app-layer adapters/presenters so
the TUI renders normalized display state instead of parsing internal payloads.

### 2026-07-11 Revalidation Summary

Of the six TUI-specific findings from 2026-05-09, **zero** have been addressed:

| Finding | Status |
|---------|--------|
| TUI builds LLMRequestMeta | **still present** |
| TUI reads LLM response internals for cost | **still present** |
| TUI implements subagent tool protocol | **still present** |
| TUI trajectory parses specific tool schemas | **still present** |
| TUI config calls default_agent_tools() | **still present** |
| TUI parses context compression message format | **still present** |

Four new violations were also found:

| Finding | Severity |
|---------|----------|
| TUI directly imports `LLMRequestMeta`, `OpenAIModel`, `LLMMessage`, `SkillLoader`, etc. | P0 |
| TUI ConfigScreen manages skill loading/discovery/symlinks directly | P1 |
| TUI runner does model/provider switching logic directly | P1 |
| TUI metadata view queries provider catalog directly | P2 |

## Priority Fixes

### LLM Layer: Codex Provider Carries AceAI Persona

- Location: `aceai/llm/openai_codex.py:18`, `aceai/llm/openai_codex.py:42`
- Code fact: `OPENAI_CODEX_DEFAULT_INSTRUCTIONS = "You are AceAI, a concise coding agent."`
  is injected into every Codex provider request.
- Boundary violation: `aceai/llm` should adapt provider wire formats. It should
  not define AceAI's product identity, app prompt, or coding-agent persona.
- Priority: priority fix.
- Status: fixed on 2026-05-09. The default instructions now live in
  `aceai/agent/ace_agent.py` and are passed explicitly into the Codex provider.
- Repair direction: move default instructions into `aceai/agent` app
  construction, or pass provider instructions explicitly from the app layer.

### LLM Layer: Codex Provider Reads Codex CLI Auth Files

- Location: `aceai/llm/openai_codex.py:93`, `aceai/llm/openai_codex.py:101`
- Code fact: the provider adapter resolves `CODEX_HOME`, reads
  `~/.codex/auth.json`, parses `tokens.access_token`, and tells the user to run
  `codex login`.
- Boundary violation: this is end-user credential discovery and CLI workflow.
  The llm adapter should receive an already-resolved access token.
- Priority: priority fix.
- Status: fixed on 2026-05-09. Codex CLI auth discovery now lives in
  `aceai/agent/provider_auth.py`; `OpenAICodex` receives a resolved token.
- Repair direction: move Codex credential discovery into `aceai/agent/provider_auth.py`
  or app config setup, then inject the token into `OpenAICodex`.

### Core Layer: OpenAI Tool Schema Is the Default Core Tool Spec

- Location: `aceai/core/tools/tool.py:16`, `aceai/core/tools/tool.py:80`,
  `aceai/core/tools/tool.py:162`, `aceai/core/tools/tool.py:201`
- Code fact: core defines `OpenAIToolSpec` and uses it as the default
  `spec_cls`.
- Boundary violation: core should own the provider-neutral tool protocol and
  strict schema behavior. OpenAI function schema serialization is
  provider-specific behavior.
- Priority: priority fix.
- Status: fixed on 2026-05-09. The core default is now `FunctionToolSpec`;
  OpenAI's top-level function wrapper is added in `aceai/llm/openai.py`.
- Repair direction: rename the default to a provider-neutral AceAI tool spec,
  or move OpenAI schema serialization to `aceai/llm/openai.py` while core
  exposes only a neutral schema contract.

### Core Layer: Skill Auto-Discovery Hardcodes App Paths

- Location: `aceai/core/skills.py:139`, `aceai/core/skills.py:141`
- Code fact: `SkillLoader.resolve_paths()` hardcodes `~/.aceai/skills` and
  `Path.cwd() / ".agents" / "skills"` for `skill_path == "auto"`.
- Boundary violation: skill loading can be a core primitive, but AceAI global
  and project discovery policy is an app/end-user workflow.
- Priority: priority fix.
- Repair direction: make core load explicit paths only. Put `"auto"` path
  resolution in `aceai/agent`, then pass resolved paths into core.

### TUI Layer: Provider Request Metadata Is Built in TUI

- Location: `aceai/agent/tui/runner.py:117-145` (previously line 124, 750)
- Code fact: five module-level functions — `_request_meta_with_reasoning_level`,
  `_reasoning_level_from_request_meta`, `_model_from_request_meta`, `_as_model`,
  `_model_options_text` — directly construct and validate `LLMRequestMeta`.
  In `AceAIInteractiveTUI.__init__` and `switch_model`, the TUI writes
  `reasoning.effort`, `reasoning.summary`, and `model` fields, and calls
  `supported_models()` / `supports_reasoning_effort()` to validate.
- Boundary violation: TUI should collect user selections and render state. It
  should not implement provider/request metadata semantics.
- Priority: priority fix.
- Status: **still present as of 2026-07-11.**
- Repair direction: move request-meta construction into `AceAgentApp` or an
  app-layer request builder. TUI should pass a config object or selection only.
  For example, add `AceAgentApp.set_model(model, reasoning_level)` that
  handles all validation and metadata construction internally.

### TUI Layer: Context Summary Display Parses Core Message Internals

- Location: `aceai/agent/tui/events.py:654-680`
- Code fact: TUI scans `ContextCompressedEvent.history`, selects system
  messages, inspects message parts (`part["type"] == "text"`), and strips
  `<aceai_context_summary>` tags via `_strip_context_summary_tags()`.
- Boundary violation: TUI is parsing core/context persistence format instead of
  consuming display-ready structured state.
- Priority: priority fix.
- Status: **still present as of 2026-07-11.**
- Repair direction: expose the summary text directly on `ContextCompressedEvent`
  or through an app-layer TUI event adapter.

### TUI Layer: Subagent Tool Protocol Is Implemented in TUI State

- Location: `aceai/agent/tui/state.py:389-470`, `aceai/agent/tui/tool_stats.py:99-141`
- Code fact: TUI special-cases `delegate_to_subagent`, decodes its arguments
  via `msg_decode(..., type=TUISubagentArguments)`, parses result JSON with
  `payload.get("type") == "subagent_audit"` branching, reads audit manifests
  from disk, and computes child tool stats.
- Boundary violation: TUI is implementing a concrete app tool's output protocol
  and artifact schema.
- Priority: priority fix.
- Status: **still present as of 2026-07-11.** In fact this has grown — `state.py`
  now contains `TUISubagentArguments`, `TUISubagentResult`, `TUISubagentToolResult`
  record types that mirror the app-layer delegation protocol.
- Repair direction: move subagent-result normalization into `aceai/agent`
  before the TUI sees it. TUI should receive a `TUISubagentState` or explicit
  subagent display event that has already been populated by the app layer.

## Code Smells

### Core Skill Registry Emits App Workflow Tools

- Location: `aceai/core/skills.py:241`, `aceai/core/skills.py:295`
- Code fact: `SkillRegistry.as_tools()` creates `skills_list` and `skill_view`,
  and `format_skills_for_prompt()` tells the model how to use them.
- Boundary concern: registry/loading is a core concern, but turning skills into
  model-facing default workflow tools is closer to an app capability bundle.
- Severity: code smell.
- Repair direction: keep skill metadata primitives in core, but let `aceai/agent`
  decide whether and how to expose skill tools and prompt fragments.

### Core LoggingExecutor Has Presentation-Flavored Logger API

- Location: `aceai/core/executor.py:241`, `aceai/core/executor.py:275`,
  `aceai/core/executor.py:286`, `aceai/core/executor.py:291`
- Code fact: `LoggingExecutor` accepts a logger with `info`, `success`, and
  `exception`, then emits tool start/fail/success text.
- Boundary concern: tracing hooks belong in core, but presentation-flavored
  success/failure text is app/UI-shaped.
- Severity: code smell.
- Repair direction: prefer structured execution events or tracing attributes in
  core; keep human-facing logging in app adapters.

### Core Context Compression Prompt Is App-Shaped

- Location: `aceai/core/context_manager.py:718`
- Code fact: the summary prompt says "AceAI runs" and preserves coding-agent
  concepts such as user goals, open tasks, file paths, remaining work, and
  unresolved errors.
- Boundary concern: context compression is a core primitive, but this policy is
  tailored to the AceAI app's coding workflow.
- Severity: code smell.
- Repair direction: make the compression prompt/policy injectable from the app
  layer while preserving a conservative framework default.

### TUI Event Adapter Reads LLM Provider Metadata for Cost

- Location: `aceai/agent/tui/events.py:425-435`
- Code fact: TUI reads `event.step.llm_response.usage`,
  `event.step.llm_response.provider_meta`, and calls `estimate_usage_cost()`.
- Boundary concern: useful display behavior, but TUI knows too much about the
  LLM response shape and provider metadata.
- Severity: code smell.
- Status: **still present as of 2026-07-11.**
- Repair direction: move cost calculation to the app/session adapter and pass
  cost as display state.

### TUI Trajectory Parses Specific App Tool Schemas

- Location: `aceai/agent/tui/trajectory.py:493-516`
- Code fact: trajectory rendering special-cases `run_shell_command`,
  `read_text_file`, `write_text_file`, and `search_text` arguments/results.
  `_tool_call_body` shows `$ {command}` for shell, `{path}` for read/write,
  `{query} in {path}` for search. `_tool_result_body` similarly special-cases
  each tool's result shape.
- Boundary concern: this is only display summarization, not execution, but it
  still binds the TUI to concrete app tool schemas.
- Severity: code smell.
- Status: **still present as of 2026-07-11.**
- Repair direction: expose tool preview/render metadata from app tools or an
  app-layer presenter, then keep trajectory generic.

### TUI Config Discovers Default App Tools Directly

- Location: `aceai/agent/tui/runner.py:1438-1462`
- Code fact: config UI calls `default_agent_tools()` and reads metadata,
  permissions (`configured_tool.metadata.require_approval`), enabled state, and
  max-call policy directly from tool objects.
- Boundary concern: TUI can render tool settings, but app capability discovery
  should come from the app layer.
- Severity: code smell.
- Status: **still present as of 2026-07-11.**
- Repair direction: have `AceAgentApp` or an app config service return
  `ToolPermissionItem`-like records.

## Non-Issues Observed

- `aceai/llm` does not directly import `aceai.agent` or `aceai.agent.tui`.
- `aceai/llm/models.py` has `LLMHostedToolSpec`, which is a provider-native
  hosted-tool contract rather than an app tool leak.
- `aceai/core` does not contain concrete filesystem, shell, browser, memory, or
  artifact tool implementations.
- `aceai/core` does not import Textual/Rich or TUI modules.
- `aceai/agent/ace_agent.py`, `aceai/agent/features/tools.py`,
  `aceai/agent/features/repo.py`, and `aceai/agent/features/delegation.py`
  are correctly placed as app-layer capability wiring.
- `aceai/agent/tui/widgets/approval.py` renders `ToolApprovalRequest` and does
  not execute tools or decide approval policy.
- `aceai/agent/app.py` (`AceAgentApp`) is a well-placed app-layer facade that
  encapsulates turn management, session service, subagent lifecycles, and idea
  storage. The repair strategy should expand this facade rather than let TUI
  bypass it.

## New Findings — 2026-07-11

### TUI Directly Imports Core/LLM/Provider Modules (New P0)

- Location: `aceai/agent/tui/runner.py:14-47`
- Code fact: the import block pulls in `LLMRequestMeta`, `LLMMessage`,
  `OpenAIModel` from `aceai.llm`; `Agent`, `AgentEvent`, `RunSuspendedEvent`,
  `Executor` from `aceai.core`; `SkillLoader`, `SkillLoadingError`,
  `SkillRegistry` from `aceai.core.skills`; and numerous app-internal modules
  (`supported_models`, `supports_reasoning_effort`, `default_api_key_for_provider`,
  `default_agent_tools`, `replace_config`, `save_config`, etc.).
- Boundary violation: TUI should only consume structured display state
  (`TUIEvent`, `TUIRunState`) and call `AceAgentApp` methods. Direct imports
  of provider-level types (`LLMRequestMeta`, `OpenAIModel`) and core framework
  types (`SkillLoader`, `Executor`) mean the TUI has full access to — and
  actively uses — implementation details from lower layers.
- Priority: priority fix.
- Repair direction: all interaction with core/llm types should be routed
  through `AceAgentApp`. Consider adding a `test_package_boundaries` rule
  that forbids `aceai.agent.tui` from importing `aceai.llm` or `aceai.core`
  (except for event types).

### TUI ConfigScreen Directly Manages Skill Loading/Discovery (New P1)

- Location: `aceai/agent/tui/setup.py` (multiple locations),
  `aceai/agent/tui/runner.py:808-830`
- Code fact: `ConfigScreen` calls `SkillLoader.load_registry()` to reload
  skills after symlink changes; `ProviderSetupScreen.__init__` calls it too.
  `_find_project_skill_dirs()` walks the filesystem with `Path.cwd().rglob("SKILL.md")`.
  `_save_skill_link()` creates symlinks in `.agents/skills/`.  `_reload_skill_items()`
  and `_try_reload_skill_items()` duplicate skill-loading logic inside the TUI.
- Boundary violation: skill discovery, loading, symlink management, and
  registry re-population are app-layer behaviors. TUI should request a current
  skill list and receive `SkillConfigItem` records.
- Priority: priority fix.
- Repair direction: add `AceAgentApp.discover_project_skills()`,
  `AceAgentApp.load_skill(path)`, and `AceAgentApp.available_skills()` methods.
  TUI screen callbacks should invoke these and re-render with returned data.

### TUI Runner Does Model/Provider Switching Logic Directly (New P1)

- Location: `aceai/agent/tui/runner.py:644-680` (`switch_model`),
  `runner.py:780-830` (`_handle_config_selection`)
- Code fact: `switch_model` validates against `supported_models()`, checks
  `supports_reasoning_effort()`, constructs new `LLMRequestMeta`, calls
  `self._agent_app.switch_model()`, then writes to session state and updates
  the status bar. `_handle_config_selection` resolves API keys from
  environment / default auth / user input, assembles an `AgentAppConfig`,
  and calls `save_config()` directly.
- Boundary violation: model validation, API key resolution, config assembly,
  and persistence are app-layer coordination logic. TUI should fire a "user
  selected provider X / model Y / api key Z" event and let the app layer
  handle all side effects.
- Priority: priority fix.
- Repair direction: expand `AceAgentApp.apply_config(config)` to own validation,
  key resolution, and persistence. TUI calls `app.apply_config(selection)` and
  re-renders from the resulting state.

### TUI Metadata View Queries Provider Catalog Directly (New P2)

- Location: `aceai/agent/tui/app.py:425-430`
- Code fact: `_metadata_sections` calls `context_window_for_model_any_provider()`
  and `supports_reasoning_effort_any_provider()` from `aceai.agent.provider_catalog`
  to display context-window usage percentage.
- Boundary violation: mild, but context-window metadata should come from app
  state rather than the TUI querying the provider catalog directly.
- Priority: code smell.
- Repair direction: let `AceAgentApp` expose `context_window` and
  `supports_reasoning` as display-ready properties, or include them in
  `TUIRunState.usage`.

## Suggested Repair Order (updated 2026-07-11)

1. Done 2026-05-09: move Codex auth discovery and default Codex instructions
   out of `aceai/llm/openai_codex.py`.
2. Done 2026-05-09: remove OpenAI-specific default tool spec naming and move
   the OpenAI `"type": "function"` wrapper to `aceai/llm/openai.py`.
3. Move skill `"auto"` discovery policy from `aceai/core/skills.py` to
   `aceai/agent`.
4. **(new)** Remove direct `aceai.llm` / `aceai.core` imports from TUI
   (`runner.py`). Route all interactions through `AceAgentApp`.
5. Add app-layer request metadata construction so TUI stops writing provider
   metadata directly (`_request_meta_with_reasoning_level` et al.).
6. Add display-ready context compression summary fields or app-layer TUI event
   normalization.
7. Add app-layer presenters/normalized records for subagent state, child tool
   stats, and tool previews.
8. **(new)** Move skill discovery/loading/symlink logic out of
   `setup.py`/`runner.py` into `AceAgentApp`.
9. **(new)** Move model/provider switching coordination (API key resolution,
   config assembly, persistence) out of `runner.py` into `AceAgentApp`.
10. Revisit the lower-priority smells: `LoggingExecutor`, app-shaped compression
    prompts, cost calculation location, config-time tool discovery, and
    trajectory tool-schema special-casing.
