# AceAI Layer Boundary Review - 2026-05-09

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
   workflow.
2. `aceai/core` still exposes provider-specific tool-schema defaults and app
   skill discovery policy.
3. `aceai/agent/tui` understands too much about provider request metadata,
   context-compression internals, and concrete app tool result schemas.

The first repair pass should remove provider/app leaks from `aceai/llm` and
`aceai/core`. The second pass should introduce app-layer adapters/presenters so
the TUI renders normalized display state instead of parsing internal payloads.

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

- Location: `aceai/agent/tui/runner.py:124`, `aceai/agent/tui/runner.py:750`
- Code fact: TUI constructs `LLMRequestMeta["reasoning"]`, writes `"summary":
  "auto"`, validates model support, and writes `request_meta["model"]`.
- Boundary violation: TUI should collect user selections and render state. It
  should not implement provider/request metadata semantics.
- Priority: priority fix.
- Repair direction: move request-meta construction into `AceAgentApp` or an
  app-layer request builder. TUI should pass a config object or selection only.

### TUI Layer: Context Summary Display Parses Core Message Internals

- Location: `aceai/agent/tui/events.py:654`
- Code fact: TUI scans `ContextCompressedEvent.history`, selects system
  messages, inspects message parts, and strips `<aceai_context_summary>` tags.
- Boundary violation: TUI is parsing core/context persistence format instead of
  consuming display-ready structured state.
- Priority: priority fix.
- Repair direction: expose the summary text directly on `ContextCompressedEvent`
  or through an app-layer TUI event adapter.

### TUI Layer: Subagent Tool Protocol Is Implemented in TUI State

- Location: `aceai/agent/tui/state.py:389`, `aceai/agent/tui/state.py:470`,
  `aceai/agent/tui/tool_stats.py:99`, `aceai/agent/tui/tool_stats.py:141`
- Code fact: TUI special-cases `delegate_to_subagent`, decodes its arguments,
  parses result JSON, reads audit manifests, and computes child tool stats.
- Boundary violation: TUI is implementing a concrete app tool's output protocol
  and artifact schema.
- Priority: priority fix.
- Repair direction: move subagent-result normalization into `aceai/agent`
  before the TUI sees it. TUI should receive a `TUISubagentState` or explicit
  subagent display event.

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

- Location: `aceai/agent/tui/events.py:433`
- Code fact: TUI reads `event.step.llm_response.usage`,
  `event.step.llm_response.provider_meta`, and calls `estimate_usage_cost()`.
- Boundary concern: useful display behavior, but TUI knows too much about the
  LLM response shape and provider metadata.
- Severity: code smell.
- Repair direction: move cost calculation to the app/session adapter and pass
  cost as display state.

### TUI Trajectory Parses Specific App Tool Schemas

- Location: `aceai/agent/tui/trajectory.py:493`,
  `aceai/agent/tui/trajectory.py:508`
- Code fact: trajectory rendering special-cases `run_shell_command`,
  `read_text_file`, `write_text_file`, and `search_text` arguments/results.
- Boundary concern: this is only display summarization, not execution, but it
  still binds the TUI to concrete app tool schemas.
- Severity: code smell.
- Repair direction: expose tool preview/render metadata from app tools or an
  app-layer presenter, then keep trajectory generic.

### TUI Config Discovers Default App Tools Directly

- Location: `aceai/agent/tui/runner.py:1438`
- Code fact: config UI calls `default_agent_tools()` and reads metadata,
  permissions, enabled state, and max-call policy.
- Boundary concern: TUI can render tool settings, but app capability discovery
  should come from the app layer.
- Severity: code smell.
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

## Suggested Repair Order

1. Done 2026-05-09: move Codex auth discovery and default Codex instructions
   out of `aceai/llm/openai_codex.py`.
2. Done 2026-05-09: remove OpenAI-specific default tool spec naming and move
   the OpenAI `"type": "function"` wrapper to `aceai/llm/openai.py`.
3. Move skill `"auto"` discovery policy from `aceai/core/skills.py` to
   `aceai/agent`.
4. Add app-layer request metadata construction so TUI stops writing provider
   metadata directly.
5. Add display-ready context compression summary fields or app-layer TUI event
   normalization.
6. Add app-layer presenters/normalized records for subagent state, child tool
   stats, and tool previews.
7. Revisit the lower-priority smells: `LoggingExecutor`, app-shaped compression
   prompts, cost calculation location, and config-time tool discovery.
