# TUI Layer Boundary Repair — Phased Plan

Drives the repair of the boundary violations documented in
`spec/layer_boundary_review_2026_05_09.md`. Each phase is independently
shippable, reviewable in ~200–400 LOC, and ratchets the boundary so the TUI
stops reaching into `aceai.llm` and `aceai.core`.

The unifying principle: expand `AceAgentApp` (`aceai/agent/app.py`) to own every
interpretation of LLM/core internals. The TUI consumes display-shaped records
(`TUIRunState`, `TUIEvent`, new app-emitted records) and never parses provider
metadata, request shapes, manifest JSON, or internal message formats.

## Status (2026-05-10)

- ✅ **Phase 1 — request-meta in app layer** (commit `20b7ef4`).
- ✅ **Phase 2 — provider/usage display state** (commit `8b2dfc5`).
- ✅ **Phase 3 — cost & context-summary helpers in app layer** (commit `4579b8f`).
- ✅ **Phase 4 — api-key resolution & config persistence in app layer** (commit `6731327`).
- ⏳ Phases 5–8 pending.

Pyright errors went from 90 baseline → 85 over Phases 1–4. All 570 tests
continue to pass after each phase.

Remaining `aceai.core` / `aceai.llm` imports in `tui/runner.py`:

- `Agent` — for the `AgentFactory` type alias (Phase 5+).
- `AgentEvent`, `RunSuspendedEvent` — for event-stream type annotations.
- `SkillLoader`, `SkillLoadingError`, `SkillRegistry` — Phase 5.
- `LLMMessage`, `OpenAIModel` — field type annotations.

## Phase 1 — Move request-meta construction into `AceAgentApp` (DONE)

Eliminated the five `_request_meta_*` / `_as_model` / `_model_options_text`
helpers in `runner.py`. `AceAgentApp.switch_model(model, *, reasoning_level)`
now owns reasoning-level lifecycle and silent downgrades. Added properties
`reasoning_level`, `model_options_text`, and method `is_model_supported`. New
module-level helpers `effective_reasoning_level`, `model_options_text_for`,
`is_model_supported` cover the configured-TUI pre-app case. Removed
`LLMRequestMeta`, `model_options`, `supported_models`,
`supports_reasoning_effort` imports from `runner.py`. Replaced the dead
`request_meta=` kwarg on `run_*_tui()` and TUI class constructors with
`reasoning_level: ReasoningLevel = "auto"`.

## Phase 2 — App-layer provider/usage display state

**Goal**: Stop `tui/app.py:_metadata_sections` and
`runner.py:_agent_metadata_sections` from calling `provider_catalog` /
inspecting `Agent` directly.

**New `AceAgentApp` API**:
- `AppRuntimeInfo` Struct: `provider_name`, `selected_model`, `default_model`,
  `reasoning_level`, `supports_reasoning`, `context_window`, `max_steps`.
- `runtime_info() -> AppRuntimeInfo`.
- `skill_summary_lines() / tool_summary_lines() / hosted_tool_summary_lines()`.
- Module helpers `context_window_for_model(model)` and
  `supports_reasoning_for_model(model)` so `tui/app.py` (no agent_app) can
  still render context-window and reasoning fields.

**TUI changes**:
- `tui/app.py:_metadata_sections` reads context_window / supports_reasoning via
  the new app helpers; drop `provider_catalog` imports.
- `runner.py:_agent_metadata_sections` becomes a thin formatter over
  `agent_app.runtime_info()` + summary lines; drop `Executor` import.

**Risks**: format of `max_steps` line (preserved as `f"{agent.max_steps}"`).

## Phase 3 — Cost & context-summary helpers in app layer (DONE)

**Done.** TUI events depending on `AgentEvent` is the correct direction, so
the conversion (`_agent_event_to_tui_event`) stayed in `tui/events.py`. Only
the deep-access bits flagged by the boundary review were extracted:

`aceai/agent/tui_adapter.py` (new) exposes three helpers:
- `usage_for_llm_completed(event)` — encapsulates `is_set(response.usage)`.
- `cost_for_llm_completed(event, *, provider_name=None)` — encapsulates the
  `estimate_usage_cost` call and the `response.provider_meta` provider
  inference fallback.
- `context_summary_text(event)` — encapsulates the `<aceai_context_summary>`
  parsing of `event.history`.

`TUIEvent.from_agent_event` and `_agent_event_to_tui_event` accept an
optional `provider_name`. `_RuntimeStreamMixin._consume_agent_stream`
snapshots `agent_app.provider_name` once per stream and passes it through, so
cost calculation uses the app's known provider directly instead of inferring
from `provider_meta`.

## Phase 4 — API-key resolution & config persistence in app layer (DONE)

**Done.** Two app-layer module helpers replace the env-var/default fallback
and the `replace_config` / `save_config` calls in TUI:

- `resolve_provider_api_key(provider) -> str` — env var (`api_key_env`) >
  default auth file (`default_api_key_for_provider`); returns `""` if neither
  source has a key.
- `normalize_user_config(config, *, persist=False) -> AgentAppConfig` —
  validates via `replace_config` and optionally saves to disk.

`runner.py:_handle_config_selection` collapses by ~40 lines and factors out
`_resolve_selection_api_key`. `apply_user_config` is one call to
`normalize_user_config(config, persist=True)`.

Dropped runner imports: `default_api_key_for_provider`, `api_key_env`,
`replace_config`, `save_config`, `os.environ` access.

**Deferred to Phase 5+**: `Agent` import (for `AgentFactory` type alias) and
the question of whether `AgentFactory` should return `AceAgentApp` instead of
`Agent`. The current shape — TUI receives `Agent` from a factory and wraps it
in `AceAgentApp` — is left intact for now.

## Phase 5 — Move skill discovery / loading / symlinks into `AceAgentApp`

**Goal**: Stop `setup.py:544` (`Path.cwd().rglob("SKILL.md")`),
`setup.py:1724` (`_save_skill_link` symlink creation), and
`runner.py:1442`/`setup.py:658,1751` (`SkillLoader.load_registry()` calls).

**New API** (in `aceai/agent/skill_view.py` or on `AceAgentApp`):
- `available_skills() -> tuple[SkillConfigItem, ...]`.
- `discover_project_skills() -> tuple[SkillConfigItem, ...]`.
- `load_skill_into_project(skill_dir: Path) -> SkillConfigItem`.
- `reload_skills_from_path(skills_path: str) -> tuple[SkillConfigItem, ...]`.
- Module-level `discover_initial_skills()` for the pre-`AceAgentApp` setup
  screen path.

**TUI changes**:
- Delete `setup.py` skill-discovery and symlink helpers.
- Replace `runner.py:_skill_config_items` / `_available_skill_items` with
  app-layer calls.
- Drop `SkillLoader` / `SkillLoadingError` / `SkillRegistry` imports.

**Risks**: `ProviderSetupScreen.__init__` runs *before* an `AceAgentApp`
exists. Resolve via free function in `aceai/agent/skill_view.py` and pass
result into the screen.

## Phase 6 — Subagent normalization in app layer

**Goal**: Stop `tui/state.py:20-41,488-504` from re-defining
`TUISubagentArguments` / `TUISubagentResult` / `TUISubagentToolResult` and
parsing the `subagent_audit` JSON branch. Stop `tool_stats.py:99-141` from
reading audit manifests.

**New API**:
- `SubagentDisplayResult` Struct (unifying audit-form and full-form result
  shapes) — fields: `thread_id, agent_id, run_id, status, summary, step_count,
  tool_results, final_answer, important_evidence`.
- `AceAgentApp.read_subagent_manifest(audit_payload) -> SubagentDisplayResult`.
- `AceAgentApp.subagent_tool_call_stats(events) -> list[ToolCallStat]`.
- `AceAgentApp.tui_event_from_session_event(session_event)` — replay path that
  pre-populates `subagent_arguments` / `subagent_result` on `TUIEvent`.
- `TUIEvent` gains optional `subagent_arguments` and `subagent_result` fields,
  populated by the app-layer adapter (Phase 3).

**TUI changes**:
- Delete `TUISubagentArguments` / `TUISubagentResult` / `TUISubagentToolResult`
  type definitions in `state.py` (re-export from app layer).
- Replace `_subagent_arguments` / `_subagent_result` parsing with reads from
  `event.subagent_arguments` / `event.subagent_result`.
- Move `tool_stats.py` audit-manifest logic to app layer.

**Risks**: must handle both audit-form (manifest reference) and full-form
(inline JSON) for legacy and new sessions. Replay path compatibility.

## Phase 7 — Tool-preview presenters & tool-permission discovery

**Goal**: Stop `trajectory.py:493-516` from special-casing
`run_shell_command` / `read_text_file` / `write_text_file` / `search_text`,
and `runner.py:1468` from calling `default_agent_tools()` directly.

**New API**:
- `ToolPresenter` record (data-driven): `call_summary_template`,
  `call_summary_fields`, `result_summary_strategy`, `result_summary_field`.
- `AceAgentApp.tool_permission_items(*, configured) -> tuple[ToolPermissionItem, ...]`.
- `AceAgentApp.tool_presenters() -> Mapping[str, ToolPresenter]`.

**TUI changes**:
- `trajectory.py` becomes generic: dispatcher consults
  `event.tool_presenter` (set by Phase 3 adapter).
- `runner.py:_available_tool_permission_items` collapses to a single app
  call.

**Risks**: data-shape (declarative) vs function-shape (callable presenters).
Recommend declarative for the four current cases; promote if a tool resists
the data shape.

## Phase 8 — Boundary test & cleanup

**Goal**: Ratchet the gains. Add a strict test in
`tests/test_package_boundaries.py` forbidding `aceai/agent/tui/` imports of
`aceai.llm` (except `aceai.llm.interface` for `Record`) and `aceai.core`
(except event types and `ToolApprovalRequest`).

**New API**: none.

**Risks**: `tui/widgets/approval.py` imports `aceai.core.models.ToolApprovalRequest`. Whitelist or move.

## Suggested merge order

`1 → 2 → 3 → 4 → 5 → 6 → 7 → 8`. Phases 1, 2 are independent. Phase 3 must
precede Phase 6 (subagent normalization needs the app-layer event adapter).
Phase 5 can run in parallel with Phase 4.

## Easy / hard breakdown

- **Easy / mechanical**: Phases 1, 2.
- **Medium / plumbing-heavy**: Phases 3, 4.
- **Hard / genuine API design**: Phases 5 (pre-app lifecycle), 6 (dual contract
  audit/full + replay), 7 (presenter shape).

## Critical files

- `aceai/agent/app.py`
- `aceai/agent/tui/runner.py`
- `aceai/agent/tui/app.py`
- `aceai/agent/tui/events.py`
- `aceai/agent/tui/state.py`
- `aceai/agent/tui/setup.py`
- `aceai/agent/tui/tool_stats.py`
- `aceai/agent/tui/trajectory.py`
- `tests/test_package_boundaries.py`
