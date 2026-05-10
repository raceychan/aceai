# Pyright Error Inventory — 2026-05-10

Snapshot of `uv run pyright` against the current `version/0.2.23` branch
(post-Phase-4). Total: **502 errors**, of which **124 in production code**
(non-test). This document categorizes them so future cleanup work can pick a
focused slice.

A spike to fix the dominant category via a `_RuntimeHost` Protocol (see
"Mixin attribute access" below) was attempted on 2026-05-10 and rolled back —
the Protocol approach reduced production errors from 124 to 70 but
introduced four `reportAbstractUsage` errors at concrete-class
instantiation sites and several `reportIncompatibleMethodOverride` errors
where the Protocol's signatures didn't perfectly match `AceAITUI` /
`textual.App` (`title`, `set_status_model`, `super().switch_session()`).
Properly fixing requires either tighter Protocol signatures or a different
pattern (e.g. `TYPE_CHECKING`-conditional inheritance from `AceAITUI`).

## By rule (production only — 124 errors)

| Count | Rule |
|---:|---|
| 68 | `reportAttributeAccessIssue` |
| 21 | `reportOptionalMemberAccess` |
| 17 | `reportArgumentType` |
| 8 | `reportOptionalCall` |
| 2 | `reportReturnType` |
| 2 | `reportOperatorIssue` |
| 2 | `reportIndexIssue` |
| 1 each | `reportIncompatibleMethodOverride`, `reportGeneralTypeIssues`, `reportTypedDictNotRequiredAccess`, `reportAssignmentType` |

## By file (production only)

| Count | File |
|---:|---|
| 84 | `aceai/agent/tui/runner.py` |
| 10 | `aceai/agent/tui/cli.py` |
| 10 | `aceai/llm/deepseek.py` |
| 8 | `aceai/agent/tui/setup.py` |
| 5 | `aceai/core/tools/tool.py` |
| 2 | `aceai/agent/builtin_skills/skill-creator/scripts/run_eval.py` |
| 1 each | `aceai/agent/session.py`, `aceai/agent/tui/metadata.py`, `aceai/agent/tui/session_adapter.py`, `aceai/core/tools/schema_generator.py`, `aceai/llm/openai_codex.py` |

## Three dominant root causes (~95 of 124)

### 1. Mixin attribute access (66 errors, 53%)

- Files: `aceai/agent/tui/runner.py` (64), `aceai/agent/tui/cli.py` (2).
- `_RuntimeStreamMixin` invokes `self.notify_session`, `self.run_worker`,
  `self.query_one`, `self.append_event`, `self.append_agent_event`, etc. —
  methods provided by `AceAITUI` (which extends `textual.App[None]`) and by
  the concrete subclasses `AceAIInteractiveTUI` / `AceAIConfiguredTUI`. The
  mixin itself only inherits `object`, so pyright flags every access.

- Top missing attributes (with hit counts):

  | Hits | Attribute | Origin |
  |---:|---|---|
  | 15 | `append_event` | `AceAITUI` |
  | 7 | `_project` | `AceAITUI` |
  | 5 | `notify_session` | `AceAITUI` |
  | 4 | `query_one` | `textual.App` |
  | 3 each | `exit_command_input`, `is_mounted`, `run_worker` | `AceAITUI` / `textual.App` |
  | 2 each | `load_events`, `query`, `session_id` | `AceAITUI` / `textual.App` |
  | 1 each | `append_agent_event`, `approve_pending_tool`, `call_after_refresh`, `clear_approval_request`, `exit`, `on_mount`, `on_unmount`, `open_*_screen`, `push_screen`, `reject_pending_tool`, `set_timer`, `show_pending_approval`, `start_run`, `_persist_session_state`, `_window_title`, `action_*`, … | various |

- **Recommended fix**: pick one of:
  - `TYPE_CHECKING`-conditional inheritance from `AceAITUI` (cleanest if we
    can match Textual's reactive-attribute signatures).
  - A `_RuntimeHost` Protocol carefully enumerating each method with
    signatures that line up with `AceAITUI` / `textual.App`. The 2026-05-10
    spike did this but got bitten by `title` (Textual reactive descriptor)
    and `set_status_model` overload mismatches.

### 2. `_agent_app: AceAgentApp | None` not narrowed (21 errors, 17%)

- Files: `aceai/agent/tui/runner.py` (19), `aceai/agent/builtin_skills/skill-creator/scripts/run_eval.py` (2).
- The configured TUI's lazy-init pattern keeps `_agent_app` Optional. Each
  `self._agent_app.something` triggers a `reportOptionalMemberAccess`.

- **Recommended fix**: add a `_require_agent_app() -> AceAgentApp` helper on
  the mixin that raises if `None`, and migrate call sites to
  `self._require_agent_app().foo`. Or override `_agent_app` to non-Optional
  on `AceAIInteractiveTUI` since it's always set in its `__init__`.

### 3. `aceai/llm/deepseek.py` `Unset[T]` access without narrowing (10 errors, 8%)

- `UnsetType` lacks `__getitem__`, `in`, `iter`, `.type` etc. The provider
  code accesses these without first calling `is_set(...)` to narrow.

- **Recommended fix**: add `is_set` guards before each access, mirroring the
  pattern used in `aceai/llm/openai.py`.

## Remaining (27 errors)

Smaller pockets, listed exhaustively for triage:

- `aceai/agent/tui/setup.py` (8): Textual `Screen[T]` generic parameters
  (`Self@ProviderSetupScreen` not `ModalScreen[object]`); `NoSelection`
  sentinel leaking into return types where `ReasoningLevel` /
  `ToolPermission` are expected; `dict[str, str]` vs
  `dict[str, ToolPermission]`.
- `aceai/agent/tui/cli.py` (8 of 10): `run_configured_tui = None` lazy
  assignment defeats `reportOptionalCall`. The other 2 are mixin-related
  (covered above).
- `aceai/core/tools/tool.py` (5): `FunctionToolSpec` doesn't conform to the
  `IToolSpec` interface contract.
- `aceai/agent/tui/runner.py` (1 line 1246): `selection.skill_selection_mode`
  typed as `str` flowing into the `ConfigScreen`'s `SkillSelectionMode`
  literal parameter.
- `aceai/agent/tui/session_adapter.py` (1): `TUIEventKind` ⊆ `SessionEventKind`
  but pyright doesn't see the relationship.
- `aceai/agent/tui/metadata.py` (1): `Screen.action_dismiss` override
  signature mismatch.
- `aceai/agent/session.py` (1): `from_payload` returns `EventLog` not
  `Self@EventLog` (msgspec idiom).
- `aceai/llm/openai_codex.py` (1): `Unset[LLMResponse]` flowing into
  `LLMResponse | None` field.
- `aceai/core/tools/schema_generator.py` (1): `Any | None` not iterable.
- `aceai/agent/builtin_skills/skill-creator/scripts/run_eval.py` (2):
  small standalone eval script (not in main runtime path).

## Cost / benefit

| Fix | Errors removed | Effort |
|---|---:|---|
| Mixin Protocol (or `TYPE_CHECKING` AceAITUI inheritance), done correctly | ~66 | medium — must match Textual reactive signatures |
| `_require_agent_app()` helper + ~20 call-site migrations | ~20 | small |
| `aceai/llm/deepseek.py` `is_set` narrowing | ~10 | small, file-local |
| Everything else | ~28 | scattered, low priority |
