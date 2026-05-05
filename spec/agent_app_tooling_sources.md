# AceAI Agent App Tooling Sources

## Goal

AceAI should grow stronger coding-agent tools by borrowing mature design from
established open-source agents, while keeping implementation ownership inside
the `aceai/agent` app layer.

This is not a plan to vendor a full external agent runtime. The app layer can
reuse ideas, protocols, prompts, and small algorithms when the license permits,
but the resulting tools should speak AceAI's native contracts:

- `aceai/core` keeps framework primitives only.
- `aceai/agent` owns product-shaped tools, permission policy, repo helpers, and
  coding-agent defaults.
- `aceai/agent/tui` renders structured events and approval state.

## Audited Sources

### Aider

- Repository: `https://github.com/Aider-AI/aider`
- Audited commit: `3ec8ec5a7d695b08a6c24fe6c0c235c8f87df9af`
- License: Apache-2.0
- Relevant files:
  - `aider/repomap.py`
  - `aider/repo.py`
  - `aider/diffs.py`
  - `aider/coders/editblock_coder.py`
  - `aider/coders/patch_coder.py`
  - `aider/coders/search_replace.py`

Useful ideas:

- Repo map as a compact source-navigation artifact, not a raw file listing.
- Git diff helpers that compare both index and working tree state.
- Multiple edit formats with validation and precise failure feedback.
- Partial/live diff rendering while whole-file edits stream.
- Search/replace fallback strategies for imperfect model edits.

Do not copy wholesale:

- Aider's coder hierarchy is tightly coupled to its chat loop, IO layer, model
  wrappers, and prompt modes.
- The repo-map implementation depends on cache, tree-sitter, grep-ast, pygments,
  and Aider-specific token counting. AceAI should start with a smaller native
  version and make heavier dependencies optional.

### OpenAI Codex CLI

- Repository: `https://github.com/openai/codex`
- Audited commit: `70807730f5a8e093d5182089ad5a4b1b4355f9fa`
- License: Apache-2.0
- Relevant files:
  - `codex-rs/tools/src/apply_patch_tool.rs`
  - `codex-rs/apply-patch/src/parser.rs`
  - `codex-rs/apply-patch/src/invocation.rs`
  - `codex-rs/core/src/exec_policy.rs`
  - `codex-rs/protocol/src/approvals.rs`
  - `codex-rs/tools/src/local_tool.rs`
  - `codex-rs/tools/src/tool_config.rs`

Useful ideas:

- `apply_patch` as a first-class edit tool with a constrained patch grammar.
- Separating patch parsing, patch verification, approval, and filesystem apply.
- Permission profile concepts for filesystem and network access.
- Explicit request-permissions flow instead of overloading shell execution.
- Command prefix rules for repeated approvals, with banned broad prefixes.
- Treat patch approval as a reviewed action over affected files.

Do not copy wholesale:

- The Rust sandbox stack, exec server, and platform-specific policies are too
  large for AceAI's current app layer.
- The Codex tool surface assumes a different protocol and client/server runtime.
  AceAI should copy the conceptual split, not the Rust architecture.

## Attribution Policy

For design-only reuse, cite the upstream project in this spec and relevant
implementation comments when helpful.

For adapted code or close translations:

- Include a file header naming the upstream repository, audited commit, source
  path, license, and summary of AceAI modifications.
- Preserve Apache-2.0 or MIT license obligations in the repository-level NOTICE
  or equivalent third-party attribution document.
- Keep adapted code isolated in small modules so later replacement is possible.

Recommended header:

```python
# Adapted from Aider:
# https://github.com/Aider-AI/aider/blob/<commit>/<path>
# License: Apache-2.0
# Modifications: rewritten for AceAI's app-layer tool API and approval model.
```

## First Implementation Slice

### 1. Native patch tools

Add app-layer patch support, preferably under `aceai/agent/features/patch.py`.

Tools:

- `preview_patch(patch: str, cwd: str = ".") -> PatchPreview`
- `apply_patch(patch: str, cwd: str = ".") -> PatchApplyResult`

Behavior:

- Use the Codex patch envelope:
  - `*** Begin Patch`
  - `*** Add File: <path>`
  - `*** Delete File: <path>`
  - `*** Update File: <path>`
  - `*** Move to: <path>`
  - `@@` hunks
  - `*** End Patch`
- Reject absolute paths.
- Resolve all paths under `cwd`.
- Generate a unified diff preview before applying.
- Require approval for `apply_patch`, with affected paths in the approval
  request payload.
- Keep `preview_patch` read-only and approval-free.

Why first:

- It upgrades AceAI from raw file writes to reviewable edits.
- It fits the existing `ToolMeta.require_approval` flow.
- It can be tested without a live model.

### 2. Git diff and status tools

Add lightweight repo tools, preferably under `aceai/agent/features/repo.py`.

Tools:

- `git_status(cwd: str = ".") -> GitStatus`
- `git_diff(cwd: str = ".", paths: list[str] = []) -> GitDiff`

Behavior:

- Use `git status --short --branch`.
- For diffs, include both staged and unstaged changes when appropriate.
- Return structured output and truncated display text separately if needed.

Why second:

- The TUI and final answer need a reliable source of changed files.
- Approval prompts become clearer when they can reference existing git state.

### 3. Workspace permission profile

Add an app-layer permission model before adding broader shell/browser tools.

Initial shape:

```python
class WorkspacePermissionProfile(Struct, frozen=True, kw_only=True):
    read_roots: list[str]
    write_roots: list[str]
    network_enabled: bool
```

Policy:

- Read tools check `read_roots`.
- Write/patch tools check `write_roots`.
- Shell command execution remains approval-required until a narrower command
  policy exists.
- A later `request_permissions` tool can request additional roots or network.

Why third:

- It prevents tool growth from becoming a pile of per-tool approval booleans.
- It keeps policy in `aceai/agent`, not `aceai/core`.

## Later Slices

### Repo map

Build a small native repo map first:

- Gather tracked files with git.
- Rank files by mentions, recent edits, and import/name matches.
- Emit compact symbols using Python `ast` for Python files.
- Defer tree-sitter support until the simple path is proven useful.

Aider's full repo map is powerful, but its dependency shape is too large for the
first AceAI implementation.

### Search/replace fallback

AceAI can later add a safer replacement tool inspired by Aider's editblock
failure feedback:

- exact match only by default;
- optional fuzzy preview, not fuzzy apply;
- failure result includes nearest match context and guidance for a corrected
  patch.

### Browser/local preview

Browser and preview tools should be their own app-layer capability bundle after
filesystem policy is real. They should not be hidden behind shell commands.

## Non-Goals

- Do not import Aider's full coder stack.
- Do not port Codex's Rust sandbox or exec server.
- Do not add app tools to `aceai/core/tools`.
- Do not add network/browser/artifact tools before workspace policy exists.

## Acceptance Criteria

- New tools live in `aceai/agent/features`, not `aceai/core`.
- `default_agent_tools()` exposes patch and repo tools only after focused tests.
- Patch apply is approval-gated and emits affected paths.
- Tests cover add, update, delete, move, absolute-path rejection, cwd escape
  rejection, preview-only behavior, and approval metadata.
- Attribution for any adapted code is present before merge.
