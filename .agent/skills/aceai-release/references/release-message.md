# Release Message Format

Use this format for AceAI release PR bodies, GitHub Release notes, release summaries, and tag notes when the user asks for structured release messaging.

## PR Title

Use:

```text
Release <x.y.z>
```

If the PR is not the final release PR, use:

```text
Prepare release <x.y.z>
```

## PR Body Template

```markdown
## Summary
- One sentence describing the release theme.

## Features
- `scope`: Describe the new user-visible capability, why it exists, and the main entry point.

## Improvements
- `scope`: Describe behavior, reliability, performance, docs, or developer-experience improvements.

## Fixes
- `scope`: Describe the bug, the condition that triggered it, and what now happens instead.

## Breaking Changes
- `scope`: Describe the removed or changed contract and what callers must do now.

## Release Operations
- Version: `<x.y.z>`
- Branch: `version/<x.y.z>`
- Tag after merge: `v<x.y.z>`
- PyPI publish: tag-triggered GitHub Actions workflow

## Verification
- `uv run pytest -v`
- `uv build`
- Any skipped or intentionally excluded checks with a reason.
```

Omit empty `Features`, `Improvements`, `Fixes`, or `Breaking Changes` sections only when the PR is small and the omitted section would be noise. Keep `Summary`, `Release Operations`, and `Verification`.

## Category Rules

### Features

Use for new capabilities, public APIs, new tools, new workflows, or newly supported user-visible behavior.

Each feature bullet should include:

- The scope or module in backticks.
- What the user can now do.
- The primary entry point, command, class, or file path when relevant.
- Any important limitation or activation condition.

Example:

```markdown
- `skills`: Add project skill loading with `skill_view`, allowing agents to load full skill instructions and supporting files on demand.
```

### Improvements

Use for better behavior that does not create a new feature: reliability, coverage, docs, CI, packaging, performance, observability, or developer ergonomics.

Each improvement bullet should include:

- The scope in backticks.
- What changed.
- Why it improves release quality or user experience.

Example:

```markdown
- `docs-ci`: Restrict GitHub Pages deployment to `master`, keeping version branches able to build docs without attempting protected deployments.
```

### Fixes

Use for bugs or incorrect behavior.

Each fix bullet should include:

- The scope in backticks.
- The broken condition.
- The corrected behavior.

Example:

```markdown
- `skill-loader`: Preserve nested skill resource paths so `references/`, `scripts/`, and `assets/` can be loaded through `skill_view`.
```

### Breaking Changes

Use for intentional incompatibility. AceAI prefers breaking changes over compatibility shims when the new contract is clearer.

Each breaking-change bullet should include:

- The scope in backticks.
- The old contract.
- The new contract.
- The required caller or operator action.

Example:

```markdown
- `agent`: Replace broad tool-result payloads with concrete `ToolExecutionResult`; callers must read `output` and `error` fields directly.
```

If there are no breaking changes, write:

```markdown
## Breaking Changes
- None.
```

## Verification Rules

Verification bullets must be copy-pasteable commands or concrete external checks.

Good:

```markdown
- `uv run pytest -v`
- `uv build`
- GitHub Actions `CI / test` passed on the release PR.
```

Bad:

```markdown
- Tested locally.
- CI should pass.
```

## Tag Note Template

```text
Release version <x.y.z>

Highlights:
- <feature/improvement/fix summary>

Verification:
- uv run pytest -v
- uv build
```

## GitHub Release Notes Template

Use the same category rules as the PR body. Keep the GitHub Release user-facing and omit branch mechanics unless they matter for operators.

```markdown
## Summary
- One sentence describing the release theme.

## Features
- `scope`: User-visible new capability.

## Improvements
- `scope`: Reliability, docs, CI, packaging, performance, or developer-experience improvement.

## Fixes
- `scope`: Broken condition and corrected behavior.

## Breaking Changes
- None.

## Verification
- `uv run pytest -v`
- `uv build`
- GitHub Actions `<tag>` CI and PyPI publish succeeded.
```
