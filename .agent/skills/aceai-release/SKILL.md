---
name: aceai-release
description: "Use for AceAI release work: preparing a version branch, committing and pushing release changes, drafting release PR messages, merging into master, tagging only after merge, and triggering PyPI publication."
---

# AceAI Release

Use this skill when the user asks to release AceAI, cut a version, prepare a release PR, tag a release, publish to PyPI, or check release readiness.

This repository publishes to PyPI from GitHub Actions when a pushed tag starts with `v`. The tag commit must already be contained in `origin/master`; otherwise the publish job fails. Therefore, do not tag the version branch before the PR is merged.

## Release Sequence

1. Inspect state.
   - Run `git status --short`, `git branch --show-current`, `git remote show origin`, and `git ls-remote --tags origin`.
   - Confirm the release branch is `version/<x.y.z>` and the base branch is `master`.
   - Check `aceai/__init__.py` has `__version__ = "<x.y.z>"`; update it if needed.
   - Check `CHANGELOG.md` if the release needs human-facing notes.

2. Validate before committing.
   - Run tests with `uv run pytest -v`.
   - Run `uv build` when packaging behavior or release metadata changed.
   - If e2e tests are intentionally excluded, say that explicitly.

3. Commit and push the release branch.
   - Stage only intended files; preserve unrelated local changes.
   - Use a focused commit message, usually `Release version <x.y.z>` for final release prep.
   - Push with `git push origin version/<x.y.z>`.

4. Open the release PR.
   - Base: `master`.
   - Head: `version/<x.y.z>`.
   - Read `references/release-message.md` before drafting the PR body.
   - The PR body must group changes by category and include verification.

5. Merge into `master`.
   - Wait for CI to pass.
   - Merge the PR using the repository's preferred merge method.
   - Do not push release tags until the merge commit is visible on `origin/master`.

6. Tag from `master`.
   - Run `git checkout master` and `git pull origin master`.
   - Confirm `aceai/__init__.py` still matches the tag version.
   - Create an annotated tag: `git tag -a v<x.y.z> -m "Release version <x.y.z>"`.
   - Push the tag: `git push origin v<x.y.z>`.

7. Verify publication.
   - Watch the tag-triggered CI publish job.
   - Confirm PyPI shows the expected version.
   - Confirm the package can be installed when appropriate.

8. Prepare next cycle.
   - Create `version/<next>` from `master` if the project wants a persistent release branch.
   - Default next-version selection:
     - Before `1.0.0`, bump patch: `0.2.0` -> `0.2.1`.
     - At and after `1.0.0`, bump minor: `1.0.0` -> `1.1.0`.
     - If the user explicitly requests patch, minor, major, or an exact version, follow the user instead.
   - Do not delete release branches unless the user asks.

## Repository-Specific Checks

- Always run tests via `uv run pytest` in this repository.
- Docs deploy is restricted to `master`; `version/**` branches may build docs but should not deploy Pages.
- The CI publish job runs only for pushed tags matching `v*`.
- The publish job checks the tag commit is contained in `origin/master`.
- `pyproject.toml` gets the package version dynamically from `aceai/__init__.py`.

## PR Message

Read `references/release-message.md` before writing a release PR title or body. Use the template and category rules there exactly unless the user asks for a different style.
