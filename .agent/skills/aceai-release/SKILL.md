---
name: aceai-release
description: "Use for AceAI release work: preparing a version branch, committing and pushing release changes, drafting release PR messages, merging into master, tagging only after merge, and triggering PyPI publication."
---

# AceAI Release

Use this skill when the user asks to release AceAI, cut a version, prepare a release PR, tag a release, publish to PyPI, or check release readiness.

Important: release work must start from this skill and its checklist. Do not treat `make release` as the release workflow entrypoint. The Makefile/script may be used only as one implementation step after this skill has been read, the repository state has been inspected, and the release sequence below has been considered.

This repository publishes to PyPI from GitHub Actions when a pushed tag starts with `v`. The tag commit must already be contained in `origin/master`; otherwise the publish job fails. Therefore, do not tag the version branch before the PR is merged.

## Release Sequence

## Fast Path

Use this path when the repository is already on `version/<x.y.z>`, the branch is
tracking `origin/version/<x.y.z>`, and the release version is the next patch after
the latest remote tag.

1. Run the minimum state inspection in parallel where possible:
   - `git status --short --branch`
   - `git branch --show-current`
   - `git ls-remote --tags origin`
   - `rg -n "__version__" aceai/__init__.py`
2. Update only release metadata that is missing:
   - `aceai/__init__.py`
   - `CHANGELOG.md`
3. Draft one release message and reuse it for:
   - the release PR body
   - GitHub Release notes
   - the final user summary
4. Run validation before commit:
   - `uv run pytest -v`
   - `uv build`
5. Commit, push, open PR, wait for PR checks, merge, tag from `master`, publish,
   verify PyPI, and create the next-cycle branch.

Avoid extra source exploration in this path unless tests fail or the version
state is inconsistent.

1. Inspect state.
   - Run `git status --short`, `git branch --show-current`, `git remote show origin`, and `git ls-remote --tags origin`.
   - Confirm the release branch is `version/<x.y.z>` and the base branch is `master`.
   - If release changes were made directly on `master`, create `version/<x.y.z>` before committing so the release PR contains the work. Do not commit release changes on `master` and then try to recover the branch afterwards.
   - Check `aceai/__init__.py` has `__version__ = "<x.y.z>"`; update it if needed.
   - Check `CHANGELOG.md` if the release needs human-facing notes.

2. Validate before committing.
   - Run tests with `uv run pytest -v`.
   - Run `uv build` when packaging behavior or release metadata changed.
   - If e2e tests are intentionally excluded, say that explicitly.

3. Commit and push the release branch.
   - Stage only intended files; preserve unrelated local changes.
   - Re-check `git status --short` before staging and leave generated scratch files untracked unless they are part of the release.
   - Use a focused commit message, usually `Release version <x.y.z>` for final release prep.
   - Push with `git push origin version/<x.y.z>`.

4. Open the release PR.
   - Base: `master`.
   - Head: `version/<x.y.z>`.
   - Read `references/release-message.md` before drafting the PR body.
   - The PR body must group changes by category and include verification.
   - Reuse the same release-message draft for the PR body and later GitHub Release notes.

5. Merge into `master`.
   - Wait for CI to pass.
   - If CI fails, inspect failed logs first with `gh run view <run-id> --log-failed`.
   - For a focused test failure, reproduce with the narrowest local `uv run pytest ...`
     command, then rerun `uv run pytest -v` before amending the release commit.
   - Prefer `git commit --amend --no-edit` plus `git push --force-with-lease` for
     release-branch fixups before merge, preserving a single release-prep commit.
   - Merge the PR using the repository's preferred merge method.
   - Do not push release tags until the merge commit is visible on `origin/master`.

6. Tag from `master`.
   - Run `git checkout master` and `git pull origin master`.
   - Confirm `aceai/__init__.py` still matches the tag version.
   - Create an annotated tag: `git tag -a v<x.y.z> -m "Release version <x.y.z>"`.
   - Push the tag: `git push origin v<x.y.z>`.

7. Create the GitHub Release.
   - Create the release after the tag is pushed and the tag-triggered CI starts.
   - Use `gh release create v<x.y.z> --title "AceAI v<x.y.z>" --notes "<release notes>"`.
   - Base release notes on the PR body and `CHANGELOG.md`.
   - Include `Summary`, grouped change categories, `Breaking Changes`, and `Verification`.
   - If publish verification is still pending, write that explicitly, then update
     the release notes after the tag CI and PyPI checks succeed.
   - If the release already exists, update it instead of creating a duplicate.

8. Verify publication.
   - Watch the tag-triggered CI publish job.
   - Confirm PyPI shows the expected version. Prefer the exact version endpoint `https://pypi.org/pypi/aceai/<x.y.z>/json`; the project-level JSON can lag briefly after publish.
   - Confirm the package can be installed when appropriate, ideally from outside the repo, and refresh the package index to avoid a false negative immediately after publish:
     `UV_CACHE_DIR=/tmp/uv-cache uvx --refresh-package aceai --from aceai==<x.y.z> python -c "import aceai; print(aceai.__version__)"`.
   - Confirm the GitHub Release exists and points at `v<x.y.z>`.

9. Prepare next cycle.
   - By default, create and push `version/<next>` from the latest `master` after the release tag publish workflow is verified. Do this unless the user explicitly says not to.
   - Default next-version selection:
     - Before `1.0.0`, bump patch: `0.2.0` -> `0.2.1`.
     - At and after `1.0.0`, bump minor: `1.0.0` -> `1.1.0`.
   - If the user explicitly requests patch, minor, major, or an exact version, follow the user instead.
   - Keep the next-cycle branch as a real remote branch with `git push -u origin version/<next>` so future work does not accidentally continue on `master`.
   - Do not delete release branches unless the user asks.

## Repository-Specific Checks

- Always run tests via `uv run pytest` in this repository.
- CI runs once for release PR commits through the `pull_request` trigger; branch
  push CI is intentionally limited to `master` to avoid duplicate PR checks.
- Docs deploy is restricted to `master`; docs builds run only when docs-related
  files change.
- The CI publish job runs only for pushed tags matching `v*`.
- The publish job checks the tag commit is contained in `origin/master`.
- `pyproject.toml` gets the package version dynamically from `aceai/__init__.py`.

## PR Message

Read `references/release-message.md` before writing a release PR title or body. Use the template and category rules there exactly unless the user asks for a different style.
