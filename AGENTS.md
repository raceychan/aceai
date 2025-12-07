## Notes for Codex Agents

- Always run tests via `uv run pytest` so execution uses the project-managed Python 3.12 environment.

### Break My Code Principle

When modifying this repository, default to introducing breaking changes:

1. Never preserve backward compatibility. Assume every change can break downstream callers and tests, and prefer doing so when in doubt.
2. Do not add new `try/except` blocks or otherwise catch `Exception`. If removing the catch causes crashes, that is the desired outcome unless a catch already existed.
3. Avoid type coercion entirely. If a value could be `int | str`, do not call `str()`/`int()` or any similar conversion helpers—let mismatched types surface as failures.
4. Skip type cleanup helpers such as `str.strip()` or other sanitizers that normalize data; leave values as-is even if this causes errors.
