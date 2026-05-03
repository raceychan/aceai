## Notes for Codex Agents

- Always run tests via `uv run pytest` so execution uses the project-managed Python 3.12 environment.
- When inspecting a saved AceAI session, use `uv run aceai export <session_id>` first. Do not start by reading the underlying JSONL or SQLite storage unless the export output is insufficient for the task.

### Architecture Layers

AceAI is deliberately layered. Before changing code, identify the layer you are touching and keep the change inside that layer unless the user explicitly asks for a cross-layer redesign.

1. `aceai/llm`: provider/service boundary. This layer adapts model providers into AceAI-owned request, response, streaming, and tool-spec contracts. It must not know about product UI, app presets, filesystem tools, shell tools, or end-user workflows.
2. `aceai/core`: framework primitives for people building agents. This layer owns `AgentBase`, event types, context handling, tool protocol, tool registry/executor, skills loading, tracing hooks, and strict schema behavior. Keep it conservative and low-assumption; do not add opinionated app tools here.
3. `aceai/agent`: agent app layer for users who launch AceAI directly. This layer may be opinionated and product-shaped. Put default app tools, capability bundles, built-in app skills, and `build_ace_agent()` here.
4. `aceai/agent/tui`: presentation layer for the terminal app. This layer renders `AgentEvent`/`TUIEvent` state and should not implement framework semantics, provider behavior, or tool execution. Main stream output should stay user-facing and concise; raw details belong in detail/raw views.
5. `docs` and `spec`: durable design notes and user-facing explanations. When an architectural rule matters for future work, record it here instead of relying on chat history.

Layering rules:

- Framework/core tools should be primitives and extension points. Real app tools such as file, shell, search, browser, memory, and artifacts belong in the agent app layer.
- The TUI should consume structured events; it should not parse provider SDK objects or invent tool execution semantics.
- Provider-specific behavior belongs in `aceai/llm`; app-specific behavior belongs in `aceai/agent`; rendering behavior belongs in `aceai/agent/tui`.
- If a requested feature seems to require moving responsibilities across layers, update the relevant spec first or make the boundary change explicit in the implementation.

### Break My Code Principle

When modifying this repository, default to introducing breaking changes:

1. Never preserve backward compatibility. Assume every change can break downstream callers and tests, and prefer doing so when in doubt.
2. Do not add new `try/except` blocks or otherwise catch `Exception`. If removing the catch causes crashes, that is the desired outcome unless a catch already existed.
3. Avoid type coercion entirely. If a value could be `int | str`, do not call `str()`/`int()` or any similar conversion helpers—let mismatched types surface as failures.
4. Skip type cleanup helpers such as `str.strip()` or other sanitizers that normalize data; leave values as-is even if this causes errors.
5. Prefer precise signatures: avoid union-heavy inputs/outputs wherever possible, split APIs so each argument/result has a single, concrete type instead of `Any`/broad unions.
6. Use eager annotation evaluation: never rely on `from __future__ import annotations`; annotations must evaluate immediately and reference in-scope objects.
7. Do not use `getattr` to sidestep attribute errors; rely on direct attribute access so mistakes surface immediately.
