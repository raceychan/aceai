---
name: aceai-real-e2e
description: "Use when validating AceAI behavior with a real provider/model and real AceAgentApp runtime, especially after agent runtime, subagent, inbox, tool orchestration, provider, or TUI-stream plumbing changes where unit tests are not enough."
---

# AceAI Real E2E

Use this skill when a change needs proof from a real AceAI query, not only fake LLM fixtures or unit tests. Typical triggers:

- subagent or background job behavior
- agent inbox delivery
- run lifecycle, cancellation, or failure handling
- provider/tool-call streaming behavior
- bugs that appeared only in an interactive session
- user asks for a "真实 query", "real e2e", or asks whether it actually ran

## Default Approach

Prefer the bundled script over manually typing into the Textual TUI:

```sh
uv run python -u .agent/skills/aceai-real-e2e/scripts/run_real_agent_query.py --query 'YOUR QUERY'
```

The script uses the repo's configured AceAI provider/model through `load_config()`, builds the normal AceAI agent with `build_agent(config)`, runs the query through `AceAgentApp.start_turn_events(...)`, and prints key events:

- provider/model/reasoning
- tool starts/completions for subagent and collection tools
- inbox item/delivered events
- tool/run failures
- completed run threads
- final summary counters

For longer prompts, pass a file:

```sh
uv run python -u .agent/skills/aceai-real-e2e/scripts/run_real_agent_query.py --query-file /tmp/aceai_real_query.txt
```

## Query Shape

Make the query read-only unless the user explicitly wants edits. For multi-agent validation, ask for concrete tool usage and a final status report:

```text
用 spawn_subagent 启动两个后台只读子 agent，不要用 delegate_to_subagent。
子 agent A：检查 <path A> 的 <risk A>。
子 agent B：检查 <path B> 的 <risk B>。
主 agent 自己检查 <path C>。
然后调用 collect_subagent_results 汇总，最后回答是否有 failed、inbox 是否进入主 agent、是否有目标错误风险。不要修改文件。
```

## TUI Caveat

Do not count a TUI paste as a valid e2e if focus was not in the input box. Textual app-level hotkeys can consume pasted characters and open unrelated UI such as idea editing. If the goal is runtime/tool/inbox proof, use the script. If the goal is visual rendering proof, use `aceai-ghostty-launch` and `aceai-tui-screenshot` after the runtime path is already known to work.

## Reporting

In the final response, say whether this was:

- unit-test validation
- real provider/runtime validation
- visual TUI validation

For real provider/runtime validation, include the important counters from the script, for example:

```text
CONFIG provider=codex model=gpt-5.5 reasoning=auto
SUMMARY_INBOX_COUNT 2
SUMMARY_DELIVERED_COUNT 2
SUMMARY_FAILED_COUNT 0
SUMMARY_COMPLETED_THREADS <child A>,<child B>,main
```

If the script itself fails because the probe code is wrong, say that clearly and fix the probe before interpreting the AceAI behavior. Do not present a probe exception as an app failure.
