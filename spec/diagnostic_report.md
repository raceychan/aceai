# AceAI Diagnostic Report

## 背景

当用户遇到工具参数校验失败、provider 异常、TUI 崩溃或不可恢复的
`run_failed` 时，AceAI 应该自动留下完整证据。不能要求用户先失败、
再记住运行另一个命令、再复制多段输出；那样最容易丢掉关键上下文。

单独截图或复制 traceback 通常不够。维护者真正需要的是：

- AceAI 版本、Python 版本、平台、当前项目和工作目录；
- session id、run id、step id、失败事件和相邻上下文；
- 用户原始输入、模型选择、工具调用参数、工具失败输出；
- 可复现命令和导出的可读 transcript；
- 明确的脱敏边界，避免把 API key 或本地密钥带进报告。

AceAI 已有 session/event-log/export 机制，因此诊断报告应属于
`aceai/agent` app 层能力；TUI 只负责触发和展示，`aceai/core` 继续只暴露结构化事件。

## 用户体验

默认体验：

1. 用户正常使用 TUI。
2. 如果某次 run 出现 `run_failed` 或 TUI 捕获到未恢复异常，AceAI 自动生成一份本地诊断报告文件。
3. TUI 主界面只显示一条简短通知，告诉用户报告文件路径。
4. 用户把这个文件发给维护者即可。

示例通知：

```text
AceAI hit an error. Diagnostic report saved:
~/.aceai/reports/2026-05-08T11-42-03Z-<session_id>-<run_id>.md
```

手动入口只是补救能力：

1. TUI 内输入 `/report`，重新生成最近一次失败的诊断报告。
2. CLI 执行 `aceai report <session_id>`，用于用户已经退出 TUI 的场景。

默认行为：

- `run_failed` 发生时自动生成，报告定位到该 run。
- TUI 进程级异常发生时自动生成，报告定位到当前 active run；如果没有 active run，则定位到当前 session。
- 如果没有失败 run，报告定位到最近一个 run，并标记 `status: completed` 或 `running`。
- 默认写入本地 Markdown 文件，并在 TUI 里显示文件路径；不要求用户执行额外命令。
- 支持 `--stdout` 输出到终端，方便 issue / chat 复制。

建议命令：

```bash
aceai report <session_id>
aceai report <session_id> --run <run_id>
aceai report <session_id> --stdout
aceai report <session_id> --include-raw
```

TUI 文案示例：

```text
Diagnostic report created:
~/.aceai/reports/2026-05-08-aceai-report-<session_id>.md

Share this file with the AceAI maintainers. It includes the failed run,
tool-call context, environment, and a sanitized session transcript.
```

## 文件形态和大小控制

报告默认是一个 Markdown 文件，扩展名 `.md`。它本质上是 failure log：
可读、可直接发给维护者、也能被 issue 系统预览。

默认路径：

```text
~/.aceai/reports/YYYY-MM-DDTHH-MM-SSZ-<session_id>-<run_id>.md
```

大小目标：

- 目标小于 200 KB；
- 硬上限 1 MB；
- 超过硬上限时保留 summary、failure、最近 user input、失败 tool context，并截断 transcript 和 raw events；
- 每个长字段按字符数截断，保留头尾，中间写明省略字节数；
- tool output 默认最多保留 32 KB；
- transcript 默认只包含目标 run；目标 run 不可切分时，保留 session 尾部最近 80 个 compact events；
- raw events 默认不写入，只有 `--include-raw` 或 debug config 开启时才写入。

报告末尾必须写明是否发生截断，例如：

```text
truncation: transcript truncated to 65536 chars; tool output truncated to 32768 chars
```

## 报告内容

报告使用 Markdown，优先让维护者一眼看懂，而不是暴露内部 JSONL。

````markdown
# AceAI diagnostic report

## Summary

- session_id:
- run_id:
- status:
- error:
- project:
- workspace:
- aceai_version:
- python:
- platform:
- created_at:
- report_path:
- report_truncated:

## Reproduction

```bash
aceai resume <session_id>
aceai export <session_id>
```

## User Input

最近一次 user message 原文。

## Failure

失败事件、错误消息、事件时间、step id。

## Tool Context

失败前后的 tool call：

- tool_name
- call_id
- arguments
- status
- output/error

## Transcript

目标 run 的可读输出。默认不导出整段 session，避免文件过大。
如果无法可靠切分 run，则只包含 session 尾部 compact transcript。

## Raw Events

仅在 `--include-raw` 时包含目标 run 的 session event JSON。
````

## 脱敏策略

报告生成器必须在写出前统一脱敏，不能依赖调用方记得处理。

默认脱敏：

- 环境变量中名称包含 `KEY`、`TOKEN`、`SECRET`、`PASSWORD`、`CREDENTIAL` 的值；
- OpenAI、Anthropic、Google、Azure 等常见 API key 形态；
- HTTP `Authorization` header；
- URL 中的 token/query secret；
- 本地 home path 可以保留，因为定位 workspace 对调试有价值。

不默认脱敏：

- 用户输入；
- 工具参数；
- 文件路径；
- 工具输出。

原因是这些字段往往正是复现问题所需证据。后续可以加
`--redact-paths` 或 `--redact-tool-output`，但第一版不自动破坏上下文。

## 架构边界

新增 app 层模块：

```text
aceai/agent/diagnostics.py
```

职责：

- 从 `SessionStore` 读取 `SessionMetadata` 和 `EventLog`；
- 选择目标 run；
- 构造 `DiagnosticReport` 数据结构；
- 渲染 Markdown；
- 执行脱敏。

TUI 层：

- 监听当前 run 的 `run_failed`，在对应事件已经写入 session 后自动生成报告；
- 在顶层异常处理处捕获 TUI/runner 未恢复异常，先 flush/finalize 当前 session，再生成报告；
- 在 command handler 中支持 `/report` 作为手动重建入口；
- 调用 app 层 report service；
- 只展示成功/失败通知和输出路径。

CLI 层：

- 在 `aceai.agent.tui.cli` 中新增 `report` command；
- 与现有 `export` 一样延迟加载 TUI/session extra；
- 支持 `--file`、`--stdout`、`--run`、`--include-raw`。

Core 层：

- 不新增诊断报告 API；
- 不读取 session；
- 不知道 Markdown、文件路径或 TUI 命令。

## 数据模型

```python
class DiagnosticReport(Struct, frozen=True, kw_only=True):
    session_id: str
    run_id: str
    status: str
    summary_error: str
    project_id: str
    project_name: str
    workspace: str
    aceai_version: str
    python_version: str
    platform: str
    created_at: str
    report_path: str
    truncated: bool
    truncation_notes: list[str]
    events: list[SessionEvent]
    transcript: str
```

第一版可以直接返回 Markdown 字符串，不需要把该 record 暴露为稳定 public API。
如果后续要上传到 issue tracker 或遥测系统，再稳定结构化 schema。

## Run 选择规则

1. 如果显式传入 `--run`，只导出该 run；不存在则失败。
2. 自动触发时使用触发失败事件的 run id。
3. 手动触发时，从 session events 逆序查找第一个 `run_failed` 或 `error` 所属 run。
4. 如果没有失败 run，选择最近一个 run。
5. 如果 session 中没有 run id，退化为当前 session 尾部 transcript。

## 自动触发点

第一版只自动生成真正失败的报告，避免用户正常工具失败时被噪音打扰：

- `run_failed`：必须自动生成；
- TUI runner 未恢复异常：必须自动生成；
- `tool_failed`：默认不单独生成，因为工具失败可能已经回给模型并恢复；如果该工具失败最终导致 run failed，会随 run failed 报告出现；
- 用户取消、approval reject、max tool calls reached：不生成。

生成顺序必须是：

1. 将失败事件记录到 `SessionRecorder`；
2. flush 当前 assistant/tool buffer；
3. 调用 report service 生成文件；
4. 在 TUI 展示路径通知。

如果 report 生成本身失败，不应掩盖原错误；TUI 只追加一条
`Diagnostic report failed: <reason>` 通知。

## 测试策略

重点测试 app 层，不把 TUI 作为主要测试入口：

- `SessionStore` 中构造包含 `tool_failed` 的 session，生成报告应包含 tool name、arguments、error、session id、run id；
- 包含 `run_failed` 的 session，报告 summary 应定位 run error；
- 没有失败事件时，报告应选择最近 run；
- 自动 `run_failed` 触发时，报告必须包含触发失败事件；
- tool failure 后模型恢复成功时，不自动生成报告；
- 报告超过大小上限时，应截断长 transcript/tool output 并写入 truncation notes；
- 脱敏函数应替换 API key、Authorization header 和 secret env value；
- CLI `aceai report <session_id> --stdout` 输出 Markdown；
- TUI 自动失败路径验证调用 report service 并产生 session notice；
- TUI `/report` command 只验证调用 report service 并产生 session notice。

## 迭代顺序

1. 新增 `aceai/agent/diagnostics.py` 和 focused tests，先实现小而完整的 Markdown report。
2. 在 TUI `run_failed` / runner exception 路径自动生成文件并通知用户。
3. 在 CLI 加 `aceai report <session_id> --stdout`，复用 `SessionStore`，作为补救入口。
4. 给 TUI 加 `/report` 命令，手动重建最近失败报告。
5. 后续再加 `--include-raw`、剪贴板复制、自动附加最近 traceback。
