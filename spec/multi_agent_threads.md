# Multi-Agent Threads

## 目标

AceAI 应该支持一个 session 内的多个可交互 agent thread。这里的目标不是给
subagent 面板换一个 UI，也不是把 `delegate_to_subagent` 做得更会总结；目标是
让 main agent 和 child agent 都成为可观察、可恢复、可继续输入的工作线。

这项能力值得做，是因为它把 subagent 从“一次性工具调用”升级成真正的协作单元：

- main thread 可以专注协调、归纳和最终回答；
- child thread 可以长期负责一个明确子任务，并保留自己的工具轨迹、上下文和状态；
- 用户可以切进某个 child thread 直接纠偏、补充上下文、批准工具或继续追问；
- 多个 child thread 的细节不会挤进 main thread，也不会默认污染 main model context；
- 失败、挂起和长任务恢复可以落在具体 thread 上，而不是都堆到 main run 的工具结果里。

## 当前问题

现有 AceAI 已经有三块基础：

- `AgentRunContext` 是一个 run 的状态对象，执行逻辑在 `aceai/core/run_loop.py`。
- session 是 `aceai/agent` app 层能力，当前是一个线性 transcript/event log。
- `delegate_to_subagent` 会创建 child agent，执行完成后把 handoff 给 main，并把大结果放进 artifact。

这些基础解决了 single-thread agent 和 bounded subagent handoff，但还不能提供
Codex 风格的 active agent thread 切换：

- child agent 的事件不是 session 内的一等工作线；
- TUI 只有一个 active stream、一个 `_active_worker` 和一个 `_active_run`；
- 用户输入默认只能发给 main agent；
- child run 完成后只能作为 tool result 被 main 消费，不能继续对话；
- approval、queued turns、context history 都是 session/main-agent 形状，不是 thread 形状。

因此，真正的设计边界必须从“subagent detail panel”提升到“session 内的
multi-thread runtime”。

## 核心决定

### 1. Thread 属于 app/session 层，不属于 core

`aceai/core` 继续只关心 agent definition、run state、tool protocol、event 和
context execution。core 可以暴露 `agent_id` 和 `run_id`，但不应该知道
`thread_id`、session picker、TUI focus 或 parent/child UI 关系。

`thread_id` 是 `aceai/agent` 的组织概念：

- 用来把 session 内的事件、上下文、审批、队列和 TUI focus 归到某条工作线；
- 用来表达 main/child/worker/verifier 等产品语义；
- 用来决定用户输入发给哪条 agent thread。

### 2. Session 是容器，Thread 是会话内的工作线

不要把 child thread 建成独立 session。一个用户问题可能产生 main thread 和多个
child thread，它们应该共享一个 session identity、project identity、成本统计和导出入口。

目标模型：

```text
Session
  main thread
    run A
    run B
  child thread 1
    run C
    run D
  child thread 2
    run E
```

Session 负责持久身份、project 归属、全局导出和删除生命周期。Thread 负责自己的
event log、effective context、active run、approval state 和 queued turns。

### 3. Run 仍然是一次输入到一次完成的执行单元

不要把 run 和 thread 混在一起。

- `thread_id`：一条可继续协作的工作线。
- `run_id`：这条工作线上的一次 agent 执行。
- `agent_id`：执行该 run 的 agent definition identity。

一个 thread 可以有多个 run。一个 run 不应该横跨多个 thread。child thread 的后续用户输入
应该创建该 child thread 上的新 run，而不是复用已经完成的 delegated run。

### 4. Main/child context 默认隔离

child thread 的 transcript、工具输出、失败和用户纠偏默认只进入 child thread 的
context。main thread 自动获得的内容必须保持 bounded：

- 初始 delegation 完成时，main 得到一个 compact handoff；
- child thread 后续继续工作时，main 不自动读取完整 child history；
- 只有显式 handoff、merge、引用或 bounded inspection tool 才能让 main 看到 child 细节。

这条规则比 UI 重要。没有 context 隔离，multi-agent thread 会很快退化成更复杂的
single-thread transcript，并重新触发 context-window 问题。

### 5. Active thread 决定输入路由

TUI 需要有 `active_thread_id`。当用户通过 `/subagents` 或 thread picker 切换后：

- 主输入框的普通消息发给 active thread；
- `/steer` 中断或重定向 active thread 的 active run；
- approval widget 显示 active thread 的 pending approval；
- 全局 pending approvals 可以在 status/metadata 中提示，但批准动作必须指向具体 thread。

如果 active thread 是 main，行为接近现在。如果 active thread 是 child，用户就在和
child agent 继续协作；main 不应该偷偷消费这条消息。

## 数据模型

### AgentThread

在 app/session 层增加 thread metadata：

```python
class AgentThread:
    thread_id: str
    session_id: str
    agent_id: str
    role: Literal["main", "subagent"]
    title: str
    status: Literal["idle", "running", "suspended", "completed", "failed"]
    parent_thread_id: str | None
    parent_run_id: str | None
    parent_tool_call_id: str | None
    created_at: datetime
    updated_at: datetime
```

后续可以加 `agent_name`、`role_label`、`handoff_policy`、`workspace_policy`，但第一版不要
把 worktree isolation、branch ownership 或 remote worker 调度塞进 thread 模型。

### SessionEvent

`SessionEvent` 应该增加 app 层 metadata：

```python
thread_id: str
agent_id: str
```

`run_id` 继续表示一次执行；`thread_id` 表示事件属于哪条工作线。旧的 main transcript
可以被视为 main thread 的事件，但实现时不需要为了兼容旧文件保留双路径。

### Context checkpoint

context checkpoint 需要按 `(session_id, thread_id)` 读取和写入。checkpoint 仍然是
effective model context 的优化，不是 transcript mutation。

main thread 的 checkpoint 不包含 child thread 原始历史。child thread 的 checkpoint
也不包含 main thread 历史，除非用户显式引用了某个 bounded context。

### App event envelope

`AceAgentApp.start_turn()` 现在直接 yield `AgentEvent`。multi-thread runtime 需要 app 层
envelope，例如：

```python
class AgentAppEvent:
    thread_id: str
    agent_id: str
    event: AgentEvent | ThreadLifecycleEvent
```

TUI、session recorder 和 diagnostics 都消费这个 app-layer envelope。core 仍然只产生
`AgentEvent`。

## Delegation 语义

### 创建 child thread

当 main model 调用 `delegate_to_subagent`：

1. app-layer delegation tool 创建一个 child thread；
2. child thread 获得自己的 agent definition、initial task、context brief 和 allowed tools；
3. child thread 启动自己的 run；
4. child run 的事件被记录到 child thread，并通过 app event envelope 送到 TUI；
5. parent tool call 等待 child run 的 handoff；
6. parent model context 只收到 compact `model_output` handoff。

这保留了 main agent 的工具调用语义：main 委派后需要一个结果才能继续。但等待期间，
child thread 已经是一等 thread，用户可以切进去观察、批准或纠偏。

### 用户介入 child thread

如果 child thread 正在运行：

- 普通输入默认入队到该 child thread；
- `/steer <message>` 取消该 child thread 当前 run，并用同一 thread history 启动新 run；
- 如果 child run 正在等待 approval，普通输入不启动新 run，TUI 应提示先批准或拒绝。

如果 parent tool call 正在等待这个 child thread 的初始 handoff，child 的最终 handoff
应该来自被用户介入后的最新有效 run，而不是已经被取消的 run。

### Child 后续继续工作

child thread 完成初始 delegated run 后仍可继续输入。后续 child runs 不会自动恢复 parent
tool call，因为 parent tool call 已经完成。

后续结果进入 main 的方式必须显式：

- 用户切回 main 后引用 child thread；
- main 调用 bounded inspection tool；
- 用户触发 handoff/merge 命令；
- child thread 生成新的 compact handoff event，main 下一轮可选择读取。

第一版可以只实现 “switch + continue child + explicit handoff text”。自动 merge 策略后置。

## Approval 与权限

当前 `delegate_to_subagent` 禁止 approval-required child tools，这是因为 child 只是工具内部的
临时执行。thread 化后，这个限制可以放宽，但 approval scope 必须变窄：

- approval request 归属具体 `(thread_id, run_id, tool_call_id)`；
- 批准一个 child thread 的工具，不会自动批准 main 或其他 child thread；
- session-level allow-all/disable-all 仍可以作为用户显式配置，但 runtime 记录中必须保留
  approval 发生在哪个 thread；
- TUI 在非 active thread 有 pending approval 时只提示，不把批准动作错发给当前 thread。

## TUI 契约

`/subagents` 应该从“显示 subagent 详情”升级为 thread picker。列表至少包含：

- `Main [default] (current)`；
- 每个 child thread 的 title、role、status、agent_id、run_id 或最新 run；
- pending approval / running / failed 的状态提示。

切换后：

- stream 主区域显示 active thread；
- topbar/status 显示 session id 和 active thread label；
- input placeholder 反映当前目标，例如 `Ask Main` 或 `Ask Subagent: <title>`；
- side panel 可以显示 thread list 或 parent/child relationship，但不能成为唯一交互入口。

TUI reducer 不能再假设只有一个 `TUIRunState`。目标形状是：

```python
class TUIThreadState:
    thread_id: str
    agent_id: str
    status: TUIRunStatus
    events: list[TUIEvent]
    subagents: list[str]
    active_run_id: str | None

class TUISessionState:
    active_thread_id: str
    threads: dict[str, TUIThreadState]
```

## 存储边界

推荐存储布局：

```text
~/.aceai/sessions/
  sessions.sqlite3
  files/
    <session-id>.jsonl              # main visible transcript or legacy main log
  <session-id>/
    threads/
      index.jsonl                   # thread metadata events
      <thread-id>.jsonl             # compact event log for this thread
    artifacts/
      ...
```

这保持 session row 仍然是顶层索引，同时允许 thread event log 独立增长。删除 session 时，
thread logs、artifacts 和 thread-scoped checkpoints 一起删除。

导出默认先保持 main readable transcript；后续加 `aceai export <session_id> --threads`
导出每条 thread 的 transcript 和 handoff 关系。

## 层级边界

- `aceai/core`
  - 保持 `AgentRunContext` state-only。
  - `AgentEvent` 可以携带 `agent_id`，但不携带 `thread_id`。
  - 不知道 session、TUI、thread picker 或 subagent UI。
- `aceai/agent`
  - 拥有 `AgentThread`、thread store、thread-scoped context history、delegation thread service。
  - 把 core `AgentEvent` 包装成带 `thread_id` 的 app event。
  - 决定 child thread 的创建、handoff 和 approval scope。
- `aceai/agent/tui`
  - 渲染 thread-aware app events。
  - 维护 active thread focus。
  - 把输入、steer、approval 发给 active thread。
- `spec` / `docs`
  - 记录 thread runtime、handoff、export、diagnostics 的约束，避免实现时靠聊天记忆。

## 实施顺序

### Phase 1: Thread metadata and event envelope

- 增加 `AgentThread` metadata。
- 创建 session 时自动创建 main thread。
- app event envelope 增加 `thread_id`。
- session recorder 能按 thread 记录事件。
- TUI state reducer 支持多 thread，但默认只显示 main。

验收：现有 main-only 行为通过同一 thread 模型运行。

### Phase 2: Delegation creates real child threads

- `delegate_to_subagent` 通过 app-layer thread service 创建 child thread。
- child run 事件实时进入 child thread log。
- parent tool result 继续只收到 bounded handoff。
- `/subagents` 可以切换 active thread，并查看 child 实时状态。

验收：main 委派一个 child 时，TUI 能切到 child thread 看到实时事件；main context 只包含
compact handoff。

### Phase 3: Input routing to active child thread

- 普通输入发给 active thread。
- queued turns 变成 thread-scoped。
- `/steer` 只影响 active thread。
- child thread 完成后可以继续对话。

验收：用户切到 child thread 后输入一条消息，不会写进 main transcript，也不会改变 main
effective context。

### Phase 4: Thread-scoped approvals

- pending approval lookup 支持 `(thread_id, run_id, tool_call_id)`。
- approval widget 操作 active thread。
- 非 active thread 的 pending approval 在 status 中提示。

验收：批准 child thread 的工具不会批准 main thread 或其他 child thread。

### Phase 5: Handoff and export

- 增加显式 handoff/merge 命令或 bounded inspection tool。
- export 支持 thread-aware 输出。
- diagnostics 能定位 session、thread、run、tool call。

验收：main 可以显式引用 child thread 的 bounded handoff；完整 child 细节仍在 child log/artifact，
不自动进入 main context。

## 非目标

- 不在第一版做独立 worktree 或 branch isolation。
- 不把 subagent thread 建成独立 OS 进程或远程 worker。
- 不让 main 自动读取所有 child transcript。
- 不把 thread 语义下沉到 `aceai/core`。
- 不为旧 session 文件设计复杂兼容层；实现时可以把旧 transcript 视为 main thread 输入。

## 必须守住的回归目标

- main-only session 行为仍然可用。
- 一个 session 可以列出 main 和多个 child threads。
- child thread 的完整事件可恢复、可导出、可诊断。
- main context 不包含 child 原始工具输出。
- child 后续输入不会进入 main transcript。
- parent tool call 等待 child handoff 时，用户介入后的最终 handoff 来自最新有效 child run。
- approval scope 不跨 thread 泄漏。
- 删除 session 会清理 thread logs、artifacts 和 thread checkpoints。
