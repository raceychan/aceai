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

## 当前实现状态

截至当前实现，AceAI 已经具备 multi-agent thread 的 MVP 主干：

- session 内有 `main` 和 `subagent` thread metadata；
- `SessionEvent` 带 `thread_id` / `agent_id`，旧事件默认归入 `main`；
- context checkpoint 按 `(session_id, thread_id)` 隔离；
- `delegate_to_subagent` 会创建真实 child thread，并把 child run 事件写入 child thread；
- parent tool result 只拿 compact handoff / audit，不自动吞入 child 原始 transcript；
- `AceAgentApp` 有 `active_thread_id`，普通输入、queue、context history 按 active thread 路由；
- child thread 完成后可以继续对话，且 session attach / app 重建后可以恢复 child agent 继续对话；
- TUI 有最小切换入口：`/subagents <thread_id>`。

当前尚未完成的是完整 realtime thread runtime。也就是说，child thread 已经是可持久化、
可恢复、可继续输入的工作线，但初始 delegation 运行期间仍然主要通过 parent tool call
同步等待 child handoff。approval-required child tools、多个 thread 同时 pending approval、
运行中 steer 后让 parent 等待最新 child handoff，这些能力需要下一阶段的 thread runtime。

## Realtime runtime 的收益

做 realtime thread runtime 之前，child thread 的能力边界是：

```text
main 委派 child
child 在 parent tool call 内运行
child 跑完后写入 thread log
用户事后可以切到 child 继续问
```

这已经解决了记录、恢复、后续继续对话和 main/child context 隔离，但 child 初始运行期间
还不是一个真正独立的 runtime actor。

做完 realtime thread runtime 之后，能力会变成：

```text
main 委派 child
child 作为独立 thread runtime 实时运行
用户可以切到 child 看进度、批准工具、/steer 纠偏
child 最终 handoff 给 parent
parent tool call 收到最新有效 handoff 后继续
```

这带来五个能力差距：

- 实时可观察：child 运行中，TUI 可以收到 `[child-thread]` 的事件，而不是只看到 parent tool call 未返回。
- 实时介入：child 跑偏时可以立刻 `/steer`，parent 最终等待的是纠偏后的 handoff。
- approval 不串线：approval 归属 `(thread_id, run_id, tool_call_id)`，批准 child 不会批准 main 或其他 child。
- 多子任务并行基础：app 可以持有多个 `ThreadRuntime`，每个 thread 有自己的 run、queue、approval 和 handoff。
- 更接近 Codex `/subagents`：用户是在多个 agent thread 之间切换工作，而不是查看一个工具结果。

代价也必须显式承认：

- app runtime 从 single active run 变成 thread runtime registry；
- TUI state 最终需要从单个 `TUIRunState` 升级为 session/thread state；
- parent 等 child handoff 要处理 suspend、approval、steer、cancel 和失败；
- approval、queue、worker lifecycle 都要 thread-scoped；
- 测试要覆盖 parent waiting + child approval + user steer 等组合。

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
    metadata: dict[str, object]
    created_at: datetime
    updated_at: datetime
```

当前 `metadata` 用来保存可恢复 child agent 所需的 app-layer 配置，例如：

```python
{
    "instructions": "...",
    "context_brief": "...",
    "allowed_tools": ["read_text_file"],
}
```

后续可以加 `agent_name`、`role_label`、`handoff_policy`、`workspace_policy`，但不要把
worktree isolation、branch ownership 或 remote worker 调度塞进第一版 thread 模型。

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

当前 `AceAgentApp.start_turn()` 仍然直接 yield `AgentEvent`，并由 app 内部按 active
thread 记录。P4 开始后，app 层增加了 envelope stream；完整 realtime multi-thread runtime
需要继续围绕这个 envelope 演进，例如：

```python
class AgentAppEvent:
    thread_id: str
    agent_id: str
    event: AgentEvent
```

后续如果需要 thread lifecycle / handoff lifecycle，可以扩展 envelope event union。TUI、session
recorder 和 diagnostics 最终都应消费这个 app-layer envelope。core 仍然只产生 `AgentEvent`，
不感知 `thread_id`。

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

## 实施顺序与状态

### Phase 1: Thread metadata and event storage - Done

- 增加 `AgentThread` metadata。
- 创建 session 时自动创建 main thread。
- `SessionEvent` 增加 `thread_id` / `agent_id`。
- session recorder 能按 thread 记录事件。
- context checkpoint 按 thread 读取和写入。
- 删除 session 时清理 thread metadata。

验收：现有 main-only 行为通过同一 thread 模型运行。

当前状态：

- 已完成。
- 全量测试通过。
- 备注：事件仍保存在 session-level JSONL 中，通过 `thread_id` 过滤成 thread log；物理拆分到
  `<thread-id>.jsonl` 可以后置。

### Phase 2: Delegation creates real child threads - Done

- `delegate_to_subagent` 通过 app-layer thread service 创建 child thread。
- child run 事件进入 child thread log。
- parent tool result 继续只收到 bounded handoff。
- subagent audit / handoff 带 `thread_id`。
- child thread metadata 持久化初始 instructions、context brief 和 allowed tools。

验收：main 委派一个 child 时，session 可以列出 child thread；child 事件只进入 child
thread；main context 只包含 compact handoff。

当前状态：

- 已完成。
- 全量测试通过。
- 备注：这里的 “real child thread” 指存储和恢复语义真实；运行期仍主要由 parent tool call
  同步等待 child handoff，实时 runtime 放到 Phase 4。

### Phase 3: Input routing and cold restore for active child thread - Done

- 普通输入发给 active thread。
- queued turns 变成 thread-scoped。
- child thread 完成后可以继续对话。
- session attach / app 重建后可以恢复 child agent，并继续同一个 child thread。
- TUI 提供最小切换入口 `/subagents <thread_id>`。

验收：用户切到 child thread 后输入一条消息，不会写进 main transcript，也不会改变 main
effective context。

当前状态：

- 已完成。
- 全量测试通过。
- 备注：`/steer` 现在按 active thread 启动新 run，但仍受 single `_active_worker` 限制；真正运行中
  child steer 和 parent handoff 等待关系放到 Phase 4。

### Phase 4: Realtime child thread runtime - Done

目标：把 child 从“parent tool call 内同步运行”升级为 app 层独立 runtime actor。

新增 runtime 形状：

```python
class ThreadRuntime:
    thread_id: str
    agent_id: str
    agent: Agent
    active_run: AgentRunContext | None
    queued_questions: list[str]
    pending_approval: PendingToolApproval | None
    worker_task: object | None
    handoff_future: object | None
```

`AceAgentApp` 需要从：

```text
_active_run
_queued_questions
_active_thread_id
```

演进为：

```text
_thread_runtimes: dict[thread_id, ThreadRuntime]
_active_thread_id: str
```

实施项：

- 引入 app-layer `AgentAppEvent(thread_id, agent_id, event)` envelope。Done: `start_turn_events()`
  已经可以实时发出 child thread events，旧 `start_turn()` 保持只返回 active thread 的裸
  `AgentEvent`。
- `delegate_to_subagent` 启动 app-layer child runtime，parent tool call 等待 runtime 的
  `handoff_future`。Done。
- child runtime 自己消费 child agent stream，实时记录事件并发给 TUI。Done。
- child failure / completion 都落在 child runtime 上，handoff future 会 resolve 或 fail，
  parent tool call 不会挂死。Done。
- child suspend 会被 child runtime 标记为 suspended，并让 parent tool call 得到明确失败。
  Done for the current approval-free child tool policy。
- 如果用户在 child 初始 handoff 前 `/steer`，parent tool call 等待最新有效 run 的 handoff。
  Moved to a later interaction phase because it requires TUI-side concurrent input while a parent turn is still
  streaming。

验收：

- child 初始运行期间，TUI 能实时看到 child events。
- parent tool call 等待 child handoff，同时 `start_turn_events()` 能先于 parent
  `ToolCompletedEvent` 发出 child events。
- child run failed / suspended 不会让 parent handoff future 永久挂起。
- 旧 `start_turn()` 仍只输出 active thread 的裸 `AgentEvent`，不会混入 child events。

### Phase 5: Thread-scoped approvals - Done

- app 层维护 per-thread active run，不再只依赖单个 `_active_run`。Done。
- `pending_approval_request(thread_id=...)` 可以读取指定 thread 的 pending approval。Done。
- `approve_tool(...)` / `reject_tool(...)` 支持显式传入 `thread_id`、`run_id`、`tool_call_id`，
  并在 resume 前校验目标匹配。Done。
- 默认 approval widget 语义仍然是操作 active thread。Done。
- approval-required child thread run 可以在 app runtime 中挂起并按指定 thread 恢复。Done。
- 非 active thread 的 pending approval 在 status 中提示。Moved to Phase 6, because it belongs to the
  `/subagents` selector/status UI.

验收：批准 child thread 的工具不会批准 main thread 或其他 child thread。

### Phase 6: Subagent panel thread selector - Done

目标：复用现有 subagent TUI，而不是立刻做全新的 thread picker。

界面策略：

- `/subagents` 继续打开现有 `SubagentStatusWidget`。Done。
- 面板主体继续显示当前 subagent detail 和分页内容。Done。
- 面板底部增加 thread selector。Done。
- selector 固定把 `Main` 排在第一项，然后列出所有 child thread。Done。
- selector label 显示 title、role/status，并标记 current。Done。
- 用户选中某个 thread 后调用 app-level `switch_thread(thread_id)`。Done。
- 切换后主 stream 显示 selected active thread 的 event log。Done。
- input、queue 和 approval target 都跟随 selected active thread。Done through app active-thread state;
  richer pending-approval badges can be added in Phase 7 diagnostics if needed.

推荐形状：

```text
subagents  3 total | 0 running | 3 done | 0 failed  page 1/3

#1 [done] Inspect version
   tools none | steps 1 | agent child-... | thread ...
   summary: ...

Active thread
[ Main                      v]
  Main [current]
  Inspect version [done]
  Check tests [done]
```

这条路线的收益：

- 复用现有 subagent detail 和分页逻辑；
- 用户已经知道 `/subagents` 会打开这个区域；
- 不需要马上把整个 TUI reducer 重写成完整 `TUISessionState`；
- Phase 5 的 approval 操作可以自然依赖这个 selector：先选中 child，再 approve/reject。

后续如果这个 panel selector 不够用，再把它扩展成完整 thread picker。目标形状仍然可以是：

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

验收：用户可以在 `/subagents` 面板底部选择 Main 或 child thread，选择后主输入和 stream 都切到
该 active thread。

### Phase 7: Handoff, merge, export, diagnostics - Done

- 初始 delegated run 已经通过 parent `delegate_to_subagent` tool result 提供 bounded handoff，
  main context 只收到 compact `model_output`。Done。
- export 默认保持 main readable transcript，不自动导出 child 原始细节。Done。
- `aceai export <session_id> --threads` 导出每条 thread 的 transcript 和 metadata。Done。
- thread-aware export 在每条事件前输出 `event_id`、`thread_id`、`agent_id`、`run_id`、
  `step_id`、`tool_call_id` 等定位信息。Done。
- 自动 merge 策略仍是非目标；后续如要做，应作为显式用户命令单独设计，而不是默认把 child
  transcript 注入 main context。

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
