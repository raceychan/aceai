# Agent 层响应结构探索

## 目标

- 在保留 service 层 `LLMResponse` 全量信息的基础上，为 agent 提供更丰富且可扩展的响应结构（reasoning trace、工具执行记录、最终答案等）。

## 已知现状

- `AgentBase.run` 为唯一入口，作为 `AsyncIterator[AgentEvent]` 直接 yield 每个事件；成功场景以 `agent.run.completed` 结束并暴露 `final_answer` 字段，失败场景触发新的 `agent.run.failed` 事件。

## 初步草案

```python
class ToolExecutionResult(Record):
    call: LLMToolCall
    output: str
    error: str | None = None

class AgentStepAnnotations(Record):
    safety: list["AgentSafetyNote"] = field(default_factory=list)
    citations: list["AgentCitationRef"] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

from datetime import datetime, timezone
from uuid import uuid4

class AgentStep(Record):
    step_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    llm_response: LLMResponse
    tool_results: list[ToolExecutionResult] = field(default_factory=list)
    annotations: AgentStepAnnotations = field(default_factory=AgentStepAnnotations)

class AgentResponse(Record):
    turns: list[AgentStep]
    final_output: str
```

- `AgentBase.run` 在每次 LLM 调用后创建 `AgentStep`，将 `LLMResponse` 原封不动塞入 `llm_response`。
- 执行工具后，向 `turn.tool_results` 写入结果；返回给模型的 tool outputs 继续通过 `LLMToolUseMessage` 发送，逻辑保持不变。
- 默认 API 仅提供 streaming 事件；若调用方需要 `AgentResponse`，需自行 collect 这些事件并构造。

## 落地策略

1. **Streaming-only**：`AgentBase.run` 直接 yield `AgentEvent`，不再在内部发布或缓存事件；需要一次性结果的调用方自己 `collect`。
2. **直接演进**：当前尚无外部用户，`AgentBase` 在语义层彻底转向 streaming，删除 `handle`/`stream`/`str` 返回值等旧接口。
3. **final tool 处理**：`final_answer` 仍作为普通 `ToolExecutionResult` 写入最后一个 turn，并在终结事件 `final_answer` 字段中携带输出，方便消费方提取。

## 下一步

1. 在 `aceai/llm/models.py` 或 `aceai/agent.py` 定义上述 record。
2. 改造 `AgentBase.run` 使其直接 yield 事件，并移除历史 `handle`/`stream` 语义。
3. 选一个 agent/client 更新调用方式，验证实战体验，再回写文档。

## 业界对标与最佳实践

- **Google ADK**：ADK 要求 provider/service/agent 全链路输出严格事件枚举与可重放 trace。我们在 `AgentStep` 中保存完整 `LLMResponse`、tool 执行产物，使得每个 turn 自包含，满足 ADK “显式事件 + 可重放” 的指导。
- **OpenAI Agents Python SDK**：该 SDK 将每次 LLM+Tool 交互序列化为不可变 step，包含调用、工具输出与最终消息。`AgentStep` 的结构化记录与 `final_answer` 作为 `ToolExecutionResult` 的策略，提供了同等级别的审计/溯源能力，同时保持我们自定义类型的可控性。
- **Streaming 接口**：两家实现都以 streaming event/iterator 暴露进行中状态。我们默认提供 `AsyncIterator[AgentEvent]`，并允许外部 `collect`，可与 ADK/OpenAI 的订阅式消费模式互操作。

## 改进建议

1. **事件枚举**：比照 ADK/OpenAI 的 `response.output_text.delta`/`step.completed`，为 agent 侧补充显式的 turn-level event taxonomy（如 `llm.completed`、`tool.started`、`tool.completed`），避免消费方用空字段推断状态。
2. **元数据增强**：为 `AgentStep`、`ToolExecutionResult` 添加可选 `id`、`timestamp` 与 `latency_ms`，以支持长程运行的暂停/恢复、审计日志以及下游的 run history。

> 目前 `AgentStep` 已内建 `step_id`（默认 `str(uuid4())`）与 `timestamp`（默认 `datetime.now(timezone.utc)`）；`ToolExecutionResult` 后续可按相同模式扩展。
3. **Safety/Citation 扩展**：对齐 ADK/OpenAI 的安全与引用注解，尽快设计 `annotations` 命名空间或轻量 schema，以承载 provider 返回的 policy/safety verdict、citation span 等信息。

### 轻量化 Safety / Citation 注解

```python
class AgentSafetyNote(Record):
    verdict: Literal["allow", "review", "block"] = "allow"
    category: str

class AgentCitationRef(Record):
    label: str
    url: str | None = None
    provider_ref: LLMCitationRef | None = None
```

- `AgentStep.annotations.safety` 存放 `AgentSafetyNote` 列表。
- `AgentStep.annotations.citations` 存放 `AgentCitationRef` 列表。
- provider → service → agent 只需传递 verdict/category 或 citation label/url；若有额外信息，可挂在 `AgentStep.annotations.extra` 或 `AgentCitationRef.provider_ref`，避免提前固定 schema。

## Agent 级事件流提案

### 数据结构

```python
class AgentLifecycleEvent(Record, kw_only=True):
    EVENT_TYPE: AgentEventType
    step_index: int
    step_id: str

    @property
    def event_type(self) -> AgentEventType: ...

class LLMStartedEvent(AgentLifecycleEvent):
    EVENT_TYPE = "agent.llm.started"

class LLMCompletedEvent(AgentLifecycleEvent):
    EVENT_TYPE = "agent.llm.completed"
    step: AgentStep

class LLMOutputDeltaEvent(AgentLifecycleEvent):
    EVENT_TYPE = "agent.llm.output_text.delta"
    text_delta: str

class ToolStartedEvent(AgentLifecycleEvent):
    EVENT_TYPE = "agent.tool.started"
    tool_call: LLMToolCall
    tool_name: str

class ToolOutputEvent(ToolStartedEvent):
    EVENT_TYPE = "agent.tool.output"
    text_delta: str

class ToolCompletedEvent(ToolStartedEvent):
    EVENT_TYPE = "agent.tool.completed"
    tool_result: ToolExecutionResult

class ToolFailedEvent(ToolStartedEvent):
    EVENT_TYPE = "agent.tool.failed"
    tool_result: ToolExecutionResult
    error: str

class StepCompletedEvent(AgentLifecycleEvent):
    EVENT_TYPE = "agent.step.completed"
    step: AgentStep

class StepFailedEvent(AgentLifecycleEvent):
    EVENT_TYPE = "agent.step.failed"
    step: AgentStep
    error: str

class RunCompletedEvent(AgentLifecycleEvent):
    EVENT_TYPE = "agent.run.completed"
    step: AgentStep
    final_answer: str

class RunFailedEvent(AgentLifecycleEvent):
    EVENT_TYPE = "agent.run.failed"
    step: AgentStep
    error: str

type AgentEvent = (
    LLMStartedEvent
    | LLMOutputDeltaEvent
    | LLMCompletedEvent
    | ToolStartedEvent
    | ToolOutputEvent
    | ToolCompletedEvent
    | ToolFailedEvent
    | StepCompletedEvent
    | StepFailedEvent
    | RunCompletedEvent
    | ...
)
```

- `AgentEvent` 联合由 `AgentBase.run()` 返回，外部消费方可按需订阅具体子类；同步体验依旧通过 collect 事件自行构造。
- `step_index`/`step_id` 是所有事件共享的最小字段，其它 payload 只出现在相应子类上，杜绝“胖接口”。
- `RunCompletedEvent` 挂载最终 `step` 与 `final_answer`；`RunFailedEvent` 则单独承载终止错误，与其它 step/工具事件一起保持精确负载。

### 事件语义

| event_type | 触发时机 | payload |
| --- | --- | --- |
| `agent.llm.started` | agent 启动某 step 的 LLM 请求 | `llm_delta=None`，`annotations` 可含请求元信息 |
| `agent.llm.output_text.delta` | 将 provider 的 `response.output_text.delta` 透传给调用方 | `llm_delta` 填文本片段，`annotations["raw_event"]` 可带底层 JSON |
| `agent.llm.completed` | provider streaming 结束且拿到完整 `LLMResponse` | `LLMCompletedEvent.step` 包含 turn 快照 |
| `agent.tool.started` | agent 决定执行工具 | `ToolStartedEvent.tool_name`（方便 UI）+ `tool_call` 用于 executor |
| `agent.tool.output` | 工具 streaming/日志输出 | `ToolOutputEvent.text_delta` 携增量字符串 |
| `agent.tool.completed` | 工具执行成功 | `ToolCompletedEvent.tool_result` 记录 output |
| `agent.tool.failed` | 工具执行报错 | `ToolFailedEvent.error` + `tool_result` 供重试/UI |
| `agent.step.completed` | LLM + 工具逻辑产生可供下一轮使用的结果 | `StepCompletedEvent.step` 提供 turn 内容 |
| `agent.step.failed` | agent 在该 step 中出现不可恢复错误 | `StepFailedEvent.error` 描述异常，`step` 仅含局部结果 |
| `agent.run.completed` | 整个 agent 任务成功结束 | `RunCompletedEvent.final_answer` 携带最终输出 |
| `agent.run.failed` | agent 运行出现不可恢复错误 | `RunFailedEvent.error` 描述失败原因 |

### 与现有模型的衔接

1. Provider 仍通过 `LLMStreamEvent` 暴露原始事件，agent 层监听它们并转换成 `agent.llm.*` 事件，保证 ADK 风格的“最大公共事件”语义。
2. `AgentEvent` 中只有 step/run 相关子类才暴露 `step` 字段，依旧与 `AgentStep` 结构共享引用，满足 trace/审计需求。
3. 上层可根据需要把 `AgentBase.run` 的事件写入任何外部 event bus（Kafka、Redis、SSE 等）或直接驱动 UI；若想保留旧的 `AgentResponse` 结构，可在外部 helper 中收集事件后组装。
4. 如果后续要扩展 planner/critic 等子流程，可在事件枚举中新增前缀（如 `agent.planner.*`），保持 closed set + 显式注册的策略。

---

本文件将作为后续讨论与实现的参考基线，如有调整请在此更新。
