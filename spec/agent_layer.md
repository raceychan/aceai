# Agent 层响应结构探索

## 目标

- 在保留 service 层 `LLMResponse` 全量信息的基础上，为 agent 提供更丰富且可扩展的响应结构（reasoning trace、工具执行记录、最终答案等）。

## 已知现状

- `AgentBase.handle` 现已直接返回 `AgentResponse`，以结构化方式暴露 llm/tool 信息（`final_output` 仍可作为便捷字段读取）。

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

- `AgentBase.handle` 在每次 LLM 调用后创建 `AgentStep`，将 `LLMResponse` 原封不动塞入 `llm_response`。
- 执行工具后，向 `turn.tool_results` 写入结果；返回给模型的 tool outputs 继续通过 `LLMToolUseMessage` 发送，逻辑保持不变。
- 默认同步 API 就返回 `AgentResponse`，消费方需要从 `final_output` 或更细粒度的 turn 数据中取值。

## 落地策略

1. **Streaming 优先**：Agent 默认提供 `async Generator[AgentStep]`（或 `AsyncIterator[AgentStep]`）接口，实时暴露每个 turn；若需要一次性结果，可在外部 `collect`。
2. **直接演进**：当前尚无外部用户，`AgentBase` 已改造为返回结构化/streaming 形式，不再保留旧的 `str` 返回值。
3. **final tool 处理**：`final_answer` 作为普通 `ToolExecutionResult` 记录在最后一个 turn 的 `tool_results` 内，同时把其输出引用到 `AgentResponse.final_output`，避免重复执行但仍保留溯源。

## 下一步

1. 在 `aceai/llm/models.py` 或 `aceai/agent.py` 定义上述 record。
2. 改造 `AgentBase.handle` 使其始终返回结构化响应。
3. 选一个 agent/client 更新调用方式，验证实战体验，再回写文档。

## 业界对标与最佳实践

- **Google ADK**：ADK 要求 provider/service/agent 全链路输出严格事件枚举与可重放 trace。我们在 `AgentStep` 中保存完整 `LLMResponse`、tool 执行产物，使得每个 turn 自包含，满足 ADK “显式事件 + 可重放” 的指导。
- **OpenAI Agents Python SDK**：该 SDK 将每次 LLM+Tool 交互序列化为不可变 step，包含调用、工具输出与最终消息。`AgentStep` 的结构化记录与 `final_answer` 作为 `ToolExecutionResult` 的策略，提供了同等级别的审计/溯源能力，同时保持我们自定义类型的可控性。
- **Streaming 接口**：两家实现都以 streaming event/iterator 暴露进行中状态。我们默认提供 `AsyncIterator[AgentStep]`，并允许外部 `collect`，可与 ADK/OpenAI 的订阅式消费模式互操作。

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
class AgentStepEvent(Record):
    event_type: Literal[
        "agent.llm.started",
        "agent.llm.output_text.delta",
        "agent.llm.completed",
        "agent.tool.started",
        "agent.tool.output",
        "agent.tool.completed",
        "agent.tool.failed",
        "agent.step.completed",
        "agent.step.failed",
        "agent.run.completed",
    ]
    step_index: int  # 0-based turn index
    step_id: str     # stable identifier, mirrors OpenAI step_id
    step: AgentStep | None = None  # populated on completed/failed/run-completed
    llm_delta: str | None = None
    tool_call: LLMToolCall | None = None
    tool_result: ToolExecutionResult | None = None
    error: str | None = None
    annotations: dict[str, Any] = field(default_factory=dict)
```

- `AgentStepEvent` 由 `AgentBase.stream()`（或 `handle_stream`）返回，外部消费方统一订阅；同步 API 仍可通过收集事件生成最终 `AgentResponse`。
- `step_index`/`step_id` 让 UI 与日志系统能稳定引用具体 step，与 OpenAI Agents「step」语义对齐。
- `step` 只在 step 或 run 终态事件中携带，避免重复复制 `LLMResponse`。

### 事件语义

| event_type | 触发时机 | payload |
| --- | --- | --- |
| `agent.llm.started` | agent 启动某 step 的 LLM 请求 | `llm_delta=None`，`annotations` 可含请求元信息 |
| `agent.llm.output_text.delta` | 将 provider 的 `response.output_text.delta` 透传给调用方 | `llm_delta` 填文本片段，`annotations["raw_event"]` 可带底层 JSON |
| `agent.llm.completed` | provider streaming 结束且拿到完整 `LLMResponse` | `step.llm_response` 已填充，`step.tool_results` 为空 |
| `agent.tool.started` | agent 决定执行工具 | `tool_call` 填充，`annotations` 可含 executor 配置 |
| `agent.tool.output` | 工具 streaming/日志输出 | `tool_result.output` 仅填增量片段 |
| `agent.tool.completed` | 工具执行成功 | `tool_result` 完整记录，`step.tool_results` append |
| `agent.tool.failed` | 工具执行报错 | `tool_result.error` 填充，供重试或 UI 呈现 |
| `agent.step.completed` | LLM + 工具逻辑产生可供下一轮使用的结果 | `step` 携带本轮全部上下文 |
| `agent.step.failed` | agent 在该 step 中出现不可恢复错误 | `error` 描述异常，`step` 视情况附上部分数据 |
| `agent.run.completed` | 整个 agent 任务结束（成功或失败） | `step` 指向最后一轮；若失败，配合 `error` |

### 与现有模型的衔接

1. Provider 仍通过 `LLMStreamEvent` 暴露原始事件，agent 层监听它们并转换成 `agent.llm.*` 事件，保证 ADK 风格的“最大公共事件”语义。
2. `AgentStepEvent` 的 `step` 字段与上文定义的 `AgentStep` 结构共享引用，确保任何 turn 的完整快照可被追溯（符合 OpenAI step trace 的 best practice）。
3. 同步 `AgentBase.handle` 可以在内部收集所有 `AgentStepEvent`，构造 `AgentResponse`；流式消费者则可以仅依赖事件驱动 UI、日志或编排逻辑。
4. 如果后续要扩展 planner/critic 等子流程，可在事件枚举中新增前缀（如 `agent.planner.*`），保持 closed set + 显式注册的策略。

---

本文件将作为后续讨论与实现的参考基线，如有调整请在此更新。
