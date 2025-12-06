## Langfuse 集成方案

### 背景
- 当前的调用链由 `AgentBase` (aceai/agent.py) 负责 Reasoning Loop，`LLMService` (aceai/llm/service.py) 统一调度 provider，`ToolExecutor` (aceai/executor.py) 执行函数工具。除了一层可选的 `LoggingToolExecutor`，目前没有统一的可观察性设施，无法记录对话级 trace、LLM 代价、工具时延或错误。
- 现有依赖较少（核心在 `openai`、`msgspec`、`ididi`），因此引入 Langfuse 时需要保持“可选依赖 + 低侵入”的设计，避免破坏纯净的核心 API。

### Langfuse 集成目标
1. **端到端 Trace**：每次 `AgentBase.handle` 产生一个 Langfuse trace，串起该轮 LLM 调用、工具执行、最终回答。
2. **LLM 观测**：在 `LLMService.complete/stream` 记录 model、提示词、tokens、错误信息，为成本分析与质量回溯提供依据。
3. **工具观测**：在 `ToolExecutor.execute_tool` 记录 span，包含工具名、入参、返回值（或摘要）、耗时、异常。
4. **最小侵入**：Langfuse 不应成为强制依赖，未配置时走 no-op 分支；集成代码需覆盖 sync & async 场景。
5. **安全治理**：允许通过配置裁剪 message/参数，以避免在 Langfuse 中泄露敏感内容。

### 代码扫描要点
- `aceai/agent.py`：`AgentBase.handle` 是会话入口，循环体中串联了 LLM -> Tool -> LLM，可在这里注入 trace 生命周期管理（开始、更新状态、结束）。
- `aceai/llm/service.py`：所有 LLM 请求（包括 `complete_json`）都走这层，是记录 Langfuse generation 的最佳位置；`stream` 需要对 chunk 做聚合以便最终写入实体。
- `aceai/executor.py`：`ToolExecutor/LoggingToolExecutor` 执行工具并计算耗时，可在这里扩展一个 `InstrumentedToolExecutor` 或给现有类加可选的 telemetry hook。
- `aceai/llm/openai.py`：provider 适配层；一般不需要直接改动，只需从 `LLMService` 传入 metadata（model、usage）给 Langfuse。
- Tests (`tests/test_agent_behavior.py` 等) 依赖轻量 stub，在落地 Langfuse 时需提供假 Telemetry stub，保证单测不引入真实网络。

### Langfuse 数据模型映射

| Langfuse 实体 | AceAI 事件/上下文 | 备注 |
| --- | --- | --- |
| `trace` | `AgentBase.handle` 一次对话 | trace id = `request_id`/外部传入；metadata 包含 agent、用户、prompt hash |
| `generation` | `LLMService.complete` 或 `complete_json` 一次响应 | 挂到当前 trace；record 数量对应 LLM 回合数 |
| `span` | `ToolExecutor.execute_tool` 一次工具调用 | parent = 触发该工具的 generation；记录耗时、依赖节点 |
| `event` | 中间状态（如 plan、error、final answer） | 可由 `AgentBase` 在关键节点追加 |

### 技术方案

#### 1. Telemetry 抽象层
1. 新增 `aceai/telemetry/base.py`，定义 `TelemetryClient`/`TraceHandle`/`SpanHandle` 协议，包含 `start_trace`, `start_span`, `log_generation`, `log_event`, `flush` 等方法；默认实现 `NoOpTelemetry`。
2. `AgentBase`, `LLMService`, `ToolExecutor` 接受可选 `telemetry: TelemetryClient`，若为 None 则回退到 NoOp。
3. 通过 `contextvars` 或注入式 `TraceContext` 关联 trace/span，避免在 API 上暴露 Langfuse 细节。

```python
# aceai/telemetry/base.py
class TraceContext(Protocol):
    trace_id: str
    user_id: str | None

class TelemetryClient(Protocol):
    def start_trace(self, *, name: str, input: dict, metadata: dict) -> TraceContext: ...
    def start_span(self, trace: TraceContext, name: str, input: dict | None = None) -> SpanHandle: ...
    def log_generation(self, trace: TraceContext, *, request: LLMRequest, response: LLMResponse, error: Exception | None = None) -> None: ...
    def log_event(self, trace: TraceContext, message: str, level: Literal["info","warning","error"]) -> None: ...
```

#### 2. Trace 生命周期（`AgentBase.handle`）
1. 入口创建/复用 Trace：若调用者传入 `trace_context`（例如多 agent 协作场景），直接使用；否则调用 `telemetry.start_trace`，name 建议使用 agent 名称。
2. 在以下节点写 event：
   - 用户问题归一化后：`log_event(level="info", message="user_question", payload=question)`.
   - 每次收到最终回答或抛错：更新 trace status（`completion`, `error`），并记录 output。
3. 将 `trace_context` 通过参数 `metadata["trace_id"]` 注入到 `LLMService`，便于下游对齐。

#### 3. LLM 调用埋点（`LLMService`）
1. `complete` 成功返回后调用 `telemetry.log_generation`，字段：
   - `name`: `"llm.complete"`
   - `model`: `metadata["model"]`
   - `input`: `messages`（可配置裁剪 system prompt & tool output）
   - `output`: `response.text`
   - `usage`: `response.usage`
2. 捕获 `OpenAIError`/`asyncio.TimeoutError`，在 `except` 分支调用 `log_generation(..., error=err)`，再抛出原异常。
3. `complete_json` / `_complete_json_with_retry`：将 schema、retry 次数写入 generation metadata，并在每次 decoder error 时记录一条 warning event，方便诊断模型输出质量。
4. `stream`：累积 `text_delta` 与 `tool_call_delta`，在 `stream` 结束时一次性调用 `log_generation`；如果 streaming 过程中出现 `ResponseErrorEvent`，立刻记录 error event + span。

#### 4. 工具调用埋点（`ToolExecutor`）
1. 在 `execute_tool` 中，围绕现有 `perf_counter` 逻辑创建 Langfuse span：

```python
span = telemetry.start_span(
    trace=context,
    name=f"tool:{tool_name}",
    input={"arguments": param_json, "deps": list(tool.signature.dep_nodes)},
)
try:
    result = await super().execute_tool(...)
    telemetry.end_span(span, output=result, status="success")
except Exception as exc:
    telemetry.end_span(span, status="error", error=str(exc))
    raise
```

2. `LoggingToolExecutor` 可继承新基类或组合 Telemetry，保持现有日志输出行为。
3. 若工具调用产生新的子 LLM 请求（未来 MCP 支持时），可以在 span metadata 中标记 `mcp_server`，便于 Langfuse 中筛选。

#### 5. Langfuse 适配实现
1. 新建 `aceai/telemetry/langfuse.py`，内部封装官方 SDK：

```python
from langfuse import Langfuse

class LangfuseTelemetry(TelemetryClient):
    def __init__(self, cfg: LangfuseConfig):
        self.client = Langfuse(
            public_key=cfg.public_key,
            secret_key=cfg.secret_key,
            host=cfg.host,
        )
    def start_trace(...):
        trace = self.client.trace(id=trace_id, name=name, metadata=metadata, input=input)
        return TraceContext(trace=trace, ...)
    def log_generation(...):
        self.client.generation(
            trace_id=trace_id,
            name=name,
            model=request["metadata"]["model"],
            input=_serialize_messages(request["messages"]),
            output=response.text,
            usage=_usage_dict(response.usage),
            metadata=metadata,
            status="error" if error else "success",
        )
```

2. `_serialize_messages` 应裁剪工具输出/大型附件，可暴露 `LangfuseRedactionPolicy` 供用户注入（例如移除 `role=="tool"` 的内容或对 `content` 做 hash）。
3. `LangfuseTelemetry.flush()` 在应用退出或单测 teardown 时调用，保证缓冲区推送完成。

#### 6. 配置与依赖
1. 在 `pyproject.toml` 添加可选依赖 `langfuse>=2.40`，并通过 `extras = {"telemetry": ["langfuse>=2.40"]}` 或 `dependency-groups` 引入；默认安装不拉取。
2. 新增 `LangfuseConfig`（env + 显式参数双通道），读取：
   - `LANGFUSE_SECRET_KEY`
   - `LANGFUSE_PUBLIC_KEY`
   - `LANGFUSE_HOST`（默认 `https://cloud.langfuse.com`）
   - `LANGFUSE_RELEASE`（可选，用于版本对齐）
3. 通过 `aceai.config` 暴露 `build_langfuse_telemetry()` 帮助函数，方便 CLI / apps 一键启用。

#### 7. 迭代节奏
1. **Phase 1**：引入 Telemetry 抽象 + Langfuse 实现（记录非 streaming LLM + 工具 span）；提供配置与单测 stub。
2. **Phase 2**：覆盖 streaming + `complete_json` 重试链路，支持事件级 redact，补充文档/示例 notebook。
3. **Phase 3**：和未来 MCP 接入打通（每个 MCP server 记录独立 span），并在 README/示例中展示 Langfuse dashboard。

### 测试与验证
- **单元测试**：为 `NoOpTelemetry`、`LangfuseTelemetry` 引入 stub client，断言在 LLM/tool 失败、重试、final answer 等路径均调用了对应 hook。
- **契约测试**：使用 Langfuse 提供的 `LANGFUSE_HOST=https://cloud.langfuse.com` sandbox key，通过 `pytest` 打上 `@pytest.mark.langfuse`，只在 CI 的 nightly 阶段跑。
- **手动验证**：运行 `examples/` 中的代理 demo，触发多轮 LLM + 工具调用，确认 Langfuse UI 中 trace -> generation -> span 结构完整，耗时/错误信息正确。

### 风险与待定事项
- **SDK 背景任务**：Langfuse Python SDK 默认为 sync + 后台线程，需要确认与 `anyio`/`pytest` 协程兼容性；必要时将提交放到 `asyncio.to_thread`。
- **敏感数据**：需要默认启用 `redact_tool_outputs=True`，并允许用户自定义 redactor，以免把凭据写入 observability 平台。
- **性能开销**：在高 QPS 场景中 Langfuse API 调用可能成为瓶颈，建议提供批量/异步队列和 `sampling_rate` 配置（例如只有 10% 的请求上报详细内容）。
- **多 provider 支持**：未来接入非 OpenAI provider 时，需要确保 `LLMResponse.usage` 仍可映射（若缺失则填 `None` 并在 Langfuse metadata 记录 provider 名称）。

---
以上方案可以在不破坏现有公共 API 的前提下，为 AceAI 引入可观测性闭环，支持 trace、LLM、工具三层的 Langfuse 上报，并留出裁剪与扩展空间。
