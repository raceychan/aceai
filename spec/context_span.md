# Context Span 串联方案（OTEL / Langfuse）

## 现状与症状

当前在 Langfuse / OTEL Trace 中看到：

- `agent.step`、`tool.*`、`llm.stream` 等没有挂在同一条 trace 下，表现为各自独立的 root span
- 尝试在 `_run_step()` / `llm_service.stream()` 等 async generator 中使用 `with tracer.start_as_current_span(...)` / `use_span(...)` 后，出现 `Failed to detach context`

根因主要有两类：

1. **没有显式建立 parent context**
   - `AgentBase._run_step()` 里用 `start_span()` 创建了 `agent.step`，但没有把它设置为 current span，也没有把它的 context 传给下游。
   - `LLMService.stream()` 内部同样 `start_span()`，在没有 parent context 的情况下也会变成 root span。
   - `ToolExecutor.execute_tool()` 使用 `start_as_current_span()`，但如果上游没有 current span（或没有显式传 parent context），也会变成 root span。

2. **async generator 的 `yield` 会跨越 span/context manager 生命周期**
   - `with tracer.start_as_current_span(...): async for ...: yield ...` 的模式会让 “span 的进入/退出” 与 “generator 的消费/中断” 交织。
   - 当外部提前停止迭代（break / cancel / GC / aclose）时，`__exit__` 可能在不同的 task/时序里运行，导致 `ContextVar` token 不匹配，触发 `Failed to detach context`。

## 约束（必须接受的事实）

- async generator 的生命周期由**消费方**决定：可能提前停止、被取消、或者从未完全消费。
- 不能依赖 “一个大 `with` 块” 覆盖整个 generator 期间的上下文保持；这类写法在提前中断时非常容易产生不平衡的 attach/detach。
- “span 串联” 的关键不是 current span，而是 **parent context** 一致。
- 仅靠 `break`/`return` 退出 `async for` 并不会保证触发被消费 generator 的 `finally`；若 span 的 `end()` 写在 generator 的 `finally` 中，需要显式 `await agen.aclose()` 才能确保 span 结束并上报。

## 结论：推荐的串联策略（显式 parent context 传递）

核心思路：**在每一层创建 span 时，都显式指定 parent context（而不是依赖 implicit current span）**。

### 1) 在 `AgentBase.run()` 建立会话级 root span（可选但推荐）

- 在 `run()` 一开始创建 `agent.run`（或类似命名）作为本次对话的 root span。
- 从该 span 派生出 `run_ctx`（一个 OTEL `Context`），后续所有 step 都使用它作为 parent。

收益：一个会话（多 step）天然在同一条 trace 下；UI 更符合“对话是一条 trace”的直觉。

### 2) 在 `_run_step()` 创建 `agent.step`，并把 `step_ctx` 显式传给下游

不要在 async generator 外层用 `start_as_current_span` 包住 `yield`。

做法：

- 用 `start_span("agent.step", context=run_ctx)` 创建 step span（只创建，不设置 current）。
- 用 `set_span_in_context(step_span, run_ctx)` 生成 `step_ctx`。
- 调用 `llm_service.stream(..., parent_context=step_ctx)`（或通过 request metadata 传递）。
- 工具执行 `executor.execute_tool(..., parent_context=step_ctx)`。

注意：`agent.step` 的 `span.end()` 放在 generator 的 `finally`，确保无论消费方是否中断都能结束 span（但不要在整个 generator 期间把它设为 current）。

### 3) 在 `LLMService.stream()` / `ToolExecutor.execute_tool()` 使用 `context=parent_ctx`

两个原则：

- 下游 span 创建时，显式传 `context=parent_ctx`（这决定了 trace/parent 关系）。
- 只在“同步/短片段”里需要 current span 时才使用 `start_as_current_span`（并确保不会跨越 `yield`）。

示例（伪代码形态）：

```python
def stream(..., parent_context):
    span = tracer.start_span("llm.stream", context=parent_context, ...)
    llm_ctx = set_span_in_context(span, parent_context)
    try:
        for event in provider_stream(...):
            # 如需把特定事件记录到当前 span：只包很短的同步段
            with tracer.start_as_current_span("llm.stream.event", context=llm_ctx):
                ...
            yield event
    finally:
        span.end()
```

工具同理：

```python
async def execute_tool(..., parent_context):
    with tracer.start_as_current_span(f"tool.{name}", context=parent_context, ...):
        ...
```

### 4) ThreadPool / `run_in_executor` 场景的补充规则

如果工具执行里引入线程池（尤其是 `loop.run_in_executor` / 自己的 threadpool）：

- 不要假设 `ContextVar` 会自动传播到线程。
- 建议把 `parent_context` 显式传入线程函数，在线程入口处 attach；线程退出时 detach（必须成对且在同一线程中执行）。

（如果使用 `asyncio.to_thread`，它会复制当前 `contextvars`，但这里依然建议以“显式 parent_context”为准，避免上游 current span 不稳定。）

## 反模式（明确不要做）

1. 在 async generator 中用一个大 `with tracer.start_as_current_span(...)` 包住 `yield`：

```python
with tracer.start_as_current_span("agent.step"):
    async for ev in stream():
        yield ev  # 外部中断时容易 Failed to detach context
```

2. 只创建 span（`start_span`）但不传 `context=`，也不把 span context 传给下游 —— 这会导致 root span 泛滥，trace 无法串联。

## 最小改动落地点（对应当前框架结构）

- `AgentBase.run()`：可选新增 `agent.run` root span，并生成 `run_ctx`
- `AgentBase._run_step()`：`agent.step` 用 `context=run_ctx` 创建，并生成 `step_ctx`；将 `step_ctx` 显式传入：
  - `llm_service.stream(..., parent_context=step_ctx)`
  - `executor.execute_tool(..., parent_context=step_ctx)`
- `LLMService.stream()`：创建 `llm.stream` 时传 `context=parent_context`
- `ToolExecutor.execute_tool()`：创建 `tool.*` 时传 `context=parent_context`

这套策略的目标不是“把 span 设成 current 一直保持”，而是确保 **每个 span 都用同一个 parent context 链接起来**，从而在 Langfuse / OTEL Trace 里稳定落在同一条 trace 下，并避免 async generator 的 `Failed to detach context`。

## TODO（可执行清单）

- [x] 在 `AgentBase.run()` 增加 `agent.run` span，并生成/保存 `run_ctx`
- [x] 在 `AgentBase._run_step()` 用 `context=run_ctx` 创建 `agent.step`，并生成 `step_ctx = set_span_in_context(step_span, run_ctx)`
- [x] 调整 `agent.step` 生命周期：覆盖 LLM + tool 执行，避免“父 span 先结束、子 span 后发生”的时间轴错位
- [x] 扩展 `ILLMService.stream()`/`LLMService.stream()` 签名：显式接收 `parent_context`（或等价结构）并用于创建 `llm.stream`（`context=parent_context`）
- [x] 扩展 `ToolExecutor.execute_tool()` 签名：显式接收 `parent_context` 并用于创建 `tool.*`（`context=parent_context`）
- [x] 在 `_run_step()` 里把同一个 `step_ctx` 传入 `llm_service.stream(..., parent_context=step_ctx)` 与 `executor.execute_tool(..., parent_context=step_ctx)`，确保它们成为 `agent.step` 的子 span
- [x] 严格禁止在 async generator 中用“大 `with start_as_current_span(...)` 包住 `yield`”的写法；如需 current span，只包住短小的同步片段
- [ ] 若引入 threadpool / `run_in_executor`：在工作线程入口显式 attach `parent_context`，线程退出时在同一线程 detach（成对）
- [x] 增加单测：断言一次 `run()` 内 `agent.run -> agent.step -> (llm.stream + tool.*)` 的 parent/trace 关系正确（不再出现多个 root span）
- [x] 增加回归单测：模拟提前中断/取消 async generator，确保不再触发 `Failed to detach context`
