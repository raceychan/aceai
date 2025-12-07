# LLM 流式推理事件实现方案

## 背景
- Agent 层目前使用 `LLMService.complete`，只有同步 `LLMCompletedEvent`，缺乏推理过程可见性。
- `LLMService.stream` 已存在，可直接暴露 `LLMStreamEvent`，但 agent 尚未消费。
- 产品诉求：在推理过程中实时发出 `agent.llm.output_text.delta`，并把增量文本写入 `AgentStep.annotations.extra["reasoning_log"]`，提高解释性。

## 核心思路
1. **切换为 streaming**：
   - `_run_step` 不再 `await llm_service.complete`，改为 `async for event in llm_service.stream(...): ...`。
   - 请求参数保持不变（`messages`、`tools`、`metadata`），仅将 streaming 结果拼装成最终 `LLMResponse`。
2. **事件映射**：
   - 收到 `event.event_type == "response.output_text.delta"` 且 `text_delta` 已设时，立即 `yield event_builder.llm_text_delta(text_delta=chunk)`。
   - 将 chunk 追加到 `reasoning_log` 缓冲字符串，写入 `step.annotations.extra`。
   - 如果拿到 `response.function_call_arguments.delta`，暂不处理（不阻塞计划），但预留 TODO。
3. **完成事件**：
   - 等 `response.completed` 事件携带 `LLMResponse`，再构造 `AgentStep` 并发 `LLMCompletedEvent`，随后沿用原有工具调用/最终答案逻辑。
4. **错误传播**：
   - 若 streaming 中出现 `response.error` 事件或迭代器抛异常，直接终止本 step，让 `AgentBase.run` 复用现有失败路径；不会再 fallback 到 `complete`。

## 数据结构调整
- `AgentStep.annotations.extra` 初始化时放入 `{"reasoning_log": ""}`，确保即使没有 delta 也有字段。
- `AgentLifecycleEvent` 等模型无需变更；只是在 `_run_step` 里多次 `yield` delta 事件。
- 可以在 `AgentStep.tool_results` 里继续按完成顺序追加，保持与 delta 顺序解耦。

## 流程示意
1. `llm_started`
2. Streaming delta → 多次 `llm_text_delta` + `reasoning_log += chunk`
3. `response.completed` → `LLMCompletedEvent`
4. 工具调用（可并发设计另行实现）
5. `StepCompleted` / `RunCompleted`

## 测试策略
- 为 `StubLLMService` 新增 `stream`，可配置 delta 序列与最终 `LLMResponse`。
- 新增测试断言：
  - `agent.llm.output_text.delta` 事件顺序正确，`reasoning_log` 拼接完整。
  - 没有 delta 时 `reasoning_log` 为空字符串。
  - streaming 抛错时，`RunFailedEvent` 仍然触发。

## 风险与取舍
- **Breaking change**：`AgentBase` 依赖 streaming，假设所有 provider 都实现 `stream`；不再 fallback 到 `complete`，符合“破坏兼容”原则。
- **事件风暴**：长回答会产生大量 delta 事件；如需节流可在后续引入 chunk 聚合。
- **内存占用**：`reasoning_log` 会保存完整推理文本；若未来需要，可改为切片或环形缓冲。
