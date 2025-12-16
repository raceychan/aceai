# LLM Response Layering Proposal

## Background

- 当前 `LLMService` 直接将 provider 原始响应压缩成 `LLMResponse`，造成 OpenAI Responses API 等富语义信息无法对外暴露。
- `AgentBase` 只能消费 `LLMResponse` 的 `text/tool_calls/usage`，无法把 service 层信息保留下来供调用者或下游拓展。
- 目标：在 provider/service/agent 三层之间建立“最大公共部分 + 可拓展”的契约，在不泄露 provider SDK 类型的前提下，完整保留底层返回。

## 设计目标

1. **信息保真**：provider 特有字段（reasoning、event trace、latency 等）都要可获得，哪怕消费者暂时用不到。
2. **Provider 封装**：对外只暴露我们自定义的类型，屏蔽 OpenAI SDK 等第三方结构。
3. **层级清晰**：provider → service → agent 每层仅做增量封装，且下层信息可在上层完整访问。
4. **公共语义优先**：抽取多家 provider 的最大公共字段，减少转换导致的信息丢失。

## 当前进度与分层范围

### 1. Provider 层

- **不再引入 `ProviderCompletion` / `ProviderStreamEvent`**。所有适配器直接填充“拓展版 `LLMResponse`”与新 `LLMStreamEvent`。我们吸收 google-adk 在事件化接口上的优点（统一 event 入口、显式类型），但不强制复制其所有 payload 形式。
- `LLMResponse` 本身具备承载富语义的字段（segments/raw events/provider meta/extras），即使调用方只关心 `text`/`tool_calls` 也能向后兼容；同时我们借鉴 liteLLM 会保留 `raw_response` 便于调试的做法，但通过自有类型屏蔽其字段细节，保持语义一致。
- `LLMStreamEvent` 统一描述 streaming 事件：直接暴露 `text_delta`/`tool_call_delta`/`response`/`error` 字段，并附带 provider 事件元信息和增量 segments，以及显式的 `event_type`；我们吸收 ADK 对事件枚举严格管理的优点（如 `response.output_text.delta`、`response.completed`），但字段命名与含义由我们主导，且事件类型是封闭集合，显式即支持，禁止落到兜底分支。该结构已在 OpenAI 适配器里落地（见 `aceai/llm/openai.py`）。
- 每个 provider adapter 需要：
  - 根据底层响应构造 `segments`（`text`, `reasoning`, `tool_call`, `citation`, `error`, `other`）；`LLMSegment.metadata` 允许挂载 `annotations`（参照 ADK 对 reasoning/safety/citation 的处理）以及 provider 自定义字段。
- 记录 `provider_meta`（目前仅 `provider_name`、`model`, `latency_ms`，其余调试属性统一放在 `extra` 字段；类型名为 `LLMProviderMeta`），吸收 liteLLM 的透明度优点，同时保持字段命名在我们控制之下。
  - 在 `raw_events` 中保留 provider 原始 JSON（裁剪敏感数据），`extras` 中写入我们命名空间下的扩展字段（如 `openai_response_metadata`），类似 liteLLM 返回 `raw_response` 的做法，方便调用者比对差异。
  - 最终 streaming 结束事件必须带上完整 `LLMResponse`（等价于 ADK 的 `response.completed`），让消费者不必重新组装文本。
  - 若 provider 产生我们暂未列出的事件类型，适配器必须显式抛错或将其提升为受支持类型；不允许塞进通用“other”分支，以贯彻 “Explicit is better than implicit”。

### 2. Service 层

- Service 层保持 API 简洁：`complete` 直接返回扩展版 `LLMResponse`，`stream` 返回 `LLMStreamEvent` 异步迭代器（`aceai/llm/service.py` 已完成接线）。
- `complete_jsons`：
  - 主返回值依旧是解码后的 schema。
  - 同时暴露最近一次 `LLMResponse`，供上层读取 segments/provider 信息。
- Service 层保障：
  - 输出类型完全由我们控制（不返回 SDK 对象）。
- 统一聚合超时/重试/failover 逻辑，但聚合信息通过 `LLMResponse.provider_meta.extra` 或 `extras` 字段暴露，不额外定义 service 专属类型。
  - 若 future 引入多 provider failover，可在 `provider_meta.extra` 记录各 attempt，或扩展 `LLMResponse` 字段。

> Agent 层暂不在本规范中设计，后续等 provider/service 抽象稳定后再补充。

## 类型定义草案

```python
    class LLMProviderMeta(Record):
        provider_name: str
        model: str
        latency_ms: float | None = None
        extra: dict[str, Any] = field(default_factory=dict)

    class LLMSegment(Record):
        type: Literal["text", "reasoning", "tool_call", "citation", "error", "other"]
        content: str
        metadata: dict[str, Any] = field(default_factory=dict)  # e.g. ADK-style annotations/citations

    class LLMResponse(Record):
        id: str | None = None
        model: str | None = None
        text: str = ""
        tool_calls: list[LLMToolCall] = field(default_factory=list[LLMToolCall])
        usage: Unset[LLMUsage] = UNSET
        segments: list[LLMSegment] = field(default_factory=list[LLMSegment])
        provider_meta: list[LLMProviderMeta] = field(default_factory=list[LLMProviderMeta])
        raw_events: list[dict[str, Any]] = field(default_factory=list)
        extras: dict[str, Any] = field(default_factory=dict)

    class LLMStreamEvent(Record):
        event_type: Literal[
            "response.output_text.delta",
            "response.function_call_arguments.delta",
            "response.completed",
            "response.error",
        ]  # mirror google-adk response event types; explicit-only, no fallback
        text_delta: Unset[str] = UNSET
        tool_call_delta: Unset[LLMToolCallDelta] = UNSET
        response: Unset[LLMResponse] = UNSET
        error: Unset[str] = UNSET
        segments: list[LLMSegment] = field(default_factory=list[LLMSegment])
        provider_meta: list[LLMProviderMeta] = field(default_factory=list[LLMProviderMeta])
        raw_event: dict[str, Any] | None = None
        extras: dict[str, Any] = field(default_factory=dict)
```

> 这些类型放在 `aceai/llm/models.py`，供 provider 与 service 层（以及未来的 agent 层）共享。

## 演进步骤

1. **接口定义**：在 `llm/models.py` 扩展 `LLMResponse`，并新增 `LLMStreamEvent`、`LLMProviderMeta`、`LLMSegment`。
2. **Service 改造**：`LLMService.complete/stream/complete_json` 全面返回新的结构，`complete_json` 暴露最近一次 `LLMResponse` 以便上层读取 segments。
3. **Provider 样板**：以 `aceai/llm/openai.py` 为首个实现者，填充 `segments`（text/tool_call/reasoning）、`raw_events`（Responses event JSON）、`provider_meta`，并确保 streaming 事件映射到 ADK 风格的 `event_type`。
4. **文档 & 测试**：在 docs/specs 中记录字段含义，新增单元测试覆盖 provider → service 信息流转。

## Open Questions（最新结论）

- `LLMSegment` 需要专门的 `citation`/`safety` 子结构。当前只做最小可行结构（如 `LLMCitationRef {id, source, span}`、`LLMSafetyAnnotation {category, level, blocked}`），等 OpenAI 等厂商提供更多细节后再拓展。
- `raw_events` 暂不做 size 限制 / 采样。它是底层调试/重放所需，过早裁剪反而增加复杂度。
- 多 provider failover 直接收敛为 `LLMProviderMeta` 列表：`provider_meta: list[LLMProviderMeta]`，记录所有尝试，调用方可查看最终使用哪家以及之前失败的原因。

## Agent 层探索

Agent 层记录已拆分至 `spec/agent_layer.md`，本文件聚焦 provider/service 层契约与演进。

---  
本文件将作为后续讨论与实现的参考基线，如有调整请在此更新。
