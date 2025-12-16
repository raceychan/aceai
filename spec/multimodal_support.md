# 多模态输入输出精简方案

## 背景
- `LLMMessage.content` 仅接受 `str`（`aceai/llm/models.py:39-60`），无法携带图片或音频等附件。
- `LLMSegment.type` 只有文本相关枚举（`aceai/llm/models.py:205-221`），无法描述模型生成的图像或音频。
- Provider 适配层缺少对自身多模态能力的声明，调用端不能提前判断“能否发图/收图”。

## 设计原则
1. **抛弃兼容性包袱**：`LLMMessage.content` 强制为结构化 part 列表，所有调用方立即迁移；仅在产品最外层（如 `Agent.ask`）允许输入 `str` 并立刻转换。
2. **只保留当前需要的模态**：先覆盖文本、图片（URL/二进制）与音频，未来再迭代视频等能力。
3. **复杂度留在 Adapter**：Service/Agent 只维护统一的数据结构，不再引入额外的资产缓存或生命周期管理。
4. **Streaming 只传有意义的事件**：保持现有 `response.delta` 语义，新增一个媒资事件即可，不需要事件矩阵。

## 核心改动

### 1. 消息内容
```python
class LLMMessagePart(Record, kw_only=True):
    type: Literal["text", "image", "audio", "file"]
    data: str | bytes | None = None   # text -> str，其余 -> bytes
    mime_type: str | None = None
    url: str | None = None            # 有现成链接就直接引用
```

- `LLMMessage.content` **仅接受 `list[LLMMessagePart]`**，禁止传入裸字符串；如果上层收到字符串输入，必须在进入 LLMService 前转换为 part。
- `LLMToolUseMessage`、`LLMToolCallMessage` 共用 `LLMMessagePart`，保证工具结果也能回传图片或音频。
- Adapter 只需关心 `type`+`data`/`url`：
  - 对 OpenAI Responses，`text` → `{"type":"text"}`，`image` → `{"type":"input_image"}`，`audio` 仅在支持时转换。
  - 对不支持的 Provider，抛出 `LLMProviderError("image input not supported")`，交由调度层处理。

### 2. 上传/引用策略
- 不再维护 `LLMAssetRef`、缓存地图或跨请求引用；谁需要上传，谁在 Adapter 内部做即可。
- 约定一个最小 helper 供 Adapter 复用：

```python
class LLMUploadedAsset(Record, kw_only=True):
    id: str
    url: str | None = None
    mime_type: str
```

- 若 Provider 需要先上传二进制，Adapter 内部可将 `LLMMessagePart.data` → `LLMUploadedAsset` 并把必要的引用写回 `part.url`，后续 retry/重发依旧读取 `url`。

### 3. 响应与 streaming
- `LLMSegment.type` 扩展 `"image" | "audio" | "file"` 三种；其他沿用原有定义。
- `LLMGeneratedMedia` 用于在响应里描述模型输出：

```python
class LLMGeneratedMedia(Record, kw_only=True):
    type: Literal["image", "audio", "file"]
    mime_type: str
    url: str | None = None
    data: bytes | None = None
```

- `LLMSegment.media: LLMGeneratedMedia | None`，文本段保持 `None`。
- Streaming 只新增一种事件：`response.media`，携带 `LLMGeneratedMedia`。文本仍通过现有 `response.delta` 传输，避免事件风暴。

### 4. Provider 能力声明
- `LLMProviderBase` 增加一个轻量属性：

```python
@dataclass(slots=True)
class LLMProviderModality:
    text_in: bool = True
    text_out: bool = True
    image_in: bool = False
    image_out: bool = False
    audio_in: bool = False
    audio_out: bool = False

@property
def modality(self) -> LLMProviderModality: ...
```

- Service 调度时只需要 `if request.contains_image and not provider.modality.image_in: raise LLMProviderError(...)` 这样的布尔判断，不再静默切换 provider。
- 未来若要支持视频，再扩展字段即可，无需枚举型合集。

### 5. Service 与 Agent 调整
- `LLMService._prepare_messages`：
  1. 直接校验消息是否为 part 列表并统计模态；如果传入字符串立即抛错，由更上层负责兼容。
  2. 根据统计结果与 provider.modality 做一次布尔校验。
- `LLMService.stream` 在收到 `response.media` 事件时，直接把 `LLMGeneratedMedia` 推给上层；Agent 若不关心可以忽略。
- Agent 输出结构无需新增字段，只要在最终响应里附带 `segments`，UI 可自行决定如何渲染。

## 迁移步骤
1. **引入新的 `LLMMessagePart` 定义**：在 `aceai/llm/models.py` 增加 class，并让 `LLMMessage.content` 仅接受 `list[LLMMessagePart]`。
2. **更新现有 provider 适配器**：最少先完成 OpenAI Vision（图像输入）和 OpenAI Responses（图像输出）的映射，其他 provider 暂时标记为不支持。
3. **扩展 `LLMSegment` / streaming`**：增加 `media` 字段、`response.media` 事件，并在 `LLMService` 中打通。
4. **补充示例与测试**：提供一个发送图片并接收图片描述的 `pytest` fixture + README 片段，覆盖最常见路径即可。

## 权衡
- 不再内置资产缓存，意味着重复上传由各 Adapter 自己决定是否优化；本阶段暂不处理。
- 依旧允许 `str`，因此不存在“全部调用方必须立刻迁移”的成本，缺点是数据类型更自由，需要在 Service 层多做一次标准化。
- Streaming 只暴露单一媒资事件，适合目前“推一张图/一个音频”的需求，未来若要做 delta，可在事件里附加 `chunk_index` 即可扩展。

---
本方案把多模态支持限定在“可落地且马上能用”的范围：统一消息结构、能发图收图、最小 streaming 扩展，其余如资产生命周期、跨 provider 缓存等留到确实需要时再迭代。
