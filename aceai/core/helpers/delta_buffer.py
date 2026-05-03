from collections import deque


class LLMDeltaChunker:
    """Aggregates tiny provider text deltas into larger agent-facing chunks."""

    __slots__ = ("_chunk_size", "_buffer", "_length")

    def __init__(self, chunk_size: int):
        self._chunk_size = chunk_size
        self._buffer: list[str] = []
        self._length = 0

    def push(self, chunk: str) -> list[str]:
        if not chunk:
            return []
        if self._chunk_size <= 0:
            return [chunk]
        self._buffer.append(chunk)
        self._length += len(chunk)
        if "\n" in chunk or self._length >= self._chunk_size:
            drained = self._drain()
            return [drained] if drained else []
        return []

    def flush(self) -> list[str]:
        if self._chunk_size <= 0:
            return []
        drained = self._drain()
        return [drained] if drained else []

    def _drain(self) -> str:
        if not self._buffer:
            return ""
        text = "".join(self._buffer)
        self._buffer.clear()
        self._length = 0
        return text


class ReasoningLogBuffer:
    """Fixed-size ring buffer that tracks the trailing reasoning_log segment."""

    __slots__ = ("_max_chars", "_enabled", "_chunks", "_length", "_truncated")

    def __init__(self, *, max_chars: int | None):
        self._max_chars = max_chars
        self._enabled = max_chars is None or max_chars > 0
        self._chunks: deque[str] = deque()
        self._length = 0
        self._truncated = False

    def append(self, chunk: str) -> None:
        if not chunk or not self._enabled:
            return
        self._chunks.append(chunk)
        self._length += len(chunk)
        if self._max_chars is not None:
            self._trim()

    def snapshot(self) -> str:
        if not self._enabled:
            return ""
        return "".join(self._chunks)

    @property
    def truncated(self) -> bool:
        return self._truncated

    def _trim(self) -> None:
        assert self._max_chars is not None
        excess = self._length - self._max_chars
        if excess <= 0:
            return
        while excess > 0 and self._chunks:
            head = self._chunks[0]
            head_len = len(head)
            if head_len <= excess:
                self._chunks.popleft()
                self._length -= head_len
                excess -= head_len
            else:
                self._chunks[0] = head[excess:]
                self._length -= excess
                excess = 0
        self._truncated = True
