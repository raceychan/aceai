"""LLM-facing tool schema contract."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class IToolSpec(Protocol):
    name: str

    def __init__(self, **kwargs: Any) -> None: ...

    def generate_schema(self) -> dict[str, Any]: ...
