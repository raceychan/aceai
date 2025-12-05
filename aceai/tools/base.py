from inspect import signature
from typing import Annotated as Annotated
from typing import Any, Callable

from msgspec import Struct
from msgspec.json import Decoder
from msgspec.json import encode as msg_encode
from msgspec.structs import asdict as msg_asdict


from ._param import ToolSignature
from .interface import ToolSpec


class Tool[**P, R]:
    def __init__(
        self,
        name: str,
        description: str,
        signature: ToolSignature,
        func: Callable[P, R],
        decoder: Callable[[bytes], Struct],
    ):
        self.name = name
        self.description = description
        self.signature = signature
        self.func = func
        self._tool_schema: ToolSpec | None = None
        self._decoder = decoder

    def encode_return(self, value: R) -> str:
        return msg_encode(value).decode("utf-8")

    def decode_params(self, data: str) -> dict[str, Any]:
        payload = self._decoder(data.encode("utf-8"))
        return msg_asdict(payload)

    def __call__(self, *args: P.args, **kwds: P.kwargs) -> R:
        return self.func(*args, **kwds)

    def reset_tool_schema(self) -> None:
        self._tool_schema = None

    @property
    def tool_schema(self) -> ToolSpec:
        if self._tool_schema is None:
            self._tool_schema = ToolSpec(
                type="function",
                name=self.name,
                description=self.description,
                parameters=self.signature.generate_params_schema(),
            )
        return self._tool_schema

    @classmethod
    def from_func(cls, func: Callable[P, R]) -> "Tool[P, R]":
        func_sig = signature(func)
        tool_signature = ToolSignature.from_signature(func_sig)
        decoder = Decoder(type=tool_signature.virtual_struct, strict=False)
        return cls(
            name=func.__name__,
            description=func.__doc__ or "",
            signature=tool_signature,
            func=func,
            decoder=decoder.decode,
        )


def tool[**P, R](func: Callable[P, R]) -> Tool[P, R]:
    return Tool.from_func(func)
