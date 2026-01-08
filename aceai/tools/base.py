from inspect import iscoroutinefunction, signature
from typing import Annotated as Annotated
from typing import Any, Callable, Protocol, TypedDict, Unpack, overload

from msgspec import Struct
from msgspec.json import Decoder
from msgspec.json import encode as msg_encode
from msgspec.structs import asdict as msg_asdict

from aceai.interface import MISSING, Maybe, is_present

from ._tool_sig import ToolSignature


class IToolSpec(Protocol):
    def __init__(
        self, *, signature: ToolSignature, name: str, description: str
    ) -> None: ...

    def generate_schema(self) -> dict[str, Any]: ...


class OpenAIToolSpec:
    def __init__(
        self, *, signature: ToolSignature, name: str, description: str
    ) -> None:
        self.signature = signature
        self.name = name
        self.description = description

    def generate_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.signature.generate_params_schema(),
            "strict": True,
        }


class IToolMeta(TypedDict, total=False):
    description: str
    """
    Human-readable description of the tool.
    """
    record_in_history: bool
    """
    Whether to record tool calls and outputs in the agent's history.
    """


class ToolMeta(Struct):
    "Every meta field should be optional."

    description: Maybe[str] = MISSING
    record_in_history: Maybe[bool] = MISSING


class Tool[**P, R]:
    def __init__(
        self,
        name: str,
        description: str,
        signature: ToolSignature,
        func: Callable[P, R],
        decoder: Callable[[bytes], Struct],
        record_in_history: Maybe[bool],
        spec_cls: type[IToolSpec] = OpenAIToolSpec,
    ):
        self.name = name
        self.description = description
        self.signature = signature
        self.func = func
        self._tool_spec_cache: dict[type[object], IToolSpec] = {}
        self._decoder = decoder
        self._record_in_history = record_in_history
        self._spec_cls = spec_cls
        self._is_async = iscoroutinefunction(func)

    @property
    def is_async(self) -> bool:
        """Whether the tool function is asynchronous."""
        return self._is_async

    def encode_return(self, value: R) -> str:
        """Serialize a tool return value to a JSON string."""
        return msg_encode(value).decode("utf-8")

    def decode_params(self, data: str) -> dict[str, Any]:
        """Decode a JSON tool arguments payload into a plain dict."""
        payload = self._decoder(data.encode("utf-8"))
        return msg_asdict(payload)

    def __call__(self, *args: P.args, **kwds: P.kwargs) -> R:
        """Invoke the wrapped tool function directly."""
        return self.func(*args, **kwds)

    def reset_tool_schema(self) -> None:
        """Drop all cached provider-specific tool specs."""
        self._tool_spec_cache.clear()

    def spec_for(self, spec_cls: type[IToolSpec]) -> IToolSpec:
        """Get or build a provider spec instance for the given spec class."""
        target_cls = spec_cls
        cached = self._tool_spec_cache.get(target_cls)
        if cached is not None:
            return cached

        spec = target_cls(
            signature=self.signature,
            name=self.name,
            description=self.description,
        )
        self._tool_spec_cache[target_cls] = spec
        return spec

    @property
    def tool_spec(self) -> IToolSpec:
        """Return the default provider spec instance."""
        return self.spec_for(self._spec_cls)

    @property
    def tool_schema(self) -> dict[str, Any]:
        """Generate a schema dict from the default provider spec."""
        return self.tool_spec.generate_schema()

    @classmethod
    def from_func(
        cls,
        func: Callable[P, R],
        meta: ToolMeta,
        spec_cls: type[IToolSpec] = OpenAIToolSpec,
    ) -> "Tool[P, R]":
        """Construct a Tool from a callable using its annotated signature."""
        func_sig = signature(func)
        tool_signature = ToolSignature.from_signature(func_sig)
        decoder = Decoder(type=tool_signature.virtual_struct, strict=False)
        if is_present(meta.description):
            description = meta.description
        else:
            description = func.__doc__ or ""
        return cls(
            name=func.__name__,
            description=description,
            signature=tool_signature,
            func=func,
            decoder=decoder.decode,
            record_in_history=meta.record_in_history,
            spec_cls=spec_cls,
        )


@overload
def tool[**P, R](
    func: Callable[P, R], *, spec_cls: type[IToolSpec] = OpenAIToolSpec
) -> Tool[P, R]: ...


@overload
def tool[**P, R](
    *,
    spec_cls: type[IToolSpec] = OpenAIToolSpec,
    **tool_meta: Unpack[IToolMeta],
) -> Callable[[Callable[P, R]], Tool[P, R]]: ...


def tool[**P, R](
    func: Maybe[Callable[P, R]] = MISSING,
    *,
    spec_cls: type[IToolSpec] = OpenAIToolSpec,
    **tool_meta: Unpack[IToolMeta],
) -> Tool[P, R] | Callable[[Callable[P, R]], Tool[P, R]]:
    if is_present(func):
        return Tool[P, R].from_func(
            func=func, meta=ToolMeta(**tool_meta), spec_cls=spec_cls
        )

    def wrapper(inner_func: Callable[P, R]) -> Tool[P, R]:
        return Tool[P, R].from_func(
            func=inner_func, meta=ToolMeta(**tool_meta), spec_cls=spec_cls
        )

    return wrapper
