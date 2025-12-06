from functools import lru_cache
from typing import Any, Literal, TypedDict, TypeGuard

from msgspec import UNSET, Struct, UnsetType
from msgspec.structs import asdict
from typing_extensions import dataclass_transform


@dataclass_transform(frozen_default=True)
class Record(Struct, frozen=True, kw_only=True):
    def asdict(self) -> dict[str, Any]:
        return asdict(self)


class _Missed:
    def __bool__(self) -> Literal[False]:
        return False

    def __repr__(self) -> str:
        return "aceai.MISSING"


type Maybe[T] = T | _Missed

MISSING = _Missed()


def is_present[T](value: Maybe[T]) -> TypeGuard[T]:
    return value is not MISSING


@lru_cache
def is_json_compatible(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (str, int, bool)):
        return True
    if isinstance(value, float):
        return value == value and value not in (float("inf"), float("-inf"))
    if isinstance(value, list):
        return all(is_json_compatible(item) for item in value)
    if isinstance(value, tuple):
        return all(is_json_compatible(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and is_json_compatible(val)
            for key, val in value.items()
        )
    return False


JSONType = Literal["string", "number", "integer", "boolean", "object", "array", "null"]


class JsonSchema(TypedDict, total=False):
    # --- Core JSON Schema ---
    type: JSONType

    # For object
    properties: dict[str, Any]
    required: list[str]
    additionalProperties: bool
    minProperties: int
    maxProperties: int

    # For array
    items: Any
    minItems: int
    maxItems: int
    uniqueItems: bool

    # For string
    minLength: int
    maxLength: int
    pattern: str
    format: str  # email, uri, date-time...

    # For numbers
    minimum: float
    maximum: float
    exclusiveMinimum: float
    exclusiveMaximum: float
    multipleOf: float

    # Enums
    enum: list[Any]

    # OpenAI extension
    enumDescriptions: list[str]

    # Value constraints
    const: Any
    default: Any

    # Composition
    allOf: list[Any]
    anyOf: list[Any]
    oneOf: list[Any]
    not_: Any

    # Conditionals
    if_: Any  # "if" is invalid identifier, so name it if_
    then: Any
    else_: Any  # same trick as if_

    # --- Meta fields your code adds ---
    description: str
    examples: list[Any]

    # From extra_json_schema
    # These are user-provided; cannot be known statically
    # but we allow arbitrary extension fields
    deprecationMessage: str  # OpenAI extension
    # Anything else from param_spec.constraint["extra_json_schema"]
    # is allowed because consumer might supply custom fields.


type Unset[T] = UnsetType | T


def is_set[T](value: Unset[T]) -> TypeGuard[T]:
    return value is not UNSET
