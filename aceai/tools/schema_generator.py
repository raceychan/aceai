from copy import deepcopy
from types import GenericAlias, UnionType
from typing import Any, Callable, cast

from msgspec.json import schema_components

from aceai.interface import MISSING, JsonSchema, Maybe, is_json_compatible, is_present

SchemaHook = Callable[[type], dict[str, Any] | None] | None
RegularTypes = type | UnionType | GenericAlias


MSGSPEC_REF_PREFIX = "#/components/schemas/"
MSGSPEC_REF_TEMPLATE = MSGSPEC_REF_PREFIX + "{name}"


def _default_schema_hook(t: type) -> dict[str, Any] | None:
    if t is object:
        # Treat bare ``object`` annotations as unconstrained payloads.
        return {"type": "object"}
    return None


def json_schema(
    type_: RegularTypes,
    schema_hook: SchemaHook = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    def _combined_hook(t: type) -> dict[str, Any] | None:
        if schema_hook is not None:
            custom_schema = schema_hook(t)
            if custom_schema is not None:
                return custom_schema
        return _default_schema_hook(t)

    (schema,), defs = schema_components(
        (type_,),
        schema_hook=_combined_hook,  # type: ignore
        ref_template=MSGSPEC_REF_TEMPLATE,
    )
    return schema, defs


def _expand(
    node: dict[str, Any] | list[Any] | str, defs: dict[str, Any], ref_prefix: str
) -> None:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith(ref_prefix):
            schema_name = ref[len(ref_prefix) :]
            extras = {k: deepcopy(v) for k, v in node.items() if k != "$ref"}
            node.clear()
            node.update(deepcopy(defs.get(schema_name, {})))
            node.update(extras)
            _expand(node, defs, ref_prefix)
            return
        for value in node.values():
            _expand(value, defs, ref_prefix)
    elif isinstance(node, list):
        for item in node:
            _expand(item, defs, ref_prefix)


def inline_schema(type_: RegularTypes, default: Maybe[Any] = MISSING) -> JsonSchema:
    """
    Given a JSON Schema `schema` that may contain `$ref` references to definitions in `defs`, return a new schema with all references expanded inline.
    Examples:

    schema = {
        "type": "object",
        "properties": {
            "person": {
                "$ref": "#/components/schemas/Person",
                "description": "Main person object"
            },
            "address": {
                "$ref": "#/components/schemas/Address"
            }
        }
    }
    defs = {
        "Person": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age":  {"type": "integer"}
            }
        },
        "Address": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "zip":  {"type": "string"}
            }
        }
    }

    inline = inline_schema(schema, defs)
    inline:
    {
        "type": "object",
        "properties": {
            "person": {
                # resolved from defs["Person"]
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age":  {"type": "integer"}
                },
                # extra field kept from original node
                "description": "Main person object"
            },
            "address": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "zip":  {"type": "string"}
                }
            }
        }
    }
    """
    schema, defs = json_schema(type_)
    if not defs:
        return cast(JsonSchema, schema)

    _expand(schema, defs, MSGSPEC_REF_PREFIX)

    if is_present(default) and is_json_compatible(default):
        schema = deepcopy(schema)
        schema.setdefault("default", default)
    return cast(JsonSchema, schema)
