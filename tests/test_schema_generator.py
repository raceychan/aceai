from aceai.interface import Record
from aceai.tools.schema_generator import (
    MSGSPEC_REF_PREFIX,
    _default_schema_hook,
    _expand,
    inline_schema,
    json_schema,
)


def test_inline_schema_sets_default_only_for_json_compatible_values() -> None:
    class Entry(Record):
        value: int

    with_default = inline_schema(list[Entry], default=("ok",), required=False)
    assert with_default["default"] == ("ok",)

    class NotSerializable:
        pass

    without_default = inline_schema(list[Entry], default=(NotSerializable(),))
    assert "default" not in without_default


def test_json_schema_uses_default_hook_for_object_type() -> None:
    schema, defs = json_schema(object)
    assert schema == {"type": "object"}
    assert defs == {}


def test_expand_inlines_refs_and_preserves_overrides() -> None:
    schema = {
        "title": "Wrapper",
        "properties": {
            "item": {
                "$ref": f"{MSGSPEC_REF_PREFIX}Item",
                "description": "Custom",
            }
        },
    }
    defs = {
        "Item": {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
        }
    }

    _expand(schema, defs, MSGSPEC_REF_PREFIX)

    assert schema["properties"]["item"]["type"] == "object"
    assert schema["properties"]["item"]["description"] == "Custom"


def test_default_schema_hook_returns_none_for_non_object_type() -> None:
    assert _default_schema_hook(int) is None


def test_json_schema_prefers_custom_schema_hook() -> None:
    class CustomPayload:
        ...

    def custom_hook(t: type) -> dict | None:
        if t is CustomPayload:
            return {"type": "string", "maxLength": 4}
        return None

    schema, _defs = json_schema(CustomPayload, schema_hook=custom_hook)

    assert schema["maxLength"] == 4
