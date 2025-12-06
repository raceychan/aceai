from aceai.interface import Record
from aceai.tools.schema_generator import (
    MSGSPEC_REF_PREFIX,
    _expand,
    inline_schema,
    json_schema,
)


def test_inline_schema_sets_default_only_for_json_compatible_values() -> None:
    class Entry(Record):
        value: int

    with_default = inline_schema(list[Entry], default=("ok",))
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
