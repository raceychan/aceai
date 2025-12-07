import math

from msgspec import Struct, to_builtins

from aceai.interface import MISSING, UNSET, is_json_compatible, is_present, is_set


def test_is_json_compatible_accepts_scalar_types() -> None:
    assert is_json_compatible(None)
    assert is_json_compatible(42)
    assert is_json_compatible(("text", True))


def test_is_json_compatible_rejects_bad_numbers_and_objects() -> None:
    assert not is_json_compatible(float("inf"))
    assert not is_json_compatible(math.nan)
    assert not is_json_compatible((object(),))


def test_is_present_and_missing_marker() -> None:
    assert is_present("value")
    assert not is_present(MISSING)


def test_is_set_uses_msgspec_unset_sentinel() -> None:
    assert not is_set(UNSET)
    assert is_set("anything")


def test_missing_marker_is_falsey() -> None:
    assert not bool(MISSING)


def test_is_json_compatible_handles_nested_lists_and_dicts(monkeypatch) -> None:
    nested = [
        {"items": [1, True, None]},
        {"meta": {"name": "entry"}},
    ]
    checker = is_json_compatible.__wrapped__
    monkeypatch.setattr("aceai.interface.is_json_compatible", checker)
    assert checker(nested)
    assert not checker({"invalid": object()})


def test_msgspec_to_builtins_recurses_into_nested_structs() -> None:
    class Child(Struct):
        value: int

    class Parent(Struct):
        child: Child

    result = to_builtins(Parent(child=Child(value=7)))

    assert result == {"child": {"value": 7}}


def test_msgspec_to_builtins_excludes_unset_fields() -> None:
    class Payload(Struct, kw_only=True):
        required: int
        optional: int = UNSET

    result = to_builtins(Payload(required=1))

    assert "optional" not in result
