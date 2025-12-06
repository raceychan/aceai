import math

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
