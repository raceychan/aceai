from aceai.core.cache import mem_cache


def test_mem_cache_supports_unhashable_value_keys() -> None:
    calls = 0

    @mem_cache(maxsize=4)
    def total(payload: dict[str, list[int]]) -> int:
        nonlocal calls
        calls += 1
        return sum(payload["items"])

    assert total({"items": [1, 2]}) == 3
    assert total({"items": [1, 2]}) == 3
    assert calls == 1
    assert total.cache_info().hits == 1  # type: ignore[attr-defined]


def test_mem_cache_can_copy_mutable_results() -> None:
    calls = 0

    @mem_cache(maxsize=4, copy_result=True)
    def build() -> dict[str, list[int]]:
        nonlocal calls
        calls += 1
        return {"items": [1]}

    first = build()
    first["items"].append(2)

    assert build() == {"items": [1]}
    assert calls == 1


def test_mem_cache_supports_identity_keys() -> None:
    calls = 0

    @mem_cache(maxsize=4, key_mode="identity")
    def size(value: list[int]) -> int:
        nonlocal calls
        calls += 1
        return len(value)

    first = [1]
    second = [1]

    assert size(first) == 1
    assert size(first) == 1
    assert size(second) == 1
    assert calls == 2


def test_mem_cache_clear_resets_entries_and_stats() -> None:
    calls = 0

    @mem_cache(maxsize=4)
    def double(value: int) -> int:
        nonlocal calls
        calls += 1
        return value * 2

    assert double(2) == 4
    assert double(2) == 4
    assert double.cache_info().hits == 1  # type: ignore[attr-defined]

    double.cache_clear()  # type: ignore[attr-defined]

    assert double.cache_info().hits == 0  # type: ignore[attr-defined]
    assert double.cache_info().currsize == 0  # type: ignore[attr-defined]
    assert double(2) == 4
    assert calls == 2
