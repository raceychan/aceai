"""Small in-memory caching helpers for hot pure-ish code paths."""

from collections import OrderedDict, namedtuple
from copy import deepcopy
from functools import wraps
from threading import RLock
from typing import Any, Callable, Hashable, Literal, ParamSpec, TypeVar, cast

P = ParamSpec("P")
R = TypeVar("R")
CacheInfo = namedtuple("CacheInfo", "hits misses maxsize currsize")
KeyMode = Literal["value", "identity"]


def mem_cache(
    maxsize: int = 1024,
    *,
    key_mode: KeyMode = "value",
    key_func: Callable[..., Hashable] | None = None,
    copy_result: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate a sync function with a bounded process-local LRU cache.

    This is similar to :func:`functools.lru_cache`, but it can build keys for
    common unhashable containers and can use object-identity keys for framework
    objects.

    Args:
        maxsize: Maximum number of cached entries. Must be positive.
        key_mode: ``"value"`` recursively freezes lists/dicts/sets into a
            hashable key. ``"identity"`` keys arguments by object identity.
        key_func: Optional custom key builder. It receives the wrapped call's
            ``*args`` and ``**kwargs`` and must return a hashable key.
        copy_result: Store and return deep copies. Use for functions returning
            mutable objects that callers may modify.
    """
    if maxsize < 1:
        raise ValueError("maxsize must be positive")
    if key_mode not in {"value", "identity"}:
        raise ValueError("key_mode must be 'value' or 'identity'")

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        cache: OrderedDict[Hashable, R] = OrderedDict()
        lock = RLock()
        hits = 0
        misses = 0

        def make_key(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Hashable:
            if key_func is not None:
                return key_func(*args, **kwargs)
            if key_mode == "identity":
                return _identity_key(args, kwargs)
            return _value_key(args, kwargs)

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            nonlocal hits, misses
            key = make_key(args, kwargs)
            with lock:
                if key in cache:
                    hits += 1
                    value = cache.pop(key)
                    cache[key] = value
                    return deepcopy(value) if copy_result else value
                misses += 1

            value = func(*args, **kwargs)
            stored = deepcopy(value) if copy_result else value
            with lock:
                cache[key] = stored
                if len(cache) > maxsize:
                    cache.popitem(last=False)
            return deepcopy(stored) if copy_result else value

        def cache_clear() -> None:
            nonlocal hits, misses
            with lock:
                cache.clear()
                hits = 0
                misses = 0

        def cache_info() -> CacheInfo:
            with lock:
                return CacheInfo(hits, misses, maxsize, len(cache))

        wrapper.cache_clear = cache_clear  # type: ignore[attr-defined]
        wrapper.cache_info = cache_info  # type: ignore[attr-defined]
        return cast(Callable[P, R], wrapper)

    return decorator


def _value_key(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Hashable:
    return (_freeze(args), _freeze(kwargs))


def _identity_key(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Hashable:
    return (
        tuple(_identity_token(arg) for arg in args),
        tuple((name, _identity_token(value)) for name, value in sorted(kwargs.items())),
    )


def _identity_token(value: Any) -> Hashable:
    if value is None or isinstance(value, (str, int, float, bool, bytes, type)):
        return (type(value), value)
    return (type(value), id(value))


def _freeze(value: Any) -> Hashable:
    if value is None or isinstance(value, (str, int, float, bool, bytes, type)):
        return cast(Hashable, value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, list):
        return (list, tuple(_freeze(item) for item in value))
    if isinstance(value, dict):
        return (
            dict,
            tuple(sorted((_freeze(key), _freeze(val)) for key, val in value.items())),
        )
    if isinstance(value, set):
        return (set, tuple(sorted(_freeze(item) for item in value)))
    if isinstance(value, frozenset):
        return (frozenset, tuple(sorted(_freeze(item) for item in value)))
    try:
        hash(value)
    except TypeError:
        return (type(value), id(value))
    return cast(Hashable, value)
