from contextvars import ContextVar

from opentelemetry.context import Context

_TRACE_CTX: ContextVar[Context | None] = ContextVar("aceai_trace_ctx", default=None)


def get_trace_ctx() -> Context | None:
    return _TRACE_CTX.get()


def set_trace_ctx(trace_ctx: Context | None) -> None:
    _TRACE_CTX.set(trace_ctx)

