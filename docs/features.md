# Features

## Tools-first signatures
Every tool parameter must use `typing.Annotated` with `spec(...)`. Missing annotations fail at registration, which makes errors visible early.

```python
from typing import Annotated
from aceai import tool, spec


@tool
def greet(name: Annotated[str, spec(description="Person to greet")]) -> str:
    return f"hi {name}"
```

## Strict decoding and encoding
AceAI uses msgspec for tool argument decoding and return encoding. If the model emits invalid types, decoding fails immediately.

```python
from msgspec import Struct, field
from typing import Annotated
from aceai import tool, spec


class User(Struct):
    id: int
    name: str
    tags: list[str] = field(default_factory=list)


@tool
def user_info(user_id: Annotated[int, spec(description="User id")]) -> User:
    return User(id=user_id, name="Ada", tags=["admin"])
```

## Dependency injection (ididi)
Mark dependencies with `ididi.use(...)` and the executor resolves them before invocation, so tools stay pure.

```python
from typing import Annotated
from ididi import use
from aceai import tool, spec


class Repo:
    def __init__(self, token: str):
        self.token = token


def build_repo() -> Repo:
    return Repo(token="...")


@tool
def fetch(
    query: Annotated[str, spec(description="Search query")],
    repo: Annotated[Repo, use(build_repo)],
) -> dict:
    return {"query": query, "token": repo.token}
```

## OpenTelemetry tracing
AceAI emits spans around agent steps, tool calls, and LLM calls. Configure OpenTelemetry however you want, then pass a tracer to `LLMService`, `Executor`, or `Agent`.

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

provider = TracerProvider()
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("agent-core")
```

## Lifecycle hooks
AceAI exposes typed lifecycle hooks through `HookRegistry`. Hooks are explicit async decision points: they can patch model requests, supply tool execution state, observe responses, or record model errors without mutating private run-loop state.

Model request hooks are split into two phases:

- `before_model_request` runs while AceAI assembles the logical request from the current context and tools.
- `before_model_call` runs after context preparation and compression, when the final `PreparedModelRequest` is about to be sent to the provider.

```python
from dataclasses import dataclass

from aceai import Agent
from aceai.core import HookContext, HookRegistry, ModelRequestPatch, PreparedModelRequest
from aceai.llm import LLMMessage


@dataclass(frozen=True)
class RuntimeContext:
    tenant_id: str


hooks = HookRegistry[RuntimeContext]()


@hooks.before_model_call(name="tenant.context_guard", order=20)
async def add_final_hint(
    ctx: HookContext[RuntimeContext],
    request: PreparedModelRequest,
) -> ModelRequestPatch | None:
    if ctx.data is None or request.step_index != 0:
        return None
    return ModelRequestPatch(
        append_messages=(
            LLMMessage.build(
                role="system",
                content=f"Use tenant policy for {ctx.data.tenant_id}.",
            ),
        ),
    )


agent = Agent(..., hook_registry=hooks)
run = agent.create_run("Summarize the account.", hook_context=RuntimeContext("acme"))
```

Use `Agent.prepare_model_request(run)` to preview the same hook path without committing hook effects.

## Provider adapters
Tool schema generation is separated from tool parsing. Implement `IToolSpec.generate_schema()` to adapt to providers that want different tool envelopes or field names.
