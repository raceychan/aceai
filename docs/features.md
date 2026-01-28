# Features

## Tools-first signatures
Every tool parameter must use `typing.Annotated` with `spec(...)`. Missing annotations fail at registration, which makes errors visible early.

```python
from typing import Annotated
from aceai.tools import tool, spec


@tool
def greet(name: Annotated[str, spec(description="Person to greet")]) -> str:
    return f"hi {name}"
```

## Strict decoding and encoding
AceAI uses msgspec for tool argument decoding and return encoding. If the model emits invalid types, decoding fails immediately.

```python
from msgspec import Struct, field
from typing import Annotated
from aceai.tools import tool, spec


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
from aceai.tools import tool, spec


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
AceAI emits spans around agent steps, tool calls, and LLM calls. Configure OpenTelemetry however you want, then pass a tracer to `LLMService`, `ToolExecutor`, or `AgentBase`.

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

provider = TracerProvider()
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("aceai-app")
```

## Provider adapters
Tool schema generation is separated from tool parsing. Implement `IToolSpec.generate_schema()` to adapt to providers that want different tool envelopes or field names.
