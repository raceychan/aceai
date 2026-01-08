# AceAI

An engineering-first agent framework: tools-first, explicit signatures, early failures, and OTLP-friendly tracing.

## Requirements & install
- Python 3.12+; `openai` SDK only if you use the OpenAI provider.
- Install: `uv add aceai`

## Why another framework?
- Precise tool calls: force `typing.Annotated` + structured schemas; no broad “magic” unions.
- Engineer-friendly: dependency injection, strict msgspec decoding, surface errors instead of hiding them.
- Observable by default: emits OpenTelemetry spans; you configure the SDK/exporter however you like (OTLP/Langfuse/etc.).
- Predictable: explicit tool/provider registration; no hidden planners or silent fallbacks.

## Quick start

Define tools with strict annotations, wire a provider, executor, and agent.

```python
import json
from typing import Annotated
from msgspec import Struct, field
from openai import AsyncOpenAI

from aceai import AgentBase, Graph, LLMService, spec, tool
from aceai.executor import ToolExecutor
from aceai.llm.openai import OpenAI


class OrderItems(Struct):
    order_id: str
    items: list[dict] = field(default_factory=list)


@tool
def lookup_order(order_id: Annotated[str, spec(description="Order ID like ORD-200")]) -> OrderItems:
    # Non-string returns are auto JSON-encoded
    return OrderItems(order_id=order_id, items=[{"sku": "sensor-kit", "qty": 2}])


def build_agent(api_key: str):
    graph = Graph()
    provider = OpenAI(
        client=AsyncOpenAI(api_key=api_key),
        default_meta={"model": "gpt-4o-mini"},
    )
    llm_service = LLMService(providers=[provider], timeout_seconds=60)
    executor = ToolExecutor(graph=graph, tools=[lookup_order])
    return AgentBase(
        sys_prompt="You are a logistics assistant.",
        default_model="gpt-4o-mini",
        llm_service=llm_service,
        executor=executor,
        max_steps=8,
    )
```

Plug your own loop/UI into `AgentBase`. See `demo.py` for a multi-tool async workflow.

## Concepts: workflow / agent / hybrid

### Workflow
A workflow is an LLM app where *you* own the control flow: what happens each step, the order, branching, and validation are all explicit in your code. The LLM is just one step (or a few steps) you call into.

In AceAI, implementing a workflow usually means calling `LLMService` directly:
- Use `LLMService.complete(...)` for plain text completion.
- Use `LLMService.complete_json(schema=...)` for structured steps, strictly decoding output into a `msgspec.Struct` (mismatches fail/retry).

Example: a two-stage workflow (extract structure first, then generate an answer).

```python
from msgspec import Struct
from openai import AsyncOpenAI

from aceai import LLMService
from aceai.llm import LLMMessage
from aceai.llm.openai import OpenAI


class Intent(Struct):
    task: str
    language: str


async def run_workflow(question: str) -> str:
    llm = LLMService(
        providers=[
            OpenAI(
                client=AsyncOpenAI(api_key="..."),
                default_meta={"model": "gpt-4o-mini"},
            )
        ],
        timeout_seconds=60,
    )

    intent = await llm.complete_json(
        schema=Intent,
        messages=[
            LLMMessage.build(role="system", content="Extract {task, language}."),
            LLMMessage.build(role="user", content=question),
        ],
    )

    resp = await llm.complete(
        messages=[
            LLMMessage.build(
                role="system",
                content=f"Answer the user. language={intent.language}; task={intent.task}.",
            ),
            LLMMessage.build(role="user", content=question),
        ],
    )
    return resp.text
```

### Agent
An agent is a workflow where the *LLM* owns the control flow: at each step it decides whether to call tools, which tool to call, and with what arguments. The framework executes tools, appends tool outputs back into context, and keeps the loop running until a final answer is produced.

In AceAI, building an agent is wiring three pieces:
- `LLMService`: talks to a concrete LLM provider (complete/stream/complete_json).
- `ToolExecutor`: strictly decodes tool args, resolves DI, runs tools, and encodes returns back to strings.
- `AgentBase`: runs the multi-step loop, maintains message history, orchestrates tool calls, and emits events.

Example: a minimal agent (one `add` tool).

```python
from typing import Annotated

from openai import AsyncOpenAI

from aceai import AgentBase, Graph, LLMService, spec, tool
from aceai.executor import ToolExecutor
from aceai.llm.openai import OpenAI


@tool
def add(
    a: Annotated[int, spec(description="Left operand")],
    b: Annotated[int, spec(description="Right operand")],
) -> int:
    return a + b


def build_agent(api_key: str) -> AgentBase:
    graph = Graph()
    llm = LLMService(
        providers=[
            OpenAI(
                client=AsyncOpenAI(api_key=api_key),
                default_meta={"model": "gpt-4o-mini"},
            )
        ],
        timeout_seconds=60,
    )
    executor = ToolExecutor(graph=graph, tools=[add])
    return AgentBase(
        sys_prompt="You are a strict calculator. Use tools when needed.",
        default_model="gpt-4o-mini",
        llm_service=llm,
        executor=executor,
        max_steps=5,
    )
```

### Hybrid
The most common production shape is hybrid: keep the deterministic parts as a workflow (call `LLMService` directly; `complete_json` is great for strict I/O), and delegate open-ended reasoning + tool use to `AgentBase`.

A simple approach is to subclass `AgentBase`, add helper methods that call `LLMService` for pre/post-processing, then hand off to `super().ask(...)`:

```python
from msgspec import Struct

from aceai.agent import AgentBase
from aceai.llm import LLMMessage


class Route(Struct):
    department: str


class RoutedAgent(AgentBase):
    async def classify(self, question: str) -> Route:
        return await self.llm_service.complete_json(
            schema=Route,
            messages=[
                LLMMessage.build(role="system", content="Classify department."),
                LLMMessage.build(role="user", content=question),
            ],
            metadata={"model": self.default_model},
        )

    async def ask(self, question: str, **request_meta) -> str:
        route = await self.classify(question)
        return await super().ask(f"[dept={route.department}] {question}", **request_meta)
```

## Features

### Tools-first
Params must be `typing.Annotated` with `spec(...)`; missing annotations fail at registration. The spec drives JSON Schema for LLM tools and docs.

Tutorial: annotate every tool parameter with a concrete type and `spec` metadata (description/alias/etc.). If you skip it, `tool(...)` raises at import time so mistakes are caught early.

```python
from typing import Annotated
from aceai.tools import tool, spec

@tool
def greet(name: Annotated[str, spec(description="Person to greet")]) -> str:
    return f"hi {name}"

# If you write: def bad(x: int): ...  -> tool(bad) will raise due to missing Annotated/spec.
```

### Strict decoding & auto JSON encoding
msgspec Struct validation enforces input types; return values are auto JSON-encoded (works for Struct/dict/primitive). LLM tool arguments are decoded into the right shapes, and outputs are encoded back to strings.

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
    # Returning Struct is fine; executor encodes to JSON string.
    return User(id=user_id, name="Ada", tags=["admin"])

# When LLM emits {"user_id":"not-int"}, msgspec decode raises immediately.
# When tool returns User(...), executor returns '{"id":1,"name":"Ada","tags":["admin"]}'.
```

### Dependency injection (`ididi.use`)
Mark dependencies with `ididi.use(...)`; the executor resolves them before invocation, so tools stay pure. A realistic chain with nested deps:

```python
from typing import Annotated
from ididi import use
from aceai.tools import tool, spec


class AsyncConnection:
    async def execute(self, query: str, params: dict) -> dict:
        return {"order_id": params["order_id"], "status": "created"}


async def get_conn(
    engine: Annotated[AsyncEngine, use(get_async_engine)]
) -> AsyncGenerator[AsyncConnection, None]:
    async with engine.connect() as conn:
        yield conn


class OrderRepo:
    def __init__(self, conn: AsyncConnection):
        self.conn = conn

    async def create(self, order_id: str, items: list[dict]) -> dict:
        return await self.conn.execute(
            "INSERT INTO orders VALUES (:order_id, :items)",
            {"order_id": order_id, "items": items},
        )


def build_repo(conn: Annotated[AsyncConnection, use(get_conn)]) -> OrderRepo:
    return OrderRepo(conn)


class OrderService:
    def __init__(self, repo: OrderRepo):
        self.repo = repo

    async def create_order(self, order_id: str, items: list[dict]) -> dict:
        return await self.repo.create(order_id, items)


def build_service(repo: Annotated[OrderRepo, use(build_repo)]) -> OrderService:
    return OrderService(repo)


@tool
async def create_order(
    order_id: Annotated[str, spec(description="New order id")],
    items: Annotated[list[dict], spec(description="Line items")],
    svc: Annotated[OrderService, use(build_service)],
) -> dict:
    return await svc.create_order(order_id, items)

```

Executor resolves AsyncConnection -> OrderRepo -> OrderService before invoking the tool.

### Observability (OpenTelemetry)
AceAI emits OpenTelemetry spans around agent steps, LLM calls, and tool calls. Configure OpenTelemetry natively (install the SDK/exporter as needed, e.g. `uv add 'aceai[otel]'`), then pass a tracer (or rely on the global tracer provider).

#### Example: Langfuse (OTLP HTTP)
```python
import base64

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

PUBLIC_KEY = "pk-lf-xxxx"
SECRET_KEY = "sk-lf-xxxx"
auth = base64.b64encode(f"{PUBLIC_KEY}:{SECRET_KEY}".encode()).decode()

otel_provider = TracerProvider()
exporter = OTLPSpanExporter(
    # EU:
    endpoint="https://cloud.langfuse.com/api/public/otel/v1/traces",
    # US:
    # endpoint="https://us.cloud.langfuse.com/api/public/otel/v1/traces",
    headers={"Authorization": f"Basic {auth}"},
)
otel_provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(otel_provider)

tracer = trace.get_tracer("aceai-app")

llm_service = LLMService(providers=[provider], timeout_seconds=60)
executor = ToolExecutor(graph=graph, tools=[greet], tracer=tracer)
agent = AgentBase(..., tracer=tracer)
```

#### Example: configure via env vars
```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="https://cloud.langfuse.com/api/public/otel"
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic <base64(pk:sk)>"
```

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

otel_provider = TracerProvider()
otel_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(otel_provider)
```

#### Example: tests (InMemorySpanExporter)
```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import InMemorySpanExporter, SimpleSpanProcessor

exporter = InMemorySpanExporter()
otel_provider = TracerProvider()
otel_provider.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(otel_provider)

# ...run your agent...
spans = exporter.get_finished_spans()
```

### Completion semantics
An agent run completes when the model returns a step with **no tool calls**; that step's `LLMResponse.text` is treated as the final answer (and can be streamed via `response.output_text.delta`).

## Code notes & caveats
- **Tool signatures**: keep types concrete; no broad unions. Unannotated params raise immediately.  
  ```python
  # Incorrect: missing Annotated/spec
  def bad(x: int): ...

  # Incorrect: overly broad type
  def maybe(val: Any): ...

  # Correct
  def good(x: Annotated[int, spec(description="Left operand")]) -> int:
      return x
  ```
- **Return encoding**: `str`/`int`/`Struct`/`dict` are encoded to JSON before sending to the LLM.  
  ```python
  # Incorrect: double encoding (LLM sees quoted JSON)
  return json.dumps({"a": 1})

  # Correct: executor encodes once
  return {"a": 1}
  # or
  return User(id=1)
  ```

- **Tracing**: AceAI emits OpenTelemetry spans; configure `TracerProvider`/exporter natively (no AceAI wrapper), then pass a `tracer=...` into `LLMService`/`ToolExecutor`/`AgentBase`.

- **Failure policy**: fail fast; no implicit retries for tools. LLM retries are up to you.
- **OpenAI dependency**: only needed if you use the OpenAI provider or `demo.py`; importing that provider without the SDK will raise a missing dependency error.

## Extensibility

### Custom agent (subclass `AgentBase`)
In real products, the core reasoning loop is rarely the whole story: you often need to inject request metadata (tenant/user ids, model selection), enforce guardrails, integrate with your UI/event system, or standardize defaults across calls. Subclassing `AgentBase` lets you wrap those concerns around the existing streaming + tool-execution loop without re-implementing it; `AgentBase` already owns message assembly, step bookkeeping, and calling into `LLMService` and the `ToolExecutor`, so delegating to `super()` keeps the behavior consistent while you add your glue. This is usually the best place to customize because it keeps product policy at the boundary and leaves your tools/providers reusable and easy to test.

```python
from aceai.agent import AgentBase
from aceai.llm.models import LLMRequestMeta
from typing import Unpack

class MyAgent(AgentBase):
    async def ask(self, question: str, **request_meta: Unpack[LLMRequestMeta]) -> str:
        # e.g., enforce defaults / attach metadata for every request
        request_meta.setdefault("model", self.default_model)
        return await super().ask(question, **request_meta)
```

### Custom executor (subclass `ToolExecutor`)
Tool calls are a natural choke point for governance: you may want to enforce an allowlist, apply rate limits, add audits, or redact arguments before storing them. Subclass `ToolExecutor` and override `execute_tool` to add pre/post hooks, then delegate to `super().execute_tool` so you still benefit from AceAI’s standard argument decoding, dependency resolution, and return encoding. Centralizing policies here is typically better than sprinkling checks across tools because it keeps tools small and makes rules consistent across the entire tool surface.

```python
from aceai.executor import ToolExecutor
from aceai.errors import AceAIValidationError

class AuditedExecutor(ToolExecutor):
    async def execute_tool(self, tool_call):
        # pre-hook (e.g., allowlist)
        if tool_call.name not in {"lookup_order", "create_order"}:
            raise AceAIValidationError(f"Tool not allowed: {tool_call.name}")
        result = await super().execute_tool(tool_call)
        # post-hook (e.g., audit log / metrics)
        return result
```

### Custom tool spec (provider schema)
When integrating a new provider, the mismatch is often the *schema envelope* rather than your tool logic: some providers want `parameters`, others want `input_schema`, some require extra flags, and some wrap tools differently. AceAI separates signature extraction (`ToolSignature` built from `typing.Annotated` + `spec`) from schema rendering; implementing `IToolSpec.generate_schema()` lets you map the same underlying JSON Schema to whatever shape your provider expects, and you can attach that renderer per tool via `@tool(spec_cls=...)`. This is the cleanest way to support multiple providers because you don’t touch tool code or signature parsing—only the adapter changes.

```python
from typing import Annotated
from aceai.tools import tool, IToolSpec, spec

class MyProviderToolSpec(IToolSpec):
    def __init__(self, *, signature, name, description):
        self.signature = signature
        self.name = name
        self.description = description

    def generate_schema(self):
        return {
            "name": self.name,
            "description": self.description,
            "params": self.signature.generate_params_schema(),
            "type": "custom",
        }

@tool(spec_cls=MyProviderToolSpec)
def hello(name: Annotated[str, spec(description="Name")]) -> str:
    return f"hi {name}"

# If your provider needs different field names (e.g., "input_schema" instead of "parameters"),
# implement that in generate_schema() here without touching the rest of the agent stack.
```

See `demo.py` for a full multi-tool agent and `agent.ipynb` for an end-to-end notebook walkthrough.
