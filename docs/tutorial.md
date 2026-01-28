# Tutorial

This tutorial builds a minimal agent with one tool. It shows the full wiring: tools, provider, executor, and agent loop.

## 1) Define a tool

```python
from typing import Annotated
from aceai import spec, tool


@tool
def add(
    a: Annotated[int, spec(description="Left operand")],
    b: Annotated[int, spec(description="Right operand")],
) -> int:
    return a + b
```

## 2) Wire the provider and service

```python
from openai import AsyncOpenAI
from aceai import LLMService
from aceai.llm.openai import OpenAI


def build_llm(api_key: str) -> LLMService:
    provider = OpenAI(
        client=AsyncOpenAI(api_key=api_key),
        default_meta={"model": "gpt-4o-mini"},
    )
    return LLMService(providers=[provider], timeout_seconds=60)
```

## 3) Build the agent

```python
from aceai import AgentBase, Graph, ToolExecutor


def build_agent(api_key: str) -> AgentBase:
    graph = Graph()
    llm_service = build_llm(api_key)
    executor = ToolExecutor(graph=graph, tools=[add])
    return AgentBase(
        sys_prompt="You are a strict calculator. Use tools when needed.",
        default_model="gpt-4o-mini",
        llm_service=llm_service,
        executor=executor,
        max_steps=5,
    )
```

## 4) Run it

```python
import asyncio


async def main() -> None:
    agent = build_agent(api_key="...")
    answer = await agent.ask("Compute 12 + 34")
    print(answer)


asyncio.run(main())
```

## 5) Next steps
- Add more tools and annotate every parameter with `spec(...)`.
- Move deterministic steps into a workflow that calls `LLMService` directly.
- Pass a tracer to `ToolExecutor` or `AgentBase` to capture spans.
