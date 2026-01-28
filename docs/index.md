# AceAI

AceAI is an engineering-first agent framework: tools-first, explicit signatures, early failures, and OTLP-friendly tracing.

## Highlights
- Tools-first, strict schemas built from `typing.Annotated` + `spec`.
- Explicit providers and dependencies, no hidden planners or retries.
- OpenTelemetry spans for agent steps, tool calls, and LLM calls.
- Clear separation: workflow (you control) vs agent loop (model controls).

## Quick start

```python
import asyncio
from typing import Annotated

from openai import AsyncOpenAI

from aceai import AgentBase, Graph, LLMService, ToolExecutor, spec, tool
from aceai.llm.openai import OpenAI


@tool
def add(
    a: Annotated[int, spec(description="Left operand")],
    b: Annotated[int, spec(description="Right operand")],
) -> int:
    return a + b


def build_agent(api_key: str) -> AgentBase:
    graph = Graph()
    provider = OpenAI(
        client=AsyncOpenAI(api_key=api_key),
        default_meta={"model": "gpt-4o-mini"},
    )
    llm_service = LLMService(providers=[provider], timeout_seconds=60)
    executor = ToolExecutor(graph=graph, tools=[add])
    return AgentBase(
        sys_prompt="You are a strict calculator. Use tools when needed.",
        default_model="gpt-4o-mini",
        llm_service=llm_service,
        executor=executor,
        max_steps=5,
    )


async def main() -> None:
    agent = build_agent(api_key="...")
    answer = await agent.ask("What is 9 + 13?")
    print(answer)


asyncio.run(main())
```

## Where to go next
- Read `docs/introduction.md` for concepts and setup.
- Follow `docs/tutorial.md` for a full agent build.
- Dive into `docs/features.md` for schema, DI, and tracing details.
