# Introduction

AceAI is designed for engineers who want explicit control, strict schemas, and observable execution.

## Requirements
- Python 3.12+
- Provider SDKs only for the adapters you use

## Install

```bash
uv add aceai
```

For built-in providers, install the matching extra:

```bash
uv add "aceai[openai]"
uv add "aceai[anthropic]"
uv add "aceai[deepseek]"
uv add "aceai[codex]"
```

## Core concepts

### Workflow
A workflow is an LLM app where you own the control flow. You decide each step and validation rule, and the LLM is just one step you call into.

In AceAI, you usually call `LLMService` directly for workflows:
- `complete(...)` for plain text
- `complete_json(schema=...)` for strict structured output

The example below uses the OpenAI adapter and requires `aceai[openai]`.

```python
from msgspec import Struct
from openai import AsyncOpenAI

from aceai import LLMService
from aceai.llm import LLMMessage
from aceai.llm.openai import OpenAI


class Intent(Struct):
    task: str
    language: str


async def classify(question: str) -> Intent:
    llm = LLMService(
        providers=[
            OpenAI(
                client=AsyncOpenAI(api_key="..."),
                default_meta={"model": "gpt-4o-mini"},
            )
        ],
        timeout_seconds=60,
    )

    return await llm.complete_json(
        schema=Intent,
        messages=[
            LLMMessage.build(role="system", content="Extract {task, language}."),
            LLMMessage.build(role="user", content=question),
        ],
    )
```

### Agent
An agent is a workflow where the LLM decides whether to call tools at each step. The agent loop executes tools, appends results, and stops when the model returns a step with no tool calls.

To build an agent you wire three pieces:
- `LLMService` to talk to providers
- `Executor` to decode and run tools
- `Agent` to manage the step loop

### Hybrid
A common production shape is hybrid: do deterministic steps in a workflow, then delegate open-ended reasoning and tool use to `Agent`. Subclassing `Agent` is a clean way to add product-specific behavior while keeping the core loop intact.

### Lifecycle hooks
Lifecycle hooks let applications extend the agent loop at typed boundaries without taking over the run loop. Register async functions with `HookRegistry` to patch model requests, inspect the final provider-bound request, refresh tool execution state, observe completed model responses, or record model errors.

Use hooks when product-specific runtime context must enter a request late, but should still remain ordered, typed, previewable, and separated from durable transcript events.
