# FAQ

## Do I need the OpenAI SDK?
Only if you use the OpenAI provider. If you do, install `openai` and pass an `AsyncOpenAI` client to `OpenAI`.

## Why does tool registration fail?
Every tool parameter must use `typing.Annotated` with `spec(...)`. Missing annotations fail fast so mistakes are obvious.

## Why did `complete_json` fail even after retries?
`complete_json` requires the first message to be a system message and validates output strictly with msgspec. If the model keeps returning invalid JSON, the call will fail.

## Where do I configure tracing?
Use OpenTelemetry directly. Configure a `TracerProvider` and exporter, then pass a tracer to `LLMService`, `ToolExecutor`, or `AgentBase`.

## Does AceAI catch tool exceptions?
No. Tool exceptions are allowed to surface so failures are visible.
