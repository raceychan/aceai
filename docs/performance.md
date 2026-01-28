# Performance

AceAI aims for predictable behavior rather than hidden optimization. Measure your workload and tune the components you own.

## What affects latency
- Model choice and provider response time.
- Prompt size and tool schema size.
- Tool execution time and dependency resolution.
- Tracing/exporter overhead.

## Practical tuning
- Keep tool schemas small and precise; avoid broad unions.
- Use smaller models for routing or classification steps.
- Reduce message history when you own the workflow.
- Batch your own external calls inside tools when possible.
- Disable or sample tracing exporters if they are expensive.

## Throughput considerations
- `LLMService` uses a per-request timeout; raise it only if needed.
- Tools run in the order requested by the model; if you need concurrency, orchestrate it in your workflow.

## Measuring
Run the same prompt and tool set with fixed inputs, and measure:
- End-to-end latency
- Tool execution time per call
- Token usage from your provider
