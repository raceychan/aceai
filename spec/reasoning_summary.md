# Reasoning Summary Capture (OpenAI Responses)

This note captures the gaps we still have when talking to OpenAI’s reasoning models (o3, o4-mini, GPT‑5) through the Responses API, plus the concrete steps we should take to unlock their visible reasoning traces.

## What the Responses payload actually looks like

The cookbook example response includes three places that matter for us:

1. `output[*]` items with `type == "reasoning"` (aka `ResponseReasoningItem`). These include an `id`, `summary` chunks (each with `text`), and status metadata.
2. A top-level `response.reasoning` object that echoes effort configuration and the generated summary (if you asked for it).
3. `usage.output_tokens_details.reasoning_tokens` which tells you how many “thinking” tokens were billed.

Unless we explicitly set `reasoning={"summary": "auto"}` (and optionally `effort`), OpenAI hides the actual summary text; we just see that reasoning tokens were spent.

## Gaps in our current adapter

* `_build_base_response_kwargs` never sets the `reasoning` param, so summaries are never requested.
* `_to_llm_response` ignores the reasoning items entirely. We only surface plain text and tool calls inside `LLMSegment`s.
* `_map_stream_event` drops the `response.reasoning_summary_*` streaming events, so even if we enable summaries we still wouldn’t see the incremental deltas.

Net result: operators see zero “Reasoning” even though the provider can expose a sanitized summary.

## Implementation plan

1. **Allow callers to request reasoning summaries.**
   * Teach `_build_base_response_kwargs` to look for `metadata["reasoning"]` and pass it through to OpenAI.
   * When no explicit value is provided and the selected model is a reasoning tier (`o3`, `o4`, `gpt-5`, etc.), default to `{"effort": "medium", "summary": "auto"}` so we always get something to show.

2. **Parse reasoning items in `_to_llm_response`.**
   * Add `_extract_reasoning_items(response)` to collect every `ResponseReasoningItem`.
   * Extend `_build_segments_from_response(...)` to accept those items and emit `LLMSegment(type="reasoning", content=<joined summary text>, metadata={"reasoning_id": item.id})`.
   * Preserve `usage.output_tokens_details.reasoning_tokens` inside `LLMResponse.extras["reasoning_tokens"]` so downstream consumers can surface billing context.

3. **(Optional but nice) stream reasoning deltas.**
   * Update `_map_stream_event` to recognize `response.reasoning_summary_text.delta`, `.added`, `.done`, etc. For `*.delta`, emit a new agent event type (or reuse `agent.llm.output_text.delta` with `LLMSegment(type="reasoning", metadata={"is_delta": True})`) so the terminal UI can show live “Reasoning:” lines before the final summary lands.
   * Buffer per-item deltas inside the provider adapter so that when `response.completed` fires we already have the concatenated summary string for `_build_segments_from_response`.

4. **Surface the summary in UI.**
   * Once `LLMResponse.segments` contains `type="reasoning"`, the existing agent pipeline can append those chunks into the reasoning log, giving the Term UI real content even when the model refuses to emit ad-hoc narration.

This spec does not change any code directly; it documents why the missing reasoning traces happen today and outlines the mechanical steps needed to expose the summaries that OpenAI already provides.***
