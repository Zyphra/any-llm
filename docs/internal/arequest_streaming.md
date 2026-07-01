# `arequest` streaming: implementation notes

`arequest` was shipped non-streaming on purpose: the goal of the first PR was
unifying multi-turn reasoning state across providers, and the streaming-event
glue is what bloats the diff. This document is the contract a follow-up PR
should implement.

## Current decision

Keep `arequest` non-streaming in the first implementation. The earlier prototype
exposed `RequestStreamEvent` in public signatures and passed provider streams
through directly, but it did not define the shared request stream type, did not
adapt Anthropic or Gemini streams into OpenResponses events, and did not cover
provider-tagged continuity state in terminal stream events.

Treat that prototype as discarded exploration. Conflict resolution should keep
the explicit `NotImplementedError` for `stream=True`, remove streaming return
types from `request` / `arequest`, and avoid exporting `RequestStreamEvent`
until the adapter and tests described here exist.

## Current implementation anchors

The non-streaming implementation already has most of the conversion pieces a
streaming follow-up should reuse:

- `src/any_llm/types/request.py` owns `RequestInput`, `RequestParams`, and
  `RequestResponse`. Add the `RequestStreamEvent` alias there only when the
  adapters exist.
- `src/any_llm/utils/request_output.py` owns `make_response_resource`,
  `make_text_message`, `make_reasoning_item`, `make_function_call_item`, and
  `finalize_request_response`. Streaming should add lifecycle-event helpers
  next to these builders instead of duplicating output-item construction in
  each provider.
- `src/any_llm/utils/request_state.py` owns provider-tagged opaque state. Every
  streamed terminal response must preserve the same `encrypted_content` tagging
  rules as the non-streaming path.
- OpenAI currently routes through `BaseOpenAIProvider._arequest`,
  `split_request_input_for_openai`, and `stamp_openai_response_state`.
  Streaming should buffer the native `response.completed` event long enough to
  stamp the final response with the synthetic OpenAI state item.
- Anthropic currently routes through `_convert_request_params`,
  `_convert_request_input`, and `_convert_request_response`. Streaming should
  reuse `_convert_request_params` for request setup and mirror
  `_convert_request_response` as content blocks close.
- Gemini currently routes through `request_input_to_completion_messages` and
  `_completion_to_request_response`. Streaming should reuse the same input
  conversion, call `_acompletion` with `stream=True`, and mirror
  `_completion_to_request_response` while collecting `thought_signature`.

Do not remove the guard in `AnyLLM.arequest` until all provider adapters return
the same OpenResponses event family and the sync `request(..., stream=True)`
wrapper is tested.

## Public surface

When `stream=True` is passed, `arequest` should return an
`AsyncIterator[RequestStreamEvent]` where `RequestStreamEvent` is a re-export
of the OpenResponses streaming union. Concretely, at minimum:

```
ResponseCreatedStreamingEvent
ResponseInProgressStreamingEvent
ResponseOutputItemAddedStreamingEvent
ResponseOutputItemDoneStreamingEvent
ResponseContentPartAddedStreamingEvent
ResponseContentPartDoneStreamingEvent
ResponseOutputTextDeltaStreamingEvent
ResponseOutputTextDoneStreamingEvent
ResponseReasoningDeltaStreamingEvent
ResponseReasoningDoneStreamingEvent
ResponseFunctionCallArgumentsDeltaStreamingEvent
ResponseFunctionCallArgumentsDoneStreamingEvent
ResponseCompletedStreamingEvent
ResponseFailedStreamingEvent
```

The first event must be `response.created`, the last must be
`response.completed` (or `response.failed`), and every output item must bracket
its deltas with `output_item.added` / `output_item.done`. Tests should assert
that, and that `sequence_number` is monotonically increasing.

## Per-provider sketch

The non-streaming converters in `providers/*/utils.py` already do the
input/output translation we need; streaming adds an event-stream adapter on top.

### OpenAI

`Responses.stream` already emits the OpenResponses event union natively, so the
adapter is essentially a pass-through:

1. Call `aresponses(..., stream=True)`.
2. Re-yield each event unchanged.
3. After the terminal `response.completed`, synthesize a trailing reasoning item
   carrying `previous_response_id` (same trick as `stamp_openai_response_state`)
   and emit a final `output_item.added` + `output_item.done` for it. The
   `response.completed` event must reflect this final output list. The
   simplest approach is to buffer the original `response.completed`, append the
   stamped item to its `response.output`, and re-emit.

### Anthropic

Anthropic's `messages.stream` yields content-block lifecycle events. Build a
small state machine that:

1. Emits `response.created` + `response.in_progress` on the first event.
2. For each `content_block_start`:
   - `thinking` → emit `output_item.added` (reasoning item with no text yet).
   - `tool_use` → emit `output_item.added` (function-call item, `arguments=""`).
   - `text` → emit `output_item.added` (message) + `content_part.added`.
3. For each `content_block_delta`:
   - `thinking_delta` → `response.reasoning.delta`.
   - `input_json_delta` → `response.function_call_arguments.delta`.
   - `text_delta` → `response.output_text.delta`.
4. For each `content_block_stop`, emit the matching `*.done` events plus
   `output_item.done`. For thinking blocks, encode `signature` into
   `encrypted_content` via `encode_provider_state(CLAUDE, ...)` exactly as the
   non-streaming path does.
5. On stream end, emit `response.completed` with the final assembled
   `ResponseResource` (build it from the accumulated items).

### Gemini

Gemini routes through `acompletion`, so streaming reuses the chat-completion
chunk stream:

1. Build the same `CompletionParams` as the non-streaming path (with
   `stream=True`) and call `_acompletion`.
2. Track per-choice state: open reasoning item, open tool calls (keyed by
   `tool_call.index`), open message text. Use UUIDs for `item_id`.
3. Map chunk deltas onto the OpenResponses event union: `delta.reasoning` →
   reasoning events; `delta.tool_calls` → function-call events;
   `delta.content` → text-output events.
4. On `finish_reason`, close any open items, then emit `response.completed`
   with a final `ResponseResource` built via `make_response_resource`.
5. After the loop, encode any captured `extra_content.google.thought_signature`
   into a trailing reasoning item's `encrypted_content` (mirroring
   `_completion_to_request_response`). The signature lives on the function
   call's chunk metadata, so the state machine needs to surface it once the
   tool call closes.

## Cross-cutting requirements

- `sequence_number` is a monotonic counter shared across all events in the
  stream; reset per call.
- Any state that travels in `encrypted_content` MUST be tagged via
  `encode_provider_state` so a downstream fallback can drop it (same rule as
  the non-streaming path).
- `response_format=<typed>` is rejected up front (`AnyLLM.arequest` does this
  already); structured-output parsing only makes sense once the full text is
  collected, which streaming sidesteps.
- The shared adapter logic (counter, item-id generation, builder helpers) is
  the natural target for a small `utils/request_streaming.py` module. Resist
  the urge to inline it in each provider.

## Tests

- Add a unit test per provider that feeds a fixed sequence of fake provider
  events and asserts the emitted OpenResponses events (types and order).
- Keep the existing integration smoke test (`test_request_streaming_smoke`)
  asserting only that the first event is `response.created` and the last is
  `response.completed`.
