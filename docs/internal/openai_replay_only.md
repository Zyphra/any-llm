# OpenAI `arequest` Replay-Only Sketch

This note captures a replay-only variant of the OpenAI `arequest` path that was explored and then intentionally not kept as the live implementation.

## Goal

Replace OpenAI's `previous_response_id` continuation with full item replay:

- do not rely on server-side stored response chaining
- replay prior OpenResponses items directly into the next `responses.create(...)` call
- request `reasoning.encrypted_content` explicitly
- round-trip OpenAI reasoning items through `RequestResponse.output`

This would be more compatible with strict stateless / zero-data-retention flows, but it is not clearly smaller than the current `previous_response_id` implementation.

## Sketch

### Provider utils

```python
from openresponses_types import ResponseResource
from openresponses_types.types import ReasoningBody

from any_llm.types.request import RequestInput, RequestParams, RequestResponse
from any_llm.utils.openresponses_compat import dump_json_model
from any_llm.utils.request_output import finalize_request_response
from any_llm.utils.request_state import OPENAI, decode_provider_state, encode_provider_state

REASONING_FAMILY = OPENAI


def build_openai_replay_input(input_data: RequestInput) -> list[dict[str, object]]:
    """Serialize request items for replay-based OpenAI Responses API continuity."""
    replay_items: list[dict[str, object]] = []

    for item in input_data:
        if not isinstance(item, ReasoningBody):
            replay_items.append(dump_json_model(item))
            continue

        encrypted_content = decode_provider_state(item.encrypted_content, REASONING_FAMILY)
        if encrypted_content is None:
            continue

        item_payload = dump_json_model(item)
        item_payload["encrypted_content"] = encrypted_content
        replay_items.append(item_payload)

    return replay_items


def finalize_openai_request_response(resource: ResponseResource, params: RequestParams) -> RequestResponse:
    """Re-tag OpenAI reasoning items so they can be replayed across future turns."""
    payload = resource.model_dump(mode="json", by_alias=True, exclude_none=False)

    tagged_output: list[dict[str, object]] = []
    for item in resource.output:
        item_payload = dump_json_model(item)
        if isinstance(item, ReasoningBody) and item.encrypted_content is not None:
            item_payload["encrypted_content"] = encode_provider_state(REASONING_FAMILY, item.encrypted_content)
        tagged_output.append(item_payload)

    payload["output"] = tagged_output
    return finalize_request_response(ResponseResource.model_validate(payload), response_format=params.response_format)
```

### Provider base

```python
@override
async def _arequest(self, params: RequestParams, **kwargs: Any) -> RequestResponse:
    """Route a request through the OpenAI Responses API using manual item replay."""
    responses_params = ResponsesParams(
        model=params.model,
        input=cast("Any", build_openai_replay_input(params.input)),
        tools=cast("Any", params.tools),
        tool_choice=params.tool_choice,
        max_output_tokens=params.max_output_tokens,
        temperature=params.temperature,
        top_p=params.top_p,
        instructions=params.instructions,
        parallel_tool_calls=params.parallel_tool_calls,
        reasoning=params.reasoning.model_dump(mode="json", exclude_none=True) if params.reasoning else None,
        presence_penalty=params.presence_penalty,
        frequency_penalty=params.frequency_penalty,
        metadata=params.metadata,
        include=["reasoning.encrypted_content"],
    )
    response = await self._aresponses(responses_params, **kwargs)
    resource = ...
    return finalize_openai_request_response(resource, params)
```

## What Changes Compared To `previous_response_id`

The replay-only variant removes:

- `_latest_openai_response_state(...)`
- `_is_openai_follow_up_item(...)`
- `split_request_input_for_openai(...)`
- `stamp_openai_response_state(...)`
- `previous_response_id` handling in `_arequest(...)`

and replaces them with:

- `build_openai_replay_input(...)`
- `finalize_openai_request_response(...)`
- `include=["reasoning.encrypted_content"]`

## Tradeoffs

### Pros

- Works better with stateless / `store=false` / zero-data-retention use cases.
- Uses the same general "replay items" story as other providers.
- Avoids synthetic "response id as reasoning item" state.

### Cons

- Not obviously fewer lines of code than the `previous_response_id` path.
- Requires explicit handling of OpenAI reasoning items and encrypted reasoning content.
- Pushes more provider-specific reasoning-item round-tripping into `any-llm`.

## Takeaway

Replay-only is viable, but it is not clearly the simpler implementation. If the goal is the smallest and most OpenAI-native first cut, the current `previous_response_id` path is still the better default.
