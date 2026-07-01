"""OpenAI Provider Utilities."""

from typing import Any

from openai.types.chat.chat_completion import ChatCompletion as OpenAIChatCompletion
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion as OpenAIParsedChatCompletion
from openresponses_types import ResponseResource
from openresponses_types.types import ReasoningBody

from any_llm.constants import REASONING_FIELD_NAMES
from any_llm.exceptions import ProviderError
from any_llm.logging import logger
from any_llm.types.completion import ChatCompletion, ParsedChatCompletion
from any_llm.types.request import RequestInput, RequestInputItemParam, RequestParams, RequestResponse
from any_llm.utils.openresponses_compat import dump_json_model
from any_llm.utils.request_output import finalize_request_response, make_reasoning_item
from any_llm.utils.request_state import OPENAI, decode_provider_state, encode_provider_state

REASONING_FAMILY = OPENAI


def _normalize_reasoning_on_message(message_dict: dict[str, Any]) -> None:
    """Mutate a message dict to move provider-specific reasoning fields to our Reasoning type."""
    if isinstance(message_dict.get("reasoning"), dict) and "content" in message_dict["reasoning"]:
        return

    possible_fields = REASONING_FIELD_NAMES
    value: Any | None = None
    for field_name in possible_fields:
        if field_name in message_dict and message_dict[field_name] is not None:
            value = message_dict[field_name]
            break

    if value is None and isinstance(message_dict.get("reasoning"), str):
        value = message_dict["reasoning"]

    if value is not None:
        message_dict["reasoning"] = {"content": str(value)}


def _normalize_openai_dict_response(response_dict: dict[str, Any]) -> dict[str, Any]:
    """Return a dict where non-standard reasoning fields are normalized.

    - For non-streaming: response.choices[*].message
    - For streaming: chunk.choices[*].delta
    """
    choices = response_dict.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            message = choice.get("message") if isinstance(choice, dict) else None
            if isinstance(message, dict):
                _normalize_reasoning_on_message(message)

            delta = choice.get("delta") if isinstance(choice, dict) else None
            if isinstance(delta, dict):
                _normalize_reasoning_on_message(delta)

    return response_dict


def _convert_chat_completion(response: OpenAIChatCompletion) -> ChatCompletion:
    if response.object != "chat.completion":
        # Force setting this here because it's a requirement Literal in the OpenAI API, but the Llama API has
        # a typo where they set it to "chat.completions". I filed a ticket with them to fix it. No harm in setting it here
        # Because this is the only accepted value anyways.
        logger.warning(
            "API returned an unexpected object type: %s. Setting to 'chat.completion'.",
            response.object,
        )
        response.object = "chat.completion"

    # Detect completely empty responses (all required fields None).
    # Some providers return 200 OK with an empty body on malformed input
    # instead of a proper error. Fail fast with a clear message.
    if response.id is None and response.choices is None and response.model is None:
        msg = (
            "Provider returned an empty response with no id, choices, or model. "
            "This usually means the provider failed to process the request."
        )
        raise ProviderError(msg)

    if not isinstance(response.created, int):
        # Sambanova returns a float instead of an int.
        logger.warning(
            "API returned an unexpected created type: %s. Setting to int.",
            type(response.created),
        )
        response.created = int(response.created)
    normalized = _normalize_openai_dict_response(response.model_dump())
    return ChatCompletion.model_validate(normalized)


def _convert_parsed_chat_completion(response: OpenAIParsedChatCompletion[Any]) -> ParsedChatCompletion[Any]:
    """Convert an OpenAI ParsedChatCompletion preserving the .parsed field on each choice."""
    base = _convert_chat_completion(response)
    parsed_completion: ParsedChatCompletion[Any] = ParsedChatCompletion.model_validate(base, from_attributes=True)
    for base_choice, parsed_choice in zip(response.choices, parsed_completion.choices, strict=True):
        parsed_choice.message.parsed = base_choice.message.parsed
    return parsed_completion


def _latest_response_state(items: list[RequestInputItemParam]):
    for index in range(len(items) - 1, -1, -1):
        item = items[index]
        if isinstance(item, ReasoningBody):
            state = decode_provider_state(item.encrypted_content, REASONING_FAMILY)
            if state is not None:
                return index, state

    return None, None


def split_request_input_for_openai(input_data: RequestInput) -> tuple[list[dict[str, object]], str | None]:
    """Strip cross-family reasoning items and extract any cached previous_response_id.

    Returns the OpenAI-ready list of input items and the latest
    ``previous_response_id`` we encoded in a prior turn (or ``None``).
    """
    last_state_index, previous_response_id = _latest_response_state(input_data)

    if last_state_index is None:
        # No prior state: forward everything except cross-family reasoning items.
        return [dump_json_model(item) for item in input_data if not isinstance(item, ReasoningBody)], None

    # OpenAI replays the prior turn server-side, so only forward what is new and drop the rest
    return [
        dump_json_model(item)
        for item in input_data[last_state_index + 1 :]
        if item.type == "function_call_output" or (item.type == "message" and item.role == "user")
    ], previous_response_id


def stamp_openai_response_state(resource: ResponseResource, params: RequestParams) -> RequestResponse:
    """Append a trailing reasoning item carrying the response id for next-turn continuity."""
    payload = resource.model_dump(mode="json", by_alias=True, exclude_none=False)
    output = [
        *resource.output,
        make_reasoning_item(text=None, encrypted_content=encode_provider_state(REASONING_FAMILY, resource.id)),
    ]
    payload["output"] = [dump_json_model(item) for item in output]
    return finalize_request_response(ResponseResource.model_validate(payload), response_format=params.response_format)
