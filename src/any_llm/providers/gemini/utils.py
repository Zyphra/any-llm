import base64
import binascii
import json
import mimetypes
from time import time
from typing import Any, Literal, cast

from google.genai import types
from google.genai.pagers import Pager
from openresponses_types import Usage
from openresponses_types.types import (
    AssistantMessageItemParam,
    DeveloperMessageItemParam,
    FunctionCallItemParam,
    InputFileContentParam,
    InputImageContentParamAutoParam,
    InputTextContentParam,
    InputTokensDetails,
    OutputTokensDetails,
    ReasoningBody,
    SpecificToolChoiceParam,
    SystemMessageItemParam,
    ToolChoiceParam,
    ToolChoiceValueEnum,
    UserMessageItemParam,
)

from any_llm.exceptions import InvalidRequestError
from any_llm.logging import logger
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageFunctionToolCall,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
    ChunkChoice,
    CompletionUsage,
    CreateEmbeddingResponse,
    Embedding,
    PromptTokensDetails,
    Reasoning,
)
from any_llm.types.model import Model
from any_llm.types.request import RequestInput, RequestParams, RequestResponse
from any_llm.utils.openresponses_compat import RequestContentLike, request_content_parts, request_content_to_text
from any_llm.utils.request_output import (
    OutputItem,
    finalize_request_response,
    make_function_call_item,
    make_reasoning_item,
    make_response_resource,
    make_text_message,
)
from any_llm.utils.request_state import GEMINI, decode_provider_state, encode_provider_state

_INLINE_SIZE_LIMIT = 20 * 1024 * 1024


def _convert_tool_spec(tools: list[dict[str, Any] | Any]) -> list[types.Tool]:
    converted_tools = []
    function_declarations = []

    for tool in tools:
        if isinstance(tool, types.Tool):
            converted_tools.append(tool)
            continue

        if tool.get("type") != "function":
            continue

        function = tool["function"]
        params: dict[str, Any] = function.get("parameters") or {}
        properties: dict[str, dict[str, Any]] = {}
        for param_name, param_info in (params.get("properties") or {}).items():
            prop: dict[str, Any] = {
                "type": param_info.get("type", "string"),
                "description": param_info.get("description", ""),
            }
            if "enum" in param_info:
                prop["enum"] = param_info["enum"]
            if "items" in param_info:
                prop["items"] = param_info["items"]
            if prop.get("type") == "array" and "items" not in prop:
                prop["items"] = {"type": "string"}
            properties[param_name] = prop

        parameters_dict = {
            "type": "object",
            "properties": properties,
            "required": params.get("required", []),
        }

        function_declarations.append(
            types.FunctionDeclaration(
                name=function["name"],
                description=function.get("description", ""),
                parameters=types.Schema(**parameters_dict),
            )
        )
    if function_declarations:
        converted_tools.append(types.Tool(function_declarations=function_declarations))
    return converted_tools


def _convert_tool_choice(tool_choice: str | dict[str, object]) -> types.ToolConfig:
    if isinstance(tool_choice, str):
        tool_choice_to_mode = {
            "required": types.FunctionCallingConfigMode.ANY,
            "auto": types.FunctionCallingConfigMode.AUTO,
        }
        return types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode=tool_choice_to_mode[tool_choice])
        )

    if tool_choice.get("type") == "function":
        function_choice = tool_choice.get("function")
        if isinstance(function_choice, dict) and isinstance(function_choice.get("name"), str):
            return types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.ANY,
                    allowed_function_names=[function_choice["name"]],
                )
            )

    msg = f"Unsupported Gemini tool_choice format: {tool_choice}"
    raise ValueError(msg)


def request_tool_choice_to_completion_tool_choice(tool_choice: ToolChoiceParam | None) -> str | dict[str, object] | None:
    if tool_choice is None:
        return None

    root_choice = tool_choice.root
    if isinstance(root_choice, ToolChoiceValueEnum):
        return root_choice.value
    if isinstance(root_choice, SpecificToolChoiceParam):
        return {"type": "function", "function": {"name": root_choice.root.name}}

    msg = f"Unsupported Gemini arequest tool_choice format: {tool_choice.model_dump(mode='json')}"
    raise ValueError(msg)


def _parse_data_uri(data_uri: str, field_name: str, provider_name: str) -> tuple[str, bytes]:
    if not data_uri.startswith("data:"):
        msg = f"{field_name} must be a data URI"
        raise InvalidRequestError(msg, provider_name=provider_name)
    if "base64," not in data_uri:
        msg = f"{field_name} must be a base64-encoded data URI"
        raise InvalidRequestError(msg, provider_name=provider_name)

    mime_part = data_uri[5:]
    semi_idx = mime_part.find(";")
    mime_type = mime_part[:semi_idx] if semi_idx != -1 else mime_part
    if not mime_type:
        msg = f"{field_name} is missing a MIME type"
        raise InvalidRequestError(msg, provider_name=provider_name)

    encoded_data = data_uri.split("base64,", 1)[1]
    if not encoded_data:
        msg = f"{field_name} is missing base64 data"
        raise InvalidRequestError(msg, provider_name=provider_name)

    try:
        raw_data = base64.b64decode(encoded_data, validate=True)
    except binascii.Error as exc:
        msg = f"{field_name} contains invalid base64 data"
        raise InvalidRequestError(msg, exc, provider_name) from exc
    return mime_type, raw_data


def _validate_inline_size(raw_data: bytes, field_name: str, provider_name: str) -> None:
    if len(raw_data) > _INLINE_SIZE_LIMIT:
        msg = f"{field_name} exceeds the 20 MB inline upload limit for {provider_name} ({len(raw_data)} bytes)"
        raise InvalidRequestError(msg, provider_name=provider_name)


def _convert_image_url_to_part(block: dict[str, Any], provider_name: str) -> types.Part:
    url = block.get("image_url", {}).get("url")
    if not isinstance(url, str) or not url:
        msg = "image_url.url is required for image content"
        raise InvalidRequestError(msg, provider_name=provider_name)

    if url.startswith("data:"):
        mime_type, raw_data = _parse_data_uri(url, "image_url.url", provider_name)
        _validate_inline_size(raw_data, "image_url.url", provider_name)
        return types.Part.from_bytes(data=raw_data, mime_type=mime_type)

    guessed_type, _ = mimetypes.guess_type(url)
    return types.Part.from_uri(file_uri=url, mime_type=guessed_type or "image/jpeg")


def _convert_file_to_part(block: dict[str, Any], provider_name: str) -> types.Part:
    file_data = block.get("file", {}).get("file_data")
    if not isinstance(file_data, str) or not file_data:
        msg = "file.file_data is required for file content"
        raise InvalidRequestError(msg, provider_name=provider_name)

    if file_data.startswith("data:"):
        mime_type, raw_data = _parse_data_uri(file_data, "file.file_data", provider_name)
        _validate_inline_size(raw_data, "file.file_data", provider_name)
        return types.Part.from_bytes(data=raw_data, mime_type=mime_type)

    guessed_type, _ = mimetypes.guess_type(file_data)
    return types.Part.from_uri(file_uri=file_data, mime_type=guessed_type or "application/octet-stream")


def _convert_messages(
    messages: list[dict[str, Any]], provider_name: str = "gemini"
) -> tuple[list[types.Content], str | None]:
    """Convert messages to Google GenAI format."""
    formatted_messages = []
    system_instruction = None

    for message in messages:
        if message["role"] == "system":
            if system_instruction is None:
                system_instruction = message["content"]
            else:
                system_instruction += f"\n{message['content']}"
        elif message["role"] == "user":
            if isinstance(message["content"], str):
                parts = [types.Part.from_text(text=message["content"])]
            else:
                parts = []
                for content in message["content"]:
                    if content["type"] == "text":
                        parts.append(types.Part.from_text(text=content["text"]))
                    elif content["type"] == "image_url":
                        parts.append(_convert_image_url_to_part(content, provider_name))
                    elif content["type"] == "file":
                        parts.append(_convert_file_to_part(content, provider_name))
                    else:
                        logger.debug("Skipping unsupported Gemini content block type: %s", content.get("type"))
            formatted_messages.append(types.Content(role="user", parts=parts))
        elif message["role"] == "assistant":
            if message.get("tool_calls"):
                parts = []
                for i, tool_call in enumerate(message["tool_calls"]):
                    function_call = tool_call["function"]
                    args = json.loads(function_call["arguments"]) if function_call["arguments"] else {}

                    # Extract thought_signature if present (OpenAI compatibility format)
                    # SDK accepts base64 string or bytes
                    thought_signature = None
                    if extra_content := tool_call.get("extra_content"):
                        if google_extra := extra_content.get("google"):
                            thought_signature = google_extra.get("thought_signature")

                    # For the first function call in parallel calls, if no thought_signature is present,
                    # use the skip validator sentinel per Google's documentation:
                    # https://ai.google.dev/gemini-api/docs/thought-signatures#faqs
                    if i == 0 and thought_signature is None:
                        thought_signature = "skip_thought_signature_validator"

                    parts.append(
                        types.Part(
                            function_call=types.FunctionCall(name=function_call["name"], args=args),
                            thought_signature=thought_signature,
                        )
                    )
            else:
                parts = [types.Part.from_text(text=message["content"])]

            formatted_messages.append(types.Content(role="model", parts=parts))
        elif message["role"] == "tool":
            try:
                content_json = json.loads(message["content"])
                part = types.Part.from_function_response(
                    name=message.get("name", "unknown"), response=_normalize_tool_response(content_json)
                )
                formatted_messages.append(types.Content(role="function", parts=[part]))
            except json.JSONDecodeError:
                part = types.Part.from_function_response(
                    name=message.get("name", "unknown"), response={"result": message["content"]}
                )
                formatted_messages.append(types.Content(role="function", parts=[part]))

    return formatted_messages, system_instruction


def _normalize_tool_response(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    return {"result": response}


def _extract_usage_dict(response: types.GenerateContentResponse) -> dict[str, Any]:
    """Extract usage from a Gemini response as a dict.

    Gemini's ``prompt_token_count`` already includes cached tokens
    (``cached_content_token_count`` is a subset).

    Reference: https://ai.google.dev/gemini-api/docs/caching
    """
    metadata = response.usage_metadata
    if metadata is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    usage: dict[str, Any] = {
        "prompt_tokens": metadata.prompt_token_count or 0,
        "completion_tokens": metadata.candidates_token_count or 0,
        "total_tokens": metadata.total_token_count or 0,
    }
    if metadata.cached_content_token_count:
        usage["prompt_tokens_details"] = PromptTokensDetails(cached_tokens=metadata.cached_content_token_count)
    return usage


def _convert_response_to_response_dict(response: types.GenerateContentResponse) -> dict[str, Any]:
    response_dict = {
        "id": "google_genai_response",
        "model": "google/genai",
        "created": 0,
        "usage": _extract_usage_dict(response),
    }

    choices: list[dict[str, Any]] = []
    if (
        response.candidates
        and len(response.candidates) > 0
        and response.candidates[0].content
        and response.candidates[0].content.parts
        and len(response.candidates[0].content.parts) > 0
    ):
        reasoning = None
        tool_calls_list: list[dict[str, Any]] = []
        text_content = None

        for part in response.candidates[0].content.parts:
            if getattr(part, "thought", None):
                reasoning = part.text
            elif function_call := getattr(part, "function_call", None):
                args_dict = {}
                if args := getattr(function_call, "args", None):
                    for key, value in args.items():
                        args_dict[key] = value

                tool_call_dict: dict[str, Any] = {
                    "id": f"call_{hash(function_call.name)}_{len(tool_calls_list)}",
                    "function": {
                        "name": function_call.name,
                        "arguments": json.dumps(args_dict),
                    },
                    "type": "function",
                }

                # Include thought_signature if present (OpenAI compatibility format)
                thought_signature = getattr(part, "thought_signature", None)
                if thought_signature is not None and isinstance(thought_signature, bytes):
                    tool_call_dict["extra_content"] = {
                        "google": {"thought_signature": base64.b64encode(thought_signature).decode("utf-8")}
                    }

                tool_calls_list.append(tool_call_dict)
            elif getattr(part, "text", None):
                text_content = part.text

        if tool_calls_list:
            choices.append(
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning": reasoning,
                        "tool_calls": tool_calls_list,
                    },
                    "finish_reason": "tool_calls",
                    "index": 0,
                }
            )
        elif text_content:
            choices.append(
                {
                    "message": {
                        "role": "assistant",
                        "content": text_content,
                        "reasoning": reasoning,
                        "tool_calls": None,
                    },
                    "finish_reason": "stop",
                    "index": 0,
                }
            )

    response_dict["choices"] = choices

    return response_dict


def _create_openai_embedding_response_from_google(
    model: str, result: types.EmbedContentResponse
) -> CreateEmbeddingResponse:
    """Convert a Google embedding response to an OpenAI-compatible format."""

    data = [
        Embedding(
            embedding=embedding.values,
            index=i,
            object="embedding",
        )
        for i, embedding in enumerate(result.embeddings or [])
        if embedding.values
    ]

    usage = Usage(prompt_tokens=0, total_tokens=0)

    return CreateEmbeddingResponse(
        data=data,
        model=model,
        object="list",
        usage=usage,
    )


def _create_openai_chunk_from_google_chunk(
    response: types.GenerateContentResponse,
) -> ChatCompletionChunk:
    """Convert a Google GenerateContentResponse to an OpenAI ChatCompletionChunk."""

    assert response.candidates
    candidate = response.candidates[0]
    assert candidate.content
    assert candidate.content.parts

    content = ""
    reasoning_content = ""
    tool_calls_list: list[ChoiceDeltaToolCall] = []

    for part in candidate.content.parts:
        if part.thought:
            reasoning_content += part.text or ""
        elif function_call := part.function_call:
            args_dict = {}
            if args := function_call.args:
                for key, value in args.items():
                    args_dict[key] = value

            tool_calls_list.append(
                ChoiceDeltaToolCall(
                    index=len(tool_calls_list),
                    id=f"call_{hash(function_call.name)}_{len(tool_calls_list)}",
                    type="function",
                    function=ChoiceDeltaToolCallFunction(
                        name=function_call.name,
                        arguments=json.dumps(args_dict),
                    ),
                )
            )
        elif part.text:
            content += part.text

    # Determine finish_reason based on what we found
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "function_call"] | None = None
    if tool_calls_list:
        finish_reason = "tool_calls"
    elif candidate.finish_reason and candidate.finish_reason.value == "STOP":
        finish_reason = "stop"

    delta = ChoiceDelta(
        content=content or None,
        role="assistant",
        reasoning=Reasoning(content=reasoning_content) if reasoning_content else None,
        tool_calls=tool_calls_list or None,
    )

    choice = ChunkChoice(
        index=0,
        delta=delta,
        finish_reason=finish_reason,
    )

    usage = None
    if response.usage_metadata:
        cached_tokens = response.usage_metadata.cached_content_token_count
        usage = CompletionUsage(
            prompt_tokens=response.usage_metadata.prompt_token_count or 0,
            completion_tokens=response.usage_metadata.candidates_token_count or 0,
            total_tokens=response.usage_metadata.total_token_count or 0,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=cached_tokens) if cached_tokens else None,
        )

    return ChatCompletionChunk(
        id=f"chatcmpl-{time()}",
        choices=[choice],
        created=int(time()),
        model=str(response.model_version),
        object="chat.completion.chunk",
        usage=usage,
    )


def _convert_models_list(models_list: Pager[types.Model]) -> list[Model]:
    return [Model(id=model.name or "Unknown", object="model", created=0, owned_by="google") for model in models_list]


# --- arequest support -------------------------------------------------------
# Gemini's ``thought_signature`` lives on individual function-call parts, not on
# a separate reasoning block. We bridge it through ``acompletion`` (which already
# carries the signature in ``extra_content.google.thought_signature`` per the
# OpenAI-compat surface) and stamp/decode it on the openresponses
# ``encrypted_content`` field for round-tripping.


def _user_content_to_completion_content(content: RequestContentLike) -> str | list[dict[str, object]] | None:
    """Translate OpenResponses user content parts to Gemini-compatible chat content."""
    parts = request_content_parts(content)
    if not parts:
        text = request_content_to_text(content)
        return text or None

    blocks: list[dict[str, object]] = []
    for part in parts:
        if isinstance(part, InputTextContentParam):
            blocks.append({"type": "text", "text": part.text})
        elif isinstance(part, InputImageContentParamAutoParam):
            if part.image_url is not None:
                blocks.append({"type": "image_url", "image_url": part.image_url.root})
        elif isinstance(part, InputFileContentParam):
            if part.file_data is not None:
                blocks.append({"type": "file", "file": {"file_data": part.file_data.root}})
            elif part.file_url is not None:
                blocks.append({"type": "file", "file": {"file_data": part.file_url}})
    if not blocks:
        return None
    if len(blocks) == 1 and blocks[0].get("type") == "text":
        return cast("str", blocks[0].get("text", ""))
    return blocks


def request_input_to_completion_messages(input_data: RequestInput) -> list[dict[str, object]]:
    """Convert OpenResponses input items to chat-completion messages for Gemini.

    Reasoning items contribute their decoded ``thought_signature`` to the next
    ``function_call``'s ``extra_content``. Any reasoning state tagged with a
    different provider (e.g. after a fallback) is silently ignored.
    """
    system_parts: list[str] = []
    messages: list[dict[str, object]] = []
    pending_reasoning_text: str | None = None
    pending_signature: str | None = None
    pending_tool_calls: list[dict[str, object]] = []
    tool_name_by_call_id: dict[str, str] = {}

    def flush_tool_calls() -> None:
        nonlocal pending_reasoning_text, pending_signature, pending_tool_calls
        if not pending_tool_calls:
            pending_reasoning_text = None
            pending_signature = None
            return
        message: dict[str, object] = {"role": "assistant", "content": None, "tool_calls": pending_tool_calls}
        if pending_reasoning_text:
            message["reasoning"] = pending_reasoning_text
        messages.append(message)
        pending_reasoning_text = None
        pending_signature = None
        pending_tool_calls = []

    for item in input_data:
        if isinstance(item, (SystemMessageItemParam, DeveloperMessageItemParam)):
            text = request_content_to_text(item.content)
            if text:
                system_parts.append(text)
        elif isinstance(item, UserMessageItemParam):
            flush_tool_calls()
            content = _user_content_to_completion_content(item.content)
            if content is not None:
                messages.append({"role": "user", "content": content})
        elif isinstance(item, AssistantMessageItemParam):
            flush_tool_calls()
            messages.append({"role": "assistant", "content": request_content_to_text(item.content) or None})
        elif isinstance(item, ReasoningBody):
            pending_reasoning_text = request_content_to_text(item.content)
            pending_signature = decode_provider_state(item.encrypted_content, GEMINI)
        elif isinstance(item, FunctionCallItemParam):
            tool_name_by_call_id[item.call_id] = item.name
            tool_call: dict[str, object] = {
                "id": item.call_id,
                "type": "function",
                "function": {"name": item.name, "arguments": item.arguments},
            }
            extra_content: dict[str, object] | None = None
            if pending_signature is not None:
                extra_content = {"google": {"thought_signature": pending_signature}}
                pending_signature = None
            elif not pending_tool_calls:
                # Per https://ai.google.dev/gemini-api/docs/thought-signatures, the first
                # function call in a turn needs the skip-validator sentinel when there is
                # no signature to forward.
                extra_content = {"google": {"thought_signature": "skip_thought_signature_validator"}}
            if extra_content is not None:
                tool_call["extra_content"] = extra_content
            pending_tool_calls.append(tool_call)
        else:
            flush_tool_calls()
            tool_message: dict[str, object] = {
                "role": "tool",
                "tool_call_id": item.call_id,
                "content": request_content_to_text(item.output),
            }
            tool_name = tool_name_by_call_id.get(item.call_id)
            if tool_name is not None:
                tool_message["name"] = tool_name
            messages.append(tool_message)

    flush_tool_calls()
    if system_parts:
        return [{"role": "system", "content": "\n".join(system_parts)}, *messages]
    return messages


def _completion_to_request_response(completion: ChatCompletion, *, params: RequestParams) -> RequestResponse:
    """Build a RequestResponse from a Gemini ChatCompletion, preserving thought signatures."""
    output: list[OutputItem] = []
    if completion.choices:
        message = completion.choices[0].message
        function_tool_calls = [
            tool_call
            for tool_call in (message.tool_calls or [])
            if isinstance(tool_call, ChatCompletionMessageFunctionToolCall)
        ]
        signature: str | None = None
        for tool_call in function_tool_calls:
            google_extra = tool_call.extra_content.get("google") if tool_call.extra_content else None
            if not isinstance(google_extra, dict):
                continue
            thought_signature = google_extra.get("thought_signature")
            if isinstance(thought_signature, str):
                signature = thought_signature
                break

        encrypted = encode_provider_state(GEMINI, signature) if signature else None
        if message.reasoning is not None or encrypted is not None:
            output.append(
                make_reasoning_item(
                    message.reasoning.content if message.reasoning else None, encrypted_content=encrypted
                )
            )
        if function_tool_calls:
            for tool_call in function_tool_calls:
                output.append(
                    make_function_call_item(
                        call_id=tool_call.id, name=tool_call.function.name, arguments=tool_call.function.arguments
                    )
                )
        if message.content:
            output.append(make_text_message(message.content))

    if completion.usage is not None:
        input_tokens = completion.usage.prompt_tokens or 0
        output_tokens = completion.usage.completion_tokens or 0
        usage = Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            input_tokens_details=InputTokensDetails(
                cached_tokens=d.cached_tokens or 0 if (d := completion.usage.prompt_tokens_details) else 0
            ),
            output_tokens_details=OutputTokensDetails(
                reasoning_tokens=d.reasoning_tokens or 0 if (d := completion.usage.completion_tokens_details) else 0
            ),
        )
    else:
        usage = Usage(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            input_tokens_details=InputTokensDetails(cached_tokens=0),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        )

    resource = make_response_resource(
        model=completion.model or params.model,
        output=output,
        tools=params.tools,
        tool_choice=params.tool_choice,
        temperature=params.temperature,
        top_p=params.top_p,
        max_output_tokens=params.max_output_tokens,
        reasoning=params.reasoning,
        usage=usage,
        instructions=params.instructions,
        metadata=params.metadata,
    )
    return finalize_request_response(resource, response_format=params.response_format)
