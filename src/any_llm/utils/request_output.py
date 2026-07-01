"""Builders for OpenResponses output items and ``RequestResponse`` finalization.

These helpers encapsulate the openresponses-required defaults (``annotations=[]``,
``logprobs=[]``, ``status="completed"``, etc.) so each provider's request
implementation stays focused on its own conversion logic. There is no
provider-specific code in this module.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, TypeAlias
from uuid import uuid4

from openresponses_types import ResponseResource
from openresponses_types.types import (
    FunctionCall,
    FunctionCallOutput,
    FunctionCallStatus,
    FunctionTool,
    JsonObjectResponseFormat,
    JsonSchemaResponseFormatParam,
    Message,
    MessageRole,
    MessageStatus,
    Object,
    OutputTextContent,
    Reasoning,
    ReasoningBody,
    ReasoningTextContent,
    TextField,
    TextResponseFormat,
    Tool,
    ToolChoiceParam,
    ToolChoiceValueEnum,
    TruncationEnum,
    Type31,
    Type34,
    Usage,
)

from any_llm.types.request import RequestResponse, RequestResponseFormatParam, RequestTool
from any_llm.utils.structured_output import is_structured_output_type, parse_json_content

if TYPE_CHECKING:
    from collections.abc import Iterable

OutputItem: TypeAlias = Message | FunctionCall | FunctionCallOutput | ReasoningBody


def make_text_message(text: str, *, item_id: str | None = None) -> Message:
    return Message(
        type="message",
        id=item_id or f"msg_{uuid4().hex}",
        status=MessageStatus.completed,
        role=MessageRole.assistant,
        content=[OutputTextContent(type="output_text", text=text, annotations=[], logprobs=[])],
    )


def make_reasoning_item(
    text: str | None, *, encrypted_content: str | None = None, item_id: str | None = None
) -> ReasoningBody:
    return ReasoningBody(
        type="reasoning",
        id=item_id or f"rs_{uuid4().hex}",
        content=[ReasoningTextContent(type="reasoning_text", text=text)] if text else None,
        summary=[],
        encrypted_content=encrypted_content,
    )


def make_function_call_item(*, call_id: str, name: str, arguments: str, item_id: str | None = None) -> FunctionCall:
    return FunctionCall(
        type="function_call",
        id=item_id or f"fc_{uuid4().hex}",
        call_id=call_id,
        name=name,
        arguments=arguments,
        status=FunctionCallStatus.completed,
    )


def _convert_tools(tools: list[RequestTool] | None) -> list[Tool]:
    converted: list[Tool] = []
    for tool in tools or []:
        function = tool["function"]
        parameters = function.get("parameters")
        converted.append(
            Tool(
                root=FunctionTool(
                    type=Type31.function,
                    name=str(function.get("name") or ""),
                    description=description if isinstance((description := function.get("description")), str) else None,
                    parameters=parameters if isinstance(parameters, dict) else None,
                    strict=strict if isinstance((strict := function.get("strict")), bool) else None,
                )
            )
        )
    return converted


def make_response_resource(
    *,
    model: str,
    output: Iterable[OutputItem],
    response_id: str | None = None,
    tools: list[RequestTool] | None = None,
    tool_choice: ToolChoiceParam | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_output_tokens: int | None = None,
    reasoning: Reasoning | None = None,
    usage: Usage | None = None,
    instructions: str | None = None,
    metadata: dict[str, str] | None = None,
) -> ResponseResource:
    """Build a ``ResponseResource`` populated with sensible openresponses defaults."""
    created_at = int(time.time())
    return ResponseResource(
        id=response_id or f"resp_{uuid4().hex}",
        object=Object.response,
        created_at=created_at,
        completed_at=created_at,
        status="completed",
        incomplete_details=None,
        model=model,
        previous_response_id=None,
        instructions=instructions,
        output=list(output),
        error=None,
        tools=_convert_tools(tools),
        tool_choice=(
            ToolChoiceValueEnum.auto
            if tool_choice is None
            else tool_choice.model_dump(mode="json", by_alias=True, exclude_none=True)
        ),
        truncation=TruncationEnum.disabled,
        parallel_tool_calls=False,
        text=TextField(format=TextResponseFormat(type=Type34.text)),
        top_p=top_p if top_p is not None else 1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        top_logprobs=0,
        temperature=temperature if temperature is not None else 1.0,
        reasoning=reasoning,
        usage=usage,
        max_output_tokens=max_output_tokens,
        max_tool_calls=None,
        store=False,
        background=False,
        service_tier="default",
        metadata=metadata or {},
        safety_identifier=None,
        prompt_cache_key=None,
    )


def finalize_request_response(
    resource: ResponseResource, *, response_format: RequestResponseFormatParam | None = None
) -> RequestResponse:
    """Wrap a ``ResponseResource`` in ``RequestResponse`` and parse structured output if requested."""
    structured_output: object | None = None
    text_parts = [
        content.text
        for item in resource.output
        if isinstance(item, Message)
        and item.role == MessageRole.assistant
        and item.status != MessageStatus.incomplete
        for content in item.content
        if isinstance(content, OutputTextContent)
    ]
    text = "".join(text_parts) if text_parts else None
    if text is not None and response_format is not None:
        if is_structured_output_type(response_format):
            structured_output = parse_json_content(response_format, text)
        elif isinstance(response_format, (JsonSchemaResponseFormatParam, JsonObjectResponseFormat)):
            try:
                structured_output = json.loads(text)
            except json.JSONDecodeError:
                structured_output = None
    return RequestResponse.model_validate(
        {**resource.model_dump(mode="json", by_alias=True, exclude_none=False), "structured_output": structured_output}
    )


__all__ = [
    "OutputItem",
    "finalize_request_response",
    "make_function_call_item",
    "make_reasoning_item",
    "make_response_resource",
    "make_text_message",
]
