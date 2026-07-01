"""Types for the experimental ``request`` / ``arequest`` API.

The request API is intentionally a thin, stateful wrapper around the OpenResponses
schema: input items, response items, and the ``Reasoning`` config are reused
verbatim from ``openresponses_types``. The only addition is ``RequestResponse``,
which extends ``ResponseResource`` with a ``structured_output`` field so callers
can use it as a drop-in for ``acompletion``.
"""

from __future__ import annotations

from typing import Literal, TypeAlias, TypedDict, cast

from openresponses_types import ResponseResource
from openresponses_types.types import (
    AssistantMessageItemParam,
    DeveloperMessageItemParam,
    FunctionCallItemParam,
    FunctionCallOutputItemParam,
    JsonObjectResponseFormat,
    JsonSchemaResponseFormatParam,
    Reasoning,
    ReasoningBody,
    ReasoningParam,
    SystemMessageItemParam,
    ToolChoiceParam,
    UserMessageItemParam,
)
from pydantic import BaseModel, ConfigDict, field_validator

# Note: `ItemReferenceParam` is intentionally excluded: providers raise on it.
RequestInputItemParam: TypeAlias = (
    ReasoningBody
    | UserMessageItemParam
    | SystemMessageItemParam
    | DeveloperMessageItemParam
    | AssistantMessageItemParam
    | FunctionCallItemParam
    | FunctionCallOutputItemParam
)

RequestInput: TypeAlias = list[RequestInputItemParam]
RequestToolChoiceString: TypeAlias = Literal["none", "auto", "required"]
RequestToolChoiceParam: TypeAlias = ToolChoiceParam | RequestToolChoiceString
RequestResponseFormatParam: TypeAlias = JsonSchemaResponseFormatParam | JsonObjectResponseFormat | type
RequestReasoningParam: TypeAlias = Reasoning | ReasoningParam


class RequestFunctionSpec(TypedDict, total=False):
    name: str
    description: str
    parameters: dict[str, object]
    strict: bool


class RequestTool(TypedDict):
    type: Literal["function"]
    function: RequestFunctionSpec


class RequestResponse(ResponseResource):
    """OpenResponses ``ResponseResource`` plus a parsed ``structured_output`` field."""

    model_config = ConfigDict(extra="forbid")

    structured_output: object | None = None


class RequestParams(BaseModel):
    """Normalized parameters for the experimental request API."""

    model_config = ConfigDict(extra="forbid")

    model: str
    input: RequestInput
    tools: list[RequestTool] | None = None
    tool_choice: ToolChoiceParam | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool | None = None
    response_format: RequestResponseFormatParam | None = None
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    parallel_tool_calls: bool | None = None
    instructions: str | None = None
    reasoning: Reasoning | None = None
    metadata: dict[str, str] | None = None

    @field_validator("reasoning", mode="before")
    @classmethod
    def _fill_reasoning_summary(cls, value: object) -> object:
        # openresponses' Reasoning requires both effort and summary
        if isinstance(value, ReasoningParam):
            value = value.model_dump(mode="json", by_alias=True, exclude_none=True)
        if isinstance(value, dict):
            value_dict = cast("dict[str, object]", value).copy()
            value_dict.setdefault("summary", None)
            return value_dict
        return value


RequestParams.model_rebuild()


__all__ = [
    "RequestInput",
    "RequestInputItemParam",
    "RequestParams",
    "RequestReasoningParam",
    "RequestResponse",
    "RequestResponseFormatParam",
    "RequestTool",
    "RequestToolChoiceParam",
]
