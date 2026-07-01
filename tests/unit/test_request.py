"""Tests for request()/arequest() SDK API."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from openresponses_types import ResponseResource
from openresponses_types.types import (
    AssistantMessageItemParam,
    JsonObjectResponseFormat,
    JsonSchemaResponseFormatParam,
    Reasoning,
    ReasoningBody,
    ReasoningEffortEnum,
    ReasoningTextContent,
    UserMessageItemParam,
)
from pydantic import BaseModel

from any_llm.any_llm import AnyLLM
from any_llm.api import arequest
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CreateEmbeddingResponse
from any_llm.types.model import Model
from any_llm.types.request import RequestInput, RequestParams
from any_llm.utils.request_output import finalize_request_response, make_response_resource, make_text_message
from any_llm.utils.request_state import CLAUDE, GEMINI, OPENAI, decode_provider_state, encode_provider_state


def test_request_params_model_creation() -> None:
    input_data: RequestInput = [UserMessageItemParam.model_validate({"type": "message", "role": "user", "content": "Hello"})]
    params = RequestParams(
        model="claude-sonnet-4-6",
        input=input_data,
        stream=False,
        response_format={"type": "json_object"},
        stop=["END"],
        presence_penalty=0.1,
        frequency_penalty=0.2,
        seed=123,
        parallel_tool_calls=True,
        reasoning=Reasoning(effort=ReasoningEffortEnum.medium, summary=None),
    )
    assert params.model == "claude-sonnet-4-6"
    assert len(params.input) == 1
    assert params.input[0].type == "message"
    assert params.input[0].role == "user"
    assert isinstance(params.response_format, JsonObjectResponseFormat)
    assert params.stop == ["END"]
    assert params.parallel_tool_calls is True
    assert isinstance(params.reasoning, Reasoning)
    assert params.reasoning.effort == "medium"
    assert params.reasoning.summary is None


def test_request_params_rejects_extra_fields() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RequestParams(
            model="gpt-5-nano",
            input=[UserMessageItemParam.model_validate({"type": "message", "role": "user", "content": "Hello"})],
            unknown_field="value",  # type: ignore[call-arg]
        )


def test_request_params_rejects_legacy_tool_choice_shape() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RequestParams(
            model="gpt-5-nano",
            input=[UserMessageItemParam.model_validate({"type": "message", "role": "user", "content": "Hello"})],
            tool_choice={"type": "function", "function": {"name": "lookup"}},
        )


def test_request_params_accepts_json_schema_wrapper_shape() -> None:
    params = RequestParams(
        model="gpt-5-nano",
        input=[UserMessageItemParam.model_validate({"type": "message", "role": "user", "content": "Hello"})],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "Foo", "schema": {"type": "object"}},
        },
    )

    assert isinstance(params.response_format, JsonSchemaResponseFormatParam)


def test_request_exported_from_init() -> None:
    import any_llm

    assert hasattr(any_llm, "request")
    assert hasattr(any_llm, "arequest")
    assert "request" in any_llm.__all__
    assert "arequest" in any_llm.__all__


def test_request_input_is_typed() -> None:
    input_data: RequestInput = [
        UserMessageItemParam.model_validate(
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}
        ),
        ReasoningBody(
            type="reasoning",
            id="rs_1",
            summary=[],
            content=[ReasoningTextContent(type="reasoning_text", text="thinking")],
            encrypted_content="anthropic:sig",
        ),
        AssistantMessageItemParam.model_validate(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "done"}],
            }
        ),
    ]

    assert isinstance(input_data, list)
    assert [type(item).__name__ for item in input_data] == [
        "UserMessageItemParam",
        "ReasoningBody",
        "AssistantMessageItemParam",
    ]


def test_provider_state_round_trips_within_family() -> None:
    blob = encode_provider_state(CLAUDE, "sig_456")
    assert decode_provider_state(blob, CLAUDE) == "sig_456"


@pytest.mark.parametrize(
    ("encoder", "decoder"),
    [(CLAUDE, GEMINI), (CLAUDE, OPENAI), (GEMINI, OPENAI), (GEMINI, CLAUDE)],
)
def test_provider_state_drops_on_family_mismatch(encoder: str, decoder: str) -> None:
    blob = encode_provider_state(encoder, "state")
    assert decode_provider_state(blob, decoder) is None


def test_provider_state_handles_missing_or_malformed_blobs() -> None:
    assert decode_provider_state(None, CLAUDE) is None
    assert decode_provider_state("", CLAUDE) is None
    assert decode_provider_state("no-colon-here", CLAUDE) is None


@pytest.mark.asyncio
async def test_arequest_parameter_capture() -> None:
    mock_provider = Mock()
    mock_provider.arequest = AsyncMock(return_value=Mock())
    input_data: RequestInput = [UserMessageItemParam.model_validate({"type": "message", "role": "user", "content": "Hello"})]

    with patch("any_llm.any_llm.AnyLLM.create") as mock_create:
        mock_create.return_value = mock_provider

        await arequest(
            model="anthropic:claude-sonnet-4-6",
            input_data=input_data,
            tools=[{"type": "function", "function": {"name": "lookup"}}],
            temperature=0.5,
            reasoning={"effort": "medium"},
            api_key="anthropic-key-123",
            api_base="https://api.anthropic.com",
            custom_param="custom_value",
        )

        mock_create.assert_called_once()
        mock_provider.arequest.assert_called_once()
        call_args = mock_provider.arequest.call_args
        assert call_args.kwargs["model"] == "claude-sonnet-4-6"
        assert call_args.kwargs["input_data"] == input_data
        assert call_args.kwargs["temperature"] == 0.5
        assert call_args.kwargs["reasoning"] == {"effort": "medium"}
        assert call_args.kwargs["custom_param"] == "custom_value"


class _StructuredResponse(BaseModel):
    city: str
    value: int


@dataclass
class _DataclassStructuredResponse:
    city: str
    value: int


def test_finalize_request_response_parses_json_schema_output() -> None:
    response = make_response_resource(
        model="test-model",
        output=[make_text_message('{"city":"Paris","value":7}')],
    )

    result = finalize_request_response(
        response,
        response_format=JsonSchemaResponseFormatParam(
            type="json_schema",
            name="StructuredResponse",
            schema={
                "type": "object",
                "properties": {"city": {"type": "string"}, "value": {"type": "integer"}},
                "required": ["city", "value"],
            },
        ),
    )

    assert result.structured_output == {"city": "Paris", "value": 7}


def test_finalize_request_response_parses_typed_output() -> None:
    response = make_response_resource(
        model="test-model",
        output=[make_text_message('{"city":"Paris","value":7}')],
    )

    result = finalize_request_response(response, response_format=_StructuredResponse)

    assert isinstance(result.structured_output, _StructuredResponse)
    assert result.structured_output.city == "Paris"
    assert result.structured_output.value == 7


def test_finalize_request_response_parses_dataclass_output() -> None:
    response = make_response_resource(
        model="test-model",
        output=[make_text_message('{"city":"Paris","value":7}')],
    )

    result = finalize_request_response(response, response_format=_DataclassStructuredResponse)

    assert isinstance(result.structured_output, _DataclassStructuredResponse)
    assert result.structured_output.city == "Paris"
    assert result.structured_output.value == 7


def test_finalize_request_response_ignores_incomplete_message_text() -> None:
    response = ResponseResource.model_validate(
        {
            "id": "resp_test",
            "object": "response",
            "created_at": 0,
            "completed_at": 0,
            "status": "completed",
            "incomplete_details": None,
            "model": "test-model",
            "previous_response_id": None,
            "instructions": None,
            "output": [
                {
                    "type": "message",
                    "id": "msg_incomplete",
                    "status": "incomplete",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": '{"city":"Par', "annotations": [], "logprobs": []}
                    ],
                },
                {
                    "type": "message",
                    "id": "msg_complete",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": '{"city":"Paris","value":7}', "annotations": [], "logprobs": []}
                    ],
                },
            ],
            "error": None,
            "tools": [],
            "tool_choice": "auto",
            "truncation": "disabled",
            "parallel_tool_calls": False,
            "text": {"format": {"type": "text"}},
            "top_p": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "top_logprobs": 0,
            "temperature": 1.0,
            "reasoning": None,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
            "max_output_tokens": None,
            "max_tool_calls": None,
            "store": False,
            "background": False,
            "service_tier": "default",
            "metadata": {},
            "safety_identifier": None,
            "prompt_cache_key": None,
        }
    )

    result = finalize_request_response(response, response_format=_StructuredResponse)

    assert isinstance(result.structured_output, _StructuredResponse)
    assert result.structured_output.city == "Paris"
    assert result.structured_output.value == 7


class _DummyRequestProvider(AnyLLM):
    PROVIDER_NAME = "dummy"
    PROVIDER_DOCUMENTATION_URL = "https://example.com"
    ENV_API_KEY_NAME = "DUMMY_API_KEY"
    SUPPORTS_COMPLETION_STREAMING = False
    SUPPORTS_COMPLETION = False
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_COMPLETION_IMAGE = False
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_EMBEDDING = False
    SUPPORTS_RESPONSES = False
    SUPPORTS_LIST_MODELS = False
    SUPPORTS_BATCH = False
    SUPPORTS_REQUESTS = True

    @staticmethod
    def _convert_completion_params(params: Any, **kwargs: Any) -> dict[str, Any]:
        return {}

    @staticmethod
    def _convert_completion_response(response: Any) -> ChatCompletion:
        raise NotImplementedError

    @staticmethod
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        raise NotImplementedError

    @staticmethod
    def _convert_embedding_params(params: Any, **kwargs: Any) -> dict[str, Any]:
        return {}

    @staticmethod
    def _convert_embedding_response(response: Any) -> CreateEmbeddingResponse:
        raise NotImplementedError

    @staticmethod
    def _convert_list_models_response(response: Any) -> Sequence[Model]:
        return []

    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        self.client = None

    async def _arequest(self, params: RequestParams, **kwargs: Any):  # type: ignore[override]
        return Mock()


@pytest.mark.asyncio
async def test_provider_arequest_rejects_streaming() -> None:
    provider = _DummyRequestProvider(api_key="test-key")
    input_data: RequestInput = [UserMessageItemParam.model_validate({"type": "message", "role": "user", "content": "Hello"})]

    with pytest.raises(NotImplementedError, match="Streaming"):
        await provider.arequest(
            model="dummy-model",
            input_data=input_data,
            stream=True,
        )
