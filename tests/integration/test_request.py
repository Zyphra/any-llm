from typing import Any

import httpx
import pytest
from google.auth.exceptions import DefaultCredentialsError
from openai import APIConnectionError
from openresponses_types import ResponseResource

from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import MissingApiKeyError
from tests.constants import EXPECTED_PROVIDERS, LOCAL_PROVIDERS

REQUEST_PROVIDERS = {
    LLMProvider.ANTHROPIC,
    LLMProvider.GEMINI,
    LLMProvider.OPENAI,
    LLMProvider.VERTEXAI,
    LLMProvider.VERTEXAIANTHROPIC,
}


def _request_tool() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "add_numbers",
                "description": "Add two integers and return the result.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                },
            },
        }
    ]


@pytest.mark.asyncio
async def test_request_multi_turn_tool_loop(
    provider: LLMProvider,
    provider_reasoning_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    if provider not in REQUEST_PROVIDERS:
        pytest.skip(f"{provider.value} is outside experimental arequest scope")

    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
        model_id = provider_reasoning_model_map[provider]
        initial_input = [
            {
                "type": "message",
                "role": "user",
                "content": "Use the add_numbers tool to add 7 and 5. Do not answer without the tool.",
            }
        ]
        first = await llm.arequest(
            model_id,
            input_data=initial_input,
            tools=_request_tool(),
            tool_choice="required",
            reasoning={"effort": "medium"},
        )
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except DefaultCredentialsError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} credentials not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS and provider not in EXPECTED_PROVIDERS:
            pytest.skip("Local Model host is not set up, skipping")
        raise

    assert isinstance(first, ResponseResource)
    function_call = next((item for item in first.output if item.type == "function_call"), None)
    assert function_call is not None

    follow_up_input = [
        *initial_input,
        *[item.model_dump(mode="json", by_alias=True, exclude_none=True) for item in first.output],
        {
            "type": "function_call_output",
            "call_id": function_call.call_id,
            "output": "12",
        },
    ]
    second = await llm.arequest(
        model_id,
        input_data=follow_up_input,
        tools=_request_tool(),
        reasoning={"effort": "medium"},
    )

    assert isinstance(second, ResponseResource)
    assert any(item.type == "message" for item in second.output)
