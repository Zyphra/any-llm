from collections.abc import Sequence
from typing import Any

from typing_extensions import override

from any_llm.exceptions import ProviderError
from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.providers.openrouter.utils import _convert_models_list, build_reasoning_directive
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams
from any_llm.types.model import Model


def _raise_for_error_finish_reason(response: Any) -> None:
    payload = response if isinstance(response, dict) else response.model_dump(warnings=False)
    if any(choice.get("finish_reason") == "error" for choice in payload.get("choices") or []):
        msg = "OpenRouter request terminated with an error"
        raise ProviderError(msg, provider_name="openrouter")


class OpenrouterProvider(BaseOpenAIProvider):
    API_BASE = "https://openrouter.ai/api/v1"
    ENV_API_KEY_NAME = "OPENROUTER_API_KEY"
    ENV_API_BASE_NAME = "OPENROUTER_API_BASE"
    PROVIDER_NAME = "openrouter"
    PROVIDER_DOCUMENTATION_URL = "https://openrouter.ai/docs"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_EMBEDDING = True
    # OpenRouter does not expose a moderation endpoint.
    SUPPORTS_MODERATION = False

    @staticmethod
    @override
    def _convert_list_models_response(response: Any) -> Sequence[Model]:
        """Convert OpenRouter list models response to valid Model objects."""
        return _convert_models_list(response)

    @staticmethod
    @override
    def _convert_completion_response(response: Any) -> ChatCompletion:
        _raise_for_error_finish_reason(response)
        return BaseOpenAIProvider._convert_completion_response(response)

    @staticmethod
    @override
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        _raise_for_error_finish_reason(response)
        return BaseOpenAIProvider._convert_completion_chunk_response(response, **kwargs)

    @staticmethod
    @override
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert CompletionParams to kwargs for OpenRouter API, including reasoning directive."""
        max_tokens = kwargs.get("max_tokens", params.max_tokens)
        max_completion_tokens = kwargs.get("max_completion_tokens", params.max_completion_tokens)
        converted_params = BaseOpenAIProvider._convert_completion_params(params, **kwargs)

        if max_tokens is not None and max_completion_tokens is None:
            converted_params["max_tokens"] = converted_params.pop("max_completion_tokens")

        reasoning_directive = build_reasoning_directive(
            reasoning=kwargs.get("reasoning"),
            reasoning_effort=params.reasoning_effort,
        )

        if reasoning_directive is not None:
            converted_params.pop("reasoning_effort", None)
            extra_body = converted_params.get("extra_body", {}).copy()
            extra_body["reasoning"] = reasoning_directive
            converted_params["extra_body"] = extra_body

        return converted_params
