import json
import os

import pytest

from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import MissingApiKeyError
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk

ZYPHRA_MODELS = [
    "deepseek-ai/DeepSeek-V3.2",
    "moonshotai/Kimi-K2.6",
    "zai-org/GLM-5.1-FP8",
    "zyphra/ZAYA1-8B",
]

# Models that populate the reasoning field. DeepSeek-V3.2 returns reasoning=null
# even with reasoning_effort set, so it's excluded from the streaming-reasoning check.
ZYPHRA_REASONING_MODELS = [
    "moonshotai/Kimi-K2.6",
    "zai-org/GLM-5.1-FP8",
    "zyphra/ZAYA1-8B",
]

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "country_code": {"type": "string"},
                "admin_area": {"type": "string"},
            },
            "required": ["city"],
        },
    },
}


@pytest.mark.parametrize("model_id", ZYPHRA_MODELS)
@pytest.mark.asyncio
async def test_zyphra_completion(model_id: str) -> None:
    """Verify each Zyphra-served model returns a non-streaming chat completion."""
    if not os.environ.get("ZYPHRA_API_KEY"):
        pytest.skip("ZYPHRA_API_KEY not set, skipping")

    try:
        llm = AnyLLM.create(LLMProvider.ZYPHRA)
    except MissingApiKeyError:
        pytest.skip("ZYPHRA_API_KEY not set, skipping")

    result = await llm.acompletion(
        model=model_id,
        messages=[{"role": "user", "content": "Reply with exactly the word: pong"}],
    )

    assert isinstance(result, ChatCompletion)
    assert result.model is not None
    assert result.choices[0].message.content is not None


@pytest.mark.parametrize("model_id", ZYPHRA_MODELS)
@pytest.mark.asyncio
async def test_zyphra_streaming(model_id: str) -> None:
    """Verify each Zyphra-served model streams chat completion chunks."""
    if not os.environ.get("ZYPHRA_API_KEY"):
        pytest.skip("ZYPHRA_API_KEY not set, skipping")

    try:
        llm = AnyLLM.create(LLMProvider.ZYPHRA)
    except MissingApiKeyError:
        pytest.skip("ZYPHRA_API_KEY not set, skipping")

    stream = await llm.acompletion(
        model=model_id,
        messages=[{"role": "user", "content": "Reply with exactly the word: pong"}],
        stream=True,
    )

    chunks: list[ChatCompletionChunk] = []
    async for chunk in stream:
        assert isinstance(chunk, ChatCompletionChunk)
        chunks.append(chunk)

    assert len(chunks) > 0
    content = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert content.strip() != ""


@pytest.mark.parametrize("model_id", ZYPHRA_REASONING_MODELS)
@pytest.mark.asyncio
async def test_zyphra_streaming_reasoning(model_id: str) -> None:
    """Verify reasoning deltas accumulate into non-empty content for reasoning-capable models."""
    if not os.environ.get("ZYPHRA_API_KEY"):
        pytest.skip("ZYPHRA_API_KEY not set, skipping")

    try:
        llm = AnyLLM.create(LLMProvider.ZYPHRA)
    except MissingApiKeyError:
        pytest.skip("ZYPHRA_API_KEY not set, skipping")

    stream = await llm.acompletion(
        model=model_id,
        messages=[{"role": "user", "content": "Briefly: what is 17 * 23?"}],
        stream=True,
        reasoning_effort="medium",
    )

    reasoning_text = ""
    async for chunk in stream:
        assert isinstance(chunk, ChatCompletionChunk)
        if not chunk.choices:
            continue
        delta_reasoning = chunk.choices[0].delta.reasoning
        if delta_reasoning is not None and delta_reasoning.content:
            reasoning_text += delta_reasoning.content

    assert reasoning_text.strip() != "", f"Expected non-empty streamed reasoning for {model_id}"


@pytest.mark.parametrize("model_id", ZYPHRA_MODELS)
@pytest.mark.asyncio
async def test_zyphra_streaming_tool_call_accumulation(model_id: str) -> None:
    """Verify streamed tool-call argument deltas reassemble into valid JSON for each model."""
    if not os.environ.get("ZYPHRA_API_KEY"):
        pytest.skip("ZYPHRA_API_KEY not set, skipping")

    try:
        llm = AnyLLM.create(LLMProvider.ZYPHRA)
    except MissingApiKeyError:
        pytest.skip("ZYPHRA_API_KEY not set, skipping")

    stream = await llm.acompletion(
        model=model_id,
        messages=[{"role": "user", "content": "What is the weather in San Francisco?"}],
        tools=[WEATHER_TOOL],
        tool_choice="auto",
        stream=True,
        temperature=0,
    )

    accumulated: dict[int, dict[str, str]] = {}
    async for chunk in stream:
        assert isinstance(chunk, ChatCompletionChunk)
        if not chunk.choices:
            continue
        tool_calls = chunk.choices[0].delta.tool_calls
        if not tool_calls:
            continue
        for tc in tool_calls:
            slot = accumulated.setdefault(tc.index, {"name": "", "arguments": ""})
            if tc.function is not None:
                if tc.function.name:
                    slot["name"] = tc.function.name
                if tc.function.arguments:
                    slot["arguments"] += tc.function.arguments

    assert accumulated, f"No tool-call deltas streamed for {model_id}"
    call = accumulated[0]
    assert call["name"] == "get_weather"
    parsed = json.loads(call["arguments"])
    assert parsed.get("city") == "San Francisco"


@pytest.mark.asyncio
async def test_zyphra_list_models() -> None:
    """Verify list_models() returns OpenAI-shaped Model entries from Zyphra's /models endpoint."""
    if not os.environ.get("ZYPHRA_API_KEY"):
        pytest.skip("ZYPHRA_API_KEY not set, skipping")

    try:
        llm = AnyLLM.create(LLMProvider.ZYPHRA)
    except MissingApiKeyError:
        pytest.skip("ZYPHRA_API_KEY not set, skipping")

    models = await llm.alist_models()

    assert len(models) > 0
    ids = {m.id for m in models}
    # Sanity-check the models we know are served. New models can be added without
    # breaking this test; the assertions only require the known set as a subset.
    assert {"deepseek-ai/DeepSeek-V3.2", "moonshotai/Kimi-K2.6", "zai-org/GLM-5.1-FP8"}.issubset(ids)
    for model in models:
        assert model.object == "model"
        assert model.id
        assert model.owned_by
