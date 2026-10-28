from contextlib import contextmanager
from unittest.mock import AsyncMock, Mock, patch

import pytest

from any_llm.providers.awsanthropic.awsanthropic import AwsanthropicProvider
from any_llm.types.completion import CompletionParams


@contextmanager
def mock_awsanthropic_provider():  # type: ignore[no-untyped-def]
    with (
        patch("any_llm.providers.awsanthropic.awsanthropic.AsyncAnthropicAWS") as mock_aws,
        patch("any_llm.providers.anthropic.base._convert_response"),
    ):
        mock_client = Mock()
        mock_aws.return_value = mock_client
        mock_client.messages.create = AsyncMock()
        yield mock_aws


def test_awsanthropic_client_created_with_api_key_workspace_and_region() -> None:
    with mock_awsanthropic_provider() as mock_aws, patch.dict("os.environ", {}, clear=True):
        AwsanthropicProvider(api_key="test-key", workspace_id="wrkspc_test", aws_region="us-west-2")

        mock_aws.assert_called_once_with(
            api_key="test-key", base_url=None, workspace_id="wrkspc_test", aws_region="us-west-2"
        )


def test_awsanthropic_client_created_without_api_key_for_sigv4() -> None:
    with mock_awsanthropic_provider() as mock_aws, patch.dict("os.environ", {}, clear=True):
        AwsanthropicProvider(workspace_id="wrkspc_test", aws_region="us-west-2", aws_profile="stage")

        mock_aws.assert_called_once_with(
            api_key=None, base_url=None, workspace_id="wrkspc_test", aws_region="us-west-2", aws_profile="stage"
        )


def test_awsanthropic_client_with_api_base_from_env() -> None:
    with (
        mock_awsanthropic_provider() as mock_aws,
        patch.dict("os.environ", {"ANTHROPIC_AWS_BASE_URL": "https://custom.aws.endpoint"}, clear=True),
    ):
        AwsanthropicProvider(api_key="test-key", workspace_id="wrkspc_test")

        mock_aws.assert_called_once_with(
            api_key="test-key", base_url="https://custom.aws.endpoint", workspace_id="wrkspc_test"
        )


def test_awsanthropic_real_client_resolves_settings_from_env() -> None:
    env = {"ANTHROPIC_AWS_API_KEY": "env-key", "ANTHROPIC_AWS_WORKSPACE_ID": "wrkspc_env", "AWS_REGION": "us-west-2"}
    with patch.dict("os.environ", env, clear=True):
        provider = AwsanthropicProvider()

    assert str(provider.client.base_url).rstrip("/") == "https://aws-external-anthropic.us-west-2.api.aws"
    assert provider.client.default_headers["anthropic-workspace-id"] == "wrkspc_env"
    assert provider.client.api_key == "env-key"


def test_awsanthropic_real_client_requires_workspace_id() -> None:
    from anthropic import AnthropicError

    with (
        patch.dict("os.environ", {"AWS_REGION": "us-west-2"}, clear=True),
        pytest.raises(AnthropicError, match="ANTHROPIC_AWS_WORKSPACE_ID"),
    ):
        AwsanthropicProvider(api_key="test-key")


@pytest.mark.asyncio
async def test_awsanthropic_completion_calls_messages_create() -> None:
    model = "claude-haiku-4-5"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_awsanthropic_provider() as mock_aws:
        provider = AwsanthropicProvider(api_key="test-key", workspace_id="wrkspc_test", aws_region="us-west-2")
        await provider._acompletion(CompletionParams(model_id=model, messages=messages))

        mock_aws.return_value.messages.create.assert_called_once()
        call_kwargs = mock_aws.return_value.messages.create.call_args[1]
        assert call_kwargs["model"] == model
        assert call_kwargs["messages"] == messages
