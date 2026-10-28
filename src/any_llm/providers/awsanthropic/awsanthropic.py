from __future__ import annotations

from typing import Any

from typing_extensions import override

from any_llm.providers.anthropic.base import BaseAnthropicProvider

MISSING_PACKAGES_ERROR = None
try:
    from anthropic import AsyncAnthropicAWS
except ImportError as e:
    MISSING_PACKAGES_ERROR = e


class AwsanthropicProvider(BaseAnthropicProvider):
    """
    AWS Anthropic Provider for Claude models through Claude Platform on AWS.

    Claude Platform on AWS is operated by Anthropic and serves the Anthropic Messages API from AWS regions.
    It is not Amazon Bedrock (use the `bedrock` provider for that). Uses Anthropic's `AsyncAnthropicAWS` client.

    Authentication uses an API key (via `ANTHROPIC_AWS_API_KEY` or the `api_key` constructor argument) or,
    without one, AWS SigV4 credentials (`aws_access_key`/`aws_secret_key`, `aws_profile`, or the default AWS
    credential chain). A workspace ID is required (via `ANTHROPIC_AWS_WORKSPACE_ID` or `workspace_id`), and the
    region comes from `aws_region` or `AWS_REGION`/`AWS_DEFAULT_REGION`.
    """

    PROVIDER_NAME = "awsanthropic"
    ENV_API_KEY_NAME = "ANTHROPIC_AWS_API_KEY"
    ENV_API_BASE_NAME = "ANTHROPIC_AWS_BASE_URL"
    PROVIDER_DOCUMENTATION_URL = "https://platform.claude.com/docs/en/build-with-claude/claude-platform-on-aws"

    SUPPORTS_LIST_MODELS = False

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    client: AsyncAnthropicAWS

    @override
    def _verify_and_set_api_key(self, api_key: str | None = None) -> str | None:
        # The API key is optional: without one the client signs requests with AWS SigV4 credentials.
        return api_key

    @override
    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        # The client resolves the API key, region, workspace ID and base URL from the environment itself.
        self.client = AsyncAnthropicAWS(api_key=api_key, base_url=api_base, **kwargs)
