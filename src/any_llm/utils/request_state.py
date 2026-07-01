"""Provider-tagged opaque state for reasoning items.

Reasoning items carry an ``encrypted_content`` blob holding provider-specific
hidden state (Anthropic ``signature``, Gemini ``thought_signature``, etc.).
When a conversation falls back to a different provider, that blob becomes
meaningless and must be discarded rather than forwarded.

We solve this with a tiny prefix protocol: ``"<provider_tag>:<state>"``. The
encoder stamps the emitting provider; the decoder returns the inner state only
when the prefix matches the expected provider, otherwise ``None``.
"""

from __future__ import annotations

from typing import Final

CLAUDE: Final = "anthropic"
GEMINI: Final = "gemini"
OPENAI: Final = "openai"


def encode_provider_state(provider_tag: str, state: str) -> str:
    """Tag opaque reasoning state with the emitting provider so we can drop it on fallback."""
    return f"{provider_tag}:{state}"


def decode_provider_state(blob: str | None, expected_provider: str) -> str | None:
    """Return the inner state if ``blob`` was tagged with ``expected_provider``, else ``None``."""
    if not blob:
        return None
    prefix = f"{expected_provider}:"
    if not blob.startswith(prefix):
        return None
    return blob[len(prefix) :]


__all__ = ["CLAUDE", "GEMINI", "OPENAI", "decode_provider_state", "encode_provider_state"]
