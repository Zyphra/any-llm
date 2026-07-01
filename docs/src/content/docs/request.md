---
title: Request
description: Experimental stateful request API for multi-turn tool-calling
---

## Overview

`request()` and `arequest()` are an experimental transport for stateful multi-turn tool-calling.

They are designed for the cases where a provider needs hidden reasoning state to be passed back on later turns:

- Gemini thought signatures
- Anthropic thinking signatures
- OpenAI Responses-style reasoning continuation

The request API intentionally stays separate from the existing APIs:

- `completion` / `acompletion` for chat-completions compatibility
- `messages` / `amessages` for Anthropic Messages compatibility
- `responses` / `aresponses` for OpenResponses / native Responses compatibility
- `request` / `arequest` for provider-native stateful tool loops

This is a proposed abstraction, not a frozen contract. If implementation experience reveals a cleaner protocol boundary, the API may evolve.

For the detailed architecture and implementation rules behind this API, see
[Request Spec](/docs/request-spec).

## Example

```python
from any_llm import arequest

tool = [
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "Add two integers.",
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

first = await arequest(
    model="gemini-2.5-flash",
    provider="gemini",
    input_data=[
        {
            "type": "message",
            "role": "user",
            "content": "Use the add_numbers tool to add 7 and 5.",
        }
    ],
    tools=tool,
    tool_choice="required",
    reasoning={"effort": "medium"},
)
```

## State Continuity

`arequest` carries provider-specific hidden reasoning state through opaque reasoning items.

That lets the transport preserve the right native continuation mechanism per provider without forcing every backend into the same low-level shape.

Foreign provider state is ignored during replay.

## Current Scope

The current experimental implementation focuses on:

- OpenAI
- Anthropic-compatible providers
- Gemini-compatible providers

Other providers continue to use the existing stable APIs.
