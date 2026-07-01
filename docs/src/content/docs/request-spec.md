---
title: Request Spec
description: Detailed implementation spec for the experimental request API
---

## Status

This is a living spec for `request()` / `arequest()`.

It captures the current architectural agreement for the experimental stateful
tool-calling transport, the implementation boundaries we want to preserve, and
the remaining work needed before the feature should be considered complete.

This document is intentionally more detailed and more opinionated than the
public overview in [Request](/docs/request).

The high-level API goal is stable:

- one AnyLLM request envelope for stateful multi-turn tool-calling
- provider-native reasoning continuity preserved across turns
- strongest existing AnyLLM endpoint reused per provider

The internal implementation is not frozen. If a cleaner abstraction appears
during implementation, we should change course rather than preserve accidental
complexity.

### Draft implementation caveat

The current draft has an unresolved input-shape mismatch.

The public examples and some tests use plain dict request items, but several
provider conversion helpers currently assume they receive already-validated
OpenResponses Pydantic models from `RequestParams.input`. That makes direct
helper-level tests fail with attribute errors such as missing `call_id`, and it
also leaves the public contract unclear for synthetic reasoning items where
OpenResponses requires an `id`.

Before this feature is hardened, we need to choose and implement one contract:

- Normalize every public `input_data` item at the SDK boundary, then make all
  provider helpers accept only typed OpenResponses models.
- Or add one shared request-input normalizer that provider helpers call before
  branching on item type.

Either path should be covered by tests for:

- user, assistant, system, and developer message dicts
- reasoning items with provider-tagged `encrypted_content`
- function call and function call output items
- OpenAI `previous_response_id`, Anthropic `signature`, and Gemini
  `thought_signature` round trips

Until that is done, treat the current branch as a draft batch: useful for
capturing the direction, but not a polished public API.

## Problem Statement

Reasoning-capable providers do not preserve multi-turn tool loops in the same
way:

- **OpenAI** uses Responses API server-side continuation via
  `previous_response_id`
- **Anthropic** uses Messages API thinking blocks with `signature`
- **Gemini** uses completion-style tool calls with `thought_signature`

This makes provider-agnostic agent loops awkward when built directly on
`acompletion`, `amessages`, or `aresponses`.

`arequest` exists to provide one shared transport for those loops without
flattening away the provider-specific continuity mechanisms that actually make
them work.

## Core Agreements

### 1. OpenResponses is the correct outer envelope

`arequest` uses the OpenResponses-style item model as its
public envelope.

That means:

- request history is represented as response-style items
- output is returned as `RequestResponse`, which extends `ResponseResource`
- request streaming is represented with response-style streaming events

This is not a claim that OpenResponses is the semantic common denominator for
all providers. It is the chosen transport envelope.

### 2. Provider-native continuity state is not standardized

Reasoning continuity is provider-specific and should stay provider-specific.

We do **not** try to normalize these into one shared explicit meaning:

- OpenAI: `previous_response_id`
- Anthropic: thinking block `signature`
- Gemini: `thought_signature`

Instead, provider-native continuity state is embedded opaquely inside the
request envelope, currently via `encrypted_content` on reasoning items.

Only the matching provider path may decode and reuse that state. Foreign state
must be ignored safely.

### 3. Translation is by endpoint family, not by provider

This is the most important implementation rule.

`arequest` should reuse existing AnyLLM machinery by **endpoint family**:

- **OpenAI-family request path** -> `aresponses`
- **Anthropic-family request path** -> `amessages`
- **Gemini-family request path** -> `acompletion`

`arequest` is allowed to perform:

- input translation from request-envelope history to the target endpoint shape
- output translation from endpoint response to request-envelope output
- opaque provider-state extraction and replay

`arequest` should **not** duplicate provider policy that already lives inside
the native endpoint path.

Examples of policy that should stay in native endpoint machinery:

- default parameter values
- provider validation
- streaming normalization inside that endpoint family
- provider-specific structured output behavior
- tool schema handling
- provider-specific request quirks

## Canonical Provider Routing

### OpenAI

`arequest` must wrap `aresponses`.

Responsibilities:

- translate request input into Responses input
- extract `previous_response_id` from opaque request state
- pass `previous_response_id` to `aresponses`
- append fresh opaque OpenAI state to the returned request output

Non-goals:

- do not emulate OpenAI reasoning continuity by replaying hidden reasoning as
  generic message items
- do not build a second OpenAI reasoning policy outside `aresponses`

### Anthropic

`arequest` must wrap `amessages`.

Responsibilities:

- translate request input into Messages input
- reconstruct assistant thinking blocks and tool-use blocks from request items
- preserve thinking signatures in opaque state
- translate Messages responses and Messages stream events back into request
  output/events

Important provider rule:

- if thinking is enabled and tool choice is forced, request handling should
  follow Anthropic constraints rather than inventing a fake universal behavior

Current agreed compatibility behavior:

- downgrade forced tool choice to `auto`
- emit a warning

Non-goals:

- do not maintain a second Anthropic thinking-budget policy if `amessages`
  already has a better internal policy

### Gemini

`arequest` must wrap `acompletion`.

Responsibilities:

- translate request input into completion messages
- reconstruct tool calls and tool outputs in the completion format
- inject preserved Gemini `thought_signature` into completion tool calls
- translate `ChatCompletion` / `ChatCompletionChunk` back into request
  output/events

Important nuance:

- Gemini reasoning continuity is not just "pass a signature"; the full request
  history still has to be converted into the completion message format
- the signature is one field inside that endpoint-family translation

Non-goals:

- do not create a second Gemini request policy beside `acompletion`
- do not fork media / tool schema / structured output handling away from the
  completion path unless strictly required

## Shared Envelope Rules

### Request input

The request envelope carries:

- user, system, developer, and assistant messages
- reasoning items
- function call items
- function call output items
- item references where supported

Opaque provider-native continuation state must stay embedded in existing fields,
not elevated into new shared top-level request concepts.

Provider-native continuation state is attached **per reasoning item / per turn**,
not globally for the whole conversation.

This matters because provider fallbacks may produce mixed histories such as:

- one turn with OpenAI `previous_response_id` state
- a later turn with Gemini `thought_signature` state
- a later turn with Anthropic thinking-signature state

That mixed history is valid.

Decoding must therefore be:

- per item
- provider-selective
- tolerant of mixed histories

Each provider path must reuse only the opaque state compatible with the current
provider and ignore foreign state without error.

### Request output

The output envelope must preserve:

- assistant text
- reasoning items
- function calls
- structured output parsing result
- usage data

`RequestResponse` extends `ResponseResource` with:

- `structured_output`

### Structured output

`arequest` should mirror `acompletion` behavior where possible:

- dict JSON schema formats are allowed
- typed structured output is allowed for non-streaming calls
- typed structured output with `stream=True` is not supported

## Streaming Rules

Streaming translation should also be by endpoint family:

- `aresponses` -> request stream events
- `amessages` -> request stream events
- `acompletion` -> request stream events

We do not want one provider-specific streaming stack per provider unless the
endpoint-family normalization itself is insufficient.

This means:

- OpenAI should mostly passthrough native Responses streaming
- Anthropic should translate Messages stream events
- Gemini should translate completion chunks

## Opaque Continuity State Rules

Opaque state must be:

- versioned
- provider-scoped
- safe to ignore when the provider does not match

It currently lives in `encrypted_content` on reasoning items.

That field is a transport carrier, not a user-facing semantic contract.

Because fallback/provider switching can mix provider-native state in one
conversation, opaque state handling must be message-local rather than
conversation-global.

## What We Know So Far

### Confirmed

- OpenAI `aresponses` already uses the OpenAI/OpenResponses-style request
  contract and converts native `Response` objects into `ResponseResource`
  when possible
- Gemini works best through `acompletion`
- Anthropic works best through `amessages`
- `arequest` should route into those strongest native endpoint families

### Confirmed implementation smell to avoid

- duplicating provider defaults or validation inside `arequest`
- pushing provider-specific replay semantics into the shared request helpers
- building large generic compatibility modules that secretly contain provider
  branches

## Unknowns That Must Be Tested

### 1. Historical replay depth

We need a **3-turn tool loop** for each provider, not just 2 turns.

That is the only reliable way to answer:

- should we resend only the most recent provider-native continuity state?
- or does the provider need the full historical chain of hidden state items?

This must be tested for:

- OpenAI
- Anthropic
- Gemini

And ideally compared in two modes:

- full historical replay
- last-turn-only replay

This question must be answered independently for each endpoint family. We should
not assume that the minimal replay set is identical across providers.

### 2. Parallel tool-calling behavior

Parallel tool calls must be treated as a separate validation problem, not as an
automatic extension of serial tool loops.

We need to verify, per provider:

- whether hidden continuity state applies to the whole assistant turn or to
  specific tool calls
- whether one assistant turn can require multiple state fragments
- whether tool-call ordering matters when replaying tool outputs
- whether only the first tool call in a parallel bundle carries required hidden
  state or whether each tool call can have its own

Known reason for caution:

- Gemini already exhibits special behavior tied to the first tool call in a
  sequence when signatures are missing

Required tests:

- one assistant turn producing 2+ tool calls
- replay of all tool outputs
- continuation into a further turn after the parallel tool response

We should run this for:

- OpenAI Responses-backed request loops
- Anthropic Messages-backed request loops
- Gemini completion-backed request loops

### 3. Streaming parity

We still need full confidence that request streaming preserves:

- reasoning deltas
- tool call deltas
- final output item structure
- provider-native continuity state where applicable

## Current Implementation Goals

The implementation should converge toward these properties:

1. `arequest` is a thin router plus endpoint-family adapters
2. provider-native hidden reasoning state is opaque and embedded
3. endpoint-family policy is reused rather than duplicated
4. public request envelope stays stable enough for agent frameworks
5. internal code stays small, typed, and easy to audit

## Remaining Work Checklist

### Correctness

- fix provider-specific live bugs that come from `arequest` drift
- verify Anthropic thinking budget / token handling through the native path
- verify 3-turn continuity per provider
- verify parallel tool-calling continuity behavior

### Cleanup

- remove dead compatibility scaffolding
- keep shared request helpers minimal and provider-agnostic
- move provider-specific replay logic out of generic helpers
- tighten intermediate typing where the bridge code still relies on loose dicts

### Validation

- focused unit coverage for each provider path
- integration tests for multi-turn tool loops
- live smoke tests for OpenAI, Anthropic, and Gemini

## Decision Rule Going Forward

If a future change makes the implementation:

- smaller
- more endpoint-family oriented
- less duplicative of provider policy
- easier to reason about

then we should prefer that change even if it means revisiting the current
implementation.

The API goal is important.
The current implementation details are not sacred.
