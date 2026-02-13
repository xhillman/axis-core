# Models Guide

axis-core resolves model strings through the model registry, then executes through the
selected adapter.

## Model Resolution

You can pass either:

- A model string identifier (for example `"claude-sonnet-4-20250514"`)
- An adapter instance (for example `AnthropicModel(...)`)

String resolution uses the adapter registry and raises `ConfigError` for unknown ids.

## Provider Setup

### Anthropic

```bash
pip install axis-core[anthropic]
export ANTHROPIC_API_KEY=...
```

### OpenAI

```bash
pip install axis-core[openai]
export OPENAI_API_KEY=...
```

### OpenRouter (OpenAI-compatible)

Use the OpenAI adapter against an OpenAI-compatible endpoint.

```bash
pip install axis-core[openrouter]
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

`OPENAI_BASE_URL` is handled by the OpenAI SDK layer.

## Responses API Routing

Some OpenAI model ids route internally to the Responses API adapter path, including:

- Codex models (for example `gpt-5-codex`, `codex-mini-latest`)
- Search models (for example `gpt-5-search-api`)
- Deep-research models
- Computer-use models

This routing is automatic based on model id.

## Fallback Chains

You can define fallback models in priority order:

```python
agent = Agent(
    model="claude-opus-4-6",
    fallback=["claude-sonnet-4-20250514", "gpt-4o"],
)
```

Runtime behavior:

- Try primary model first
- On recoverable model errors, try next fallback
- Stop immediately on non-recoverable errors
- Emit telemetry when fallback transitions occur

## Adapter Instances vs Model Strings

String-based:

```python
agent = Agent(model="gpt-4o")
```

Instance-based:

```python
from axis_core.adapters.models.openai import OpenAIModel

agent = Agent(model=OpenAIModel(model_id="gpt-4o", temperature=0.2, max_tokens=800))
```

Use instance-based config when you need provider-specific constructor options.

## Model Parameters

Common adapter constructor options include:

- `model_id`
- `api_key`
- `temperature`
- `max_tokens`

## Pricing and External Limits

Do not hardcode provider prices in project docs. Provider pricing and limits change often.
Link directly to provider pricing pages in user-facing materials.
