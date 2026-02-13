# Environment Variables Reference

This table documents variables currently read by axis-core runtime code.

## Providers and Credentials

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | empty | Used by `AnthropicModel` when `api_key` is not passed. |
| `OPENAI_API_KEY` | empty | Used by `OpenAIModel`/`OpenAIResponsesModel` when `api_key` is not passed. |
| `OPENAI_BASE_URL` | SDK default | Optional OpenAI SDK endpoint override (for OpenRouter/openai-compatible gateways). |

## Global Defaults (`axis_core.config`)

| Variable | Default | Purpose |
|---|---|---|
| `AXIS_DEFAULT_MODEL` | `claude-sonnet-4-20250514` | Default model when `Agent(model=...)` is not provided. |
| `AXIS_DEFAULT_PLANNER` | `auto` | Default planner when `Agent(planner=...)` is not provided. |
| `AXIS_DEFAULT_MEMORY` | `ephemeral` | Default memory backend when `Agent(memory=...)` is not provided. |
| `AXIS_TELEMETRY` | `true` | Default telemetry-enabled flag in global config. |
| `AXIS_VERBOSE` | `false` | Default verbose flag in global config. |
| `AXIS_DEBUG` | `false` | Default debug flag in global config. |

## Telemetry Sink Selection

| Variable | Default | Purpose |
|---|---|---|
| `AXIS_TELEMETRY_SINK` | `none` | Sink type: `none`, `console`, `file`, `callback`. |
| `AXIS_TELEMETRY_REDACT` | `true` | Redact sensitive values in telemetry output. |
| `AXIS_TELEMETRY_COMPACT` | `false` | Compact output mode for console sink. |
| `AXIS_TELEMETRY_FILE` | `./axis_trace.jsonl` | File path for file sink output. |
| `AXIS_TELEMETRY_BATCH_SIZE` | `100` | Batch size for buffered file sink writes. |
| `AXIS_TELEMETRY_BUFFER_MODE` | `batched` | Buffer mode: `immediate`, `batched`, `phase`, `end`. |
| `AXIS_TELEMETRY_CALLBACK` | empty | Callback ref in `module:function` form for callback sink. |

## Privacy / Persistence Controls

| Variable | Default | Purpose |
|---|---|---|
| `AXIS_PERSIST_SENSITIVE_TOOL_DATA` | `false` | Include raw sensitive tool args/results in persisted run state (debug use only). |

## Context Assembly Controls

| Variable | Default | Purpose |
|---|---|---|
| `AXIS_CONTEXT_STRATEGY` | `smart` | Context history strategy in act phase (`smart`, `full`, `minimal`). |
| `AXIS_MAX_CYCLE_CONTEXT` | `5` | Max prior cycles included when strategy is `smart`. |

## Notes

- `.env` loading is attempted automatically when `python-dotenv` is installed.
- Some variables listed in `.env.example` are roadmap or compatibility placeholders and
  may not be consumed by current runtime code.
