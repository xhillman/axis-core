# Environment Variables Reference

This table documents variables currently read by axis-core runtime code.

## Providers and Credentials

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | empty | Used by `AnthropicModel` when `api_key` is not passed. |
| `OPENAI_API_KEY` | empty | Used by `OpenAIModel`/`OpenAIResponsesModel` when `api_key` is not passed. |
| `OPENAI_BASE_URL` | SDK default | Optional OpenAI SDK endpoint override (for OpenRouter/openai-compatible gateways). |

## Global Defaults (`axis_core.config.Config`, import time)

| Variable | Default | Purpose |
|---|---|---|
| `AXIS_DEFAULT_MODEL` | `claude-sonnet-4-20250514` | Default model when `Agent(model=...)` is not provided. |
| `AXIS_DEFAULT_PLANNER` | `auto` | Default planner when `Agent(planner=...)` is not provided. |
| `AXIS_DEFAULT_MEMORY` | `ephemeral` | Default memory backend when `Agent(memory=...)` is not provided (`ephemeral`, `sqlite`, `redis`, `synaptic`). |
| `AXIS_TELEMETRY` | `true` | Default telemetry-enabled flag in global config. |
| `AXIS_VERBOSE` | `false` | Default verbose flag in global config. |
| `AXIS_DEBUG` | `false` | Default debug flag in global config. |

## Run-Start Runtime Boundary (`axis_core.config.resolve_runtime_settings`)

These values are resolved once at run start and carried into execution via `ResolvedConfig`.

| Variable | Default | Purpose |
|---|---|---|
| `AXIS_TRANSCRIPT_STRICT` | `false` | Reject unresolved tool-call/tool-result pairing instead of best-effort repair/drop behavior. |
| `AXIS_MAX_TOOL_RESULT_CHARS` | unset | Cap persisted tool-result content passed to model calls. |
| `AXIS_CONTEXT_STRATEGY` | `smart` | Context history strategy in act phase (`smart`, `full`, `minimal`). |
| `AXIS_MAX_CYCLE_CONTEXT` | `5` | Max prior cycles included when strategy is `smart`. |
| `AXIS_CONTEXT_GUARD_ENABLED` | `false` | Enable token-threshold checks before model calls. |
| `AXIS_CONTEXT_WINDOW_TOKENS` | unset | Context-window token budget used by guard/pruning checks. |
| `AXIS_CONTEXT_GUARD_WARN_TOKENS` | `32000` | Warning threshold for low remaining tokens. |
| `AXIS_CONTEXT_GUARD_BLOCK_TOKENS` | `16000` | Hard-block threshold for low remaining tokens. |
| `AXIS_CONTEXT_PRUNE_ENABLED` | `false` | Enable tool-result-first pruning before block decisions. |

## Constructor-Time Helper Env Reads

These env vars are not part of `ResolvedConfig`; they are consumed by narrower helpers during
construction or adapter setup.

### Telemetry Sink Selection

| Variable | Default | Purpose |
|---|---|---|
| `AXIS_TELEMETRY_SINK` | `none` | Sink type: `none`, `console`, `file`, `callback`. |
| `AXIS_TELEMETRY_REDACT` | `true` | Redact sensitive values in telemetry output. |
| `AXIS_TELEMETRY_COMPACT` | `false` | Compact output mode for console sink. |
| `AXIS_TELEMETRY_FILE` | `./axis_trace.jsonl` | File path for file sink output. |
| `AXIS_TELEMETRY_BATCH_SIZE` | `100` | Batch size for buffered file sink writes. |
| `AXIS_TELEMETRY_BUFFER_MODE` | `batched` | Buffer mode: `immediate`, `batched`, `phase`, `end`. |
| `AXIS_TELEMETRY_CALLBACK` | empty | Callback ref in `module:function` form for callback sink. |

### Privacy / Persistence Controls

| Variable | Default | Purpose |
|---|---|---|
| `AXIS_PERSIST_SENSITIVE_TOOL_DATA` | `false` | Include raw sensitive tool args/results in persisted run state (debug use only). |

### Memory Adapter Paths

| Variable | Default | Purpose |
|---|---|---|
| `AXIS_SYNAPTIC_PATH` | `synaptic.db` | SQLite file path used by the `synaptic` memory adapter. |

## Notes

- `.env` loading is attempted automatically when `python-dotenv` is installed.
- `AXIS_TELEMETRY`, `AXIS_VERBOSE`, and `AXIS_DEBUG` are loaded into the global `config` singleton.
- Transcript/context precedence is `step payload -> run-start ResolvedConfig -> built-in default`.
