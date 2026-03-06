# Adapters Map

> **When to open:** Adding/modifying model providers, memory backends, planners, telemetry sinks, or adapter registration.

## Directory Layout

```text
axis_core/adapters/
├── models/
│   ├── __init__.py              # Registers model IDs and aliases via lazy factories
│   ├── anthropic.py             # Anthropic chat/messages adapter
│   ├── openai.py                # OpenAI chat-completions adapter
│   ├── openai_responses.py      # OpenAI Responses API adapter
│   ├── openai_error_utils.py    # Shared OpenAI error normalization helpers
│   └── provider_helpers.py      # Shared provider parsing / usage helpers
├── memory/
│   ├── __init__.py              # Registers ephemeral, sqlite, redis, synaptic
│   ├── ephemeral.py             # In-memory adapter
│   ├── sqlite.py                # SQLite/FTS-backed adapter
│   ├── redis.py                 # Redis-backed adapter
│   └── synaptic.py              # Synaptic-backed adapter
├── planners/
│   ├── __init__.py              # Registers sequential, auto, react
│   ├── sequential.py
│   ├── auto.py
│   └── react.py
└── telemetry/
    ├── __init__.py              # Exports console, file, and callback sinks
    ├── console.py
    ├── file.py
    └── callback.py
```

## Registration Pattern

Each adapter family registers lazy factories in its `__init__.py` using
`make_lazy_factory()` from `axis_core/engine/registry.py`.

1. Wrap the concrete class with `make_lazy_factory("module_path", "ClassName")`
2. Register the wrapper in the relevant registry
3. Keep optional-dependency install guidance close to the registration site
4. Update tests for resolution and protocol behavior

## Model Adapters

- `AnthropicModel` handles Anthropic message conversion, tool calls, and pricing estimation.
- `OpenAIModel` handles chat-completions models plus routing logic for OpenAI-compatible behavior.
- `OpenAIResponsesModel` owns the Responses API path for models registered against that surface.
- Shared provider utilities live in `provider_helpers.py` and `openai_error_utils.py`; if you change token/cost/error normalization, review all three model modules together.

## Memory Adapters

- `EphemeralMemory`, `SQLiteMemory`, `RedisMemory`, and `SynapticMemory` all implement the memory/session contracts.
- SQLite owns FTS-backed keyword search.
- Redis owns namespace + TTL behavior.
- Synaptic adds the optional `synaptic-core` integration path and should be treated like any other first-class memory backend.

## Planner Adapters

- `SequentialPlanner` is deterministic and the fallback baseline.
- `AutoPlanner` generates a structured plan from a model and falls back when parsing/validation fails.
- `ReActPlanner` runs the thought/action/observation planning style.

## Telemetry Adapters

- `ConsoleSink` is the human-readable stdout/stderr sink.
- `FileSink` writes JSONL traces with batching support.
- `CallbackSink` invokes caller-provided sync or async callbacks.

## Common Change Patterns

- **New model provider** → add a new module under `adapters/models/`, register IDs in `__init__.py`, update pricing/error helpers as needed
- **New memory backend** → implement `MemoryAdapter` and `SessionStore`, then register it in `memory/__init__.py`
- **New telemetry sink** → add the sink module, export it in `telemetry/__init__.py`, and add protocol tests
- **Protocol change** → review every concrete adapter plus `tests/protocols/`
- **Registry/lazy-factory change** → review every adapter `__init__.py`

## Sharp Edges

- Provider/model metadata is split across registration, pricing, and helper modules; keep those in sync
- Optional dependency failures should raise actionable `ConfigError` messages
- Redis search uses cursor iteration and namespace metadata; preserve that behavior
- Synaptic support is optional but active, not experimental
