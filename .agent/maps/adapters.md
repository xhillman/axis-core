# Adapters Map

> **When to open:** Adding/modifying model providers, memory backends, planners, or telemetry sinks.

## Directory Layout

```
axis_core/adapters/
├── models/
│   ├── __init__.py      # Registers 10 Claude + 17 OpenAI model IDs via lazy factories
│   ├── anthropic.py     # AnthropicModel (618L) — Claude API adapter
│   └── openai.py        # OpenAIModel (642L) — OpenAI/GPT API adapter
├── memory/
│   ├── __init__.py      # Registers ephemeral, sqlite, redis
│   ├── ephemeral.py     # EphemeralMemory (220L) — in-memory dict
│   ├── sqlite.py        # SQLiteMemory (306L) — FTS5 full-text search
│   └── redis.py         # RedisMemory (302L) — TTL, namespaces, SCAN search
├── planners/
│   ├── __init__.py      # Registers sequential, auto, react
│   ├── sequential.py    # SequentialPlanner (129L) — deterministic, always succeeds
│   ├── auto.py          # AutoPlanner (410L) — LLM-based planning + fallback
│   └── react.py         # ReActPlanner (469L) — Thought/Action/Observation loop
└── telemetry/
    ├── __init__.py      # Exports ConsoleSink
    └── console.py       # ConsoleSink (119L) — pretty/compact output, redaction
```

## Registration Pattern (all adapters follow this)

Each `__init__.py` uses `make_lazy_factory()` from `engine/registry.py`:
1. Calls `make_lazy_factory("module_path", "ClassName")` → creates wrapper class
2. Registers wrapper with `{category}_registry.register(name, wrapper)`
3. Wrapper defers actual import until first instantiation
4. Optional deps: try/except at import time → `ConfigError` with install instructions

**Reference implementation:** `axis_core/adapters/models/__init__.py`

## Model Adapters

| Registered Name | Class | Optional Dep |
|---|---|---|
| `claude-sonnet-4-20250514`, `claude-opus-4-*`, etc. (10 IDs) | `AnthropicModel` | `anthropic>=0.18` |
| `claude-haiku`, `claude-sonnet`, `claude-opus` (3 aliases) | `AnthropicModel` | `anthropic>=0.18` |
| `gpt-5-*`, `gpt-4o-*`, `o1-*`, `o3-*`, etc. (17 IDs) | `OpenAIModel` | `openai>=1.0` |

**Key internals:**
- Both adapters have `MODEL_PRICING` dicts for cost estimation
- Both convert between axis-core message format and provider-specific format
- `OpenAIModel` has `_COMPLETION_TOKENS_MODELS` set (13 models using `max_completion_tokens`)
- Implements `ModelAdapter` protocol: `complete()`, `stream()`, `estimate_tokens()`, `estimate_cost()`

## Memory Adapters

| Registered Name | Class | Capabilities | Optional Dep |
|---|---|---|---|
| `ephemeral` | `EphemeralMemory` | KEYWORD_SEARCH | None |
| `sqlite` | `SQLiteMemory` | KEYWORD_SEARCH (FTS5) | `aiosqlite>=0.19` |
| `redis` | `RedisMemory` | KEYWORD_SEARCH, TTL, NAMESPACES | `redis>=5.0` |

**Key internals:**
- All implement `MemoryAdapter` + `SessionStore` protocols
- SQLite uses FTS5 virtual tables with triggers for full-text search
- Redis uses `axis:` prefix, `__meta__` suffix for metadata, SCAN for search
- Session support: `store_session()`, `retrieve_session()`, `update_session()` with optimistic locking

## Planner Adapters

| Registered Name | Class | Uses Model? |
|---|---|---|
| `sequential` | `SequentialPlanner` | No |
| `auto` | `AutoPlanner` | Yes (generates JSON plan) |
| `react` | `ReActPlanner` | Yes (Thought/Action/Observation) |

**Key internals:**
- `SequentialPlanner`: deterministic, wraps each tool as one step — ideal fallback
- `AutoPlanner`: sends planning prompt to LLM, parses JSON plan, falls back to Sequential on any error (AD-016)
- `ReActPlanner`: implements ReAct loop with `_DEFAULT_MAX_ITERATIONS = 10`, explicit reasoning traces

## Common Change Patterns

- **New model provider** → create `adapters/models/new_provider.py`, register IDs in `__init__.py`, add pricing dict
- **New memory backend** → implement `MemoryAdapter` + `SessionStore`, register in `__init__.py`
- **New planner** → implement `Planner` protocol, register in `__init__.py`
- **If you change a Protocol** → check ALL adapters implementing it (see [protocols_types.md](protocols_types.md))
- **If you change `make_lazy_factory()`** → all 4 adapter `__init__.py` files are affected

## Sharp Edges

- Model pricing dicts are hardcoded — must be manually updated when providers change pricing
- `AutoPlanner` JSON parsing is fragile — uses regex extraction, not guaranteed valid JSON
- Redis adapter uses SCAN (not KEYS) for search — correct for production but returns results incrementally
- Telemetry `ConsoleSink` is the only sink — no file/remote sinks exist yet
- `loadouts/` directory exists but is empty (placeholder for pre-configured templates)
