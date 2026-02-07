# Domain Core Map

> **When to open:** Working on agent.py, context/state management, sessions, tools, attachments, or cancellation.

## Key Files

| File | Lines | Responsibility |
|---|---|---|
| `axis_core/agent.py` | 783 | `Agent` class — run/stream (sync+async), builds engine, resolves adapters |
| `axis_core/context.py` | 965 | `RunContext` (mutable state during execution), `RunState` (append-only history) |
| `axis_core/session.py` | 345 | `Session` multi-turn history, `Message`, optimistic locking, persistence |
| `axis_core/tool.py` | 445 | `@tool` decorator, `ToolManifest`, JSON schema generation, `ToolContext` |
| `axis_core/attachments.py` | 119 | `Image`, `PDF`, `Attachment` — eager-loaded, 10MB limit |
| `axis_core/cancel.py` | 29 | `CancelToken` — cooperative cancellation (AD-028) |
| `axis_core/redaction.py` | 55 | `redact_sensitive_data()`, `is_sensitive_key()` |

## Agent Public API

```python
Agent(
    model="claude-sonnet-4-20250514",  # string auto-resolves via registry
    tools=[...],                        # @tool-decorated functions
    memory="ephemeral",                 # string or MemoryAdapter instance
    planner="sequential",               # string or Planner instance
    budget=Budget(...),                  # resource limits
    system_prompt="...",                 # optional
    fallback_models=[...],              # model fallback chain (AD-013)
    telemetry=[...],                    # TelemetrySink list
)

# Sync
result = agent.run("prompt")
for event in agent.stream("prompt"):
    ...

# Async
result = await agent.run_async("prompt")
async for event in agent.stream_async("prompt"):
    ...
```

**Internal flow:** `Agent._build_engine()` → resolves all string adapters → creates `LifecycleEngine` → calls `engine.execute()`

## Tool System

```python
@tool
def my_tool(query: str, limit: int = 10) -> str:
    """Tool description shown to the model."""
    return "result"

# ToolManifest generated automatically from type hints + docstring
# JSON schema generated via generate_tool_schema()
# ToolContext passed to tools that request it (dependency injection)
```

- `Capability` enum: `READ`, `WRITE`, `EXECUTE`, `NETWORK`, `FILESYSTEM`
- `RateLimiter`: defined but NOT wired into lifecycle (unused infrastructure)

## Context & State

- `RunContext`: mutable during execution, holds model/memory/planner refs, budget state, cancel token
- `RunState`: append-only execution history (cycles, errors, model calls)
- `CycleState`: snapshot of one lifecycle cycle (observation, plan, execution result, eval)
- Size limits: `WARN_CONTEXT_SIZE = 50MB`, `MAX_CONTEXT_SIZE = 100MB`

## Session System

- `Session`: ordered list of `Message` objects with `version` for optimistic locking
- `Message`: role (user/assistant/system/tool) + `ContentPart` list
- Session persistence via `SessionStore` protocol (memory adapters implement it)
- `SESSION_PREFIX = "session:"`, `SESSION_CONTEXT_KEY = "__session_history__"`

## Common Change Patterns

- **If you change Agent constructor args** → update `agent.py` + `config.py` (env var defaults)
- **If you change tool schema generation** → update `tool.py:generate_tool_schema()` + model adapters' `_convert_tools_*` methods
- **If you change Session/Message format** → update all 3 memory adapters' session methods
- **If you change RunContext fields** → update all 6 phase modules

## Sharp Edges

- `agent.py` uses `_coerce()` generic for type coercion (Budget, Session, CancelToken)
- `context.py` has append-only semantics — `_errors`, `_cycles` etc. are private lists exposed as tuples
- `session.py` optimistic locking: `update_session()` checks version match, raises `ConcurrencyError` on mismatch
- `attachments.py` uses eager loading (reads file at construction time, not lazy)
