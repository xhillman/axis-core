# Domain Core Map

> **When to open:** Working on public entrypoints, context/state management, sessions, tools, checkpoint/output handling, attachments, or cancellation.

## Key Files

| File | Responsibility |
|---|---|
| `axis_core/agent.py` | `Agent` public API, sync/async entrypoints, engine construction |
| `axis_core/context/__init__.py` | Public context exports |
| `axis_core/context/types.py` | `RunContext`, `RunState`, `CycleState`, phase result types |
| `axis_core/context/transcript.py` | Transcript repair, pruning, context-window helpers |
| `axis_core/context/codec.py` | Context serialization helpers |
| `axis_core/session.py` | `Session`, `Message`, optimistic locking, persistence helpers |
| `axis_core/tool.py` | `@tool`, manifests, schema generation, policy metadata |
| `axis_core/checkpoint.py` | Checkpoint payload/model helpers |
| `axis_core/output_schema.py` | Output-schema normalization/validation |
| `axis_core/cli.py` | CLI surface over the public API |
| `axis_core/attachments.py` | `Image`, `PDF`, `Attachment` helpers |
| `axis_core/cancel.py` | `CancelToken` |
| `axis_core/redaction.py` | Sensitive-data redaction helpers |

## Agent Public API

```python
Agent(
    model="claude-sonnet-4-20250514",
    tools=[...],
    memory="ephemeral",
    planner="sequential",
    budget=Budget(...),
    system_prompt="...",
    fallback_models=[...],
    telemetry=[...],
)
```

Sync entrypoints: `run()` and `stream()`

Async entrypoints: `run_async()` and `stream_async()`

**Internal flow:** `Agent._build_engine()` resolves adapters and runtime settings, creates a
`LifecycleEngine`, then delegates execution.

## Tool System

```python
@tool
def my_tool(query: str, limit: int = 10) -> str:
    """Tool description shown to the model."""
    return "result"
```

- `ToolManifest` is generated from type hints + docstring
- `ToolContext` supports dependency injection for tools that request it
- `Capability` models destructive/privileged tool behavior

## Context & State

- `RunContext`: mutable runtime state during execution
- `RunState`: append-only execution history
- `CycleState`: one lifecycle-cycle snapshot
- Transcript/window repair and pruning live in `context/transcript.py`
- Serialization helpers live in `context/codec.py`

## Session System

- `Session` stores ordered `Message` objects with optimistic locking via `version`
- Memory adapters implement session persistence through the `SessionStore` protocol
- Session history is injected into runtime context when present

## Common Change Patterns

- **Agent constructor change** → update `agent.py`, `config.py`, and user-facing docs/examples
- **Tool schema/policy change** → update `tool.py`, model adapter conversion paths, and tool tests
- **Session/message change** → update all memory adapters' session methods
- **RunContext or transcript change** → update all lifecycle phases plus checkpoint/serialization paths
- **CLI change** → update `axis_core/cli.py` and `tests/test_cli.py`

## Sharp Edges

- `agent.py` is large and mixes public API with engine construction; keep public behavior stable
- `RunState` in `context/types.py` is append-only and exposes tuples over private lists
- `session.py` optimistic locking raises `ConcurrencyError` on version mismatch
- `attachments.py` uses eager loading at construction time
