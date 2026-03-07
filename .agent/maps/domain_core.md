# Domain Core Map

> **When to open:** Working on public entrypoints, context/state management, sessions, tools, checkpoint/output handling, attachments, or cancellation.

## Key Files

| File | Responsibility |
|---|---|
| `axis_core/agent.py` | `Agent` public API, sync/async entrypoints, engine construction |
| `axis_core/_agent_construction.py` | Constructor normalization, config coercion, telemetry sink instantiation |
| `axis_core/_agent_runtime.py` | Shared run/stream result handling, timeout/failure helpers |
| `axis_core/_agent_checkpoint.py` | Agent-facing checkpoint persistence/loading helpers |
| `axis_core/context/__init__.py` | Public context exports |
| `axis_core/context/types.py` | `RunContext`, `RunState`, `CycleState`, phase result types |
| `axis_core/context/transcript.py` | Transcript repair, pruning, context-window helpers |
| `axis_core/context/codec.py` | Context serialization helpers |
| `axis_core/session.py` | `Session`, `Message`, optimistic locking, persistence helpers |
| `axis_core/tool.py` | Public tool API facade over the internal `_tool_*` modules |
| `axis_core/_tool_decorator.py` | `@tool` metadata attachment and manifest creation |
| `axis_core/_tool_schema.py` | Tool input/output schema inference |
| `axis_core/_tool_runtime.py` | `ToolContext`, idempotency helpers, `RateLimiter` |
| `axis_core/_tool_types.py` | `ToolManifest`, `Capability`, `ToolCallRecord` |
| `axis_core/checkpoint.py` | Checkpoint envelopes, phase-boundary validation, resume-state preparation |
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
    system="...",
    fallback=[...],
    telemetry=[...],
)
```

Sync entrypoints: `run()` and `stream()`

Async entrypoints: `run_async()` and `stream_async()`

**Internal flow:** `Agent.__init__()` normalizes constructor inputs through
`_agent_construction.py`, `run*`/`stream*` share execution helpers in `_agent_runtime.py`, and
`_build_engine()` is the final `LifecycleEngine` assembly point.

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
- `axis_core/tool.py` is the stable import surface; implementation now lives in `_tool_*` modules

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

- **Agent constructor/runtime change** → update `agent.py`, the relevant `_agent_*.py` helper, `config.py`, and user-facing docs/examples
- **Tool schema/decorator/runtime change** → start at `tool.py`, then update the owning `_tool_*.py` module, model adapter conversion paths, and tool tests
- **Session/message change** → update all memory adapters' session methods
- **Checkpoint/resume payload change** → update `checkpoint.py`, `agent.py`, and lifecycle resume integration
- **RunContext or transcript change** → update all lifecycle phases plus checkpoint/serialization paths
- **CLI change** → update `axis_core/cli.py` and `tests/test_cli.py`

## Sharp Edges

- `agent.py` is still the public facade, but construction/runtime/checkpoint helpers now live in `_agent_*`; keep those boundaries aligned instead of re-inlining logic
- `tool.py` is a facade; route implementation changes into the matching `_tool_*` module instead of growing the facade
- `RunState` in `context/types.py` is append-only and exposes tuples over private lists
- `session.py` optimistic locking raises `ConcurrencyError` on version mismatch
- `attachments.py` uses eager loading at construction time
