# Testing & Quality Map

> **When to open:** Adding/fixing tests, running quality checks, or understanding test conventions.

## Test Structure

```
tests/
├── test_agent.py                          # Agent public API tests
├── test_attachments.py                    # Attachment loading/validation
├── test_budget.py                         # Budget limits and state
├── test_cancel.py                         # CancelToken
├── test_config.py                         # Config loading, env vars, merge
├── test_context.py                        # RunContext, RunState, serialization
├── test_errors.py                         # Error hierarchy
├── test_lockfile.py                       # requirements.lock validity
├── test_result.py                         # RunResult, StreamEvent
├── test_session.py                        # Session, Message, optimistic locking
├── test_tool.py                           # @tool decorator, schema generation
├── engine/
│   ├── test_lifecycle.py                  # LifecycleEngine.execute()
│   ├── test_first_cycle_model_calling.py  # First-cycle model behavior
│   ├── test_multi_cycle_integration.py    # Multi-cycle execution
│   ├── test_tool_integration.py           # Tool execution in engine
│   ├── test_registry.py                   # Adapter registration
│   ├── test_resolver.py                   # String → adapter resolution
│   └── test_trace_collector.py            # Trace event collection
├── adapters/
│   ├── models/
│   │   ├── test_anthropic.py              # AnthropicModel
│   │   └── test_openai.py                 # OpenAIModel
│   ├── memory/
│   │   ├── test_ephemeral.py              # EphemeralMemory
│   │   ├── test_sqlite.py                 # SQLiteMemory
│   │   └── test_redis.py                  # RedisMemory
│   ├── planners/
│   │   ├── test_auto.py                   # AutoPlanner
│   │   ├── test_react.py                  # ReActPlanner
│   │   └── test_sequential.py             # SequentialPlanner
│   └── telemetry/
│       └── test_console.py                # ConsoleSink
└── protocols/
    ├── test_memory.py                     # MemoryAdapter protocol
    ├── test_model.py                      # ModelAdapter protocol
    ├── test_planner.py                    # Planner protocol
    └── test_telemetry.py                  # TelemetrySink protocol
```

## Commands

| Command | Purpose |
|---|---|
| `pytest <affected-tests>` | Sub-task gate: validate touched behavior |
| `ruff check <touched-paths>` | Sub-task gate: lint touched scope |
| `mypy <touched-python-paths>` | Sub-task gate: type-check touched scope |
| `pytest` | Run all 633 tests (~5s) |
| `pytest --cov=axis_core` | With coverage |
| `pytest -m "not slow"` | Skip slow tests |
| `pytest tests/engine/test_lifecycle.py` | Single file |
| `ruff check axis_core tests` | Parent-task gate: full lint |
| `mypy axis_core --strict` | Parent-task gate: full type check |

## Gate Levels

- **Sub-task gate:** run touched-scope tests/lint/types before marking sub-task complete.
- **Parent-task gate:** run full `pytest`, `ruff check axis_core tests`, and `mypy axis_core --strict` before marking parent complete.

## Testing Rules

**Public-contract testing only.** Allowed surfaces:
- `Agent.run()`, `run_async()`, `stream()`, `stream_async()`
- `LifecycleEngine.execute()`
- Lifecycle phase functions (`initialize`, `observe`, `plan`, `act`, `evaluate`, `finalize`)
- Adapter protocol methods (`complete`, `stream`, `store`, `retrieve`, `plan`)
- `resolve_adapter()` and registry APIs

**Do NOT test:** internal helpers (`_try_models_with_fallback`, `_execute_model_step`, `_build_engine`, etc.)

## Conventions

- Markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`
- Async: `@pytest.mark.asyncio` (asyncio_mode=auto in pytest.ini)
- Pattern: `test_*.py` files, `Test*` classes, `test_*` functions
- Absolute imports only: `from axis_core.* import ...`

## Common Change Patterns

- **New adapter** → add `tests/adapters/{category}/test_new_adapter.py`
- **New protocol method** → add tests in `tests/protocols/test_{protocol}.py`
- **Bug fix** → write failing test FIRST (TDD), then fix
- **If you add a config option** → add test in `tests/test_config.py`
