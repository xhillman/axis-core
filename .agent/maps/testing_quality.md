# Testing & Quality Map

> **When to open:** Adding/fixing tests, running quality checks, or understanding test conventions.

## Test Structure

```text
tests/
├── test_agent.py                          # Agent public API and runtime behavior
├── test_cli.py                            # CLI surface
├── test_context.py                        # Context package public behavior
├── test_doc_policy_consistency.py         # Doc-policy checker coverage
├── test_lockfile.py                       # requirements.lock validity
├── test_tool.py                           # @tool decorator, schema + policy behavior
├── engine/
│   ├── test_lifecycle.py                  # LifecycleEngine.execute()
│   ├── test_act_phase.py                  # Act-phase policies / runtime config
│   ├── test_first_cycle_model_calling.py  # First-cycle model behavior
│   ├── test_multi_cycle_integration.py    # Multi-cycle execution
│   ├── test_tool_integration.py           # Tool execution in engine
│   ├── test_registry.py                   # Adapter registration
│   ├── test_resolver.py                   # String → adapter resolution
│   ├── test_trace_collector.py            # Trace event collection
│   └── test_real_llm_integration.py       # Live-provider integration coverage
├── adapters/
│   ├── models/
│   │   ├── test_anthropic.py
│   │   ├── test_openai.py
│   │   ├── test_openai_responses.py
│   │   └── test_openai_responses_integration.py
│   ├── memory/
│   │   ├── test_ephemeral.py
│   │   ├── test_sqlite.py
│   │   ├── test_redis.py
│   │   ├── test_synaptic.py
│   │   └── test_synaptic_integration.py
│   ├── planners/
│   │   ├── test_auto.py
│   │   ├── test_react.py
│   │   └── test_sequential.py
│   └── telemetry/
│       ├── test_console.py
│       ├── test_file.py
│       └── test_callback.py
├── budget/
│   └── test_usage_normalization.py
├── context/
│   └── test_context_window_guard.py
├── protocols/
│   ├── test_memory.py
│   ├── test_model.py
│   ├── test_planner.py
│   └── test_telemetry.py
└── tool/
    └── test_idempotency.py
```

## Commands

| Command | Purpose |
|---|---|
| `pytest <affected-tests>` | Sub-task gate: validate touched behavior |
| `ruff check <touched-paths>` | Sub-task gate: lint touched scope |
| `mypy <touched-python-paths>` | Sub-task gate: type-check touched scope |
| `pytest` | Run the full suite |
| `pytest --cov=axis_core` | With coverage |
| `pytest -m "not slow"` | Skip slow tests |
| `pytest tests/engine/test_lifecycle.py` | Single file |
| `ruff check axis_core tests` | Parent-task gate: full lint |
| `mypy axis_core --strict` | Parent-task gate: full type check |

Current baseline: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest --collect-only -q` collects `831` items with `3` skipped in this repo.

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

**Do NOT test:** internal helpers like `_build_engine`, `_execute_model_step`, or fallback internals directly.

## Conventions

- Markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`
- Async: `@pytest.mark.asyncio` (asyncio mode is configured in `pytest.ini`)
- Pattern: `test_*.py` files, `Test*` classes, `test_*` functions
- Prefer absolute imports from `axis_core.*`

## Common Change Patterns

- **New adapter** → add `tests/adapters/{category}/test_new_adapter.py`
- **New protocol method** → add tests in `tests/protocols/test_{protocol}.py`
- **Bug fix** → write a failing public-contract test first
- **Config option change** → cover it in `tests/test_config.py`
