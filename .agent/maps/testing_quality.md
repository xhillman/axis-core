# Testing & Quality Map

> **When to open:** Adding/fixing tests, running quality checks, or understanding test conventions.

## Test Structure

```text
tests/
├── test_agent.py                          # Agent public API and runtime behavior
├── test_cli.py                            # CLI surface
├── test_config.py                         # Config/env resolution and bootstrap behavior
├── test_context.py                        # Context package public behavior
├── test_acceptance_contracts.py           # Contract-shape checker coverage
├── test_doc_policy_consistency.py         # Doc-policy checker coverage
├── test_lockfile.py                       # requirements.lock validity
├── test_test_runner_script.py             # `./scripts/test.sh` wrapper behavior
├── test_tool.py                           # Public tool facade, schema + policy behavior
├── test_attachments.py / test_budget.py / test_cancel.py
├── test_errors.py / test_package_exports.py / test_redaction.py
├── test_result.py / test_session.py
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
│   │   ├── test_pricing.py
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
├── budget/                                # Budget normalization helpers
├── context/                               # Context-window guard behavior
├── protocols/
│   ├── test_memory.py
│   ├── test_model.py
│   ├── test_planner.py
│   └── test_telemetry.py
└── tool/                                  # Tool-runtime helpers such as idempotency
```

## Commands

| Command | Purpose |
|---|---|
| `./scripts/test.sh <affected-tests>` | Sub-task gate: validate touched behavior in the project `.venv` |
| `ruff check <touched-paths>` | Sub-task gate: lint touched scope |
| `mypy <touched-python-paths>` | Sub-task gate: type-check touched scope |
| `./scripts/test.sh` | Run the full suite |
| `./scripts/test.sh --cov=axis_core` | With coverage |
| `./scripts/test.sh -m "not slow"` | Skip slow tests |
| `./scripts/test.sh tests/engine/test_lifecycle.py` | Single file |
| `ruff check axis_core tests` | Parent-task gate: full lint |
| `mypy axis_core --strict` | Parent-task gate: full type check |

Use the wrapper to avoid broken global pytest plugins and ensure `pytest-asyncio` loads from the
project environment.

## Gate Levels

- **Sub-task gate:** run touched-scope tests/lint/types before marking sub-task complete.
- **Parent-task gate:** run full `./scripts/test.sh`, `ruff check axis_core tests`, and `mypy axis_core --strict` before marking parent complete.

## Testing Rules

**Public-contract testing only.** Allowed surfaces:
- `Agent.run()`, `run_async()`, `stream()`, `stream_async()`
- `LifecycleEngine.execute()`
- Documented lifecycle extension points on `LifecycleEngine` (`_initialize`, `_observe`, `_plan`, `_act`, `_evaluate`, `_finalize`)
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
- **Doc/process checker change** → cover it in `tests/test_doc_policy_consistency.py` or `tests/test_acceptance_contracts.py`
