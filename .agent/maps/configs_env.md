# Configuration & Environment Map

> **When to open:** Changing config loading, budget defaults, environment variables, or timeouts.

## Key Files

| File | Responsibility |
|---|---|
| `axis_core/config.py` | Config singleton, dataclasses, env loading, runtime defaults |
| `axis_core/budget.py` | `Budget` limits and `BudgetState` tracking |
| `.env.example` | Supported environment variables |
| `pyproject.toml` | Package metadata, optional extras, ruff/mypy config |
| `requirements.lock` | Pinned deps for reproducible installs |

## Config Resolution Order

`defaults → env vars → constructor args → runtime args`

Implemented via `deep_merge()` in `config.py`.

## Config Dataclasses

```text
Config (singleton: axis_core.config.config)
├── ResolvedConfig
├── Timeouts
├── RetryPolicy
├── RateLimits
└── CacheConfig
```

## Environment Variables (key groups)

| Prefix | Variables | Loaded By |
|---|---|---|
| `AXIS_DEFAULT_*` | `MODEL`, `MEMORY`, `PLANNER` | `config.py` |
| `AXIS_MAX_*` | `CYCLES`, `TOOL_CALLS`, `MODEL_CALLS`, `TOKENS`, `COST_USD`, `WALL_TIME` | `config.py` → `Budget` |
| `AXIS_TIMEOUT_*` | per-phase and total timeouts | `config.py` → `Timeouts` |
| `AXIS_RETRY_*` | retry ceilings/backoff | `config.py` → `RetryPolicy` |
| `AXIS_RATE_*` | request rate limits | `config.py` → `RateLimits` |
| `AXIS_CACHE_*` | cache TTL and sizing | `config.py` → `CacheConfig` |
| `AXIS_TELEMETRY`, `AXIS_VERBOSE`, `AXIS_DEBUG` | runtime toggles | `config.py` |
| `*_API_KEY` | `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` | Model adapters directly |

## Budget System

```python
Budget(
    max_cycles=10,
    max_tool_calls=50,
    max_model_calls=20,
    max_tokens=100_000,
    max_cost_usd=1.0,
    max_wall_time=300.0,
)
```

`BudgetState` tracks the mutable counters. Budget exhaustion is enforced in
`axis_core/engine/phases/evaluate.py`.

## Common Change Patterns

- **New env var** → add to `.env.example`, load it in `config.py`, and update user-facing docs/guidance
- **New budget limit** → update `Budget`, `BudgetState`, and evaluation checks
- **New timeout** → update `Timeouts` plus the relevant phase/runtime call site
- **Dependency change** → update `pyproject.toml` and regenerate `requirements.lock`

## Sharp Edges

- `config` is a module-level singleton
- `.env` is loaded via `python-dotenv` at import time
- `budget.py` stores data; enforcement lives in `evaluate.py`
- `tests/test_lockfile.py` covers lockfile consistency
