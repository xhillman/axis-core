# Configuration & Environment Map

> **When to open:** Changing config loading, budget defaults, environment variables, or timeouts.

## Key Files

| File | Lines | Responsibility |
|---|---|---|
| `axis_core/config.py` | 320 | Config singleton, dataclasses, env var loading, `deep_merge()` |
| `axis_core/budget.py` | 115 | `Budget` (limits), `BudgetState` (tracking) |
| `.env.example` | 188 | All supported environment variables (documented) |
| `pyproject.toml` | 85 | Package metadata, dependencies, tool config (ruff, mypy) |
| `requirements.lock` | — | Pinned deps for reproducible builds |

## Config Resolution Order

`defaults → env vars → constructor args → runtime args`

Implemented via `deep_merge()` (AD-015) in `config.py`.

## Config Dataclasses

```
Config (singleton: axis_core.config.config)
├── ResolvedConfig        # Fully merged config for a run
├── Timeouts              # Per-phase + total timeouts
├── RetryPolicy           # Max retries, backoff settings
├── RateLimits            # Requests per minute/hour
└── CacheConfig           # Cache TTL, max size
```

## Environment Variables (key groups)

| Prefix | Variables | Loaded By |
|---|---|---|
| `AXIS_DEFAULT_*` | `MODEL`, `MEMORY`, `PLANNER` | `config.py` |
| `AXIS_MAX_*` | `CYCLES`, `TOOL_CALLS`, `MODEL_CALLS`, `TOKENS`, `COST_USD`, `WALL_TIME` | `config.py` → `Budget` |
| `AXIS_TIMEOUT_*` | `OBSERVE`, `PLAN`, `ACT`, `EVALUATE`, `FINALIZE`, `TOTAL` | `config.py` → `Timeouts` |
| `AXIS_RETRY_*` | `MAX_RETRIES`, `BACKOFF_BASE`, `BACKOFF_MAX` | `config.py` → `RetryPolicy` |
| `AXIS_RATE_*` | `RPM`, `RPH` | `config.py` → `RateLimits` |
| `AXIS_CACHE_*` | `TTL`, `MAX_SIZE` | `config.py` → `CacheConfig` |
| `AXIS_CONTEXT_*` | `STRATEGY`, `MAX_CYCLE_CONTEXT` | `config.py` |
| `AXIS_TELEMETRY_*` | `SINK`, `REDACT`, `PERSIST_SENSITIVE`, `BUFFER_MODE` | `config.py` |
| `*_API_KEY` | `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` | Model adapters directly |
| `AXIS_DEBUG` | Debug mode | `config.py` |

## Budget System

```python
Budget(
    max_cycles=10,           # Lifecycle loop iterations
    max_tool_calls=50,       # Total tool invocations
    max_model_calls=20,      # Total model API calls
    max_tokens=100_000,      # Total tokens (input + output)
    max_cost_usd=1.0,        # Cost ceiling
    max_wall_time=300.0,     # Wall clock seconds
)

BudgetState  # Mutable tracking: cycles_used, tokens_used, cost_usd, etc.
```

Budget checked in `phases/evaluate.py` → `identify_exhausted_resource()`.

## Common Change Patterns

- **New env var** → add to `.env.example` + `config.py` loader + document in CLAUDE.md
- **New budget limit** → add to `Budget` + `BudgetState` + `evaluate.py` check
- **New timeout** → add to `Timeouts` dataclass + relevant phase module
- **Dependency change** → update `pyproject.toml` → regenerate `requirements.lock`

## Sharp Edges

- `config` is a module-level singleton — imported as `from axis_core.config import config`
- `.env` is loaded via `python-dotenv` at import time
- Budget enforcement is in `evaluate.py`, NOT in `budget.py` (budget.py is just data)
- `requirements.lock` validity is tested by `tests/test_lockfile.py`
