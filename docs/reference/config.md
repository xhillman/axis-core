# Config API Reference

This page covers typed configuration objects and the global `config` singleton.

## `Timeouts`

```python
Timeouts(
    observe: float = 10.0,
    plan: float = 30.0,
    act: float = 60.0,
    evaluate: float = 5.0,
    finalize: float = 30.0,
    total: float = 300.0,
)
```

## `RetryPolicy`

```python
RetryPolicy(
    max_attempts: int = 3,
    backoff: str = "exponential",
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    retry_on: list[str] | None = None,
)
```

## `RateLimits`

```python
RateLimits(
    model_calls: str | None = None,
    tool_calls: str | None = None,
    requests: str | None = None,
)
```

`parse_rate(field_name)` converts `count/period` strings into `(count, period_seconds)`.

## `CacheConfig`

```python
CacheConfig(
    enabled: bool = True,
    model_responses: bool = True,
    tool_results: bool = True,
    ttl: int = 3600,
    backend: str = "memory",
    max_size_mb: int = 100,
)
```

## Budget Types

See also `axis_core.budget`:

- `Budget` for limits
- `BudgetState` for consumption tracking and helpers

## `ResolvedConfig`

`ResolvedConfig` is the fully materialized run config passed into runtime execution.

## Global `config` Singleton

`axis_core.config.config` fields:

- `default_model`
- `default_planner`
- `default_memory`
- `anthropic_api_key`
- `openai_api_key`
- `telemetry`
- `verbose`
- `debug`

Methods:

- `reset()` restores values to environment-derived defaults.

## Resolution Order

Runtime uses this order:

- library defaults
- environment-derived config singleton values
- constructor arguments
- per-call runtime overrides
