# Budget and Limits Guide

Use budget and runtime policies to control cost, latency, and failure behavior.

## `Budget`

`Budget` defines hard run limits. Hitting any limit stops execution.

Fields:

- `max_cycles` (default `10`)
- `max_tool_calls` (default `50`)
- `max_model_calls` (default `20`)
- `max_cost_usd` (default `1.00`)
- `max_wall_time_seconds` (default `300.0`)
- `max_input_tokens` (default `None`)
- `max_output_tokens` (default `None`)
- `warn_at_cost_usd` (default `0.80`)

## `BudgetState`

`BudgetState` tracks actual consumption and exposes helpers:

- `total_tokens`
- `is_exhausted(budget)`
- `should_warn(budget)`
- `cost_remaining_usd(budget)`
- `cycles_remaining(budget)`
- `tool_calls_remaining(budget)`
- `model_calls_remaining(budget)`

## `Timeouts`

Defaults (seconds):

- `observe=10.0`
- `plan=30.0`
- `act=60.0`
- `evaluate=5.0`
- `finalize=30.0`
- `total=300.0`

## `RetryPolicy`

Defaults:

- `max_attempts=3`
- `backoff="exponential"`
- `initial_delay=1.0`
- `max_delay=60.0`
- `jitter=True`
- `retry_on=None`

## `RateLimits`

`RateLimits` string format is `count/period` where period is:

- `second`
- `minute`
- `hour`

Examples:

- `"60/minute"`
- `"10/second"`

Fields:

- `model_calls`
- `tool_calls`
- `requests`

## `CacheConfig`

Defaults:

- `enabled=True`
- `model_responses=True`
- `tool_results=True`
- `ttl=3600`
- `backend="memory"`
- `max_size_mb=100`

## Recommended Baseline

```python
from axis_core import Agent, Budget, Timeouts, RetryPolicy, RateLimits, CacheConfig

agent = Agent(
    model="claude-sonnet-4-20250514",
    budget=Budget(max_cost_usd=1.00, max_cycles=8),
    timeouts=Timeouts(total=180.0, act=45.0),
    retry=RetryPolicy(max_attempts=3, backoff="exponential"),
    rate_limits=RateLimits(model_calls="60/minute", tool_calls="30/minute"),
    cache=CacheConfig(enabled=True, ttl=1800),
)
```

## Environment Variable Notes

Global defaults currently come from config/env for:

- `AXIS_DEFAULT_MODEL`
- `AXIS_DEFAULT_PLANNER`
- `AXIS_DEFAULT_MEMORY`
- telemetry and debug flags

Most budget/timeout/retry/rate/cache values are configured through constructor params
and typed config objects.
