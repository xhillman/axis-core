# Errors API Reference

axis-core exposes a structured error system rooted at `AxisError`.

## `ErrorClass`

Enum values:

- `INPUT`
- `CONFIG`
- `PLAN`
- `TOOL`
- `MODEL`
- `BUDGET`
- `TIMEOUT`
- `CANCELLED`
- `RUNTIME`

## Base Type: `AxisError`

Fields:

- `message: str`
- `error_class: ErrorClass`
- `phase: str | None`
- `cycle: int | None`
- `step_id: str | None`
- `recoverable: bool`
- `retry_after: float | None`
- `details: dict[str, object]`
- `cause: Exception | None`

## Core Error Types

- `InputError`
- `ConfigError`
- `PlanError`
- `ToolError` (includes `tool_name`)
- `ModelError` (includes `model_id`)
- `BudgetError` (includes `resource`, `used`, `limit`)
- `TimeoutError`
- `CancelledError`
- `ConcurrencyError` (includes version mismatch context)

## Recoverability

`ModelError.from_exception()` classifies many transient provider errors as recoverable,
which enables retry/fallback behavior.

Recoverability is surfaced via:

- `error.recoverable`
- `error.retry_after` (when available)

## `BudgetError` Suggestions

`BudgetError` enriches error text with an actionable suggestion when usage and limit
context are available.

## Error Records

`ErrorRecord` stores immutable error history entries with:

- `error`
- `timestamp`
- `phase`
- `cycle`
- `recovered`
