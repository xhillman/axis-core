from __future__ import annotations

import hashlib
import json
from typing import Any

from axis_core.budget import BudgetState
from axis_core.context import RunState
from axis_core.errors import AxisError, ErrorClass
from axis_core.errors import TimeoutError as AxisTimeoutError
from axis_core.output_schema import coerce_to_output_schema
from axis_core.result import RunResult, RunStats


def build_run_result(
    raw: dict[str, Any],
    duration_ms: float,
    *,
    trace: list[Any] | None = None,
    output_schema: type[Any] | None = None,
) -> RunResult:
    """Convert lifecycle engine raw result dict into a RunResult."""
    budget_state: BudgetState = raw.get("budget_state", BudgetState())
    errors: list[Any] = raw.get("errors", [])

    stats = RunStats(
        cycles=raw.get("cycles_completed", 0),
        tool_calls=budget_state.tool_calls,
        model_calls=budget_state.model_calls,
        input_tokens=budget_state.input_tokens,
        output_tokens=budget_state.output_tokens,
        total_tokens=budget_state.total_tokens,
        cost_usd=budget_state.cost_usd,
        duration_ms=duration_ms,
    )

    error = raw.get("error")
    output = raw.get("output")
    output_raw = raw.get("output_raw", "")
    success = raw.get("success", False)

    if output_schema is not None and success:
        try:
            output = coerce_to_output_schema(
                output=output,
                output_raw=output_raw,
                schema=output_schema,
            )
        except AxisError as schema_error:
            success = False
            error = schema_error
            output = None

    memory_error_str = raw.get("memory_error")
    memory_error: AxisError | None = None
    if memory_error_str:
        memory_error = AxisError(
            message=str(memory_error_str),
            error_class=ErrorClass.RUNTIME,
        )

    return RunResult(
        output=output,
        output_raw=output_raw,
        success=success,
        error=error,
        had_recoverable_errors=any(getattr(item, "recovered", False) for item in errors),
        stats=stats,
        trace=trace or [],
        state=raw.get("state", RunState()),
        run_id=raw.get("run_id", ""),
        memory_error=memory_error,
    )


def build_failure_result(
    error: AxisError,
    duration_ms: float,
    *,
    trace: list[Any] | None = None,
) -> RunResult:
    """Build a failed RunResult when execution aborts before finalize."""
    return RunResult(
        output=None,
        output_raw="",
        success=False,
        error=error,
        had_recoverable_errors=False,
        stats=RunStats(
            cycles=0,
            tool_calls=0,
            model_calls=0,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
            duration_ms=duration_ms,
        ),
        trace=trace or [],
        state=RunState(),
        run_id="",
        memory_error=None,
    )


def config_fingerprint(
    *,
    model: Any,
    tools: dict[str, Any],
    system: str | None,
) -> str:
    """Generate a stable fingerprint of the current agent config."""
    model_id = model if isinstance(model, str) else getattr(model, "model_id", None)
    config_data = {
        "tools": sorted(tools.keys()),
        "system": system,
        "model": model_id,
    }
    canonical = json.dumps(config_data, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def effective_timeout(timeout: float | None, default_timeout: float | None) -> float | None:
    """Resolve runtime timeout using explicit override first."""
    if timeout is not None:
        return timeout
    return default_timeout


def build_timeout_error(
    timeout: float | None,
    *,
    default_timeout: float | None,
) -> AxisTimeoutError:
    """Create a normalized timeout error payload."""
    timeout_seconds = timeout if timeout is not None else default_timeout
    return AxisTimeoutError(
        message=f"Run exceeded timeout of {timeout_seconds:.3f} seconds",
        details={"timeout_seconds": timeout_seconds},
    )
