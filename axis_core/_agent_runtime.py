from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

from axis_core.budget import Budget, BudgetState
from axis_core.config import (
    CacheConfig,
    RateLimits,
    ResolvedConfig,
    RetryPolicy,
    Timeouts,
    ToolPolicy,
)
from axis_core.context import RunState
from axis_core.errors import AxisError, ErrorClass
from axis_core.errors import TimeoutError as AxisTimeoutError
from axis_core.output_schema import coerce_to_output_schema
from axis_core.result import RunResult, RunStats

logger = logging.getLogger("axis_core.agent")


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


def resolve_runtime_config(
    *,
    model: Any,
    planner: Any,
    memory: Any,
    budget: Budget,
    timeouts: Timeouts,
    rate_limits: RateLimits | None,
    retry: RetryPolicy | None,
    cache: CacheConfig | None,
    tool_policy: ToolPolicy | None,
    confirmation_handler: Any,
    telemetry_enabled: bool,
    verbose: bool,
) -> ResolvedConfig:
    """Build the runtime config passed into the lifecycle engine."""
    raw_context_strategy = os.getenv("AXIS_CONTEXT_STRATEGY")
    context_strategy = coerce_context_strategy(raw_context_strategy)
    if raw_context_strategy is not None and context_strategy is None:
        logger.warning(
            "Invalid AXIS_CONTEXT_STRATEGY='%s'; falling back to 'smart'",
            raw_context_strategy,
        )
        context_strategy = "smart"
    if context_strategy is None:
        context_strategy = "smart"

    raw_max_cycle_context = os.getenv("AXIS_MAX_CYCLE_CONTEXT")
    max_cycle_context = coerce_env_non_negative_int(raw_max_cycle_context)
    if raw_max_cycle_context is not None and max_cycle_context is None:
        logger.warning(
            "Invalid AXIS_MAX_CYCLE_CONTEXT='%s'; falling back to 5",
            raw_max_cycle_context,
        )
        max_cycle_context = 5
    if max_cycle_context is None:
        max_cycle_context = 5

    transcript_strict = coerce_env_bool(
        os.getenv("AXIS_TRANSCRIPT_STRICT"),
        default=False,
    )
    max_tool_result_chars = coerce_env_positive_int(
        os.getenv("AXIS_MAX_TOOL_RESULT_CHARS")
    )
    context_guard_enabled = coerce_env_bool(
        os.getenv("AXIS_CONTEXT_GUARD_ENABLED"),
        default=False,
    )
    context_window_tokens = coerce_env_positive_int(
        os.getenv("AXIS_CONTEXT_WINDOW_TOKENS")
    )
    context_warn_tokens = (
        coerce_env_positive_int(os.getenv("AXIS_CONTEXT_GUARD_WARN_TOKENS"))
        or 32_000
    )
    context_block_tokens = (
        coerce_env_positive_int(os.getenv("AXIS_CONTEXT_GUARD_BLOCK_TOKENS"))
        or 16_000
    )
    context_pruning_enabled = coerce_env_bool(
        os.getenv("AXIS_CONTEXT_PRUNE_ENABLED"),
        default=False,
    )

    return ResolvedConfig(
        model=model,
        planner=planner,
        memory=memory,
        budget=budget,
        timeouts=timeouts,
        rate_limits=rate_limits,
        retry=retry,
        cache=cache,
        context_strategy=context_strategy,
        max_cycle_context=max_cycle_context,
        transcript_strict=transcript_strict,
        max_tool_result_chars=max_tool_result_chars,
        context_window_guard_enabled=context_guard_enabled,
        context_window_tokens=context_window_tokens,
        context_window_warn_tokens=context_warn_tokens,
        context_window_block_tokens=context_block_tokens,
        context_pruning_enabled=context_pruning_enabled,
        tool_policy=tool_policy,
        confirmation_handler=confirmation_handler,
        telemetry_enabled=telemetry_enabled,
        verbose=verbose,
    )


def coerce_env_bool(value: str | None, *, default: bool = False) -> bool:
    """Coerce boolean env-var style values."""
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def coerce_env_positive_int(value: str | None) -> int | None:
    """Coerce positive integer env-var values."""
    if value is None:
        return None
    try:
        parsed = int(value.strip())
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def coerce_env_non_negative_int(value: str | None) -> int | None:
    """Coerce non-negative integer env-var values."""
    if value is None:
        return None
    try:
        parsed = int(value.strip())
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def coerce_context_strategy(value: str | None) -> str | None:
    """Validate context strategy values read from the environment."""
    if value is None:
        return None
    candidate = value.strip().lower()
    if candidate in {"smart", "full", "minimal"}:
        return candidate
    return None
