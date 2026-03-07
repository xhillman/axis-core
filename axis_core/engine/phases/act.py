"""Act phase: execute plan steps with dependency handling (AD-003, AD-042)."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

from axis_core.context import (
    ExecutionResult,
    RunContext,
)
from axis_core.engine.phases.act_model_execution import ActModelExecutionService
from axis_core.engine.phases.act_tool_execution import ActToolExecutionService
from axis_core.errors import (
    AxisError,
    ErrorClass,
    ErrorRecord,
    ToolError,
)
from axis_core.protocols.planner import Plan, PlanStep, StepType
from axis_core.redaction import redact_sensitive_data

if TYPE_CHECKING:
    from axis_core.engine.lifecycle import LifecycleEngine

logger = logging.getLogger("axis_core.engine")


async def act(engine: LifecycleEngine, ctx: RunContext, plan_obj: Plan) -> ExecutionResult:
    """Act phase: execute plan steps with dependency handling.

    Per AD-003, steps execute serially. Per AD-042, independent steps
    continue on failure while dependent steps are skipped.

    Args:
        engine: The lifecycle engine instance
        ctx: Current run context
        plan_obj: Plan to execute

    Returns:
        ExecutionResult with results, errors, and skipped steps
    """
    from axis_core.engine.lifecycle import Phase

    phase_start = time.monotonic()
    await engine._emit(
        "phase_entered",
        run_id=ctx.run_id,
        phase=Phase.ACT.value,
        cycle=ctx.cycle_count,
    )

    results: dict[str, Any] = {}
    errors: dict[str, AxisError] = {}
    skipped: set[str] = set()

    for step in plan_obj.steps:
        # AD-042: Skip if any dependency failed
        if step.dependencies:
            failed_deps = [d for d in step.dependencies if d in errors or d in skipped]
            if failed_deps:
                skipped.add(step.id)
                logger.info(
                    "Skipping step %s: dependencies failed/skipped: %s",
                    step.id,
                    failed_deps,
                )
                continue

        # Execute based on step type
        try:
            if step.type == StepType.TOOL:
                result = await _execute_tool_step(engine, ctx, step)
                results[step.id] = result
            elif step.type == StepType.MODEL:
                result = await _execute_model_step(engine, ctx, step)
                results[step.id] = result
            elif step.type == StepType.TERMINAL:
                # Terminal steps produce the final output
                output = step.payload.get("output", "")
                if output:
                    ctx.state.output = output
                    ctx.state.output_raw = str(output)
                results[step.id] = output
            elif step.type == StepType.TRANSFORM:
                result = step.payload.get("transform_result", step.payload)
                results[step.id] = result
        except Exception as e:
            axis_error = _wrap_error(e, step)
            errors[step.id] = axis_error
            redacted_error = str(redact_sensitive_data(str(axis_error)))

            # Record error in state
            ctx.state.append_error(
                ErrorRecord(
                    error=axis_error,
                    timestamp=datetime.utcnow(),
                    phase=Phase.ACT.value,
                    cycle=ctx.cycle_count,
                    recovered=True,  # We continue execution
                )
            )

            await engine._emit(
                "tool_failed" if step.type == StepType.TOOL else "step_failed",
                run_id=ctx.run_id,
                phase=Phase.ACT.value,
                cycle=ctx.cycle_count,
                step_id=step.id,
                data={"error": redacted_error},
            )

            logger.warning("Step %s failed: %s", step.id, redacted_error)

    execution_result = ExecutionResult(
        results=results,
        errors=errors,
        skipped=frozenset(skipped),
        duration_ms=(time.monotonic() - phase_start) * 1000,
    )

    ctx.state.current_execution = execution_result

    await engine._emit(
        "phase_exited",
        run_id=ctx.run_id,
        phase=Phase.ACT.value,
        cycle=ctx.cycle_count,
        duration_ms=execution_result.duration_ms,
        data={
            "results_count": len(results),
            "errors_count": len(errors),
            "skipped_count": len(skipped),
        },
    )

    return execution_result


async def _execute_tool_step(
    engine: LifecycleEngine,
    ctx: RunContext,
    step: PlanStep,
) -> Any:
    """Execute a single tool step via the dedicated tool execution service."""
    return await ActToolExecutionService(engine, ctx, step).execute()


async def _execute_model_step(
    engine: LifecycleEngine,
    ctx: RunContext,
    step: PlanStep,
) -> Any:
    """Execute a single model step via the dedicated model execution service."""
    return await ActModelExecutionService(engine, ctx, step).execute()


def _wrap_error(e: Exception, step: PlanStep) -> AxisError:
    """Wrap an exception into an appropriate AxisError.

    Args:
        e: Original exception
        step: Step that failed

    Returns:
        Wrapped AxisError
    """
    if isinstance(e, AxisError):
        return e
    if step.type == StepType.TOOL:
        return ToolError(
            message=str(redact_sensitive_data(f"Tool step '{step.id}' failed: {e}")),
            tool_name=step.payload.get("tool"),
            cause=e,
        )
    return AxisError(
        message=str(redact_sensitive_data(f"Step '{step.id}' failed: {e}")),
        error_class=ErrorClass.RUNTIME,
        cause=e,
    )
