"""Evaluate phase: check termination conditions."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from axis_core.context import EvalDecision, ExecutionResult, RunContext
from axis_core.errors import BudgetError, CancelledError
from axis_core.protocols.planner import Plan, StepType

if TYPE_CHECKING:
    from axis_core.engine.lifecycle import LifecycleEngine


async def evaluate(
    engine: LifecycleEngine,
    ctx: RunContext,
    plan_obj: Plan,
    execution: ExecutionResult,
) -> EvalDecision:
    """Evaluate phase: check termination conditions.

    Checks (in order):
    1. Cancellation
    2. Terminal plan step completed
    3. Budget exhaustion
    4. Unrecoverable errors

    Args:
        engine: The lifecycle engine instance
        ctx: Current run context
        plan_obj: Current plan
        execution: Execution results

    Returns:
        EvalDecision indicating whether to continue or stop
    """
    from axis_core.engine.lifecycle import Phase

    phase_start = time.monotonic()
    await engine._emit(
        "phase_entered",
        run_id=ctx.run_id,
        phase=Phase.EVALUATE.value,
        cycle=ctx.cycle_count,
    )

    decision: EvalDecision

    # 1. Check cancellation
    if ctx.cancel_token and ctx.cancel_token.is_cancelled:
        reason = _cancel_reason(ctx.cancel_token)
        decision = EvalDecision(
            done=True,
            error=CancelledError(message=reason),
            reason=f"Cancelled: {reason}",
        )
    # 2. Check for terminal step
    elif _has_terminal_step(plan_obj):
        decision = EvalDecision(
            done=True,
            reason="Plan completed with terminal step",
        )
    # 3. Check budget
    elif ctx.state.budget_state.is_exhausted(ctx.budget):
        resource = identify_exhausted_resource(ctx)
        decision = EvalDecision(
            done=True,
            error=BudgetError(
                message=f"Budget exhausted: {resource}",
                resource=resource,
            ),
            reason=f"Budget exhausted: {resource}",
        )
    # 4. Check for all-error execution
    elif execution.errors and not execution.results:
        decision = EvalDecision(
            done=True,
            error=list(execution.errors.values())[0],
            recoverable=False,
            reason="All steps failed",
        )
    else:
        # Continue cycling
        decision = EvalDecision(
            done=False,
            reason="Continue to next cycle",
        )

    duration_ms = (time.monotonic() - phase_start) * 1000
    await engine._emit(
        "phase_exited",
        run_id=ctx.run_id,
        phase=Phase.EVALUATE.value,
        cycle=ctx.cycle_count,
        duration_ms=duration_ms,
        data={"done": decision.done, "reason": decision.reason},
    )

    return decision


def _cancel_reason(cancel_token: Any) -> str:
    """Extract cancellation reason from a CancelToken."""
    return (
        getattr(cancel_token, "reason", None)
        or getattr(cancel_token, "_reason", None)
        or "Cancelled"
    )


def _has_terminal_step(plan_obj: Plan) -> bool:
    """Check if plan contains a TERMINAL step."""
    return any(step.type == StepType.TERMINAL for step in plan_obj.steps)


def identify_exhausted_resource(ctx: RunContext) -> str:
    """Identify which budget resource was exhausted."""
    bs = ctx.state.budget_state
    b = ctx.budget
    if bs.cycles >= b.max_cycles:
        return "cycles"
    if bs.cost_usd >= b.max_cost_usd:
        return "cost_usd"
    if bs.tool_calls >= b.max_tool_calls:
        return "tool_calls"
    if bs.model_calls >= b.max_model_calls:
        return "model_calls"
    if b.max_input_tokens is not None and bs.input_tokens >= b.max_input_tokens:
        return "input_tokens"
    if b.max_output_tokens is not None and bs.output_tokens >= b.max_output_tokens:
        return "output_tokens"
    if bs.wall_time_seconds >= b.max_wall_time_seconds:
        return "wall_time"
    return "unknown"
