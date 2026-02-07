"""Plan phase: call planner, validate plan (AD-006)."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from axis_core.context import Observation, RunContext
from axis_core.errors import PlanError
from axis_core.protocols.planner import Plan, StepType

if TYPE_CHECKING:
    from axis_core.engine.lifecycle import LifecycleEngine

logger = logging.getLogger("axis_core.engine")


async def plan(engine: LifecycleEngine, ctx: RunContext, observation: Observation) -> Plan:
    """Plan phase: call planner, validate plan (AD-006).

    Args:
        engine: The lifecycle engine instance
        ctx: Current run context
        observation: Current observation

    Returns:
        Validated Plan

    Raises:
        PlanError: If plan validation fails (unknown tools, invalid deps)
    """
    from axis_core.engine.lifecycle import Phase

    phase_start = time.monotonic()
    await engine._emit(
        "phase_entered",
        run_id=ctx.run_id,
        phase=Phase.PLAN.value,
        cycle=ctx.cycle_count,
    )

    # Provide tool information to planners via context
    # (AutoPlanner uses this for LLM-based planning)
    tool_descriptions: dict[str, Any] = {}
    for tool_name, tool_fn in engine.tools.items():
        if hasattr(tool_fn, "_axis_manifest"):
            manifest = tool_fn._axis_manifest
            tool_descriptions[tool_name] = manifest.description
    ctx.context["__tools__"] = tool_descriptions

    result: Plan = await engine.planner.plan(observation, ctx)

    # AD-016: Emit telemetry if planner fell back to sequential
    if result.metadata.get("fallback"):
        await engine._emit(
            "planner_fallback",
            run_id=ctx.run_id,
            phase=Phase.PLAN.value,
            cycle=ctx.cycle_count,
            data={
                "original_planner": result.metadata.get("planner", "unknown"),
                "reason": result.metadata.get("fallback_reason", "unknown"),
            },
        )

    # AD-006: Strict plan validation
    _validate_plan(result, engine.tools)

    ctx.state.current_plan = result

    duration_ms = (time.monotonic() - phase_start) * 1000
    await engine._emit(
        "phase_exited",
        run_id=ctx.run_id,
        phase=Phase.PLAN.value,
        cycle=ctx.cycle_count,
        duration_ms=duration_ms,
        data={"plan_id": result.id, "step_count": len(result.steps)},
    )

    return result


def _validate_plan(plan_obj: Plan, tools: dict[str, Any]) -> None:
    """Validate plan per AD-006: all tools exist, dependencies valid.

    Args:
        plan_obj: Plan to validate
        tools: Available tools dict

    Raises:
        PlanError: If validation fails
    """
    step_ids = {step.id for step in plan_obj.steps}

    for step in plan_obj.steps:
        # Validate tool steps reference existing tools
        if step.type == StepType.TOOL:
            tool_name = step.payload.get("tool")
            if tool_name and tool_name not in tools:
                raise PlanError(
                    message=(
                        f"Plan validation failed: tool '{tool_name}' not found. "
                        f"Available tools: {list(tools.keys())}"
                    ),
                )

        # Validate dependencies reference existing step IDs
        if step.dependencies:
            for dep_id in step.dependencies:
                if dep_id not in step_ids:
                    raise PlanError(
                        message=(
                            f"Plan validation failed: invalid dependency "
                            f"'{dep_id}' in step '{step.id}'. "
                            f"Available step IDs: {list(step_ids)}"
                        ),
                    )
