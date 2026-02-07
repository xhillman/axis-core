"""Observe phase: gather input, load memory, assess state."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

from axis_core.context import Observation, RunContext
from axis_core.protocols.planner import StepType

if TYPE_CHECKING:
    from axis_core.engine.lifecycle import LifecycleEngine

logger = logging.getLogger("axis_core.engine")


async def observe(engine: LifecycleEngine, ctx: RunContext) -> Observation:
    """Observe phase: gather input, load memory, assess state.

    Args:
        engine: The lifecycle engine instance
        ctx: Current run context

    Returns:
        Observation with input, memory context, and prior cycle summaries
    """
    from axis_core.engine.lifecycle import Phase

    phase_start = time.monotonic()
    await engine._emit(
        "phase_entered",
        run_id=ctx.run_id,
        phase=Phase.OBSERVE.value,
        cycle=ctx.cycle_count,
    )

    # Gather memory context
    memory_context: dict[str, Any] = {}
    if engine.memory is not None:
        try:
            results = await engine.memory.search(
                query=ctx.input.text,
                limit=5,
            )
            if results:
                memory_context["relevant_memories"] = [
                    {"key": item.key, "value": item.value}
                    for item in results
                ]
        except Exception:
            logger.warning("Memory search failed during observe", exc_info=True)

    # Summarize previous cycles
    previous_cycles: tuple[dict[str, Any], ...] = tuple(
        {
            "cycle": c.cycle_number,
            "goal": c.plan.goal if c.plan else "",
            "done": c.evaluation.done if c.evaluation else False,
        }
        for c in ctx.state.cycles
    )

    # Check if previous cycle executed tools
    # If it did, we need to call the model again (don't reuse tool_requests)
    previous_cycle_had_tools = False
    if ctx.state._cycles:
        last_cycle = ctx.state._cycles[-1]
        if last_cycle.plan:
            previous_cycle_had_tools = any(
                step.type == StepType.TOOL for step in last_cycle.plan.steps
            )

    # Pull previous model response if available (for subsequent cycles)
    # BUT: If previous cycle executed tools, clear tool_requests so planner calls model again
    last_response = ctx.state.last_model_response
    if previous_cycle_had_tools:
        # Previous cycle executed tools - need to call model again with results
        tool_requests = None
        response = None
    else:
        # Use last response as-is
        tool_requests = last_response.tool_calls if last_response else None
        response = last_response.content if last_response else None

    observation = Observation(
        input=ctx.input,
        memory_context=memory_context,
        previous_cycles=previous_cycles,
        tool_requests=tool_requests,
        response=response,
        timestamp=datetime.utcnow(),
    )

    ctx.state.current_observation = observation

    duration_ms = (time.monotonic() - phase_start) * 1000
    await engine._emit(
        "phase_exited",
        run_id=ctx.run_id,
        phase=Phase.OBSERVE.value,
        cycle=ctx.cycle_count,
        duration_ms=duration_ms,
    )

    return observation
