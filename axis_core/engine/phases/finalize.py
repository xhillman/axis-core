"""Finalize phase: persist memory, emit summary, clean up (AD-007)."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from axis_core.context import RunContext

if TYPE_CHECKING:
    from axis_core.engine.lifecycle import LifecycleEngine

logger = logging.getLogger("axis_core.engine")


async def finalize(
    engine: LifecycleEngine,
    ctx: RunContext,
    error: Exception | None = None,
) -> dict[str, Any]:
    """Finalize phase: persist memory, emit summary, clean up.

    Per AD-007, memory persistence failures are non-fatal. The run
    succeeds but the memory_error field is populated.

    Args:
        engine: The lifecycle engine instance
        ctx: Current run context
        error: Error that caused termination (if any)

    Returns:
        Result dict with output, success, stats, and optional memory_error
    """
    from axis_core.engine.lifecycle import Phase

    phase_start = time.monotonic()
    await engine._emit(
        "phase_entered",
        run_id=ctx.run_id,
        phase=Phase.FINALIZE.value,
    )

    memory_error: str | None = None

    # Persist to memory (AD-007: non-fatal on failure)
    if engine.memory is not None:
        try:
            await engine.memory.store(
                key=f"run:{ctx.run_id}:output",
                value=ctx.state.output,
                metadata={
                    "agent_id": ctx.agent_id,
                    "run_id": ctx.run_id,
                    "cycles": ctx.cycle_count,
                },
            )
        except Exception as e:
            memory_error = str(e)
            logger.warning(
                "Memory persistence failed during finalize: %s", e, exc_info=True
            )

    # Build result
    success = error is None and ctx.state.output is not None
    result: dict[str, Any] = {
        "output": ctx.state.output,
        "output_raw": ctx.state.output_raw,
        "success": success,
        "error": error,
        "memory_error": memory_error,
        "run_id": ctx.run_id,
        "cycles_completed": ctx.cycle_count,
        "budget_state": ctx.state.budget_state,
        "errors": ctx.state.errors,
        "state": ctx.state,  # Include full state for debugging/replay
    }

    # Flush and close telemetry
    for sink in engine.telemetry:
        try:
            await sink.flush()
            await sink.close()
        except Exception:
            logger.warning("Telemetry sink cleanup failed", exc_info=True)

    duration_ms = (time.monotonic() - phase_start) * 1000
    await engine._emit(
        "phase_exited",
        run_id=ctx.run_id,
        phase=Phase.FINALIZE.value,
        duration_ms=duration_ms,
    )

    return result
