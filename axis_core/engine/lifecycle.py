"""Lifecycle engine for axis-core agent execution.

This module implements the core execution loop:
    Initialize → [Observe → Plan → Act → Evaluate]* → Finalize

Phase logic is implemented in per-phase modules under ``axis_core.engine.phases``.
This module orchestrates phase sequencing, adapter resolution, and the main
execution loop.

Architecture Decisions:
- AD-003: Serial tool execution within Act phase
- AD-005: Checkpoint at phase boundaries
- AD-006: Strict plan validation (tools exist, schemas match, deps valid)
- AD-007: Memory persistence failures are non-fatal in Finalize
- AD-028: Cooperative cancellation checked at phase boundaries
- AD-042: Continue independent steps, skip dependent ones on failure
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from enum import Enum
from typing import Any, TypeVar

from axis_core.attachments import AttachmentLike
from axis_core.budget import Budget
from axis_core.cancel import CancelToken
from axis_core.context import (
    CycleState,
    EvalDecision,
    ExecutionResult,
    Observation,
    RunContext,
)
from axis_core.engine.phases.act import act as _act_phase
from axis_core.engine.phases.evaluate import evaluate as _evaluate_phase
from axis_core.engine.phases.evaluate import identify_exhausted_resource
from axis_core.engine.phases.finalize import finalize as _finalize_phase
from axis_core.engine.phases.initialize import initialize as _initialize_phase
from axis_core.engine.phases.observe import observe as _observe_phase
from axis_core.engine.phases.plan import plan as _plan_phase
from axis_core.engine.registry import memory_registry, model_registry, planner_registry
from axis_core.engine.resolver import resolve_adapter
from axis_core.errors import (
    AxisError,
    BudgetError,
    CancelledError,
    ConfigError,
    ErrorClass,
)
from axis_core.protocols.planner import Plan
from axis_core.protocols.telemetry import TraceEvent
from axis_core.redaction import redact_sensitive_data

logger = logging.getLogger("axis_core.engine")
T = TypeVar("T")


class Phase(Enum):
    """Lifecycle execution phases."""

    INITIALIZE = "initialize"
    OBSERVE = "observe"
    PLAN = "plan"
    ACT = "act"
    EVALUATE = "evaluate"
    FINALIZE = "finalize"


class LifecycleEngine:
    """Core execution engine implementing the agent lifecycle.

    Orchestrates the observe→plan→act→evaluate cycle, manages adapters,
    enforces budgets, validates plans, and emits telemetry at phase boundaries.

    Attributes:
        model: LLM model adapter for completions
        memory: Memory adapter for state persistence (optional)
        planner: Planning strategy adapter
        telemetry: List of telemetry sinks
        tools: Dict mapping tool names to callable functions
    """

    def __init__(
        self,
        model: Any,
        planner: Any,
        memory: Any | None = None,
        telemetry: list[Any] | None = None,
        tools: dict[str, Any] | None = None,
        system: str | None = None,
        fallback: list[Any] | None = None,
    ) -> None:
        # Resolve adapters from strings or pass through instances (Task 16.2)
        resolved_model = resolve_adapter(model, model_registry)
        resolved_planner = resolve_adapter(planner, planner_registry)
        resolved_memory = resolve_adapter(memory, memory_registry)

        # Model and planner are required (won't be None after resolution)
        if resolved_model is None:
            raise ConfigError("Model adapter is required")
        if resolved_planner is None:
            raise ConfigError("Planner adapter is required")

        self.model: Any = resolved_model
        self.planner: Any = resolved_planner
        self.memory: Any | None = resolved_memory
        self.telemetry: list[Any] = telemetry or []
        self.tools: dict[str, Any] = tools or {}
        self.system = system
        self._token_callback: Any | None = None

        # Resolve fallback models (Task 15.0)
        self.fallback: list[Any] = []
        if fallback:
            for fallback_model in fallback:
                resolved_fallback = resolve_adapter(fallback_model, model_registry)
                if resolved_fallback is not None:
                    self.fallback.append(resolved_fallback)

    # =========================================================================
    # Telemetry helpers
    # =========================================================================

    async def _emit(
        self,
        event_type: str,
        run_id: str,
        phase: str | None = None,
        cycle: int | None = None,
        step_id: str | None = None,
        data: dict[str, Any] | None = None,
        duration_ms: float | None = None,
    ) -> None:
        """Emit a telemetry event to all sinks."""
        redacted_data = redact_sensitive_data(data or {})
        event_data = (
            redacted_data
            if isinstance(redacted_data, dict)
            else {"value": redacted_data}
        )
        event = TraceEvent(
            type=event_type,
            timestamp=datetime.utcnow(),
            run_id=run_id,
            phase=phase,
            cycle=cycle,
            step_id=step_id,
            data=event_data,
            duration_ms=duration_ms,
        )
        for sink in self.telemetry:
            try:
                await sink.emit(event)
            except Exception:
                logger.warning("Telemetry sink failed to emit event", exc_info=True)

    # =========================================================================
    # Tool manifest extraction
    # =========================================================================

    def _get_tool_manifests(self) -> list[Any]:
        """Extract tool manifests from registered tools.

        Returns protocol-defined ToolManifest objects. The model adapter
        is responsible for converting these to provider-specific formats.

        Returns:
            List of ToolManifest objects (protocol layer)
        """
        if not self.tools:
            return []

        manifests: list[Any] = []

        for tool_name, tool_fn in self.tools.items():
            # Check if tool has manifest (created by @tool decorator)
            if not hasattr(tool_fn, "_axis_manifest"):
                logger.warning(
                    "Tool '%s' missing _axis_manifest, skipping",
                    tool_name,
                )
                continue

            manifest = tool_fn._axis_manifest
            manifests.append(manifest)

        return manifests

    # =========================================================================
    # Phase delegates — thin wrappers for backward compatibility
    # =========================================================================

    async def _initialize(
        self,
        input_text: str,
        agent_id: str,
        budget: Budget,
        context: dict[str, Any] | None = None,
        attachments: list[AttachmentLike] | None = None,
        cancel_token: CancelToken | None = None,
        config: Any | None = None,
    ) -> RunContext:
        """Initialize phase: create RunContext, validate config."""
        return await _initialize_phase(
            engine=self,
            input_text=input_text,
            agent_id=agent_id,
            budget=budget,
            context=context,
            attachments=attachments,
            cancel_token=cancel_token,
            config=config,
        )

    async def _observe(self, ctx: RunContext) -> Observation:
        """Observe phase: gather input, load memory, assess state."""
        return await _observe_phase(engine=self, ctx=ctx)

    async def _plan(self, ctx: RunContext, observation: Observation) -> Plan:
        """Plan phase: call planner, validate plan (AD-006)."""
        return await _plan_phase(engine=self, ctx=ctx, observation=observation)

    async def _act(self, ctx: RunContext, plan: Plan) -> ExecutionResult:
        """Act phase: execute plan steps with dependency handling."""
        return await _act_phase(engine=self, ctx=ctx, plan_obj=plan)

    async def _evaluate(
        self,
        ctx: RunContext,
        plan: Plan,
        execution: ExecutionResult,
    ) -> EvalDecision:
        """Evaluate phase: check termination conditions."""
        return await _evaluate_phase(
            engine=self, ctx=ctx, plan_obj=plan, execution=execution,
        )

    async def _finalize(
        self,
        ctx: RunContext,
        error: Exception | None = None,
    ) -> dict[str, Any]:
        """Finalize phase: persist memory, emit summary, clean up."""
        return await _finalize_phase(engine=self, ctx=ctx, error=error)

    @staticmethod
    def _update_wall_time(ctx: RunContext, run_started_monotonic: float) -> None:
        """Refresh tracked wall-clock budget consumption."""
        elapsed = max(0.0, time.monotonic() - run_started_monotonic)
        ctx.state.budget_state.wall_time_seconds = elapsed

    @staticmethod
    def _wall_time_budget_error(ctx: RunContext) -> BudgetError:
        """Create a wall-time budget exhaustion error."""
        return BudgetError(
            message="Budget exhausted: wall_time",
            resource="wall_time",
            used=ctx.state.budget_state.wall_time_seconds,
            limit=ctx.budget.max_wall_time_seconds,
        )

    # =========================================================================
    # Main execution loop (7.8)
    # =========================================================================

    async def execute(
        self,
        input_text: str,
        agent_id: str,
        budget: Budget,
        context: dict[str, Any] | None = None,
        attachments: list[AttachmentLike] | None = None,
        cancel_token: CancelToken | None = None,
        config: Any | None = None,
        token_callback: Any | None = None,
    ) -> dict[str, Any]:
        """Execute the full lifecycle: Initialize → [Observe→Plan→Act→Evaluate]* → Finalize.

        Args:
            input_text: User input text
            agent_id: Agent identifier
            budget: Budget limits
            context: Optional context dict
            attachments: Optional attachments
            cancel_token: Optional cancellation token
            config: Optional resolved config
            token_callback: Optional callback for streaming model tokens

        Returns:
            Result dict from _finalize
        """
        self._token_callback = token_callback
        run_started_monotonic = time.monotonic()
        try:
            # Initialize
            ctx = await self._initialize(
                input_text=input_text,
                agent_id=agent_id,
                budget=budget,
                context=context,
                attachments=attachments,
                cancel_token=cancel_token,
                config=config,
            )
            self._update_wall_time(ctx, run_started_monotonic)

            async def _run_with_wall_budget(
                operation: Callable[[], Awaitable[T]],
            ) -> T:
                self._update_wall_time(ctx, run_started_monotonic)
                remaining = (
                    ctx.budget.max_wall_time_seconds
                    - ctx.state.budget_state.wall_time_seconds
                )
                if remaining <= 0:
                    raise self._wall_time_budget_error(ctx)
                try:
                    result = await asyncio.wait_for(operation(), timeout=remaining)
                except asyncio.TimeoutError as e:
                    self._update_wall_time(ctx, run_started_monotonic)
                    raise self._wall_time_budget_error(ctx) from e
                self._update_wall_time(ctx, run_started_monotonic)
                return result

            await self._emit(
                "run_started",
                run_id=ctx.run_id,
                data={"agent_id": agent_id},
            )

            termination_error: Exception | None = None

            try:
                # Main cycle loop
                while True:
                    cycle_start = time.monotonic()
                    self._update_wall_time(ctx, run_started_monotonic)
                    await self._emit(
                        "cycle_started",
                        run_id=ctx.run_id,
                        cycle=ctx.cycle_count,
                    )

                    # Check cancellation at cycle start (AD-028)
                    if ctx.cancel_token and ctx.cancel_token.is_cancelled:
                        from axis_core.engine.phases.evaluate import _cancel_reason

                        termination_error = CancelledError(
                            message=_cancel_reason(ctx.cancel_token)
                        )
                        break

                    # Check budget before starting cycle
                    if ctx.state.budget_state.is_exhausted(ctx.budget):
                        resource = identify_exhausted_resource(ctx)
                        termination_error = BudgetError(
                            message=f"Budget exhausted: {resource}",
                            resource=resource,
                        )
                        break

                    cycle_start_time = datetime.utcnow()

                    # Observe
                    observation = await _run_with_wall_budget(
                        lambda: self._observe(ctx)
                    )
                    if ctx.state.budget_state.is_exhausted(ctx.budget):
                        resource = identify_exhausted_resource(ctx)
                        termination_error = BudgetError(
                            message=f"Budget exhausted: {resource}",
                            resource=resource,
                        )
                        break

                    # Plan
                    plan = await _run_with_wall_budget(
                        lambda: self._plan(ctx, observation)
                    )
                    if ctx.state.budget_state.is_exhausted(ctx.budget):
                        resource = identify_exhausted_resource(ctx)
                        termination_error = BudgetError(
                            message=f"Budget exhausted: {resource}",
                            resource=resource,
                        )
                        break

                    # Act
                    execution = await _run_with_wall_budget(
                        lambda: self._act(ctx, plan)
                    )
                    if ctx.state.budget_state.is_exhausted(ctx.budget):
                        resource = identify_exhausted_resource(ctx)
                        termination_error = BudgetError(
                            message=f"Budget exhausted: {resource}",
                            resource=resource,
                        )
                        break

                    # Evaluate
                    decision = await _run_with_wall_budget(
                        lambda: self._evaluate(ctx, plan, execution)
                    )
                    if ctx.state.budget_state.is_exhausted(ctx.budget):
                        resource = identify_exhausted_resource(ctx)
                        decision = EvalDecision(
                            done=True,
                            error=BudgetError(
                                message=f"Budget exhausted: {resource}",
                                resource=resource,
                            ),
                            reason=f"Budget exhausted: {resource}",
                        )

                    cycle_end_time = datetime.utcnow()

                    # Record completed cycle
                    cycle_state = CycleState(
                        cycle_number=ctx.cycle_count,
                        observation=observation,
                        plan=plan,
                        execution=execution,
                        evaluation=decision,
                        started_at=cycle_start_time,
                        ended_at=cycle_end_time,
                    )
                    ctx.state.append_cycle(cycle_state)

                    # Increment cycle count and budget
                    ctx.cycle_count += 1
                    ctx.state.budget_state.cycles += 1

                    cycle_duration_ms = (time.monotonic() - cycle_start) * 1000
                    await self._emit(
                        "cycle_completed",
                        run_id=ctx.run_id,
                        cycle=ctx.cycle_count - 1,
                        duration_ms=cycle_duration_ms,
                        data={"done": decision.done},
                    )

                    if decision.done:
                        termination_error = decision.error
                        break

                    # Reset current-cycle state for next cycle
                    ctx.state.current_observation = None
                    ctx.state.current_plan = None
                    ctx.state.current_execution = None

            except AxisError as e:
                termination_error = e
            except Exception as e:
                termination_error = AxisError(
                    message=f"Unexpected error: {e}",
                    error_class=ErrorClass.RUNTIME,
                    cause=e,
                )

            # Finalize (always runs)
            self._update_wall_time(ctx, run_started_monotonic)
            result = await self._finalize(ctx, error=termination_error)
            self._update_wall_time(ctx, run_started_monotonic)

            event_type = "run_completed" if result["success"] else "run_failed"
            await self._emit(
                event_type,
                run_id=ctx.run_id,
                data={
                    "success": result["success"],
                    "cycles": result["cycles_completed"],
                },
            )

            return result
        finally:
            self._token_callback = None


__all__ = [
    "LifecycleEngine",
    "Phase",
]
