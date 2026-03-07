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

import inspect
import logging
import time
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Any

from axis_core.attachments import AttachmentLike
from axis_core.budget import Budget
from axis_core.cancel import CancelToken
from axis_core.checkpoint import (
    CheckpointResumeState,
    create_checkpoint,
    prepare_checkpoint_resume,
)
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
from axis_core.engine.runtime_policy import LifecycleRuntimePolicyServices
from axis_core.errors import (
    AxisError,
    BudgetError,
    CancelledError,
    ConfigError,
    ErrorClass,
)
from axis_core.errors import (
    TimeoutError as AxisTimeoutError,
)
from axis_core.protocols.planner import Plan
from axis_core.protocols.telemetry import TraceEvent
from axis_core.redaction import redact_sensitive_data

logger = logging.getLogger("axis_core.engine")


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
        checkpoint_handler: Callable[[dict[str, Any]], Any] | None = None,
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
        self._checkpoint_handler = checkpoint_handler

        # Resolve fallback models (Task 15.0)
        self.fallback: list[Any] = []
        if fallback:
            for fallback_model in fallback:
                resolved_fallback = resolve_adapter(fallback_model, model_registry)
                if resolved_fallback is not None:
                    self.fallback.append(resolved_fallback)

        # Runtime execution policy state (Task 17.0)
        self._runtime_policies = LifecycleRuntimePolicyServices()
        self._tools_missing_manifest_warned: set[str] = set()

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
                if tool_name not in self._tools_missing_manifest_warned:
                    logger.warning(
                        "Tool '%s' missing _axis_manifest, skipping",
                        tool_name,
                    )
                    self._tools_missing_manifest_warned.add(tool_name)
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

    def _configure_runtime_policies(self, config: Any | None) -> None:
        """Resolve active timeout/retry/rate-limit/cache policies for this run."""
        self._runtime_policies.configure(config, tools=self.tools)

    async def acquire_model_slot(
        self,
        ctx: RunContext,
        step_id: str | None = None,
    ) -> None:
        """Apply model rate-limit token acquisition if configured."""
        await self._runtime_policies.rate_limits.acquire_model_slot(
            emit=self._emit,
            ctx=ctx,
            step_id=step_id,
        )

    async def acquire_tool_slot(
        self,
        ctx: RunContext,
        tool_name: str,
        step_id: str | None = None,
    ) -> None:
        """Apply global and per-tool rate-limit token acquisition if configured."""
        await self._runtime_policies.rate_limits.acquire_tool_slot(
            emit=self._emit,
            ctx=ctx,
            tool_name=tool_name,
            step_id=step_id,
        )

    def cache_enabled_for_models(self) -> bool:
        """Whether model response cache is active."""
        return self._runtime_policies.cache.enabled_for_models()

    def cache_enabled_for_tools(self) -> bool:
        """Whether tool result cache is active."""
        return self._runtime_policies.cache.enabled_for_tools()

    def default_cache_ttl_seconds(self) -> int:
        """Default cache TTL from active config."""
        return self._runtime_policies.cache.default_ttl_seconds()

    def compute_cache_key(self, namespace: str, payload: dict[str, Any]) -> str:
        """Compute deterministic cache key for a namespace + payload."""
        return self._runtime_policies.cache.compute_key(namespace, payload)

    def cache_get(self, key: str) -> tuple[bool, Any]:
        """Get cache entry by key, evicting expired entries."""
        return self._runtime_policies.cache.get(key)

    def cache_set(
        self,
        key: str,
        value: Any,
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        """Set cache entry with TTL and max-size eviction."""
        self._runtime_policies.cache.set(key, value, ttl_seconds=ttl_seconds)

    @staticmethod
    def _build_failed_result(ctx: RunContext, error: Exception) -> dict[str, Any]:
        """Build fallback result when finalize fails or times out."""
        return {
            "output": ctx.state.output,
            "output_raw": ctx.state.output_raw,
            "success": False,
            "error": error,
            "memory_error": None,
            "run_id": ctx.run_id,
            "cycles_completed": ctx.cycle_count,
            "budget_state": ctx.state.budget_state,
            "errors": ctx.state.errors,
            "state": ctx.state,
        }

    async def _cleanup_telemetry(self) -> None:
        """Best-effort telemetry flush/close outside finalize phase."""
        for sink in self.telemetry:
            try:
                await sink.flush()
                await sink.close()
            except Exception:
                logger.warning("Telemetry sink cleanup failed", exc_info=True)

    async def _persist_checkpoint(
        self,
        ctx: RunContext,
        *,
        phase: Phase,
        next_phase: Phase | None,
    ) -> None:
        """Persist a checkpoint envelope if a handler is configured."""
        if self._checkpoint_handler is None:
            return

        checkpoint = create_checkpoint(
            ctx,
            phase=phase.value,
            next_phase=next_phase.value if next_phase else None,
        )
        try:
            result = self._checkpoint_handler(checkpoint)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.warning(
                "Checkpoint persistence failed at phase '%s'",
                phase.value,
                exc_info=True,
            )

    @staticmethod
    def _budget_exhaustion_error(ctx: RunContext) -> BudgetError | None:
        """Return a budget error when the run budget is exhausted."""
        if not ctx.state.budget_state.is_exhausted(ctx.budget):
            return None

        resource = identify_exhausted_resource(ctx)
        return BudgetError(
            message=f"Budget exhausted: {resource}",
            resource=resource,
        )

    @staticmethod
    def _cycle_boundary_error(ctx: RunContext) -> Exception | None:
        """Return cancellation or budget errors checked at cycle boundaries."""
        if ctx.cancel_token and ctx.cancel_token.is_cancelled:
            from axis_core.engine.phases.evaluate import _cancel_reason

            return CancelledError(message=_cancel_reason(ctx.cancel_token))

        return LifecycleEngine._budget_exhaustion_error(ctx)

    # =========================================================================
    # Main execution loop (7.8)
    # =========================================================================

    async def _execute_from_context(
        self,
        ctx: RunContext,
        *,
        run_started_monotonic: float,
        resume_state: CheckpointResumeState | None = None,
    ) -> dict[str, Any]:
        """Continue lifecycle execution from a prepared RunContext."""
        self._update_wall_time(ctx, run_started_monotonic)

        await self._emit(
            "run_started",
            run_id=ctx.run_id,
            data={"agent_id": ctx.agent_id},
        )

        termination_error: Exception | None = None

        try:
            resume_state = resume_state or CheckpointResumeState()

            while True:
                cycle_start = time.monotonic()
                self._update_wall_time(ctx, run_started_monotonic)
                await self._emit(
                    "cycle_started",
                    run_id=ctx.run_id,
                    cycle=ctx.cycle_count,
                )

                termination_error = self._cycle_boundary_error(ctx)
                if termination_error is not None:
                    break

                cycle_start_time = datetime.utcnow()
                observation = resume_state.observation
                if observation is None:
                    observation = await self._runtime_policies.timeouts.run_with_budget(
                        Phase.OBSERVE.value,
                        lambda: self._observe(ctx),
                        ctx=ctx,
                        run_started_monotonic=run_started_monotonic,
                        update_wall_time=self._update_wall_time,
                        wall_time_budget_error=self._wall_time_budget_error,
                    )
                    await self._persist_checkpoint(
                        ctx,
                        phase=Phase.OBSERVE,
                        next_phase=Phase.PLAN,
                    )

                termination_error = self._budget_exhaustion_error(ctx)
                if termination_error is not None:
                    break

                plan = resume_state.plan
                if plan is None:
                    plan = await self._runtime_policies.timeouts.run_with_budget(
                        Phase.PLAN.value,
                        lambda: self._plan(ctx, observation),
                        ctx=ctx,
                        run_started_monotonic=run_started_monotonic,
                        update_wall_time=self._update_wall_time,
                        wall_time_budget_error=self._wall_time_budget_error,
                    )
                    await self._persist_checkpoint(
                        ctx,
                        phase=Phase.PLAN,
                        next_phase=Phase.ACT,
                    )

                termination_error = self._budget_exhaustion_error(ctx)
                if termination_error is not None:
                    break

                execution = resume_state.execution
                if execution is None:
                    execution = await self._runtime_policies.timeouts.run_with_budget(
                        Phase.ACT.value,
                        lambda: self._act(ctx, plan),
                        ctx=ctx,
                        run_started_monotonic=run_started_monotonic,
                        update_wall_time=self._update_wall_time,
                        wall_time_budget_error=self._wall_time_budget_error,
                    )
                    await self._persist_checkpoint(
                        ctx,
                        phase=Phase.ACT,
                        next_phase=Phase.EVALUATE,
                    )

                termination_error = self._budget_exhaustion_error(ctx)
                if termination_error is not None:
                    break

                decision = await self._runtime_policies.timeouts.run_with_budget(
                    Phase.EVALUATE.value,
                    lambda: self._evaluate(ctx, plan, execution),
                    ctx=ctx,
                    run_started_monotonic=run_started_monotonic,
                    update_wall_time=self._update_wall_time,
                    wall_time_budget_error=self._wall_time_budget_error,
                )
                budget_error = self._budget_exhaustion_error(ctx)
                if budget_error is not None:
                    decision = EvalDecision(
                        done=True,
                        error=budget_error,
                        reason=budget_error.message,
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

                next_phase = Phase.FINALIZE if decision.done else Phase.OBSERVE
                await self._persist_checkpoint(
                    ctx,
                    phase=Phase.EVALUATE,
                    next_phase=next_phase,
                )

                if decision.done:
                    termination_error = decision.error
                    break

                # Reset current-cycle state for next cycle after checkpoint capture.
                ctx.state.current_observation = None
                ctx.state.current_plan = None
                ctx.state.current_execution = None
                resume_state = CheckpointResumeState()

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
        try:
            result = await self._runtime_policies.timeouts.run_with_budget(
                Phase.FINALIZE.value,
                lambda: self._finalize(ctx, error=termination_error),
                ctx=ctx,
                run_started_monotonic=run_started_monotonic,
                update_wall_time=self._update_wall_time,
                wall_time_budget_error=self._wall_time_budget_error,
            )
        except (BudgetError, AxisTimeoutError) as e:
            termination_error = e
            result = self._build_failed_result(ctx, e)
            await self._cleanup_telemetry()
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
        """Execute the full lifecycle from Initialize through Finalize."""
        self._token_callback = token_callback
        run_started_monotonic = time.monotonic()
        try:
            self._configure_runtime_policies(config)
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
            await self._persist_checkpoint(
                ctx,
                phase=Phase.INITIALIZE,
                next_phase=Phase.OBSERVE,
            )
            return await self._execute_from_context(
                ctx,
                run_started_monotonic=run_started_monotonic,
                resume_state=CheckpointResumeState(),
            )
        finally:
            self._token_callback = None

    async def resume(
        self,
        checkpoint: dict[str, Any],
        *,
        cancel_token: CancelToken | None = None,
        config: Any | None = None,
        token_callback: Any | None = None,
    ) -> dict[str, Any]:
        """Resume lifecycle execution from a checkpoint payload."""
        self._token_callback = token_callback
        run_started_monotonic = time.monotonic()
        try:
            self._configure_runtime_policies(config)
            prepared_resume = prepare_checkpoint_resume(
                checkpoint,
                cancel_token=cancel_token,
                config=config,
            )
            return await self._execute_from_context(
                prepared_resume.context,
                run_started_monotonic=run_started_monotonic,
                resume_state=prepared_resume.resume_state,
            )
        finally:
            self._token_callback = None


__all__ = [
    "LifecycleEngine",
    "Phase",
]
