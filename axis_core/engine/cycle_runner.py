"""Lifecycle cycle orchestration helpers."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from axis_core.checkpoint import CheckpointResumeState
from axis_core.context import (
    CycleState,
    EvalDecision,
    ExecutionResult,
    Observation,
    RunContext,
)
from axis_core.engine.runtime_policy import LifecycleRuntimePolicyServices
from axis_core.errors import AxisError, BudgetError, ErrorClass
from axis_core.errors import TimeoutError as AxisTimeoutError
from axis_core.protocols.planner import Plan

ObserveFn = Callable[[RunContext], Awaitable[Observation]]
PlanFn = Callable[[RunContext, Observation], Awaitable[Plan]]
ActFn = Callable[[RunContext, Plan], Awaitable[ExecutionResult]]
EvaluateFn = Callable[[RunContext, Plan, ExecutionResult], Awaitable[EvalDecision]]
FinalizeFn = Callable[[RunContext, Exception | None], Awaitable[dict[str, Any]]]
EmitFn = Callable[..., Awaitable[None]]
PersistCheckpointFn = Callable[[RunContext, str, str | None], Awaitable[None]]
UpdateWallTimeFn = Callable[[RunContext, float], None]
WallTimeBudgetErrorFn = Callable[[RunContext], BudgetError]
BudgetExhaustionErrorFn = Callable[[RunContext], BudgetError | None]
CycleBoundaryErrorFn = Callable[[RunContext], Exception | None]
BuildFailedResultFn = Callable[[RunContext, Exception], dict[str, Any]]
CleanupTelemetryFn = Callable[[], Awaitable[None]]


class LifecycleCycleRunner:
    """Run the steady-state lifecycle loop with injected engine services."""

    _OBSERVE_PHASE = "observe"
    _PLAN_PHASE = "plan"
    _ACT_PHASE = "act"
    _EVALUATE_PHASE = "evaluate"
    _FINALIZE_PHASE = "finalize"

    def __init__(
        self,
        *,
        emit: EmitFn,
        runtime_policies: LifecycleRuntimePolicyServices,
        observe: ObserveFn,
        plan: PlanFn,
        act: ActFn,
        evaluate: EvaluateFn,
        finalize: FinalizeFn,
        persist_checkpoint: PersistCheckpointFn,
        update_wall_time: UpdateWallTimeFn,
        wall_time_budget_error: WallTimeBudgetErrorFn,
        budget_exhaustion_error: BudgetExhaustionErrorFn,
        cycle_boundary_error: CycleBoundaryErrorFn,
        build_failed_result: BuildFailedResultFn,
        cleanup_telemetry: CleanupTelemetryFn,
    ) -> None:
        self._emit = emit
        self._runtime_policies = runtime_policies
        self._observe = observe
        self._plan = plan
        self._act = act
        self._evaluate = evaluate
        self._finalize = finalize
        self._persist_checkpoint = persist_checkpoint
        self._update_wall_time = update_wall_time
        self._wall_time_budget_error = wall_time_budget_error
        self._budget_exhaustion_error = budget_exhaustion_error
        self._cycle_boundary_error = cycle_boundary_error
        self._build_failed_result = build_failed_result
        self._cleanup_telemetry = cleanup_telemetry

    async def run(
        self,
        ctx: RunContext,
        *,
        run_started_monotonic: float,
        resume_state: CheckpointResumeState | None = None,
    ) -> dict[str, Any]:
        """Continue lifecycle execution from a prepared run context."""
        self._update_wall_time(ctx, run_started_monotonic)
        await self._emit(
            "run_started",
            run_id=ctx.run_id,
            data={"agent_id": ctx.agent_id},
        )

        termination_error: Exception | None = None

        try:
            active_resume_state = resume_state or CheckpointResumeState()

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

                cycle_started_at = datetime.utcnow()
                observation = await self._resolve_observation(
                    ctx,
                    resume_state=active_resume_state,
                    run_started_monotonic=run_started_monotonic,
                )
                termination_error = self._budget_exhaustion_error(ctx)
                if termination_error is not None:
                    break

                plan = await self._resolve_plan(
                    ctx,
                    observation=observation,
                    resume_state=active_resume_state,
                    run_started_monotonic=run_started_monotonic,
                )
                termination_error = self._budget_exhaustion_error(ctx)
                if termination_error is not None:
                    break

                execution = await self._resolve_execution(
                    ctx,
                    plan=plan,
                    resume_state=active_resume_state,
                    run_started_monotonic=run_started_monotonic,
                )
                termination_error = self._budget_exhaustion_error(ctx)
                if termination_error is not None:
                    break

                decision = await self._resolve_decision(
                    ctx,
                    plan=plan,
                    execution=execution,
                    run_started_monotonic=run_started_monotonic,
                )
                cycle_ended_at = datetime.utcnow()

                self._record_completed_cycle(
                    ctx,
                    observation=observation,
                    plan=plan,
                    execution=execution,
                    decision=decision,
                    started_at=cycle_started_at,
                    ended_at=cycle_ended_at,
                )

                await self._emit_cycle_completed(
                    ctx,
                    cycle_start=cycle_start,
                    done=decision.done,
                )

                next_phase = (
                    self._FINALIZE_PHASE if decision.done else self._OBSERVE_PHASE
                )
                await self._persist_checkpoint(
                    ctx,
                    self._EVALUATE_PHASE,
                    next_phase,
                )

                if decision.done:
                    termination_error = decision.error
                    break

                self._reset_current_cycle_state(ctx)
                active_resume_state = CheckpointResumeState()

        except AxisError as error:
            termination_error = error
        except Exception as error:
            termination_error = AxisError(
                message=f"Unexpected error: {error}",
                error_class=ErrorClass.RUNTIME,
                cause=error,
            )

        return await self._finalize_run(
            ctx,
            run_started_monotonic=run_started_monotonic,
            termination_error=termination_error,
        )

    async def _resolve_observation(
        self,
        ctx: RunContext,
        *,
        resume_state: CheckpointResumeState,
        run_started_monotonic: float,
    ) -> Observation:
        if resume_state.observation is not None:
            return resume_state.observation

        observation = await self._runtime_policies.timeouts.run_with_budget(
            self._OBSERVE_PHASE,
            lambda: self._observe(ctx),
            ctx=ctx,
            run_started_monotonic=run_started_monotonic,
            update_wall_time=self._update_wall_time,
            wall_time_budget_error=self._wall_time_budget_error,
        )
        await self._persist_checkpoint(
            ctx,
            self._OBSERVE_PHASE,
            self._PLAN_PHASE,
        )
        return observation

    async def _resolve_plan(
        self,
        ctx: RunContext,
        *,
        observation: Observation,
        resume_state: CheckpointResumeState,
        run_started_monotonic: float,
    ) -> Plan:
        if resume_state.plan is not None:
            return resume_state.plan

        plan = await self._runtime_policies.timeouts.run_with_budget(
            self._PLAN_PHASE,
            lambda: self._plan(ctx, observation),
            ctx=ctx,
            run_started_monotonic=run_started_monotonic,
            update_wall_time=self._update_wall_time,
            wall_time_budget_error=self._wall_time_budget_error,
        )
        await self._persist_checkpoint(
            ctx,
            self._PLAN_PHASE,
            self._ACT_PHASE,
        )
        return plan

    async def _resolve_execution(
        self,
        ctx: RunContext,
        *,
        plan: Plan,
        resume_state: CheckpointResumeState,
        run_started_monotonic: float,
    ) -> ExecutionResult:
        if resume_state.execution is not None:
            return resume_state.execution

        execution = await self._runtime_policies.timeouts.run_with_budget(
            self._ACT_PHASE,
            lambda: self._act(ctx, plan),
            ctx=ctx,
            run_started_monotonic=run_started_monotonic,
            update_wall_time=self._update_wall_time,
            wall_time_budget_error=self._wall_time_budget_error,
        )
        await self._persist_checkpoint(
            ctx,
            self._ACT_PHASE,
            self._EVALUATE_PHASE,
        )
        return execution

    async def _resolve_decision(
        self,
        ctx: RunContext,
        *,
        plan: Plan,
        execution: ExecutionResult,
        run_started_monotonic: float,
    ) -> EvalDecision:
        decision = await self._runtime_policies.timeouts.run_with_budget(
            self._EVALUATE_PHASE,
            lambda: self._evaluate(ctx, plan, execution),
            ctx=ctx,
            run_started_monotonic=run_started_monotonic,
            update_wall_time=self._update_wall_time,
            wall_time_budget_error=self._wall_time_budget_error,
        )
        budget_error = self._budget_exhaustion_error(ctx)
        if budget_error is not None:
            return EvalDecision(
                done=True,
                error=budget_error,
                reason=budget_error.message,
            )
        return decision

    @staticmethod
    def _record_completed_cycle(
        ctx: RunContext,
        *,
        observation: Observation,
        plan: Plan,
        execution: ExecutionResult,
        decision: EvalDecision,
        started_at: datetime,
        ended_at: datetime,
    ) -> None:
        cycle_state = CycleState(
            cycle_number=ctx.cycle_count,
            observation=observation,
            plan=plan,
            execution=execution,
            evaluation=decision,
            started_at=started_at,
            ended_at=ended_at,
        )
        ctx.state.append_cycle(cycle_state)
        ctx.cycle_count += 1
        ctx.state.budget_state.cycles += 1

    async def _emit_cycle_completed(
        self,
        ctx: RunContext,
        *,
        cycle_start: float,
        done: bool,
    ) -> None:
        cycle_duration_ms = (time.monotonic() - cycle_start) * 1000
        await self._emit(
            "cycle_completed",
            run_id=ctx.run_id,
            cycle=ctx.cycle_count - 1,
            duration_ms=cycle_duration_ms,
            data={"done": done},
        )

    @staticmethod
    def _reset_current_cycle_state(ctx: RunContext) -> None:
        ctx.state.current_observation = None
        ctx.state.current_plan = None
        ctx.state.current_execution = None

    async def _finalize_run(
        self,
        ctx: RunContext,
        *,
        run_started_monotonic: float,
        termination_error: Exception | None,
    ) -> dict[str, Any]:
        self._update_wall_time(ctx, run_started_monotonic)
        try:
            result = await self._runtime_policies.timeouts.run_with_budget(
                self._FINALIZE_PHASE,
                lambda: self._finalize(ctx, termination_error),
                ctx=ctx,
                run_started_monotonic=run_started_monotonic,
                update_wall_time=self._update_wall_time,
                wall_time_budget_error=self._wall_time_budget_error,
            )
        except (BudgetError, AxisTimeoutError) as error:
            result = self._build_failed_result(ctx, error)
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
