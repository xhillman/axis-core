"""Lifecycle engine for axis-core agent execution.

This module implements the core execution loop:
    Initialize → [Observe → Plan → Act → Evaluate]* → Finalize

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
import inspect
import json
import logging
import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from axis_core.budget import Budget
from axis_core.context import (
    CycleState,
    EvalDecision,
    ExecutionResult,
    NormalizedInput,
    Observation,
    RunContext,
    RunState,
)
from axis_core.engine.registry import memory_registry, model_registry, planner_registry
from axis_core.engine.resolver import resolve_adapter
from axis_core.errors import (
    AxisError,
    BudgetError,
    CancelledError,
    ConfigError,
    ErrorClass,
    ErrorRecord,
    ModelError,
    PlanError,
    ToolError,
)
from axis_core.protocols.model import ModelResponse, ToolCall, UsageStats
from axis_core.protocols.planner import Plan, PlanStep, StepType
from axis_core.protocols.telemetry import TraceEvent
from axis_core.tool import ToolContext

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
        event = TraceEvent(
            type=event_type,
            timestamp=datetime.utcnow(),
            run_id=run_id,
            phase=phase,
            cycle=cycle,
            step_id=step_id,
            data=data or {},
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

        Example:
            >>> manifests = engine._get_tool_manifests()
            >>> manifests[0].name
            "get_weather"
            >>> manifests[0].input_schema
            {"type": "object", "properties": {...}}
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
    # Initialize phase (7.2)
    # =========================================================================

    async def _initialize(
        self,
        input_text: str,
        agent_id: str,
        budget: Budget,
        context: dict[str, Any] | None = None,
        attachments: list[Any] | None = None,
        cancel_token: Any | None = None,
        config: Any | None = None,
    ) -> RunContext:
        """Initialize phase: create RunContext, validate config.

        Args:
            input_text: User input text
            agent_id: Agent identifier
            budget: Budget limits for this run
            context: Optional context dict for sharing state
            attachments: Optional list of attachments
            cancel_token: Optional cancellation token
            config: Optional resolved configuration

        Returns:
            Initialized RunContext

        Raises:
            ConfigError: If input is empty or config is invalid
        """
        if not input_text or not input_text.strip():
            raise ConfigError(message="Input must not be empty")

        run_id = str(uuid.uuid4())

        normalized_input = NormalizedInput(
            text=input_text.strip(),
            original=input_text,
        )

        ctx = RunContext(
            run_id=run_id,
            agent_id=agent_id,
            input=normalized_input,
            context=context or {},
            attachments=attachments or [],
            config=config,
            budget=budget,
            state=RunState(),
            trace=None,
            started_at=datetime.utcnow(),
            cycle_count=0,
            cancel_token=cancel_token,
        )

        await self._emit(
            "phase_entered",
            run_id=run_id,
            phase=Phase.INITIALIZE.value,
            data={"agent_id": agent_id, "input_length": len(input_text)},
        )

        logger.debug("Initialized run %s for agent %s", run_id, agent_id)

        await self._emit(
            "phase_exited",
            run_id=run_id,
            phase=Phase.INITIALIZE.value,
        )

        return ctx

    # =========================================================================
    # Observe phase (7.3)
    # =========================================================================

    async def _observe(self, ctx: RunContext) -> Observation:
        """Observe phase: gather input, load memory, assess state.

        Args:
            ctx: Current run context

        Returns:
            Observation with input, memory context, and prior cycle summaries
        """
        phase_start = time.monotonic()
        await self._emit(
            "phase_entered",
            run_id=ctx.run_id,
            phase=Phase.OBSERVE.value,
            cycle=ctx.cycle_count,
        )

        # Gather memory context
        memory_context: dict[str, Any] = {}
        if self.memory is not None:
            try:
                results = await self.memory.search(
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
        await self._emit(
            "phase_exited",
            run_id=ctx.run_id,
            phase=Phase.OBSERVE.value,
            cycle=ctx.cycle_count,
            duration_ms=duration_ms,
        )

        return observation

    # =========================================================================
    # Plan phase (7.4, AD-006)
    # =========================================================================

    async def _plan(self, ctx: RunContext, observation: Observation) -> Plan:
        """Plan phase: call planner, validate plan (AD-006).

        Args:
            ctx: Current run context
            observation: Current observation

        Returns:
            Validated Plan

        Raises:
            PlanError: If plan validation fails (unknown tools, invalid deps)
        """
        phase_start = time.monotonic()
        await self._emit(
            "phase_entered",
            run_id=ctx.run_id,
            phase=Phase.PLAN.value,
            cycle=ctx.cycle_count,
        )

        # Provide tool information to planners via context
        # (AutoPlanner uses this for LLM-based planning)
        tool_descriptions = {}
        for tool_name, tool_fn in self.tools.items():
            if hasattr(tool_fn, "_axis_manifest"):
                manifest = tool_fn._axis_manifest
                tool_descriptions[tool_name] = manifest.description
        ctx.context["__tools__"] = tool_descriptions

        plan: Plan = await self.planner.plan(observation, ctx)

        # AD-016: Emit telemetry if planner fell back to sequential
        if plan.metadata.get("fallback"):
            await self._emit(
                "planner_fallback",
                run_id=ctx.run_id,
                phase=Phase.PLAN.value,
                cycle=ctx.cycle_count,
                data={
                    "original_planner": plan.metadata.get("planner", "unknown"),
                    "reason": plan.metadata.get("fallback_reason", "unknown"),
                },
            )

        # AD-006: Strict plan validation
        await self._validate_plan(plan)

        ctx.state.current_plan = plan

        duration_ms = (time.monotonic() - phase_start) * 1000
        await self._emit(
            "phase_exited",
            run_id=ctx.run_id,
            phase=Phase.PLAN.value,
            cycle=ctx.cycle_count,
            duration_ms=duration_ms,
            data={"plan_id": plan.id, "step_count": len(plan.steps)},
        )

        return plan

    async def _validate_plan(self, plan: Plan) -> None:
        """Validate plan per AD-006: all tools exist, dependencies valid.

        Args:
            plan: Plan to validate

        Raises:
            PlanError: If validation fails
        """
        step_ids = {step.id for step in plan.steps}

        for step in plan.steps:
            # Validate tool steps reference existing tools
            if step.type == StepType.TOOL:
                tool_name = step.payload.get("tool")
                if tool_name and tool_name not in self.tools:
                    raise PlanError(
                        message=(
                            f"Plan validation failed: tool '{tool_name}' not found. "
                            f"Available tools: {list(self.tools.keys())}"
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

    # =========================================================================
    # Act phase (7.5, AD-003, AD-042)
    # =========================================================================

    async def _act(self, ctx: RunContext, plan: Plan) -> ExecutionResult:
        """Act phase: execute plan steps with dependency handling.

        Per AD-003, steps execute serially. Per AD-042, independent steps
        continue on failure while dependent steps are skipped.

        Args:
            ctx: Current run context
            plan: Plan to execute

        Returns:
            ExecutionResult with results, errors, and skipped steps
        """
        phase_start = time.monotonic()
        await self._emit(
            "phase_entered",
            run_id=ctx.run_id,
            phase=Phase.ACT.value,
            cycle=ctx.cycle_count,
        )

        results: dict[str, Any] = {}
        errors: dict[str, AxisError] = {}
        skipped: set[str] = set()

        for step in plan.steps:
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
                    result = await self._execute_tool_step(ctx, step)
                    results[step.id] = result
                elif step.type == StepType.MODEL:
                    result = await self._execute_model_step(ctx, step)
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
                axis_error = self._wrap_error(e, step)
                errors[step.id] = axis_error

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

                await self._emit(
                    "tool_failed" if step.type == StepType.TOOL else "step_failed",
                    run_id=ctx.run_id,
                    phase=Phase.ACT.value,
                    cycle=ctx.cycle_count,
                    step_id=step.id,
                    data={"error": str(e)},
                )

                logger.warning("Step %s failed: %s", step.id, e)

        execution_result = ExecutionResult(
            results=results,
            errors=errors,
            skipped=frozenset(skipped),
            duration_ms=(time.monotonic() - phase_start) * 1000,
        )

        ctx.state.current_execution = execution_result

        await self._emit(
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

    async def _execute_tool_step(self, ctx: RunContext, step: PlanStep) -> Any:
        """Execute a single tool step.

        Args:
            ctx: Current run context
            step: Tool step to execute

        Returns:
            Tool execution result

        Raises:
            ToolError: If tool execution fails
        """
        tool_name = step.payload.get("tool", "")
        args = step.payload.get("args", {})

        if tool_name not in self.tools:
            raise ToolError(
                message=f"Tool '{tool_name}' not found",
                tool_name=tool_name,
            )

        tool_fn = self.tools[tool_name]

        await self._emit(
            "tool_called",
            run_id=ctx.run_id,
            phase=Phase.ACT.value,
            cycle=ctx.cycle_count,
            step_id=step.id,
            data={"tool": tool_name, "args": args},
        )

        start = time.monotonic()
        try:
            tool_kwargs = dict(args)
            if "ctx" in tool_kwargs:
                tool_kwargs.pop("ctx")
            try:
                supports_ctx = "ctx" in inspect.signature(tool_fn).parameters
            except (TypeError, ValueError):
                supports_ctx = False
            if supports_ctx:
                tool_ctx = ToolContext(
                    run_id=ctx.run_id,
                    agent_id=ctx.agent_id,
                    cycle=ctx.cycle_count,
                    context=ctx.context,
                    budget=ctx.budget,
                    budget_state=ctx.state.budget_state,
                )
                result = await tool_fn(ctx=tool_ctx, **tool_kwargs)
            else:
                result = await tool_fn(**tool_kwargs)
        except Exception as e:
            raise ToolError(
                message=f"Tool '{tool_name}' failed: {e}",
                tool_name=tool_name,
                cause=e,
            ) from e

        duration_ms = (time.monotonic() - start) * 1000

        # Track budget
        ctx.state.budget_state.tool_calls += 1

        await self._emit(
            "tool_returned",
            run_id=ctx.run_id,
            phase=Phase.ACT.value,
            cycle=ctx.cycle_count,
            step_id=step.id,
            data={"tool": tool_name, "duration_ms": duration_ms},
            duration_ms=duration_ms,
        )

        return result

    async def _call_model_with_fallback(
        self,
        ctx: RunContext,
        messages: Any,
        system: str | None,
        tools: Any | None,
    ) -> Any:
        """Call model with fallback chain on recoverable errors (AD-013).

        Attempts to call the primary model first. On recoverable errors,
        tries each fallback model in order. Preserves exact request parameters
        across all attempts per AD-013.

        Args:
            ctx: Current run context
            messages: Messages to send to model
            system: System prompt
            tools: Tool manifests

        Returns:
            ModelResponse from first successful model

        Raises:
            ModelError: If all models (primary + fallbacks) fail
        """
        # Build chain of models to try: primary first, then fallbacks
        models_to_try = [self.model] + self.fallback
        errors: list[Exception] = []

        for idx, model in enumerate(models_to_try):
            is_fallback = idx > 0
            model_id = getattr(model, "model_id", "unknown")

            try:
                response = await model.complete(
                    messages=messages,
                    system=system,
                    tools=tools,
                )

                # Success - emit fallback event if we used a fallback
                if is_fallback:
                    previous_model_id = getattr(
                        models_to_try[idx - 1], "model_id", "unknown"
                    )
                    await self._emit(
                        "model_fallback",
                        run_id=ctx.run_id,
                        phase=Phase.ACT.value,
                        cycle=ctx.cycle_count,
                        data={
                            "from_model": previous_model_id,
                            "to_model": model_id,
                            "attempt": idx + 1,
                        },
                    )

                return response

            except Exception as e:
                # Classify error as recoverable or not
                model_error = ModelError.from_exception(e, model_id)
                errors.append(model_error)

                # If error is not recoverable, stop trying fallbacks
                if not model_error.recoverable:
                    logger.warning(
                        "Non-recoverable error from model %s: %s",
                        model_id,
                        model_error.message,
                    )
                    raise model_error

                # Log recoverable error and continue to next fallback
                logger.info(
                    "Recoverable error from model %s (attempt %d/%d): %s",
                    model_id,
                    idx + 1,
                    len(models_to_try),
                    model_error.message,
                )

                # If this was the last model, we're out of options
                if idx == len(models_to_try) - 1:
                    break

        # All models failed with recoverable errors
        error_messages = [str(e) for e in errors]
        combined_error = ModelError(
            message=(
                f"All models failed after {len(models_to_try)} attempts. "
                f"Errors: {'; '.join(error_messages)}"
            ),
            model_id="fallback_chain",
            recoverable=False,
            cause=errors[-1] if errors else None,
        )
        raise combined_error

    async def _call_model_with_fallback_stream(
        self,
        ctx: RunContext,
        messages: Any,
        system: str | None,
        tools: Any | None,
        token_callback: Any,
    ) -> ModelResponse:
        """Call model with fallback chain using streaming responses."""
        models_to_try = [self.model] + self.fallback
        errors: list[Exception] = []

        for idx, model in enumerate(models_to_try):
            is_fallback = idx > 0
            model_id = getattr(model, "model_id", "unknown")

            try:
                response = await self._stream_model_response(
                    model=model,
                    messages=messages,
                    system=system,
                    tools=tools,
                    token_callback=token_callback,
                )

                if is_fallback:
                    previous_model_id = getattr(
                        models_to_try[idx - 1], "model_id", "unknown"
                    )
                    await self._emit(
                        "model_fallback",
                        run_id=ctx.run_id,
                        phase=Phase.ACT.value,
                        cycle=ctx.cycle_count,
                        data={
                            "from_model": previous_model_id,
                            "to_model": model_id,
                            "attempt": idx + 1,
                        },
                    )

                return response

            except Exception as e:
                model_error = e if isinstance(e, ModelError) else ModelError.from_exception(e, model_id)
                errors.append(model_error)

                if not model_error.recoverable:
                    logger.warning(
                        "Non-recoverable error from model %s: %s",
                        model_id,
                        model_error.message,
                    )
                    raise model_error

                logger.info(
                    "Recoverable error from model %s (attempt %d/%d): %s",
                    model_id,
                    idx + 1,
                    len(models_to_try),
                    model_error.message,
                )

                if idx == len(models_to_try) - 1:
                    break

        error_messages = [str(e) for e in errors]
        combined_error = ModelError(
            message=(
                f"All models failed after {len(models_to_try)} attempts. "
                f"Errors: {'; '.join(error_messages)}"
            ),
            model_id="fallback_chain",
            recoverable=False,
            cause=errors[-1] if errors else None,
        )
        raise combined_error

    async def _stream_model_response(
        self,
        model: Any,
        messages: Any,
        system: str | None,
        tools: Any | None,
        token_callback: Any,
    ) -> ModelResponse:
        """Stream a model response and aggregate into a ModelResponse."""
        content_parts: list[str] = []
        tool_calls_by_index: dict[int, dict[str, Any]] = {}
        requires_complete = False

        async for chunk in model.stream(
            messages=messages,
            system=system,
            tools=tools,
        ):
            if chunk.content:
                content_parts.append(chunk.content)
                await token_callback(chunk.content)

            if chunk.tool_call_delta:
                delta = chunk.tool_call_delta
                if "function" in delta:
                    idx = int(delta.get("index", 0))
                    entry = tool_calls_by_index.setdefault(
                        idx, {"id": None, "name": None, "arguments": ""}
                    )
                    if "id" in delta:
                        entry["id"] = delta["id"]
                    func = delta.get("function") or {}
                    name = func.get("name")
                    if name:
                        entry["name"] = name
                    args_text = func.get("arguments")
                    if args_text:
                        entry["arguments"] += args_text
                elif "partial_json" in delta:
                    requires_complete = True

            if chunk.is_final:
                break

        if requires_complete:
            return await model.complete(messages=messages, system=system, tools=tools)

        tool_calls: tuple[ToolCall, ...] | None = None
        if tool_calls_by_index:
            calls: list[ToolCall] = []
            for idx in sorted(tool_calls_by_index):
                entry = tool_calls_by_index[idx]
                name = entry.get("name")
                if not name:
                    requires_complete = True
                    break
                args: dict[str, Any] = {}
                args_text = entry.get("arguments", "")
                if args_text:
                    try:
                        args = json.loads(args_text)
                    except json.JSONDecodeError:
                        args = {"_raw": args_text}
                calls.append(
                    ToolCall(
                        id=entry.get("id") or f"call_{idx}",
                        name=name,
                        arguments=args,
                    )
                )

            if requires_complete:
                return await model.complete(messages=messages, system=system, tools=tools)

            tool_calls = tuple(calls) if calls else None

        content = "".join(content_parts)
        input_tokens = await self._estimate_tokens_for_messages(
            model=model,
            messages=messages,
            system=system,
        )
        output_tokens = await self._estimate_tokens(model=model, text=content)
        usage = UsageStats(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )
        cost = await self._estimate_cost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            cost_usd=cost,
        )

    async def _estimate_tokens(self, model: Any, text: str) -> int:
        """Estimate token count using model adapter if available."""
        estimator = getattr(model, "estimate_tokens", None)
        if callable(estimator):
            try:
                value = estimator(text)
                if asyncio.iscoroutine(value):
                    value = await value
                return int(value)
            except Exception:
                pass
        return max(1, len(text) // 4) if text else 0

    async def _estimate_tokens_for_messages(
        self,
        model: Any,
        messages: Any,
        system: str | None,
    ) -> int:
        parts: list[str] = []
        if system:
            parts.append(system)
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(str(part) for part in content)
            parts.append(str(content))
        return await self._estimate_tokens(model=model, text="\n".join(parts))

    async def _estimate_cost(
        self,
        model: Any,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        estimator = getattr(model, "estimate_cost", None)
        if callable(estimator):
            try:
                value = estimator(input_tokens, output_tokens)
                if asyncio.iscoroutine(value):
                    value = await value
                return float(value)
            except Exception:
                return 0.0
        return 0.0

    async def _execute_model_step(self, ctx: RunContext, step: PlanStep) -> Any:
        """Execute a model (LLM) step.

        Args:
            ctx: Current run context
            step: Model step to execute

        Returns:
            Model response content
        """
        # Build messages if not explicitly provided
        if "messages" not in step.payload:
            # Get context strategy from environment or use default
            import os
            strategy = os.getenv("AXIS_CONTEXT_STRATEGY", "smart")
            max_cycles = int(os.getenv("AXIS_MAX_CYCLE_CONTEXT", "5"))
            messages = ctx.state.build_messages(ctx, strategy=strategy, max_cycles=max_cycles)
        else:
            messages = step.payload["messages"]

        system = step.payload.get("system", self.system)

        # Get tool manifests (protocol objects) - adapter will convert to its format
        tool_manifests = self._get_tool_manifests()
        tools = tool_manifests if tool_manifests else None

        await self._emit(
            "model_called",
            run_id=ctx.run_id,
            phase=Phase.ACT.value,
            cycle=ctx.cycle_count,
            step_id=step.id,
        )

        start = time.monotonic()
        # Use fallback chain if configured (Task 15.0)
        if self._token_callback is not None:
            response = await self._call_model_with_fallback_stream(
                ctx=ctx,
                messages=messages,
                system=system,
                tools=tools,
                token_callback=self._token_callback,
            )
        else:
            response = await self._call_model_with_fallback(
                ctx=ctx,
                messages=messages,
                system=system,
                tools=tools,
            )
        duration_ms = (time.monotonic() - start) * 1000

        # Track budget
        ctx.state.budget_state.model_calls += 1
        ctx.state.budget_state.input_tokens += response.usage.input_tokens
        ctx.state.budget_state.output_tokens += response.usage.output_tokens
        ctx.state.budget_state.cost_usd += response.cost_usd

        await self._emit(
            "model_returned",
            run_id=ctx.run_id,
            phase=Phase.ACT.value,
            cycle=ctx.cycle_count,
            step_id=step.id,
            data={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cost_usd": response.cost_usd,
            },
            duration_ms=duration_ms,
        )

        # Store raw response as potential output
        ctx.state.output_raw = response.content

        # Store full response for next Observe phase
        ctx.state.last_model_response = response

        return response.content

    def _wrap_error(self, e: Exception, step: PlanStep) -> AxisError:
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
                message=f"Tool step '{step.id}' failed: {e}",
                tool_name=step.payload.get("tool"),
                cause=e,
            )
        return AxisError(
            message=f"Step '{step.id}' failed: {e}",
            error_class=ErrorClass.RUNTIME,
            cause=e,
        )

    # =========================================================================
    # Evaluate phase (7.6)
    # =========================================================================

    async def _evaluate(
        self,
        ctx: RunContext,
        plan: Plan,
        execution: ExecutionResult,
    ) -> EvalDecision:
        """Evaluate phase: check termination conditions.

        Checks (in order):
        1. Cancellation
        2. Terminal plan step completed
        3. Budget exhaustion
        4. Unrecoverable errors

        Args:
            ctx: Current run context
            plan: Current plan
            execution: Execution results

        Returns:
            EvalDecision indicating whether to continue or stop
        """
        phase_start = time.monotonic()
        await self._emit(
            "phase_entered",
            run_id=ctx.run_id,
            phase=Phase.EVALUATE.value,
            cycle=ctx.cycle_count,
        )

        decision: EvalDecision

        # 1. Check cancellation
        if ctx.cancel_token and ctx.cancel_token.is_cancelled:
            reason = getattr(ctx.cancel_token, "_reason", None) or "Cancelled"
            decision = EvalDecision(
                done=True,
                error=CancelledError(message=reason),
                reason=f"Cancelled: {reason}",
            )
        # 2. Check for terminal step
        elif self._has_terminal_step(plan):
            decision = EvalDecision(
                done=True,
                reason="Plan completed with terminal step",
            )
        # 3. Check budget
        elif ctx.state.budget_state.is_exhausted(ctx.budget):
            resource = self._identify_exhausted_resource(ctx)
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
        await self._emit(
            "phase_exited",
            run_id=ctx.run_id,
            phase=Phase.EVALUATE.value,
            cycle=ctx.cycle_count,
            duration_ms=duration_ms,
            data={"done": decision.done, "reason": decision.reason},
        )

        return decision

    def _has_terminal_step(self, plan: Plan) -> bool:
        """Check if plan contains a TERMINAL step."""
        return any(step.type == StepType.TERMINAL for step in plan.steps)

    def _identify_exhausted_resource(self, ctx: RunContext) -> str:
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

    # =========================================================================
    # Finalize phase (7.7, AD-007)
    # =========================================================================

    async def _finalize(
        self,
        ctx: RunContext,
        error: Exception | None = None,
    ) -> dict[str, Any]:
        """Finalize phase: persist memory, emit summary, clean up.

        Per AD-007, memory persistence failures are non-fatal. The run
        succeeds but the memory_error field is populated.

        Args:
            ctx: Current run context
            error: Error that caused termination (if any)

        Returns:
            Result dict with output, success, stats, and optional memory_error
        """
        phase_start = time.monotonic()
        await self._emit(
            "phase_entered",
            run_id=ctx.run_id,
            phase=Phase.FINALIZE.value,
        )

        memory_error: str | None = None

        # Persist to memory (AD-007: non-fatal on failure)
        if self.memory is not None:
            try:
                await self.memory.store(
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
        for sink in self.telemetry:
            try:
                await sink.flush()
                await sink.close()
            except Exception:
                logger.warning("Telemetry sink cleanup failed", exc_info=True)

        duration_ms = (time.monotonic() - phase_start) * 1000
        await self._emit(
            "phase_exited",
            run_id=ctx.run_id,
            phase=Phase.FINALIZE.value,
            duration_ms=duration_ms,
        )

        return result

    # =========================================================================
    # Main execution loop (7.8)
    # =========================================================================

    async def execute(
        self,
        input_text: str,
        agent_id: str,
        budget: Budget,
        context: dict[str, Any] | None = None,
        attachments: list[Any] | None = None,
        cancel_token: Any | None = None,
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
                    await self._emit(
                        "cycle_started",
                        run_id=ctx.run_id,
                        cycle=ctx.cycle_count,
                    )

                    # Check cancellation at cycle start (AD-028)
                    if ctx.cancel_token and ctx.cancel_token.is_cancelled:
                        reason = getattr(ctx.cancel_token, "_reason", None) or "Cancelled"
                        termination_error = CancelledError(message=reason)
                        break

                    # Check budget before starting cycle
                    if ctx.state.budget_state.is_exhausted(ctx.budget):
                        resource = self._identify_exhausted_resource(ctx)
                        termination_error = BudgetError(
                            message=f"Budget exhausted: {resource}",
                            resource=resource,
                        )
                        break

                    cycle_start_time = datetime.utcnow()

                    # Observe
                    observation = await self._observe(ctx)

                    # Plan
                    plan = await self._plan(ctx, observation)

                    # Act
                    execution = await self._act(ctx, plan)

                    # Evaluate
                    decision = await self._evaluate(ctx, plan, execution)

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
            result = await self._finalize(ctx, error=termination_error)

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
