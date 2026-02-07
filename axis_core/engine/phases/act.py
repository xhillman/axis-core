"""Act phase: execute plan steps with dependency handling (AD-003, AD-042)."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from axis_core.context import ExecutionResult, ModelCallRecord, RunContext
from axis_core.errors import (
    AxisError,
    ErrorClass,
    ErrorRecord,
    ModelError,
    ToolError,
)
from axis_core.protocols.model import ModelResponse, ToolCall, UsageStats
from axis_core.protocols.planner import Plan, PlanStep, StepType
from axis_core.tool import ToolCallRecord, ToolContext

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
    """Execute a single tool step.

    Args:
        engine: The lifecycle engine instance
        ctx: Current run context
        step: Tool step to execute

    Returns:
        Tool execution result

    Raises:
        ToolError: If tool execution fails
    """
    from axis_core.engine.lifecycle import Phase

    tool_name = step.payload.get("tool", "")
    args = step.payload.get("args", {})

    if tool_name not in engine.tools:
        raise ToolError(
            message=f"Tool '{tool_name}' not found",
            tool_name=tool_name,
        )

    tool_fn = engine.tools[tool_name]

    await engine._emit(
        "tool_called",
        run_id=ctx.run_id,
        phase=Phase.ACT.value,
        cycle=ctx.cycle_count,
        step_id=step.id,
        data={"tool": tool_name, "args": args},
    )

    start = time.monotonic()
    result: Any = None
    error_msg: str | None = None
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
        error_msg = f"Tool '{tool_name}' failed: {e}"
        duration_ms = (time.monotonic() - start) * 1000

        # Record failed tool call for observability/checkpointing
        ctx.state.append_tool_call(ToolCallRecord(
            tool_name=tool_name,
            call_id=step.id,
            args=dict(args),
            result=None,
            error=error_msg,
            cached=False,
            duration_ms=duration_ms,
            timestamp=time.time(),
        ))

        raise ToolError(
            message=error_msg,
            tool_name=tool_name,
            cause=e,
        ) from e

    duration_ms = (time.monotonic() - start) * 1000

    # Track budget
    ctx.state.budget_state.tool_calls += 1

    # Record successful tool call for observability/checkpointing
    ctx.state.append_tool_call(ToolCallRecord(
        tool_name=tool_name,
        call_id=step.id,
        args=dict(args),
        result=result,
        error=None,
        cached=False,
        duration_ms=duration_ms,
        timestamp=time.time(),
    ))

    await engine._emit(
        "tool_returned",
        run_id=ctx.run_id,
        phase=Phase.ACT.value,
        cycle=ctx.cycle_count,
        step_id=step.id,
        data={"tool": tool_name, "duration_ms": duration_ms},
        duration_ms=duration_ms,
    )

    return result


async def try_models_with_fallback(
    engine: LifecycleEngine,
    ctx: RunContext,
    call_fn: Any,
) -> Any:
    """Try primary model then fallbacks on recoverable errors (AD-013).

    Args:
        engine: The lifecycle engine instance
        ctx: Current run context
        call_fn: Async callable(model) -> ModelResponse

    Returns:
        ModelResponse from first successful model

    Raises:
        ModelError: If all models (primary + fallbacks) fail
    """
    from axis_core.engine.lifecycle import Phase

    models_to_try = [engine.model] + engine.fallback
    errors: list[Exception] = []

    for idx, model in enumerate(models_to_try):
        model_id = getattr(model, "model_id", "unknown")

        try:
            response = await call_fn(model)

            if idx > 0:
                previous_model_id = getattr(
                    models_to_try[idx - 1], "model_id", "unknown"
                )
                await engine._emit(
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
            model_error = (
                e if isinstance(e, ModelError)
                else ModelError.from_exception(e, model_id)
            )
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

    error_messages = [str(e) for e in errors]
    raise ModelError(
        message=(
            f"All models failed after {len(models_to_try)} attempts. "
            f"Errors: {'; '.join(error_messages)}"
        ),
        model_id="fallback_chain",
        recoverable=False,
        cause=errors[-1] if errors else None,
    )


async def stream_model_response(
    engine: LifecycleEngine,
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
        return cast(
            ModelResponse,
            await model.complete(messages=messages, system=system, tools=tools),
        )

    tool_calls: tuple[ToolCall, ...] | None = None
    if tool_calls_by_index:
        calls: list[ToolCall] = []
        for idx in sorted(tool_calls_by_index):
            entry = tool_calls_by_index[idx]
            name = entry.get("name")
            if not name:
                requires_complete = True
                break
            parsed_args: dict[str, Any] = {}
            args_text = entry.get("arguments", "")
            if args_text:
                try:
                    parsed_args = json.loads(args_text)
                except json.JSONDecodeError:
                    parsed_args = {"_raw": args_text}
            calls.append(
                ToolCall(
                    id=entry.get("id") or f"call_{idx}",
                    name=name,
                    arguments=parsed_args,
                )
            )

        if requires_complete:
            return cast(
                ModelResponse,
                await model.complete(messages=messages, system=system, tools=tools),
            )

        tool_calls = tuple(calls) if calls else None

    content = "".join(content_parts)
    input_tokens = await _estimate_tokens_for_messages(
        model=model,
        messages=messages,
        system=system,
    )
    output_tokens = await _estimate_tokens(model=model, text=content)
    usage = UsageStats(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )
    cost = await _estimate_cost(
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


async def _estimate_tokens(model: Any, text: str) -> int:
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
    return await _estimate_tokens(model=model, text="\n".join(parts))


async def _estimate_cost(
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


async def _execute_model_step(
    engine: LifecycleEngine,
    ctx: RunContext,
    step: PlanStep,
) -> Any:
    """Execute a model (LLM) step.

    Args:
        engine: The lifecycle engine instance
        ctx: Current run context
        step: Model step to execute

    Returns:
        Model response content
    """
    from axis_core.engine.lifecycle import Phase

    # Build messages if not explicitly provided
    if "messages" not in step.payload:
        # Get context strategy from environment or use default
        import os
        strategy = os.getenv("AXIS_CONTEXT_STRATEGY", "smart")
        max_cycles = int(os.getenv("AXIS_MAX_CYCLE_CONTEXT", "5"))
        messages = ctx.state.build_messages(ctx, strategy=strategy, max_cycles=max_cycles)
    else:
        messages = step.payload["messages"]

    system = step.payload.get("system", engine.system)

    # Get tool manifests (protocol objects) - adapter will convert to its format
    tool_manifests = engine._get_tool_manifests()
    tools = tool_manifests if tool_manifests else None

    await engine._emit(
        "model_called",
        run_id=ctx.run_id,
        phase=Phase.ACT.value,
        cycle=ctx.cycle_count,
        step_id=step.id,
    )

    start = time.monotonic()
    # Use fallback chain if configured (Task 15.0)
    if engine._token_callback is not None:
        token_cb = engine._token_callback

        async def _stream_call(m: Any) -> Any:
            return await stream_model_response(
                engine=engine, model=m, messages=messages, system=system,
                tools=tools, token_callback=token_cb,
            )

        response = await try_models_with_fallback(engine, ctx, _stream_call)
    else:
        async def _complete_call(m: Any) -> Any:
            return await m.complete(
                messages=messages, system=system, tools=tools,
            )

        response = await try_models_with_fallback(engine, ctx, _complete_call)
    duration_ms = (time.monotonic() - start) * 1000

    # Track budget
    ctx.state.budget_state.model_calls += 1
    ctx.state.budget_state.input_tokens += response.usage.input_tokens
    ctx.state.budget_state.output_tokens += response.usage.output_tokens
    ctx.state.budget_state.cost_usd += response.cost_usd

    # Record detailed model call for observability/checkpointing
    ctx.state.append_model_call(ModelCallRecord(
        model_id=getattr(engine.model, "model_id", "unknown"),
        call_id=step.id,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cost_usd=response.cost_usd,
        duration_ms=duration_ms,
        timestamp=time.time(),
    ))

    await engine._emit(
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
            message=f"Tool step '{step.id}' failed: {e}",
            tool_name=step.payload.get("tool"),
            cause=e,
        )
    return AxisError(
        message=f"Step '{step.id}' failed: {e}",
        error_class=ErrorClass.RUNTIME,
        cause=e,
    )
