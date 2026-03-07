"""Act phase: execute plan steps with dependency handling (AD-003, AD-042)."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from axis_core.context import (
    ContextWindowGuard,
    ExecutionResult,
    ModelCallRecord,
    RunContext,
    normalize_transcript_messages,
    prune_messages_for_context_window,
)
from axis_core.engine.phases.act_execution_utils import (
    is_retryable_model_error,
    record_retry_attempt,
    resolve_retry_policy,
    sleep_for_retry,
)
from axis_core.engine.phases.act_runtime_settings import ActRuntimeSettingsResolver
from axis_core.engine.phases.act_tool_execution import ActToolExecutionService
from axis_core.errors import (
    AxisError,
    ErrorClass,
    ErrorRecord,
    ModelError,
    ToolError,
)
from axis_core.protocols.model import ModelResponse, ToolCall, UsageStats
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


async def try_models_with_fallback(
    engine: LifecycleEngine,
    ctx: RunContext,
    call_fn: Any,
    step: PlanStep | None = None,
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
    errors: list[ModelError] = []
    effective_step = step or PlanStep(id="model-call", type=StepType.MODEL)
    retry_policy = resolve_retry_policy(ctx, effective_step)
    max_attempts = max(1, retry_policy.max_attempts)

    for idx, model in enumerate(models_to_try):
        model_id = getattr(model, "model_id", "unknown")
        last_error: ModelError | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                await engine.acquire_model_slot(ctx, step_id=effective_step.id)
                response = await call_fn(model)

                if idx > 0:
                    previous_error = errors[-1] if errors else None
                    previous_model_id = getattr(
                        models_to_try[idx - 1], "model_id", "unknown"
                    )
                    telemetry_data: dict[str, Any] = {
                        "from_model": previous_model_id,
                        "to_model": model_id,
                        "attempt": idx + 1,
                    }
                    if previous_error is not None:
                        telemetry_data["reason"] = previous_error.reason
                        telemetry_data["status_code"] = previous_error.status_code
                        telemetry_data["provider_code"] = previous_error.provider_code
                    await engine._emit(
                        "model_fallback",
                        run_id=ctx.run_id,
                        phase=Phase.ACT.value,
                        cycle=ctx.cycle_count,
                        data=telemetry_data,
                    )

                ctx.state._retry_state.pop(effective_step.id, None)
                return response

            except Exception as e:
                model_error = (
                    e if isinstance(e, ModelError)
                    else ModelError.from_exception(e, model_id)
                )
                errors.append(model_error)
                last_error = model_error

                if not ModelError.is_reason_recoverable(model_error.reason):
                    logger.warning(
                        "Non-recoverable reason from model %s (%s): %s",
                        model_id,
                        model_error.reason,
                        model_error.message,
                    )
                    raise model_error

                record_retry_attempt(ctx, effective_step)
                should_retry = (
                    attempt < max_attempts
                    and is_retryable_model_error(model_error, retry_policy)
                )
                logger.info(
                    "Recoverable error from model %s (model #%d, retry %d/%d): %s",
                    model_id,
                    idx + 1,
                    attempt,
                    max_attempts,
                    model_error.message,
                )
                if should_retry:
                    await sleep_for_retry(retry_policy, attempt)
                    continue
                break

        if last_error is not None and not ModelError.is_reason_recoverable(last_error.reason):
            raise last_error

    error_messages = [str(e) for e in errors]
    final_error = errors[-1] if errors else None
    raise ModelError(
        message=(
            f"All models failed after {len(models_to_try)} attempts. "
            f"Errors: {'; '.join(error_messages)}"
        ),
        model_id="fallback_chain",
        reason="fallback_exhausted",
        recoverable=False,
        status_code=final_error.status_code if final_error is not None else None,
        provider_code=final_error.provider_code if final_error is not None else None,
        cause=final_error,
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

    settings = ActRuntimeSettingsResolver(ctx, step)

    # Build messages if not explicitly provided
    if "messages" not in step.payload:
        message_context = settings.message_context()
        messages = ctx.state.build_messages(
            ctx,
            strategy=message_context.strategy,
            max_cycles=message_context.max_cycle_context,
        )
    else:
        messages = step.payload["messages"]

    transcript_settings = settings.transcript()
    if isinstance(messages, list):
        try:
            messages = normalize_transcript_messages(
                messages,
                strict=transcript_settings.strict,
                max_tool_result_chars=transcript_settings.max_tool_result_chars,
            )
        except ValueError as exc:
            raise ModelError(
                message=str(exc),
                model_id=getattr(engine.model, "model_id", "unknown"),
                reason="invalid_request",
                recoverable=False,
                cause=exc,
            ) from exc

    system = step.payload.get("system", engine.system)

    context_window_settings = settings.context_window()
    if isinstance(messages, list) and context_window_settings.guard_enabled:
        if context_window_settings.tokens is not None:
            guard = ContextWindowGuard(
                warn_threshold_tokens=context_window_settings.warn_tokens,
                block_threshold_tokens=context_window_settings.block_tokens,
            )

            estimated_tokens = await _estimate_tokens_for_messages(
                model=engine.model,
                messages=messages,
                system=system,
            )
            target_tokens = max(
                1,
                context_window_settings.tokens
                - max(
                    context_window_settings.warn_tokens,
                    context_window_settings.block_tokens,
                ),
            )
            if context_window_settings.pruning_enabled and estimated_tokens > target_tokens:
                pruned_messages, pruned_count = prune_messages_for_context_window(
                    messages,
                    target_tokens=target_tokens,
                )
                if pruned_count > 0:
                    pruned_estimated_tokens = await _estimate_tokens_for_messages(
                        model=engine.model,
                        messages=pruned_messages,
                        system=system,
                    )
                    await engine._emit(
                        "context_window_pruned",
                        run_id=ctx.run_id,
                        phase=Phase.ACT.value,
                        cycle=ctx.cycle_count,
                        step_id=step.id,
                        data={
                            "target_tokens": target_tokens,
                            "dropped_messages": pruned_count,
                            "estimated_tokens_before": estimated_tokens,
                            "estimated_tokens_after": pruned_estimated_tokens,
                        },
                    )
                    messages = pruned_messages
                    estimated_tokens = pruned_estimated_tokens

            assessment = guard.evaluate(
                estimated_tokens=estimated_tokens,
                context_window_tokens=context_window_settings.tokens,
            )
            if assessment.should_warn:
                await engine._emit(
                    "context_window_warning",
                    run_id=ctx.run_id,
                    phase=Phase.ACT.value,
                    cycle=ctx.cycle_count,
                    step_id=step.id,
                    data={
                        "estimated_tokens": assessment.estimated_tokens,
                        "context_window_tokens": assessment.context_window_tokens,
                        "remaining_tokens": assessment.remaining_tokens,
                        "warn_threshold_tokens": context_window_settings.warn_tokens,
                    },
                )

            if assessment.should_block:
                raise ModelError(
                    message=(
                        "Context window guard blocked model call: "
                        f"estimated_tokens={assessment.estimated_tokens}, "
                        f"remaining_tokens={assessment.remaining_tokens}, "
                        f"block_threshold={context_window_settings.block_tokens}, "
                        f"context_window_tokens={assessment.context_window_tokens}"
                    ),
                    model_id=getattr(engine.model, "model_id", "unknown"),
                    reason="context_window_exceeded",
                    recoverable=False,
                )

    # Get tool manifests (protocol objects) - adapter will convert to its format
    tool_manifests = engine._get_tool_manifests()
    tools = tool_manifests if tool_manifests else None

    cache_key: str | None = None
    if engine.cache_enabled_for_models():
        cache_key = engine.compute_cache_key(
            "model",
            {
                "model": getattr(engine.model, "model_id", "unknown"),
                "messages": messages,
                "system": system,
                "tools": [str(m) for m in tool_manifests],
                "stream": engine._token_callback is not None,
            },
        )
        cache_hit, cached_response = engine.cache_get(cache_key)
        if cache_hit:
            response = cast(ModelResponse, cached_response)
            ctx.state.output_raw = response.content
            ctx.state.last_model_response = response
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
                    "cached": True,
                },
                duration_ms=0.0,
            )
            return response.content

    await engine._emit(
        "model_called",
        run_id=ctx.run_id,
        phase=Phase.ACT.value,
        cycle=ctx.cycle_count,
        step_id=step.id,
    )

    start = time.monotonic()
    timeout_seconds = step.payload.get("timeout")

    async def _with_timeout(operation: Any) -> Any:
        if isinstance(timeout_seconds, (int, float)) and timeout_seconds > 0:
            return await asyncio.wait_for(operation, timeout=float(timeout_seconds))
        return await operation

    # Use fallback chain if configured (Task 15.0)
    if engine._token_callback is not None:
        token_cb = engine._token_callback

        async def _stream_call(m: Any) -> Any:
            return await _with_timeout(
                stream_model_response(
                    engine=engine,
                    model=m,
                    messages=messages,
                    system=system,
                    tools=tools,
                    token_callback=token_cb,
                )
            )

        response = await try_models_with_fallback(engine, ctx, _stream_call, step=step)
    else:
        async def _complete_call(m: Any) -> Any:
            return await _with_timeout(
                m.complete(
                    messages=messages,
                    system=system,
                    tools=tools,
                )
            )

        response = await try_models_with_fallback(engine, ctx, _complete_call, step=step)
    duration_ms = (time.monotonic() - start) * 1000

    # Track budget
    ctx.state.budget_state.record_model_usage(usage=response.usage, cost_usd=response.cost_usd)

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

    if cache_key is not None:
        engine.cache_set(
            cache_key,
            response,
            ttl_seconds=engine.default_cache_ttl_seconds(),
        )

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
            message=str(redact_sensitive_data(f"Tool step '{step.id}' failed: {e}")),
            tool_name=step.payload.get("tool"),
            cause=e,
        )
    return AxisError(
        message=str(redact_sensitive_data(f"Step '{step.id}' failed: {e}")),
        error_class=ErrorClass.RUNTIME,
        cause=e,
    )
