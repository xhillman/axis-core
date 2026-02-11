"""Act phase: execute plan steps with dependency handling (AD-003, AD-042)."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import random
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from axis_core.config import RetryPolicy
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
from axis_core.redaction import redact_sensitive_data
from axis_core.tool import Capability, ToolCallRecord, ToolContext

if TYPE_CHECKING:
    from axis_core.engine.lifecycle import LifecycleEngine

logger = logging.getLogger("axis_core.engine")


def _resolve_retry_policy(
    ctx: RunContext,
    step: PlanStep,
    tool_retry: RetryPolicy | None = None,
) -> RetryPolicy:
    """Resolve effective retry policy for a step."""
    if step.retry_policy is not None:
        return step.retry_policy
    if tool_retry is not None:
        return tool_retry

    config_retry = getattr(getattr(ctx, "config", None), "retry", None)
    if isinstance(config_retry, RetryPolicy):
        return config_retry

    return RetryPolicy(max_attempts=1, jitter=False, initial_delay=0.0, max_delay=0.0)


def _matches_retry_filter(error: Exception, retry_policy: RetryPolicy) -> bool:
    """Return True when retry_on filter allows retry for this error."""
    if retry_policy.retry_on is None:
        return True

    filters = {entry.lower() for entry in retry_policy.retry_on}
    error_name = type(error).__name__.lower()
    return any(token in error_name for token in filters)


def _is_retryable_tool_error(error: Exception, retry_policy: RetryPolicy) -> bool:
    """Determine whether a tool failure should be retried."""
    if not _matches_retry_filter(error, retry_policy):
        return False

    if isinstance(error, AxisError):
        return error.recoverable

    if isinstance(error, (TypeError, ValueError, KeyError)):
        return False

    return True


def _is_retryable_model_error(error: ModelError, retry_policy: RetryPolicy) -> bool:
    """Determine whether a model failure should be retried."""
    if not error.recoverable:
        return False
    return _matches_retry_filter(error.cause or error, retry_policy)


def _retry_delay_seconds(retry_policy: RetryPolicy, attempt: int) -> float:
    """Compute retry delay before the next attempt."""
    if retry_policy.backoff == "fixed":
        delay = retry_policy.initial_delay
    elif retry_policy.backoff == "linear":
        delay = retry_policy.initial_delay * attempt
    else:
        delay = retry_policy.initial_delay * (2 ** max(0, attempt - 1))

    delay = min(delay, retry_policy.max_delay)
    if retry_policy.jitter and delay > 0:
        delay *= 0.5 + random.random()
    return max(0.0, delay)


async def _sleep_for_retry(retry_policy: RetryPolicy, attempt: int) -> None:
    """Sleep based on retry policy for an attempt number."""
    delay = _retry_delay_seconds(retry_policy, attempt)
    if delay > 0:
        await asyncio.sleep(delay)


def _record_retry_attempt(ctx: RunContext, step: PlanStep) -> None:
    """Track retry attempts in run state (not persisted by design)."""
    retry_state = ctx.state._retry_state.get(step.id, {"attempts": 0})
    retry_state["attempts"] = int(retry_state.get("attempts", 0)) + 1
    ctx.state._retry_state[step.id] = retry_state


async def _confirm_destructive_tool(
    ctx: RunContext,
    *,
    tool_name: str,
    args: Any,
    capabilities: tuple[Capability, ...] | None,
) -> None:
    """Require explicit approval before destructive tool execution."""
    if Capability.DESTRUCTIVE not in (capabilities or ()):
        return

    config = getattr(ctx, "config", None)
    confirmation_handler = getattr(config, "confirmation_handler", None)

    if confirmation_handler is None:
        raise ToolError(
            message=(
                f"Tool '{tool_name}' requires confirmation handler for "
                "Capability.DESTRUCTIVE"
            ),
            tool_name=tool_name,
            recoverable=False,
        )

    if not callable(confirmation_handler):
        raise ToolError(
            message=(
                f"Confirmation handler for tool '{tool_name}' is not callable"
            ),
            tool_name=tool_name,
            recoverable=False,
        )

    confirmation_args = args if isinstance(args, dict) else {}

    try:
        decision = confirmation_handler(tool_name, confirmation_args)
        if inspect.isawaitable(decision):
            decision = await cast(Any, decision)
    except Exception as e:
        raise ToolError(
            message=f"Confirmation handler failed for tool '{tool_name}': {e}",
            tool_name=tool_name,
            cause=e,
            recoverable=False,
        ) from e

    if not isinstance(decision, bool):
        raise ToolError(
            message=(
                f"Confirmation handler for tool '{tool_name}' must return bool, "
                f"got {type(decision).__name__}"
            ),
            tool_name=tool_name,
            recoverable=False,
        )

    if not decision:
        raise ToolError(
            message=f"Tool '{tool_name}' execution rejected by confirmation handler",
            tool_name=tool_name,
            recoverable=False,
        )


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
    manifest = getattr(tool_fn, "_axis_manifest", None)
    capabilities = cast(
        tuple[Capability, ...] | None,
        getattr(manifest, "capabilities", None),
    )

    await _confirm_destructive_tool(
        ctx,
        tool_name=tool_name,
        args=args,
        capabilities=capabilities,
    )

    retry_policy = _resolve_retry_policy(
        ctx,
        step,
        tool_retry=getattr(manifest, "retry", None),
    )
    max_attempts = max(1, retry_policy.max_attempts)

    await engine._emit(
        "tool_called",
        run_id=ctx.run_id,
        phase=Phase.ACT.value,
        cycle=ctx.cycle_count,
        step_id=step.id,
        data={"tool": tool_name, "args": args},
    )

    start = time.monotonic()
    cache_key: str | None = None
    tool_cache_ttl = getattr(manifest, "cache_ttl", None)
    if (
        engine.cache_enabled_for_tools()
        and isinstance(tool_cache_ttl, int)
        and tool_cache_ttl > 0
    ):
        cache_key = engine.compute_cache_key(
            "tool",
            {
                "tool": tool_name,
                "args": args,
            },
        )
        cache_hit, cached_result = engine.cache_get(cache_key)
        if cache_hit:
            duration_ms = (time.monotonic() - start) * 1000
            ctx.state.append_tool_call(ToolCallRecord(
                tool_name=tool_name,
                call_id=step.id,
                args=dict(args),
                result=cached_result,
                error=None,
                cached=True,
                duration_ms=duration_ms,
                timestamp=time.time(),
            ))
            await engine._emit(
                "tool_returned",
                run_id=ctx.run_id,
                phase=Phase.ACT.value,
                cycle=ctx.cycle_count,
                step_id=step.id,
                data={"tool": tool_name, "duration_ms": duration_ms, "cached": True},
                duration_ms=duration_ms,
            )
            return cached_result

    async def _invoke_tool_once() -> Any:
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
            return await tool_fn(ctx=tool_ctx, **tool_kwargs)
        return await tool_fn(**tool_kwargs)

    timeout_seconds = step.payload.get("timeout", getattr(manifest, "timeout", None))
    last_error: Exception | None = None
    result: Any = None

    for attempt in range(1, max_attempts + 1):
        try:
            await engine.acquire_tool_slot(ctx, tool_name=tool_name, step_id=step.id)
            if isinstance(timeout_seconds, (int, float)) and timeout_seconds > 0:
                result = await asyncio.wait_for(
                    _invoke_tool_once(),
                    timeout=float(timeout_seconds),
                )
            else:
                result = await _invoke_tool_once()
            last_error = None
            break
        except Exception as e:
            last_error = e
            _record_retry_attempt(ctx, step)
            if attempt >= max_attempts or not _is_retryable_tool_error(e, retry_policy):
                break
            await _sleep_for_retry(retry_policy, attempt)

    if last_error is not None:
        error_msg = str(redact_sensitive_data(
            f"Tool '{tool_name}' failed: {last_error}"
        ))
        duration_ms = (time.monotonic() - start) * 1000
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
            cause=last_error,
            recoverable=_is_retryable_tool_error(last_error, retry_policy),
        ) from last_error

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

    if cache_key is not None and isinstance(tool_cache_ttl, int) and tool_cache_ttl > 0:
        engine.cache_set(cache_key, result, ttl_seconds=tool_cache_ttl)

    await engine._emit(
        "tool_returned",
        run_id=ctx.run_id,
        phase=Phase.ACT.value,
        cycle=ctx.cycle_count,
        step_id=step.id,
        data={"tool": tool_name, "duration_ms": duration_ms, "cached": False},
        duration_ms=duration_ms,
    )

    return result


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
    retry_policy = _resolve_retry_policy(ctx, effective_step)
    max_attempts = max(1, retry_policy.max_attempts)

    for idx, model in enumerate(models_to_try):
        model_id = getattr(model, "model_id", "unknown")
        last_error: ModelError | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                await engine.acquire_model_slot(ctx, step_id=effective_step.id)
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

                ctx.state._retry_state.pop(effective_step.id, None)
                return response

            except Exception as e:
                model_error = (
                    e if isinstance(e, ModelError)
                    else ModelError.from_exception(e, model_id)
                )
                errors.append(model_error)
                last_error = model_error

                if not model_error.recoverable:
                    logger.warning(
                        "Non-recoverable error from model %s: %s",
                        model_id,
                        model_error.message,
                    )
                    raise model_error

                _record_retry_attempt(ctx, effective_step)
                should_retry = (
                    attempt < max_attempts
                    and _is_retryable_model_error(model_error, retry_policy)
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
                    await _sleep_for_retry(retry_policy, attempt)
                    continue
                break

        if last_error is not None and not last_error.recoverable:
            raise last_error

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
        if strategy not in {"smart", "full", "minimal"}:
            logger.warning(
                "Invalid AXIS_CONTEXT_STRATEGY='%s'; falling back to 'smart'",
                strategy,
            )
            strategy = "smart"

        raw_max_cycles = os.getenv("AXIS_MAX_CYCLE_CONTEXT", "5")
        try:
            max_cycles = int(raw_max_cycles)
        except ValueError:
            logger.warning(
                "Invalid AXIS_MAX_CYCLE_CONTEXT='%s'; falling back to 5",
                raw_max_cycles,
            )
            max_cycles = 5
        if max_cycles < 0:
            logger.warning(
                "Negative AXIS_MAX_CYCLE_CONTEXT=%d; falling back to 5",
                max_cycles,
            )
            max_cycles = 5
        messages = ctx.state.build_messages(ctx, strategy=strategy, max_cycles=max_cycles)
    else:
        messages = step.payload["messages"]

    system = step.payload.get("system", engine.system)

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
