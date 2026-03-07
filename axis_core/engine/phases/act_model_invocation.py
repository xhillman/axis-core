"""Shared helpers for act-phase model fallback and streaming aggregation."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, cast

from axis_core.context import RunContext
from axis_core.engine.phases.act_execution_utils import (
    is_retryable_model_error,
    record_retry_attempt,
    resolve_retry_policy,
    sleep_for_retry,
)
from axis_core.errors import ModelError
from axis_core.protocols.model import ModelResponse, ToolCall, UsageStats
from axis_core.protocols.planner import PlanStep, StepType

if TYPE_CHECKING:
    from axis_core.engine.lifecycle import LifecycleEngine

logger = logging.getLogger("axis_core.engine")


async def try_models_with_fallback(
    engine: LifecycleEngine,
    ctx: RunContext,
    call_fn: Any,
    step: PlanStep | None = None,
) -> Any:
    """Try primary model then fallbacks on recoverable errors."""
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
                        models_to_try[idx - 1],
                        "model_id",
                        "unknown",
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
            except Exception as exc:
                model_error = (
                    exc
                    if isinstance(exc, ModelError)
                    else ModelError.from_exception(exc, model_id)
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

    error_messages = [str(error) for error in errors]
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
    del engine

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
                    idx,
                    {"id": None, "name": None, "arguments": ""},
                )
                if "id" in delta:
                    entry["id"] = delta["id"]
                function_payload = delta.get("function") or {}
                name = function_payload.get("name")
                if name:
                    entry["name"] = name
                args_text = function_payload.get("arguments")
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
    input_tokens = await estimate_tokens_for_messages(
        model=model,
        messages=messages,
        system=system,
    )
    output_tokens = await estimate_tokens(model=model, text=content)
    usage = UsageStats(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )
    cost = await estimate_cost(
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


async def estimate_tokens(model: Any, text: str) -> int:
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


async def estimate_tokens_for_messages(
    model: Any,
    messages: Any,
    system: str | None,
) -> int:
    parts: list[str] = []
    if system:
        parts.append(system)
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(part) for part in content)
        parts.append(str(content))
    return await estimate_tokens(model=model, text="\n".join(parts))


async def estimate_cost(
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


__all__ = [
    "estimate_cost",
    "estimate_tokens",
    "estimate_tokens_for_messages",
    "stream_model_response",
    "try_models_with_fallback",
]
