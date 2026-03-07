"""Internal act-phase service for executing model steps."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from axis_core.context import (
    ContextWindowGuard,
    ModelCallRecord,
    RunContext,
    normalize_transcript_messages,
    prune_messages_for_context_window,
)
from axis_core.engine.phases.act_model_invocation import (
    estimate_tokens_for_messages,
    stream_model_response,
    try_models_with_fallback,
)
from axis_core.engine.phases.act_runtime_settings import (
    ActRuntimeSettingsResolver,
    ContextWindowRuntimeSettings,
    TranscriptRuntimeSettings,
)
from axis_core.errors import ModelError
from axis_core.protocols.model import ModelResponse
from axis_core.protocols.planner import PlanStep

if TYPE_CHECKING:
    from axis_core.engine.lifecycle import LifecycleEngine


@dataclass(frozen=True)
class ModelExecutionRequest:
    """Resolved inputs and policy for a single model step."""

    messages: Any
    system: str | None
    tools: Any | None
    tool_manifests: list[Any]
    cache_key: str | None
    timeout_seconds: float | int | None


class ActModelExecutionService:
    """Execute one model step while preserving existing observable behavior."""

    def __init__(self, engine: LifecycleEngine, ctx: RunContext, step: PlanStep) -> None:
        self._engine = engine
        self._ctx = ctx
        self._step = step

    async def execute(self) -> Any:
        request = await self._build_request()

        cache_hit, cached_content = await self._maybe_get_cached_response(request)
        if cache_hit:
            return cached_content

        await self._emit_model_called()
        start = time.monotonic()
        response = await self._invoke_model(request)
        duration_ms = (time.monotonic() - start) * 1000

        self._ctx.state.budget_state.record_model_usage(
            usage=response.usage,
            cost_usd=response.cost_usd,
        )
        self._ctx.state.append_model_call(
            ModelCallRecord(
                model_id=self._model_id(self._engine.model),
                call_id=self._step.id,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cost_usd=response.cost_usd,
                duration_ms=duration_ms,
                timestamp=time.time(),
            )
        )

        await self._engine._emit(
            "model_returned",
            run_id=self._ctx.run_id,
            phase=self._phase_value(),
            cycle=self._ctx.cycle_count,
            step_id=self._step.id,
            data={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cost_usd": response.cost_usd,
            },
            duration_ms=duration_ms,
        )

        self._ctx.state.output_raw = response.content
        self._ctx.state.last_model_response = response

        if request.cache_key is not None:
            self._engine.cache_set(
                request.cache_key,
                response,
                ttl_seconds=self._engine.default_cache_ttl_seconds(),
            )

        return response.content

    async def _build_request(self) -> ModelExecutionRequest:
        settings = ActRuntimeSettingsResolver(self._ctx, self._step)
        system = self._step.payload.get("system", self._engine.system)
        messages = await self._resolve_messages(
            transcript_settings=settings.transcript(),
            context_window_settings=settings.context_window(),
            system=system,
            settings=settings,
        )
        tool_manifests = self._engine._get_tool_manifests()
        tools = tool_manifests if tool_manifests else None

        return ModelExecutionRequest(
            messages=messages,
            system=system,
            tools=tools,
            tool_manifests=tool_manifests,
            cache_key=self._build_cache_key(
                messages=messages,
                system=system,
                tool_manifests=tool_manifests,
            ),
            timeout_seconds=self._step.payload.get("timeout"),
        )

    async def _resolve_messages(
        self,
        *,
        transcript_settings: TranscriptRuntimeSettings,
        context_window_settings: ContextWindowRuntimeSettings,
        system: str | None,
        settings: ActRuntimeSettingsResolver,
    ) -> Any:
        if "messages" not in self._step.payload:
            message_context = settings.message_context()
            messages = self._ctx.state.build_messages(
                self._ctx,
                strategy=message_context.strategy,
                max_cycles=message_context.max_cycle_context,
            )
        else:
            messages = self._step.payload["messages"]

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
                    model_id=self._model_id(self._engine.model),
                    reason="invalid_request",
                    recoverable=False,
                    cause=exc,
                ) from exc

        return await self._apply_context_window_guard(
            messages=messages,
            system=system,
            settings=context_window_settings,
        )

    async def _apply_context_window_guard(
        self,
        *,
        messages: Any,
        system: str | None,
        settings: ContextWindowRuntimeSettings,
    ) -> Any:
        if not isinstance(messages, list) or not settings.guard_enabled:
            return messages

        if settings.tokens is None:
            return messages

        guard = ContextWindowGuard(
            warn_threshold_tokens=settings.warn_tokens,
            block_threshold_tokens=settings.block_tokens,
        )
        estimated_tokens = await estimate_tokens_for_messages(
            model=self._engine.model,
            messages=messages,
            system=system,
        )
        target_tokens = max(
            1,
            settings.tokens - max(settings.warn_tokens, settings.block_tokens),
        )
        if settings.pruning_enabled and estimated_tokens > target_tokens:
            pruned_messages, pruned_count = prune_messages_for_context_window(
                messages,
                target_tokens=target_tokens,
            )
            if pruned_count > 0:
                pruned_estimated_tokens = await estimate_tokens_for_messages(
                    model=self._engine.model,
                    messages=pruned_messages,
                    system=system,
                )
                await self._engine._emit(
                    "context_window_pruned",
                    run_id=self._ctx.run_id,
                    phase=self._phase_value(),
                    cycle=self._ctx.cycle_count,
                    step_id=self._step.id,
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
            context_window_tokens=settings.tokens,
        )
        if assessment.should_warn:
            await self._engine._emit(
                "context_window_warning",
                run_id=self._ctx.run_id,
                phase=self._phase_value(),
                cycle=self._ctx.cycle_count,
                step_id=self._step.id,
                data={
                    "estimated_tokens": assessment.estimated_tokens,
                    "context_window_tokens": assessment.context_window_tokens,
                    "remaining_tokens": assessment.remaining_tokens,
                    "warn_threshold_tokens": settings.warn_tokens,
                },
            )

        if assessment.should_block:
            raise ModelError(
                message=(
                    "Context window guard blocked model call: "
                    f"estimated_tokens={assessment.estimated_tokens}, "
                    f"remaining_tokens={assessment.remaining_tokens}, "
                    f"block_threshold={settings.block_tokens}, "
                    f"context_window_tokens={assessment.context_window_tokens}"
                ),
                model_id=self._model_id(self._engine.model),
                reason="context_window_exceeded",
                recoverable=False,
            )

        return messages

    def _build_cache_key(
        self,
        *,
        messages: Any,
        system: str | None,
        tool_manifests: list[Any],
    ) -> str | None:
        if not self._engine.cache_enabled_for_models():
            return None

        return self._engine.compute_cache_key(
            "model",
            {
                "model": self._model_id(self._engine.model),
                "messages": messages,
                "system": system,
                "tools": [str(manifest) for manifest in tool_manifests],
                "stream": self._engine._token_callback is not None,
            },
        )

    async def _maybe_get_cached_response(
        self,
        request: ModelExecutionRequest,
    ) -> tuple[bool, Any]:
        if request.cache_key is None:
            return False, None

        cache_hit, cached_response = self._engine.cache_get(request.cache_key)
        if not cache_hit:
            return False, None

        response = cast(ModelResponse, cached_response)
        self._ctx.state.output_raw = response.content
        self._ctx.state.last_model_response = response
        await self._engine._emit(
            "model_returned",
            run_id=self._ctx.run_id,
            phase=self._phase_value(),
            cycle=self._ctx.cycle_count,
            step_id=self._step.id,
            data={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cost_usd": response.cost_usd,
                "cached": True,
            },
            duration_ms=0.0,
        )
        return True, response.content

    async def _invoke_model(self, request: ModelExecutionRequest) -> ModelResponse:
        async def _with_timeout(operation: Any) -> Any:
            if (
                isinstance(request.timeout_seconds, (int, float))
                and request.timeout_seconds > 0
            ):
                return await asyncio.wait_for(
                    operation,
                    timeout=float(request.timeout_seconds),
                )
            return await operation

        if self._engine._token_callback is not None:
            token_callback = self._engine._token_callback

            async def _stream_call(model: Any) -> Any:
                return await _with_timeout(
                    stream_model_response(
                        engine=self._engine,
                        model=model,
                        messages=request.messages,
                        system=request.system,
                        tools=request.tools,
                        token_callback=token_callback,
                    )
                )

            return cast(
                ModelResponse,
                await try_models_with_fallback(
                    self._engine,
                    self._ctx,
                    _stream_call,
                    step=self._step,
                ),
            )

        async def _complete_call(model: Any) -> Any:
            return await _with_timeout(
                model.complete(
                    messages=request.messages,
                    system=request.system,
                    tools=request.tools,
                )
            )

        return cast(
            ModelResponse,
            await try_models_with_fallback(
                self._engine,
                self._ctx,
                _complete_call,
                step=self._step,
            ),
        )

    async def _emit_model_called(self) -> None:
        await self._engine._emit(
            "model_called",
            run_id=self._ctx.run_id,
            phase=self._phase_value(),
            cycle=self._ctx.cycle_count,
            step_id=self._step.id,
        )

    def _model_id(self, model: Any) -> str:
        return getattr(model, "model_id", "unknown")

    def _phase_value(self) -> str:
        from axis_core.engine.lifecycle import Phase

        return Phase.ACT.value


__all__ = [
    "ActModelExecutionService",
    "ModelExecutionRequest",
    "stream_model_response",
    "try_models_with_fallback",
]
