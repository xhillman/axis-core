"""Internal act-phase service for executing tool steps."""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from axis_core.config import RetryPolicy, ToolPolicy
from axis_core.context import RunContext
from axis_core.engine.phases.act_execution_utils import (
    is_retryable_tool_error,
    record_retry_attempt,
    resolve_retry_policy,
    sleep_for_retry,
)
from axis_core.errors import ToolError
from axis_core.protocols.planner import PlanStep
from axis_core.redaction import redact_sensitive_data
from axis_core.tool import Capability, ToolCallRecord, ToolContext, build_idempotency_key

if TYPE_CHECKING:
    from axis_core.engine.lifecycle import LifecycleEngine


@dataclass(frozen=True)
class ToolExecutionRequest:
    """Resolved inputs and policy for a single tool step."""

    tool_name: str
    args: Any
    tool_fn: Any
    capabilities: tuple[Capability, ...] | None
    retry_policy: RetryPolicy
    idempotency_key: str | None
    supports_ctx: bool
    supports_idempotency_key: bool
    timeout_seconds: float | int | None
    cache_key: str | None
    cache_ttl: int | None


class ActToolExecutionService:
    """Execute one tool step while preserving existing observable behavior."""

    def __init__(self, engine: LifecycleEngine, ctx: RunContext, step: PlanStep) -> None:
        self._engine = engine
        self._ctx = ctx
        self._step = step

    async def execute(self) -> Any:
        request = self._build_request()

        self._enforce_tool_policy(tool_name=request.tool_name)
        await self._confirm_destructive_tool(request)
        await self._emit_tool_called(request)

        start = time.monotonic()
        cache_hit, cached_result = await self._maybe_get_cached_result(request, start)
        if cache_hit:
            return cached_result

        result, last_error = await self._invoke_with_retry(request)
        duration_ms = (time.monotonic() - start) * 1000

        if last_error is not None:
            error_msg = str(
                redact_sensitive_data(f"Tool '{request.tool_name}' failed: {last_error}")
            )
            self._record_tool_call(
                request=request,
                result=None,
                error=error_msg,
                cached=False,
                duration_ms=duration_ms,
            )
            raise ToolError(
                message=error_msg,
                tool_name=request.tool_name,
                cause=last_error,
                recoverable=is_retryable_tool_error(last_error, request.retry_policy),
            ) from last_error

        self._ctx.state.budget_state.tool_calls += 1
        self._record_tool_call(
            request=request,
            result=result,
            error=None,
            cached=False,
            duration_ms=duration_ms,
        )
        self._store_cached_result(request, result)
        await self._emit_tool_returned(
            tool_name=request.tool_name,
            duration_ms=duration_ms,
            cached=False,
        )
        return result

    def _build_request(self) -> ToolExecutionRequest:
        tool_name = self._step.payload.get("tool", "")
        args = self._step.payload.get("args", {})

        if tool_name not in self._engine.tools:
            raise ToolError(
                message=f"Tool '{tool_name}' not found",
                tool_name=tool_name,
            )

        tool_fn = self._engine.tools[tool_name]
        manifest = getattr(tool_fn, "_axis_manifest", None)
        capabilities = cast(
            tuple[Capability, ...] | None,
            getattr(manifest, "capabilities", None),
        )
        retry_policy = resolve_retry_policy(
            self._ctx,
            self._step,
            tool_retry=getattr(manifest, "retry", None),
        )
        supports_ctx, supports_idempotency_key = self._inspect_tool_signature(tool_fn)
        cache_ttl = getattr(manifest, "cache_ttl", None)

        return ToolExecutionRequest(
            tool_name=tool_name,
            args=args,
            tool_fn=tool_fn,
            capabilities=capabilities,
            retry_policy=retry_policy,
            idempotency_key=self._resolve_tool_idempotency_key(tool_name=tool_name),
            supports_ctx=supports_ctx,
            supports_idempotency_key=supports_idempotency_key,
            timeout_seconds=self._step.payload.get("timeout", getattr(manifest, "timeout", None)),
            cache_key=self._build_cache_key(tool_name=tool_name, args=args, cache_ttl=cache_ttl),
            cache_ttl=cache_ttl,
        )

    def _inspect_tool_signature(self, tool_fn: Any) -> tuple[bool, bool]:
        try:
            tool_signature = inspect.signature(tool_fn)
        except (TypeError, ValueError):
            return False, False

        return (
            "ctx" in tool_signature.parameters,
            "idempotency_key" in tool_signature.parameters,
        )

    async def _maybe_get_cached_result(
        self,
        request: ToolExecutionRequest,
        start: float,
    ) -> tuple[bool, Any]:
        if request.cache_key is None:
            return False, None

        cache_hit, cached_result = self._engine.cache_get(request.cache_key)
        if not cache_hit:
            return False, None

        duration_ms = (time.monotonic() - start) * 1000
        self._record_tool_call(
            request=request,
            result=cached_result,
            error=None,
            cached=True,
            duration_ms=duration_ms,
        )
        await self._emit_tool_returned(
            tool_name=request.tool_name,
            duration_ms=duration_ms,
            cached=True,
        )
        return True, cached_result

    async def _invoke_with_retry(
        self,
        request: ToolExecutionRequest,
    ) -> tuple[Any, Exception | None]:
        max_attempts = max(1, request.retry_policy.max_attempts)
        result: Any = None
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                await self._engine.acquire_tool_slot(
                    self._ctx,
                    tool_name=request.tool_name,
                    step_id=self._step.id,
                )
                result = await self._invoke_once(request, attempt)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                record_retry_attempt(self._ctx, self._step)
                if (
                    attempt >= max_attempts
                    or not is_retryable_tool_error(exc, request.retry_policy)
                ):
                    break
                await sleep_for_retry(request.retry_policy, attempt)

        return result, last_error

    async def _invoke_once(self, request: ToolExecutionRequest, attempt: int) -> Any:
        tool_kwargs = dict(request.args)
        if "ctx" in tool_kwargs:
            tool_kwargs.pop("ctx")
        if (
            request.supports_idempotency_key
            and "idempotency_key" not in tool_kwargs
            and request.idempotency_key is not None
        ):
            tool_kwargs["idempotency_key"] = request.idempotency_key

        if request.supports_ctx:
            invoke_result = request.tool_fn(
                ctx=ToolContext(
                    run_id=self._ctx.run_id,
                    agent_id=self._ctx.agent_id,
                    cycle=self._ctx.cycle_count,
                    context=self._ctx.context,
                    budget=self._ctx.budget,
                    budget_state=self._ctx.state.budget_state,
                    idempotency_key=request.idempotency_key,
                    retry_attempt=attempt,
                ),
                **tool_kwargs,
            )
        else:
            invoke_result = request.tool_fn(**tool_kwargs)

        if isinstance(request.timeout_seconds, (int, float)) and request.timeout_seconds > 0:
            return await asyncio.wait_for(
                invoke_result,
                timeout=float(request.timeout_seconds),
            )
        return await invoke_result

    def _record_tool_call(
        self,
        *,
        request: ToolExecutionRequest,
        result: Any,
        error: str | None,
        cached: bool,
        duration_ms: float,
    ) -> None:
        self._ctx.state.append_tool_call(
            ToolCallRecord(
                tool_name=request.tool_name,
                call_id=self._step.id,
                args=dict(request.args),
                result=result,
                error=error,
                cached=cached,
                duration_ms=duration_ms,
                timestamp=time.time(),
            )
        )

    def _store_cached_result(self, request: ToolExecutionRequest, result: Any) -> None:
        if request.cache_key is None:
            return
        if not isinstance(request.cache_ttl, int) or request.cache_ttl <= 0:
            return
        self._engine.cache_set(request.cache_key, result, ttl_seconds=request.cache_ttl)

    def _build_cache_key(
        self,
        *,
        tool_name: str,
        args: Any,
        cache_ttl: int | None,
    ) -> str | None:
        if not (
            self._engine.cache_enabled_for_tools()
            and isinstance(cache_ttl, int)
            and cache_ttl > 0
        ):
            return None
        return self._engine.compute_cache_key(
            "tool",
            {
                "tool": tool_name,
                "args": args,
            },
        )

    def _resolve_tool_idempotency_key(self, *, tool_name: str) -> str | None:
        if "idempotency_key" in self._step.payload:
            raw_key = self._step.payload.get("idempotency_key")
            if raw_key is None:
                return None
            if isinstance(raw_key, str):
                key = raw_key.strip()
                return key if key else None
            return str(raw_key)

        return build_idempotency_key(
            run_id=self._ctx.run_id,
            cycle=self._ctx.cycle_count,
            step_id=self._step.id,
            tool_name=tool_name,
        )

    async def _confirm_destructive_tool(self, request: ToolExecutionRequest) -> None:
        if Capability.DESTRUCTIVE not in (request.capabilities or ()):
            return

        config = getattr(self._ctx, "config", None)
        confirmation_handler = getattr(config, "confirmation_handler", None)
        if confirmation_handler is None:
            raise ToolError(
                message=(
                    f"Tool '{request.tool_name}' requires confirmation handler for "
                    "Capability.DESTRUCTIVE"
                ),
                tool_name=request.tool_name,
                recoverable=False,
            )
        if not callable(confirmation_handler):
            raise ToolError(
                message=f"Confirmation handler for tool '{request.tool_name}' is not callable",
                tool_name=request.tool_name,
                recoverable=False,
            )

        confirmation_args = request.args if isinstance(request.args, dict) else {}
        try:
            decision = confirmation_handler(request.tool_name, confirmation_args)
            if inspect.isawaitable(decision):
                decision = await cast(Any, decision)
        except Exception as exc:
            raise ToolError(
                message=f"Confirmation handler failed for tool '{request.tool_name}': {exc}",
                tool_name=request.tool_name,
                cause=exc,
                recoverable=False,
            ) from exc

        if not isinstance(decision, bool):
            raise ToolError(
                message=(
                    f"Confirmation handler for tool '{request.tool_name}' must return bool, "
                    f"got {type(decision).__name__}"
                ),
                tool_name=request.tool_name,
                recoverable=False,
            )

        if not decision:
            raise ToolError(
                message=(
                    f"Tool '{request.tool_name}' execution rejected by confirmation handler"
                ),
                tool_name=request.tool_name,
                recoverable=False,
            )

    def _enforce_tool_policy(self, *, tool_name: str) -> None:
        runtime_config = getattr(self._ctx, "config", None)
        tool_policy = getattr(runtime_config, "tool_policy", None)
        if tool_policy is None:
            return
        if not isinstance(tool_policy, ToolPolicy):
            raise ToolError(
                message="Invalid runtime config: tool_policy must be ToolPolicy or None",
                tool_name=tool_name,
                recoverable=False,
            )

        allowed, reason = tool_policy.evaluate(tool_name)
        if allowed:
            return

        detail = f" ({reason})" if reason else ""
        raise ToolError(
            message=f"Tool '{tool_name}' blocked by tool policy{detail}",
            tool_name=tool_name,
            recoverable=False,
        )

    async def _emit_tool_called(self, request: ToolExecutionRequest) -> None:
        from axis_core.engine.lifecycle import Phase

        await self._engine._emit(
            "tool_called",
            run_id=self._ctx.run_id,
            phase=Phase.ACT.value,
            cycle=self._ctx.cycle_count,
            step_id=self._step.id,
            data={
                "tool": request.tool_name,
                "args": request.args,
                "idempotency_key": request.idempotency_key,
            },
        )

    async def _emit_tool_returned(
        self,
        *,
        tool_name: str,
        duration_ms: float,
        cached: bool,
    ) -> None:
        from axis_core.engine.lifecycle import Phase

        await self._engine._emit(
            "tool_returned",
            run_id=self._ctx.run_id,
            phase=Phase.ACT.value,
            cycle=self._ctx.cycle_count,
            step_id=self._step.id,
            data={"tool": tool_name, "duration_ms": duration_ms, "cached": cached},
            duration_ms=duration_ms,
        )


__all__ = ["ActToolExecutionService", "ToolExecutionRequest"]
