"""Internal runtime primitives for tool execution."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

from axis_core.budget import Budget, BudgetState


@dataclass
class ToolContext:
    """Runtime context passed to tool functions."""

    run_id: str
    agent_id: str
    cycle: int
    context: dict[str, object]
    budget: Budget
    budget_state: BudgetState
    idempotency_key: str | None = None
    retry_attempt: int = 1
    _initialized: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Mark context as initialized to enable read-only protection."""
        object.__setattr__(self, "_initialized", True)

    def __setattr__(self, name: str, value: object) -> None:
        """Enforce read-only fields after initialization."""
        if getattr(self, "_initialized", False):
            if name in (
                "run_id",
                "agent_id",
                "cycle",
                "budget",
                "budget_state",
                "idempotency_key",
                "retry_attempt",
            ):
                raise AttributeError(f"ToolContext.{name} is read-only")
        object.__setattr__(self, name, value)


_IDEMPOTENCY_RESULTS_CONTEXT_KEY = "__axis_idempotency_results__"


def build_idempotency_key(*, run_id: str, cycle: int, step_id: str, tool_name: str) -> str:
    """Build a stable tool idempotency key for a specific run/cycle/step."""
    return f"axis:{run_id}:{cycle}:{step_id}:{tool_name}"


def _get_idempotency_store(ctx: ToolContext) -> dict[str, Any]:
    raw_store = ctx.context.get(_IDEMPOTENCY_RESULTS_CONTEXT_KEY)
    if isinstance(raw_store, dict):
        return cast(dict[str, Any], raw_store)

    store: dict[str, Any] = {}
    ctx.context[_IDEMPOTENCY_RESULTS_CONTEXT_KEY] = store
    return store


def get_idempotent_result(
    ctx: ToolContext,
    *,
    key: str | None = None,
) -> tuple[bool, Any]:
    """Return cached idempotent result for key or context key."""
    effective_key = key if key is not None else ctx.idempotency_key
    if effective_key is None:
        return (False, None)

    store = _get_idempotency_store(ctx)
    if effective_key not in store:
        return (False, None)
    return (True, store[effective_key])


def set_idempotent_result(
    ctx: ToolContext,
    result: Any,
    *,
    key: str | None = None,
) -> None:
    """Persist idempotent result in tool context using key or context key."""
    effective_key = key if key is not None else ctx.idempotency_key
    if effective_key is None:
        return

    store = _get_idempotency_store(ctx)
    store[effective_key] = result


async def run_idempotent(
    ctx: ToolContext,
    operation: Callable[[], Any],
    *,
    key: str | None = None,
) -> Any:
    """Run operation and memoize result by idempotency key when available."""
    found, cached_result = get_idempotent_result(ctx, key=key)
    if found:
        return cached_result

    result = operation()
    if inspect.isawaitable(result):
        result = await result
    set_idempotent_result(ctx, result, key=key)
    return result


class RateLimiter:
    """Token bucket rate limiter for controlling request rates."""

    def __init__(self, count: int, period_seconds: float) -> None:
        self._count = count
        self._period_seconds = period_seconds
        self._tokens = float(count)
        self._last_refill = time.time()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time since last refill."""
        now = time.time()
        elapsed = now - self._last_refill

        if elapsed > 0 and self._period_seconds > 0:
            tokens_to_add = (elapsed / self._period_seconds) * self._count
            self._tokens = min(self._count, self._tokens + tokens_to_add)
            self._last_refill = now

    def try_acquire(self) -> bool:
        """Try to acquire a token without waiting."""
        self._refill()

        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True

        return False

    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary."""
        async with self._lock:
            while True:
                self._refill()

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return

                if self._count > 0 and self._period_seconds > 0:
                    time_per_token = self._period_seconds / self._count
                    await asyncio.sleep(time_per_token / 2)
                else:
                    await asyncio.sleep(0.1)


__all__ = [
    "RateLimiter",
    "ToolContext",
    "build_idempotency_key",
    "get_idempotent_result",
    "run_idempotent",
    "set_idempotent_result",
]
