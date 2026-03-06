"""Internal runtime policy helpers for lifecycle execution."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from axis_core.config import CacheConfig, RateLimits, Timeouts
from axis_core.context import RunContext
from axis_core.errors import BudgetError, ConfigError
from axis_core.errors import (
    TimeoutError as AxisTimeoutError,
)
from axis_core.tool import RateLimiter

logger = logging.getLogger("axis_core.engine")
T = TypeVar("T")


@dataclass
class _CacheEntry:
    """Internal in-memory cache entry."""

    value: Any
    expires_at: float
    size_bytes: int


class RuntimeCacheService:
    """Manage per-run in-memory caching for model responses and tool results."""

    def __init__(self) -> None:
        self._active_cache: CacheConfig | None = None
        self._cache_store: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._cache_size_bytes = 0

    def configure(self, cache: CacheConfig | None) -> None:
        """Apply cache settings for the active run."""
        if cache is not None and cache.enabled and cache.backend != "memory":
            logger.warning(
                "Cache backend '%s' is not supported for engine runtime cache. "
                "Using in-memory cache only.",
                cache.backend,
            )
            cache = CacheConfig(
                enabled=cache.enabled,
                model_responses=cache.model_responses,
                tool_results=cache.tool_results,
                ttl=cache.ttl,
                backend="memory",
                max_size_mb=cache.max_size_mb,
            )
        self._active_cache = cache

    @staticmethod
    def _estimate_cache_size(value: Any) -> int:
        """Estimate cache entry size in bytes."""
        try:
            serialized = json.dumps(value, sort_keys=True, default=str)
        except Exception:
            serialized = str(value)
        return len(serialized.encode("utf-8"))

    def _cache_max_bytes(self) -> int:
        if self._active_cache is None:
            return 0
        return max(0, int(self._active_cache.max_size_mb * 1024 * 1024))

    def _is_cache_active(self) -> bool:
        return (
            self._active_cache is not None
            and self._active_cache.enabled
            and self._active_cache.backend == "memory"
        )

    def enabled_for_models(self) -> bool:
        return self._is_cache_active() and bool(
            self._active_cache and self._active_cache.model_responses
        )

    def enabled_for_tools(self) -> bool:
        return self._is_cache_active() and bool(
            self._active_cache and self._active_cache.tool_results
        )

    def default_ttl_seconds(self) -> int:
        if self._active_cache is None:
            return 0
        return max(0, self._active_cache.ttl)

    def compute_key(self, namespace: str, payload: dict[str, Any]) -> str:
        canonical = json.dumps(
            {"namespace": namespace, "payload": payload},
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"{namespace}:{digest}"

    def get(self, key: str) -> tuple[bool, Any]:
        if not self._is_cache_active():
            return False, None

        entry = self._cache_store.get(key)
        if entry is None:
            return False, None

        now = time.monotonic()
        if entry.expires_at <= now:
            self._cache_store.pop(key, None)
            self._cache_size_bytes = max(0, self._cache_size_bytes - entry.size_bytes)
            return False, None

        self._cache_store.move_to_end(key)
        return True, entry.value

    def set(
        self,
        key: str,
        value: Any,
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        if not self._is_cache_active():
            return

        ttl = self.default_ttl_seconds() if ttl_seconds is None else max(0, ttl_seconds)
        if ttl <= 0:
            return

        size_bytes = self._estimate_cache_size(value)
        max_bytes = self._cache_max_bytes()
        if max_bytes > 0 and size_bytes > max_bytes:
            return

        existing = self._cache_store.get(key)
        if existing is not None:
            self._cache_size_bytes = max(0, self._cache_size_bytes - existing.size_bytes)
            self._cache_store.pop(key, None)

        self._cache_store[key] = _CacheEntry(
            value=value,
            expires_at=time.monotonic() + ttl,
            size_bytes=size_bytes,
        )
        self._cache_size_bytes += size_bytes
        self._cache_store.move_to_end(key)

        while max_bytes > 0 and self._cache_size_bytes > max_bytes and self._cache_store:
            _, evicted = self._cache_store.popitem(last=False)
            self._cache_size_bytes = max(0, self._cache_size_bytes - evicted.size_bytes)


class RateLimitPolicyService:
    """Build and acquire runtime rate-limiters for model and tool calls."""

    def __init__(self) -> None:
        self._model_rate_limiter: RateLimiter | None = None
        self._tool_rate_limiter: RateLimiter | None = None
        self._tool_specific_rate_limiters: dict[str, RateLimiter] = {}

    @staticmethod
    def parse_rate_limit(rate_spec: str, field_name: str) -> tuple[int, float]:
        """Parse a rate string like '10/minute' into count/period seconds."""
        if "/" not in rate_spec:
            raise ConfigError(
                message=(
                    f"Invalid rate format for {field_name}: '{rate_spec}'. "
                    "Expected format: 'count/period' (e.g., '60/minute')"
                )
            )

        count_raw, period_raw = rate_spec.split("/", 1)
        try:
            count = int(count_raw)
        except ValueError as e:
            raise ConfigError(
                message=(
                    f"Invalid rate count for {field_name}: '{count_raw}'. "
                    "Count must be an integer."
                )
            ) from e

        period_map = {
            "second": 1.0,
            "minute": 60.0,
            "hour": 3600.0,
        }
        period_seconds = period_map.get(period_raw)
        if period_seconds is None:
            raise ConfigError(
                message=(
                    f"Invalid rate period for {field_name}: '{period_raw}'. "
                    "Must be 'second', 'minute', or 'hour'."
                )
            )

        return count, period_seconds

    def configure(self, rate_limits: RateLimits | None, tools: dict[str, Any]) -> None:
        """Apply rate-limit settings for the active run."""
        self._model_rate_limiter = None
        self._tool_rate_limiter = None
        self._tool_specific_rate_limiters = {}

        if rate_limits is None:
            return

        if rate_limits.model_calls is not None:
            count, period = self.parse_rate_limit(
                rate_limits.model_calls,
                "model_calls",
            )
            self._model_rate_limiter = RateLimiter(count=count, period_seconds=period)

        if rate_limits.tool_calls is not None:
            count, period = self.parse_rate_limit(
                rate_limits.tool_calls,
                "tool_calls",
            )
            self._tool_rate_limiter = RateLimiter(count=count, period_seconds=period)

        for tool_name, tool_fn in tools.items():
            manifest = getattr(tool_fn, "_axis_manifest", None)
            rate_spec = getattr(manifest, "rate_limit", None)
            if rate_spec is None:
                continue
            count, period = self.parse_rate_limit(rate_spec, f"tool:{tool_name}")
            self._tool_specific_rate_limiters[tool_name] = RateLimiter(
                count=count,
                period_seconds=period,
            )

    async def acquire_model_slot(
        self,
        *,
        emit: Callable[..., Awaitable[None]],
        ctx: RunContext,
        step_id: str | None = None,
    ) -> None:
        """Apply model-call rate limiting and emit telemetry if active."""
        if self._model_rate_limiter is None:
            return
        await self._model_rate_limiter.acquire()
        await emit(
            "rate_limit_acquired",
            run_id=ctx.run_id,
            phase="act",
            cycle=ctx.cycle_count,
            step_id=step_id,
            data={"target": "model"},
        )

    async def acquire_tool_slot(
        self,
        *,
        emit: Callable[..., Awaitable[None]],
        ctx: RunContext,
        tool_name: str,
        step_id: str | None = None,
    ) -> None:
        """Apply tool-call rate limiting and emit telemetry if active."""
        if self._tool_rate_limiter is not None:
            await self._tool_rate_limiter.acquire()
        tool_limiter = self._tool_specific_rate_limiters.get(tool_name)
        if tool_limiter is not None:
            await tool_limiter.acquire()
        if self._tool_rate_limiter is not None or tool_limiter is not None:
            await emit(
                "rate_limit_acquired",
                run_id=ctx.run_id,
                phase="act",
                cycle=ctx.cycle_count,
                step_id=step_id,
                data={"target": "tool", "tool": tool_name},
            )


class PhaseTimeoutPolicyService:
    """Resolve per-phase timeout settings and enforce runtime time budgets."""

    def __init__(self) -> None:
        self._timeouts: Timeouts | None = None

    def configure(self, timeouts: Timeouts | None) -> None:
        """Apply timeout settings for the active run."""
        self._timeouts = timeouts

    def seconds_for(self, phase_name: str) -> float | None:
        if self._timeouts is None:
            return None
        return getattr(self._timeouts, phase_name, None)

    @staticmethod
    def timeout_error(phase_name: str, timeout_seconds: float) -> AxisTimeoutError:
        return AxisTimeoutError(
            message=(
                f"Phase '{phase_name}' exceeded timeout of {timeout_seconds:.3f} seconds"
            ),
            phase=phase_name,
            details={"phase": phase_name, "timeout_seconds": timeout_seconds},
        )

    async def run_with_budget(
        self,
        phase_name: str,
        operation: Callable[[], Awaitable[T]],
        *,
        ctx: RunContext,
        run_started_monotonic: float,
        update_wall_time: Callable[[RunContext, float], None],
        wall_time_budget_error: Callable[[RunContext], BudgetError],
    ) -> T:
        """Run one phase operation within wall-clock and per-phase limits."""
        update_wall_time(ctx, run_started_monotonic)
        remaining = ctx.budget.max_wall_time_seconds - ctx.state.budget_state.wall_time_seconds
        if remaining <= 0:
            raise wall_time_budget_error(ctx)

        phase_timeout = self.seconds_for(phase_name)
        if phase_timeout is not None and phase_timeout <= 0:
            raise self.timeout_error(phase_name, phase_timeout)

        timeout_budget = remaining
        timeout_source = "wall"
        if phase_timeout is not None and phase_timeout < timeout_budget:
            timeout_budget = phase_timeout
            timeout_source = "phase"

        try:
            result = await asyncio.wait_for(operation(), timeout=timeout_budget)
        except asyncio.TimeoutError as e:
            update_wall_time(ctx, run_started_monotonic)
            if timeout_source == "phase":
                raise self.timeout_error(phase_name, timeout_budget) from e
            raise wall_time_budget_error(ctx) from e

        update_wall_time(ctx, run_started_monotonic)
        return result


class LifecycleRuntimePolicyServices:
    """Aggregate runtime policy services used by lifecycle orchestration."""

    def __init__(self) -> None:
        self.cache = RuntimeCacheService()
        self.rate_limits = RateLimitPolicyService()
        self.timeouts = PhaseTimeoutPolicyService()

    def configure(self, config: Any | None, *, tools: dict[str, Any]) -> None:
        """Resolve and validate runtime policies for the active run."""
        timeouts = getattr(config, "timeouts", None)
        if timeouts is not None and not isinstance(timeouts, Timeouts):
            raise ConfigError(message="config.timeouts must be Timeouts or None")
        self.timeouts.configure(timeouts)

        rate_limits = getattr(config, "rate_limits", None)
        if rate_limits is not None and not isinstance(rate_limits, RateLimits):
            raise ConfigError(message="config.rate_limits must be RateLimits or None")
        self.rate_limits.configure(rate_limits, tools)

        cache = getattr(config, "cache", None)
        if cache is not None and not isinstance(cache, CacheConfig):
            raise ConfigError(message="config.cache must be CacheConfig or None")
        self.cache.configure(cache)
