"""Agent class — the primary public API for axis-core.

Provides run(), run_async(), stream(), and stream_async() methods for
executing agent tasks against LLMs with tool support.

Architecture Decisions:
- AD-008: Single-execution constraint via asyncio.Lock
- AD-010: Stream event ordering via asyncio.Queue
- AD-027: Sync wrappers use asyncio.run()
- AD-034: Runtime type validation on public APIs
- AD-036: RunResult is frozen/immutable
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from datetime import datetime
from typing import Any

from axis_core.budget import Budget, BudgetState
from axis_core.config import CacheConfig, RateLimits, RetryPolicy, Timeouts
from axis_core.context import RunState
from axis_core.engine.lifecycle import LifecycleEngine
from axis_core.errors import AxisError
from axis_core.result import RunResult, RunStats, StreamEvent

logger = logging.getLogger("axis_core.agent")


def _coerce_budget(value: dict[str, Any] | Budget | None) -> Budget:
    """Coerce a dict or Budget into a Budget instance."""
    if value is None:
        return Budget()
    if isinstance(value, Budget):
        return value
    if isinstance(value, dict):
        return Budget(**value)
    raise TypeError(
        f"Argument 'budget' must be Budget or dict, got {type(value).__name__}"
    )


def _coerce_timeouts(value: dict[str, Any] | Timeouts | None) -> Timeouts:
    """Coerce a dict or Timeouts into a Timeouts instance."""
    if value is None:
        return Timeouts()
    if isinstance(value, Timeouts):
        return value
    if isinstance(value, dict):
        return Timeouts(**value)
    raise TypeError(
        f"Argument 'timeouts' must be Timeouts or dict, got {type(value).__name__}"
    )


def _coerce_retry(value: dict[str, Any] | RetryPolicy | None) -> RetryPolicy | None:
    """Coerce a dict or RetryPolicy into a RetryPolicy instance."""
    if value is None:
        return None
    if isinstance(value, RetryPolicy):
        return value
    if isinstance(value, dict):
        return RetryPolicy(**value)
    raise TypeError(
        f"Argument 'retry' must be RetryPolicy or dict, got {type(value).__name__}"
    )


def _coerce_rate_limits(
    value: dict[str, Any] | RateLimits | None,
) -> RateLimits | None:
    """Coerce a dict or RateLimits into a RateLimits instance."""
    if value is None:
        return None
    if isinstance(value, RateLimits):
        return value
    if isinstance(value, dict):
        return RateLimits(**value)
    raise TypeError(
        f"Argument 'rate_limits' must be RateLimits or dict, got {type(value).__name__}"
    )


def _coerce_cache(value: dict[str, Any] | CacheConfig | None) -> CacheConfig | None:
    """Coerce a dict or CacheConfig into a CacheConfig instance."""
    if value is None:
        return None
    if isinstance(value, CacheConfig):
        return value
    if isinstance(value, dict):
        return CacheConfig(**value)
    raise TypeError(
        f"Argument 'cache' must be CacheConfig or dict, got {type(value).__name__}"
    )


def _resolve_telemetry_sinks() -> list[Any]:
    """Resolve telemetry sinks from environment variables.

    Reads AXIS_TELEMETRY_SINK env var and creates appropriate sink instances.

    Supported values:
        - "console": Creates ConsoleSink for stdout output
        - "none": Returns empty list (telemetry disabled)
        - "file": Not yet implemented (logs warning)
        - "callback": Not yet implemented (logs warning)

    Returns:
        List of telemetry sink instances
    """
    sink_type = os.getenv("AXIS_TELEMETRY_SINK", "none").lower()

    if sink_type == "none":
        return []

    if sink_type == "console":
        # Import here to avoid circular imports
        from axis_core.adapters.telemetry.console import ConsoleSink

        # Check if compact mode is requested via env var
        compact = os.getenv("AXIS_TELEMETRY_COMPACT", "false").lower() == "true"
        return [ConsoleSink(compact=compact)]

    if sink_type == "file":
        # File sink not yet implemented
        file_path = os.getenv("AXIS_TELEMETRY_FILE", "./axis_trace.jsonl")
        logger.warning(
            "AXIS_TELEMETRY_SINK=file is not yet implemented. "
            f"File path would be: {file_path}. Using no telemetry."
        )
        return []

    if sink_type == "callback":
        logger.warning(
            "AXIS_TELEMETRY_SINK=callback requires passing sinks directly to Agent. "
            "Using no telemetry."
        )
        return []

    # Unknown sink type
    logger.warning(
        f"Unknown AXIS_TELEMETRY_SINK value: '{sink_type}'. "
        f"Supported values: console, file, callback, none. Using no telemetry."
    )
    return []


class Agent:
    """Primary API for executing AI agent tasks.

    Agents coordinate a model, planner, optional memory, and tools to execute
    tasks through the observe→plan→act→evaluate lifecycle.

    Args:
        tools: List of @tool-decorated callables
        system: System prompt text
        persona: Named persona or Persona object
        model: Model adapter instance or string identifier
        fallback: Fallback model(s) if primary fails
        memory: Memory adapter instance or string identifier
        planner: Planner instance or string identifier
        budget: Resource constraints (Budget or dict)
        timeouts: Per-phase timeouts (Timeouts or dict)
        rate_limits: Rate limiting config (RateLimits or dict)
        retry: Retry policy (RetryPolicy or dict)
        cache: Cache config (CacheConfig or dict)
        telemetry: True (collect silently), False (disabled), or list of sinks
        verbose: Print events to console
        auth: Per-tool credentials
    """

    def __init__(
        self,
        tools: list[Callable[..., Any]] | None = None,
        *,
        system: str | None = None,
        persona: str | None = None,
        model: Any = None,
        fallback: list[Any] | None = None,
        memory: Any = None,
        planner: Any = None,
        budget: dict[str, Any] | Budget | None = None,
        timeouts: dict[str, Any] | Timeouts | None = None,
        rate_limits: dict[str, Any] | RateLimits | None = None,
        retry: dict[str, Any] | RetryPolicy | None = None,
        cache: dict[str, Any] | CacheConfig | None = None,
        telemetry: bool | list[Any] = True,
        verbose: bool = False,
        auth: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        # ----- AD-034: Runtime type validation -----
        if tools is not None and not isinstance(tools, list):
            raise TypeError(
                f"Argument 'tools' must be a list of callables, "
                f"got {type(tools).__name__}"
            )
        if system is not None and not isinstance(system, str):
            raise TypeError(
                f"Argument 'system' must be str, got {type(system).__name__}"
            )
        if verbose is not None and not isinstance(verbose, bool):
            raise TypeError(
                f"Argument 'verbose' must be bool, got {type(verbose).__name__}"
            )

        # ----- Store configuration -----
        self._agent_id = str(uuid.uuid4())
        self._system = system
        self._persona = persona
        self._model = model
        self._fallback: list[Any] = fallback or []
        self._memory = memory
        self._planner = planner
        self._budget = _coerce_budget(budget)
        self._timeouts = _coerce_timeouts(timeouts)
        self._rate_limits = _coerce_rate_limits(rate_limits)
        self._retry = _coerce_retry(retry)
        self._cache = _coerce_cache(cache)
        self._verbose = verbose
        self._auth = auth

        # Telemetry - resolve sinks from env vars when enabled
        if isinstance(telemetry, bool):
            self._telemetry_enabled = telemetry
            # When telemetry=True, resolve sinks from AXIS_TELEMETRY_SINK env var
            self._telemetry_sinks: list[Any] = (
                _resolve_telemetry_sinks() if telemetry else []
            )
        elif isinstance(telemetry, list):
            self._telemetry_enabled = True
            self._telemetry_sinks = telemetry
        else:
            raise TypeError(
                f"Argument 'telemetry' must be bool or list, "
                f"got {type(telemetry).__name__}"
            )

        # Build tool registry
        self._tools: dict[str, Any] = {}
        if tools:
            for t in tools:
                if hasattr(t, "_axis_manifest"):
                    self._tools[t._axis_manifest.name] = t
                elif hasattr(t, "__name__"):
                    self._tools[t.__name__] = t
                else:
                    self._tools[str(t)] = t

        # AD-008: Single-execution constraint
        self._lock = asyncio.Lock()
        self._running = False

    # =========================================================================
    # Internal: build engine and execute
    # =========================================================================

    def _build_engine(self) -> LifecycleEngine:
        """Create a LifecycleEngine with current agent configuration."""
        return LifecycleEngine(
            model=self._model,
            planner=self._planner,
            memory=self._memory,
            telemetry=self._telemetry_sinks if self._telemetry_enabled else [],
            tools=self._tools,
            system=self._system,
        )

    def _build_result(
        self,
        raw: dict[str, Any],
        duration_ms: float,
    ) -> RunResult:
        """Convert lifecycle engine raw result dict into a RunResult."""
        budget_state: BudgetState = raw.get("budget_state", BudgetState())
        errors: list[Any] = raw.get("errors", [])

        stats = RunStats(
            cycles=raw.get("cycles_completed", 0),
            tool_calls=budget_state.tool_calls,
            model_calls=budget_state.model_calls,
            input_tokens=budget_state.input_tokens,
            output_tokens=budget_state.output_tokens,
            total_tokens=budget_state.total_tokens,
            cost_usd=budget_state.cost_usd,
            duration_ms=duration_ms,
        )

        error = raw.get("error")
        memory_error_str = raw.get("memory_error")
        memory_error: AxisError | None = None
        if memory_error_str:
            from axis_core.errors import ErrorClass

            memory_error = AxisError(
                message=str(memory_error_str),
                error_class=ErrorClass.RUNTIME,
            )

        return RunResult(
            output=raw.get("output"),
            output_raw=raw.get("output_raw", ""),
            success=raw.get("success", False),
            error=error,
            had_recoverable_errors=any(
                getattr(e, "recovered", False) for e in errors
            ),
            stats=stats,
            trace=[],  # Trace collection will be enhanced in task 10
            state=RunState(),
            run_id=raw.get("run_id", ""),
            memory_error=memory_error,
        )

    # =========================================================================
    # run_async — native async (8.3, AD-008)
    # =========================================================================

    async def run_async(
        self,
        input: str | list[Any],
        *,
        context: dict[str, Any] | None = None,
        attachments: list[Any] | None = None,
        output_schema: type | None = None,
        timeout: float | None = None,
        cancel_token: Any | None = None,
    ) -> RunResult:
        """Execute agent asynchronously.

        Args:
            input: Text or multimodal input
            context: Arbitrary context dict passed to tools
            attachments: Images, PDFs, etc.
            output_schema: Force structured output (Pydantic model)
            timeout: Override default timeout
            cancel_token: For external cancellation

        Returns:
            RunResult with output, stats, and state

        Raises:
            TypeError: If input type is invalid
            RuntimeError: If agent is already executing (AD-008)
        """
        # AD-034: Validate input type
        if not isinstance(input, (str, list)):
            raise TypeError(
                f"Argument 'input' must be str or list, got {type(input).__name__}"
            )

        # AD-008: Single-execution constraint
        if self._lock.locked():
            raise RuntimeError(
                "Agent is already executing. "
                "Create multiple Agent instances for concurrent execution."
            )

        async with self._lock:
            self._running = True
            start = time.monotonic()
            try:
                engine = self._build_engine()

                input_text = input if isinstance(input, str) else str(input)

                try:
                    raw = await engine.execute(
                        input_text=input_text,
                        agent_id=self._agent_id,
                        budget=self._budget,
                        context=context,
                        attachments=attachments,
                        cancel_token=cancel_token,
                    )
                except AxisError as e:
                    # Engine raised before finalize (e.g. empty input)
                    duration_ms = (time.monotonic() - start) * 1000
                    return RunResult(
                        output=None,
                        output_raw="",
                        success=False,
                        error=e,
                        had_recoverable_errors=False,
                        stats=RunStats(
                            cycles=0,
                            tool_calls=0,
                            model_calls=0,
                            input_tokens=0,
                            output_tokens=0,
                            total_tokens=0,
                            cost_usd=0.0,
                            duration_ms=duration_ms,
                        ),
                        trace=[],
                        state=RunState(),
                        run_id="",
                        memory_error=None,
                    )

                duration_ms = (time.monotonic() - start) * 1000
                return self._build_result(raw, duration_ms)
            finally:
                self._running = False

    # =========================================================================
    # run — sync wrapper (8.4, AD-027)
    # =========================================================================

    def run(
        self,
        input: str | list[Any],
        *,
        context: dict[str, Any] | None = None,
        attachments: list[Any] | None = None,
        output_schema: type | None = None,
        timeout: float | None = None,
        cancel_token: Any | None = None,
    ) -> RunResult:
        """Execute agent synchronously. Blocks until complete.

        Wraps run_async() with asyncio.run() per AD-027.

        Raises:
            RuntimeError: If called from an async context
        """
        # Detect if we're already in an async context
        try:
            asyncio.get_running_loop()
            raise RuntimeError(
                "agent.run() cannot be called from async context. "
                "Use await agent.run_async() instead."
            )
        except RuntimeError as e:
            if "cannot be called from async context" in str(e):
                raise
            # No event loop running — safe to proceed

        return asyncio.run(
            self.run_async(
                input,
                context=context,
                attachments=attachments,
                output_schema=output_schema,
                timeout=timeout,
                cancel_token=cancel_token,
            )
        )

    # =========================================================================
    # stream_async — async streaming (8.5, AD-010)
    # =========================================================================

    async def stream_async(
        self,
        input: str | list[Any],
        *,
        context: dict[str, Any] | None = None,
        attachments: list[Any] | None = None,
        output_schema: type | None = None,
        timeout: float | None = None,
        cancel_token: Any | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Execute agent with async streaming. Yields events as they occur.

        Per AD-010, uses asyncio.Queue for event ordering.

        Raises:
            TypeError: If input type is invalid
            RuntimeError: If agent is already executing (AD-008)
        """
        if not isinstance(input, (str, list)):
            raise TypeError(
                f"Argument 'input' must be str or list, got {type(input).__name__}"
            )

        if self._lock.locked():
            raise RuntimeError(
                "Agent is already executing. "
                "Create multiple Agent instances for concurrent execution."
            )

        async with self._lock:
            self._running = True
            start = time.monotonic()
            try:
                engine = self._build_engine()
                input_text = input if isinstance(input, str) else str(input)

                # Emit start event
                yield StreamEvent(
                    type="run_started",
                    timestamp=datetime.utcnow(),
                    data={"agent_id": self._agent_id},
                    sequence=0,
                )

                raw = await engine.execute(
                    input_text=input_text,
                    agent_id=self._agent_id,
                    budget=self._budget,
                    context=context,
                    attachments=attachments,
                    cancel_token=cancel_token,
                )

                duration_ms = (time.monotonic() - start) * 1000
                result = self._build_result(raw, duration_ms)

                # Emit final event
                event_type = "run_completed" if result.success else "run_failed"
                yield StreamEvent(
                    type=event_type,
                    timestamp=datetime.utcnow(),
                    data={
                        "success": result.success,
                        "output": result.output,
                        "run_id": result.run_id,
                    },
                    sequence=1,
                )
            finally:
                self._running = False

    # =========================================================================
    # stream — sync wrapper (8.6, AD-027)
    # =========================================================================

    def stream(
        self,
        input: str | list[Any],
        *,
        context: dict[str, Any] | None = None,
        attachments: list[Any] | None = None,
        output_schema: type | None = None,
        timeout: float | None = None,
        cancel_token: Any | None = None,
    ) -> Iterator[StreamEvent]:
        """Synchronous streaming. Yields StreamEvents.

        Wraps stream_async() per AD-027.
        """
        loop = asyncio.new_event_loop()
        try:
            gen = self.stream_async(
                input,
                context=context,
                attachments=attachments,
                output_schema=output_schema,
                timeout=timeout,
                cancel_token=cancel_token,
            )
            while True:
                try:
                    yield loop.run_until_complete(gen.__anext__())
                except StopAsyncIteration:
                    break
        finally:
            loop.close()


__all__ = ["Agent"]
