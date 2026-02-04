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
import dataclasses
import hashlib
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from datetime import datetime
from typing import Any

from axis_core.budget import Budget, BudgetState
from axis_core.config import CacheConfig, RateLimits, RetryPolicy, Timeouts, config
from axis_core.context import RunState
from axis_core.engine.lifecycle import LifecycleEngine
from axis_core.engine.registry import memory_registry
from axis_core.engine.resolver import resolve_adapter
from axis_core.engine.trace_collector import TraceCollector
from axis_core.errors import AxisError
from axis_core.protocols.telemetry import BufferMode, TraceEvent
from axis_core.result import RunResult, RunStats, StreamEvent
from axis_core.session import Session, generate_session_id, load_session

logger = logging.getLogger("axis_core.agent")

# Sentinel value to distinguish "not provided" from "explicitly None"
_UNSET = object()
_STREAM_DONE = object()


def _trace_event_to_dict(event: TraceEvent) -> dict[str, Any]:
    """Serialize TraceEvent to a dict for streaming."""
    return {
        "type": event.type,
        "timestamp": event.timestamp.isoformat(),
        "run_id": event.run_id,
        "phase": event.phase,
        "cycle": event.cycle,
        "step_id": event.step_id,
        "data": event.data,
        "duration_ms": event.duration_ms,
    }


class _StreamTelemetrySink:
    """Telemetry sink that forwards trace events into a stream queue."""

    def __init__(self, queue: asyncio.Queue[Any]) -> None:
        self._queue = queue

    @property
    def buffering(self) -> BufferMode:
        return BufferMode.IMMEDIATE

    async def emit(self, event: TraceEvent) -> None:
        await self._queue.put(
            StreamEvent(
                type="telemetry",
                timestamp=event.timestamp,
                data={"event": _trace_event_to_dict(event)},
            )
        )

    async def flush(self) -> None:
        pass

    async def close(self) -> None:
        pass


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
        # Redaction is enabled by default for security (MED-1)
        redact = os.getenv("AXIS_TELEMETRY_REDACT", "true").lower() == "true"
        return [ConsoleSink(compact=compact, redact=redact)]

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
        model: Any = _UNSET,
        fallback: list[Any] | None = None,
        memory: Any = _UNSET,
        planner: Any = _UNSET,
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
        # Fall back to config defaults when not provided (config resolution order)
        # Use sentinel to distinguish "not provided" from "explicitly None"
        self._model = config.default_model if model is _UNSET else model
        self._fallback: list[Any] = fallback or []
        self._memory = config.default_memory if memory is _UNSET else memory
        self._planner = config.default_planner if planner is _UNSET else planner
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

    def _build_engine(self, extra_sinks: list[Any] | None = None) -> LifecycleEngine:
        """Create a LifecycleEngine with current agent configuration."""
        sinks: list[Any] = (
            list(self._telemetry_sinks) if self._telemetry_enabled else []
        )
        if extra_sinks:
            sinks.extend(extra_sinks)
        return LifecycleEngine(
            model=self._model,
            planner=self._planner,
            memory=self._memory,
            telemetry=sinks,
            tools=self._tools,
            system=self._system,
            fallback=self._fallback,
        )

    def _build_result(
        self,
        raw: dict[str, Any],
        duration_ms: float,
        trace: list[Any] | None = None,
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
            trace=trace or [],
            state=raw.get("state", RunState()),
            run_id=raw.get("run_id", ""),
            memory_error=memory_error,
        )

    def _get_config_fingerprint(self) -> str:
        """Generate fingerprint of current agent config (AD-044)."""
        model_value = self._model
        model_id = (
            model_value
            if isinstance(model_value, str)
            else getattr(model_value, "model_id", None)
        )
        config_data = {
            "tools": sorted(self._tools.keys()),
            "system": self._system,
            "model": model_id,
        }
        canonical = json.dumps(config_data, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    async def session_async(
        self,
        id: str | None = None,
        *,
        max_history: int = 100,
    ) -> Session:
        """Create or resume a session."""
        session_id = id or generate_session_id()
        memory = resolve_adapter(self._memory, memory_registry)

        session: Session | None = None
        if memory is not None:
            try:
                session = await load_session(memory, session_id)
            except Exception as e:
                logger.error("Failed to load session %s: %s", session_id, e)

        current_fingerprint = self._get_config_fingerprint()

        if session is not None:
            session.max_history = max_history
            if session.config_fingerprint and session.config_fingerprint != current_fingerprint:
                logger.warning(
                    "Session %s was created with different agent configuration. "
                    "Tools or system prompt may have changed. "
                    "Continuing with current configuration.",
                    session_id,
                )
                session.config_fingerprint = current_fingerprint
        else:
            session = Session(
                id=session_id,
                max_history=max_history,
                agent_id=self._agent_id,
                config_fingerprint=current_fingerprint,
            )

        if session.agent_id is None:
            session.agent_id = self._agent_id

        session.attach(self, memory)
        return session

    def session(
        self,
        id: str | None = None,
        *,
        max_history: int = 100,
    ) -> Session:
        """Create or resume a session (sync wrapper)."""
        try:
            asyncio.get_running_loop()
            raise RuntimeError(
                "agent.session() cannot be called from async context. "
                "Use await agent.session_async() instead."
            )
        except RuntimeError as e:
            if "cannot be called from async context" in str(e):
                raise

        return asyncio.run(self.session_async(id=id, max_history=max_history))

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
                trace_collector = TraceCollector() if self._telemetry_enabled else None
                extra_sinks = [trace_collector] if trace_collector else []
                engine = self._build_engine(extra_sinks=extra_sinks)

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
                    trace = trace_collector.get_events() if trace_collector else []
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
                        trace=trace,
                        state=RunState(),
                        run_id="",
                        memory_error=None,
                    )

                duration_ms = (time.monotonic() - start) * 1000
                trace = trace_collector.get_events() if trace_collector else []
                return self._build_result(raw, duration_ms, trace=trace)
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
        stream_telemetry: bool = False,
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
            queue: asyncio.Queue[Any] = asyncio.Queue()
            try:
                trace_collector = TraceCollector() if self._telemetry_enabled else None
                extra_sinks: list[Any] = [trace_collector] if trace_collector else []
                if stream_telemetry:
                    extra_sinks.append(_StreamTelemetrySink(queue))

                engine = self._build_engine(extra_sinks=extra_sinks)
                input_text = input if isinstance(input, str) else str(input)

                async def _on_token(token: str) -> None:
                    if token:
                        await queue.put(
                            StreamEvent(
                                type="model_token",
                                timestamp=datetime.utcnow(),
                                data={"token": token},
                            )
                        )

                # Emit start event
                yield StreamEvent(
                    type="run_started",
                    timestamp=datetime.utcnow(),
                    data={"agent_id": self._agent_id},
                    sequence=0,
                )

                async def _run_engine() -> dict[str, Any]:
                    try:
                        return await engine.execute(
                            input_text=input_text,
                            agent_id=self._agent_id,
                            budget=self._budget,
                            context=context,
                            attachments=attachments,
                            cancel_token=cancel_token,
                            token_callback=_on_token,
                        )
                    finally:
                        await queue.put(_STREAM_DONE)

                task = asyncio.create_task(_run_engine())

                while True:
                    item = await queue.get()
                    if item is _STREAM_DONE:
                        break
                    yield item

                raw = await task
                duration_ms = (time.monotonic() - start) * 1000
                trace = trace_collector.get_events() if trace_collector else []
                result = self._build_result(raw, duration_ms, trace=trace)

                # Emit final event
                event_type = "run_completed" if result.success else "run_failed"
                yield StreamEvent(
                    type=event_type,
                    timestamp=datetime.utcnow(),
                    data={
                        "success": result.success,
                        "output": result.output,
                        "run_id": result.run_id,
                        "stats": dataclasses.asdict(result.stats),
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
        stream_telemetry: bool = False,
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
                stream_telemetry=stream_telemetry,
            )
            while True:
                try:
                    yield loop.run_until_complete(gen.__anext__())
                except StopAsyncIteration:
                    break
        finally:
            loop.close()


__all__ = ["Agent"]
