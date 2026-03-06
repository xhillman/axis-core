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
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from axis_core._agent_checkpoint import (
    load_checkpoint_payload,
    persist_checkpoint,
)
from axis_core._agent_construction import build_agent_construction
from axis_core._agent_runtime import (
    build_failure_result,
    build_run_result,
    build_timeout_error,
    config_fingerprint,
    effective_timeout,
    resolve_runtime_config,
)
from axis_core.attachments import AttachmentLike
from axis_core.budget import Budget
from axis_core.cancel import CancelToken
from axis_core.config import (
    CacheConfig,
    RateLimits,
    ResolvedConfig,
    RetryPolicy,
    Timeouts,
    ToolPolicy,
)
from axis_core.engine.lifecycle import LifecycleEngine
from axis_core.engine.registry import memory_registry
from axis_core.engine.resolver import resolve_adapter
from axis_core.engine.trace_collector import TraceCollector
from axis_core.errors import AxisError, ErrorClass
from axis_core.errors import TimeoutError as AxisTimeoutError
from axis_core.output_schema import parse_stream_partial, schema_name
from axis_core.protocols.telemetry import BufferMode, TraceEvent
from axis_core.result import RunResult, StreamEvent
from axis_core.session import Session, generate_session_id, load_session

logger = logging.getLogger("axis_core.agent")

# Sentinel value to distinguish "not provided" from "explicitly None"
_UNSET = object()
_STREAM_DONE = object()
ConfirmationHandler = Callable[[str, dict[str, Any]], bool | Awaitable[bool]]
ExecutionOperation = Callable[
    [LifecycleEngine, ResolvedConfig],
    Awaitable[dict[str, Any]],
]


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


@dataclasses.dataclass(slots=True)
class _ExecutionSession:
    timeout: float | None
    started_at: float
    trace_collector: TraceCollector | None
    engine: Any
    runtime_config: ResolvedConfig

    def duration_ms(self) -> float:
        return (time.monotonic() - self.started_at) * 1000

    def trace(self) -> list[Any]:
        return self.trace_collector.get_events() if self.trace_collector else []


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
        auth: Deprecated. Credentials must be managed inside tools.
        confirmation_handler: Optional approval callback for destructive tools.
        tool_policy: Optional per-agent allow/deny policy for tool names.
        checkpoint: Enable automatic phase-boundary checkpoint persistence.
        checkpoint_dir: Directory where checkpoints are stored when enabled.
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
        tool_policy: dict[str, Any] | ToolPolicy | None = None,
        telemetry: bool | list[Any] = True,
        verbose: bool = False,
        auth: dict[str, dict[str, Any]] | None = None,
        confirmation_handler: ConfirmationHandler | None = None,
        checkpoint: bool = False,
        checkpoint_dir: str = "./checkpoints",
    ) -> None:
        construction = build_agent_construction(
            tools=tools,
            system=system,
            persona=persona,
            model=model,
            fallback=fallback,
            memory=memory,
            planner=planner,
            budget=budget,
            timeouts=timeouts,
            rate_limits=rate_limits,
            retry=retry,
            cache=cache,
            tool_policy=tool_policy,
            telemetry=telemetry,
            verbose=verbose,
            auth=auth,
            confirmation_handler=confirmation_handler,
            checkpoint=checkpoint,
            checkpoint_dir=checkpoint_dir,
            unset=_UNSET,
        )

        self._agent_id = str(uuid.uuid4())
        self._system = construction.system
        self._persona = construction.persona
        self._model = construction.model
        self._fallback = construction.fallback
        self._memory = construction.memory
        self._planner = construction.planner
        self._budget = construction.budget
        self._timeouts = construction.timeouts
        self._rate_limits = construction.rate_limits
        self._retry = construction.retry
        self._cache = construction.cache
        self._tool_policy = construction.tool_policy
        self._verbose = construction.verbose
        self._confirmation_handler = construction.confirmation_handler
        self._checkpoint_enabled = construction.checkpoint_enabled
        self._checkpoint_dir = construction.checkpoint_dir
        self._telemetry_enabled = construction.telemetry_enabled
        self._telemetry_sinks = construction.telemetry_sinks
        self._tools = construction.tools

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
            checkpoint_handler=(
                self._persist_checkpoint if self._checkpoint_enabled else None
            ),
        )

    async def _persist_checkpoint(self, payload: dict[str, Any]) -> None:
        """Persist a checkpoint envelope to disk."""
        await persist_checkpoint(self._checkpoint_dir, payload)

    @staticmethod
    def _load_checkpoint_payload(checkpoint: str | dict[str, Any]) -> dict[str, Any]:
        """Load a checkpoint envelope from dict payload or JSON file path."""
        return load_checkpoint_payload(checkpoint)

    def _build_result(
        self,
        raw: dict[str, Any],
        duration_ms: float,
        trace: list[Any] | None = None,
        output_schema: type[Any] | None = None,
    ) -> RunResult:
        """Convert lifecycle engine raw result dict into a RunResult."""
        return build_run_result(
            raw,
            duration_ms,
            trace=trace,
            output_schema=output_schema,
        )

    def _get_config_fingerprint(self) -> str:
        """Generate fingerprint of current agent config (AD-044)."""
        return config_fingerprint(
            model=self._model,
            tools=self._tools,
            system=self._system,
        )

    def _effective_timeout(self, timeout: float | None) -> float | None:
        """Resolve runtime timeout: explicit override first, then configured total."""
        return effective_timeout(timeout, self._timeouts.total)

    def _resolved_config(self) -> ResolvedConfig:
        """Build the resolved runtime config passed into the lifecycle engine."""
        return resolve_runtime_config(
            model=self._model,
            planner=self._planner,
            memory=self._memory,
            budget=self._budget,
            timeouts=self._timeouts,
            rate_limits=self._rate_limits,
            retry=self._retry,
            cache=self._cache,
            tool_policy=self._tool_policy,
            confirmation_handler=self._confirmation_handler,
            telemetry_enabled=self._telemetry_enabled,
            verbose=self._verbose,
        )

    def on_confirm(self, handler: ConfirmationHandler) -> Agent:
        """Register a destructive-tool confirmation handler and return self."""
        if not callable(handler):
            raise TypeError(
                f"Argument 'handler' must be callable, got {type(handler).__name__}"
            )
        self._confirmation_handler = handler
        return self

    def _build_failure_result(
        self,
        error: AxisError,
        duration_ms: float,
        trace: list[Any] | None = None,
    ) -> RunResult:
        """Build a failed RunResult when execution aborts before finalize."""
        return build_failure_result(error, duration_ms, trace=trace)

    @staticmethod
    async def _await_operation(
        operation: Awaitable[dict[str, Any]],
        timeout: float | None,
    ) -> dict[str, Any]:
        """Await an operation with an optional wall-clock timeout."""
        if timeout is None:
            return await operation
        return await asyncio.wait_for(operation, timeout=timeout)

    def _build_timeout_error(self, timeout: float | None) -> AxisTimeoutError:
        """Create a normalized timeout error payload."""
        return build_timeout_error(
            timeout,
            default_timeout=self._timeouts.total,
        )

    @asynccontextmanager
    async def _execution_session(
        self,
        *,
        timeout: float | None,
        extra_sinks: list[Any] | None = None,
    ) -> AsyncIterator[_ExecutionSession]:
        """Create a shared execution session with lock, engine, config, and trace state."""
        if self._lock.locked():
            raise RuntimeError(
                "Agent is already executing. "
                "Create multiple Agent instances for concurrent execution."
            )

        async with self._lock:
            self._running = True
            try:
                trace_collector = TraceCollector() if self._telemetry_enabled else None
                sinks: list[Any] = [trace_collector] if trace_collector else []
                if extra_sinks:
                    sinks.extend(extra_sinks)
                yield _ExecutionSession(
                    timeout=self._effective_timeout(timeout),
                    started_at=time.monotonic(),
                    trace_collector=trace_collector,
                    engine=self._build_engine(extra_sinks=sinks),
                    runtime_config=self._resolved_config(),
                )
            finally:
                self._running = False

    def _build_result_from_outcome(
        self,
        *,
        execution: _ExecutionSession,
        raw: dict[str, Any] | None = None,
        error: AxisError | None = None,
        output_schema: type[Any] | None = None,
    ) -> RunResult:
        """Map raw execution outcomes into the normalized RunResult contract."""
        if raw is None:
            if error is None:
                error = AxisError(
                    message="Run failed",
                    error_class=ErrorClass.RUNTIME,
                )
            return self._build_failure_result(
                error=error,
                duration_ms=execution.duration_ms(),
                trace=execution.trace(),
            )

        return self._build_result(
            raw,
            execution.duration_ms(),
            trace=execution.trace(),
            output_schema=output_schema,
        )

    async def _execute_guarded(
        self,
        *,
        operation: ExecutionOperation,
        timeout: float | None,
        output_schema: type[Any] | None = None,
        prepare: Callable[[], None] | None = None,
    ) -> RunResult:
        """Execute run/resume operations with shared lock + timeout/error handling."""
        async with self._execution_session(timeout=timeout) as execution:
            try:
                if prepare is not None:
                    prepare()

                raw = await self._await_operation(
                    operation(execution.engine, execution.runtime_config),
                    execution.timeout,
                )
            except AxisError as error:
                return self._build_result_from_outcome(
                    execution=execution,
                    error=error,
                )
            except asyncio.TimeoutError:
                return self._build_result_from_outcome(
                    execution=execution,
                    error=self._build_timeout_error(execution.timeout),
                )

            return self._build_result_from_outcome(
                execution=execution,
                raw=raw,
                output_schema=output_schema,
            )

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
        attachments: list[AttachmentLike] | None = None,
        output_schema: type | None = None,
        timeout: float | None = None,
        cancel_token: CancelToken | None = None,
    ) -> RunResult:
        """Execute agent asynchronously.

        Args:
            input: Text or multimodal input
            context: Arbitrary context dict passed to tools
            attachments: Images, PDFs, etc.
            output_schema: Structured output schema to enforce on final output.
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

        input_text = input if isinstance(input, str) else str(input)
        return await self._execute_guarded(
            operation=lambda engine, runtime_config: engine.execute(
                input_text=input_text,
                agent_id=self._agent_id,
                budget=self._budget,
                context=context,
                attachments=attachments,
                cancel_token=cancel_token,
                config=runtime_config,
            ),
            timeout=timeout,
            output_schema=output_schema,
        )

    # =========================================================================
    # run — sync wrapper (8.4, AD-027)
    # =========================================================================

    async def resume_async(
        self,
        checkpoint: str | dict[str, Any],
        *,
        timeout: float | None = None,
        cancel_token: CancelToken | None = None,
    ) -> RunResult:
        """Resume an agent run from a persisted checkpoint."""
        if not isinstance(checkpoint, (str, dict)):
            raise TypeError(
                "Argument 'checkpoint' must be str path or dict payload, "
                f"got {type(checkpoint).__name__}"
            )

        checkpoint_payload: dict[str, Any] = {}

        def _prepare_checkpoint() -> None:
            nonlocal checkpoint_payload
            checkpoint_payload = self._load_checkpoint_payload(checkpoint)

        return await self._execute_guarded(
            operation=lambda engine, runtime_config: engine.resume(
                checkpoint=checkpoint_payload,
                cancel_token=cancel_token,
                config=runtime_config,
            ),
            timeout=timeout,
            prepare=_prepare_checkpoint,
        )

    def resume(
        self,
        checkpoint: str | dict[str, Any],
        *,
        timeout: float | None = None,
        cancel_token: CancelToken | None = None,
    ) -> RunResult:
        """Resume an agent run from checkpoint (sync wrapper)."""
        try:
            asyncio.get_running_loop()
            raise RuntimeError(
                "agent.resume() cannot be called from async context. "
                "Use await agent.resume_async() instead."
            )
        except RuntimeError as e:
            if "cannot be called from async context" in str(e):
                raise

        return asyncio.run(
            self.resume_async(
                checkpoint,
                timeout=timeout,
                cancel_token=cancel_token,
            )
        )

    def run(
        self,
        input: str | list[Any],
        *,
        context: dict[str, Any] | None = None,
        attachments: list[AttachmentLike] | None = None,
        output_schema: type | None = None,
        timeout: float | None = None,
        cancel_token: CancelToken | None = None,
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
        attachments: list[AttachmentLike] | None = None,
        output_schema: type | None = None,
        timeout: float | None = None,
        cancel_token: CancelToken | None = None,
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

        queue: asyncio.Queue[Any] = asyncio.Queue()
        structured_buffer: list[str] = []
        last_structured_fingerprint: str | None = None
        extra_sinks: list[Any] = []
        if stream_telemetry:
            extra_sinks.append(_StreamTelemetrySink(queue))

        async with self._execution_session(
            timeout=timeout,
            extra_sinks=extra_sinks,
        ) as execution:
            input_text = input if isinstance(input, str) else str(input)

            async def _on_token(token: str) -> None:
                nonlocal last_structured_fingerprint
                if token:
                    if output_schema is not None:
                        structured_buffer.append(token)
                        partial = parse_stream_partial(
                            "".join(structured_buffer),
                            output_schema,
                        )
                        if partial is not None:
                            fingerprint = _structured_payload_fingerprint(partial)
                            if fingerprint != last_structured_fingerprint:
                                last_structured_fingerprint = fingerprint
                                await queue.put(
                                    StreamEvent(
                                        type="structured_partial",
                                        timestamp=datetime.utcnow(),
                                        data={
                                            "schema": schema_name(output_schema),
                                            "output": partial,
                                        },
                                    )
                                )
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
                    return await self._await_operation(
                        execution.engine.execute(
                            input_text=input_text,
                            agent_id=self._agent_id,
                            budget=self._budget,
                            context=context,
                            attachments=attachments,
                            cancel_token=cancel_token,
                            config=execution.runtime_config,
                            token_callback=_on_token,
                        ),
                        execution.timeout,
                    )
                finally:
                    await queue.put(_STREAM_DONE)

            task = asyncio.create_task(_run_engine())

            while True:
                item = await queue.get()
                if item is _STREAM_DONE:
                    break
                yield item

            run_error: AxisError | None = None
            raw: dict[str, Any] | None = None

            try:
                raw = await task
            except AxisError as error:
                run_error = error
            except asyncio.TimeoutError:
                run_error = self._build_timeout_error(execution.timeout)

            result = self._build_result_from_outcome(
                execution=execution,
                raw=raw,
                error=run_error,
                output_schema=output_schema,
            )

            if output_schema is not None and result.success:
                yield StreamEvent(
                    type="structured_final",
                    timestamp=datetime.utcnow(),
                    data={
                        "schema": schema_name(output_schema),
                        "output": result.output,
                    },
                )

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
                    "error": str(result.error) if result.error else None,
                },
                sequence=1,
            )

    # =========================================================================
    # stream — sync wrapper (8.6, AD-027)
    # =========================================================================

    def stream(
        self,
        input: str | list[Any],
        *,
        context: dict[str, Any] | None = None,
        attachments: list[AttachmentLike] | None = None,
        output_schema: type | None = None,
        timeout: float | None = None,
        cancel_token: CancelToken | None = None,
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


def _structured_payload_fingerprint(payload: Any) -> str:
    try:
        return json.dumps(payload, sort_keys=True, default=str)
    except Exception:
        return repr(payload)
