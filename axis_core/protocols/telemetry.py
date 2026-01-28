"""Telemetry sink protocol and associated dataclasses.

This module defines the TelemetrySink protocol interface for observability backends,
along with enums for buffering modes and dataclasses for trace events.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class BufferMode(Enum):
    """Buffering strategies for telemetry events.

    Attributes:
        IMMEDIATE: Emit events immediately (no buffering)
        BATCHED: Buffer events and emit in batches
        PHASE: Buffer events until phase completion
        END: Buffer all events until run completion
    """

    IMMEDIATE = "immediate"
    BATCHED = "batched"
    PHASE = "phase"
    END = "end"


@dataclass(frozen=True)
class TraceEvent:
    """A single telemetry event in the trace.

    Attributes:
        type: Event type (e.g., "phase_start", "tool_call", "model_response")
        timestamp: When this event occurred
        run_id: Unique identifier for the agent run
        phase: Execution phase (observe, plan, act, evaluate, finalize)
        cycle: Cycle number (for multi-cycle runs)
        step_id: Step identifier (for step-level events)
        data: Event-specific data payload
        duration_ms: Duration in milliseconds (for span events)
    """

    type: str
    timestamp: datetime
    run_id: str
    phase: str | None = None
    cycle: int | None = None
    step_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    duration_ms: float | None = None


@runtime_checkable
class TelemetrySink(Protocol):
    """Protocol for telemetry/observability sinks.

    Telemetry sinks collect trace events from agent execution and forward them to
    observability backends (stdout, files, OpenTelemetry, LangSmith, etc.).

    Implementations must provide:
    - buffering property returning the buffer strategy
    - emit() for recording events
    - flush() for forcing buffered events to be sent
    - close() for cleanup and final flush
    """

    @property
    def buffering(self) -> BufferMode:
        """Buffering strategy used by this sink."""
        ...

    async def emit(self, event: TraceEvent) -> None:
        """Emit a trace event.

        Args:
            event: Event to emit

        Note:
            Depending on the buffering mode, this may not immediately send the event.
            Call flush() to ensure buffered events are sent.
        """
        ...

    async def flush(self) -> None:
        """Flush any buffered events to the backend.

        This is a no-op for IMMEDIATE mode sinks.
        """
        ...

    async def close(self) -> None:
        """Close the sink and flush any remaining events.

        Should be called at the end of agent execution to ensure all events are sent.
        After calling close(), no more events should be emitted.
        """
        ...
