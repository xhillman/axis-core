"""TraceCollector for accumulating telemetry events.

This module provides the TraceCollector class, which implements TelemetrySink
to accumulate all trace events during execution. Used to populate RunResult.trace.
"""

from __future__ import annotations

from axis_core.protocols.telemetry import BufferMode, TraceEvent


class TraceCollector:
    """Event accumulator for agent execution traces.

    TraceCollector implements the TelemetrySink protocol with BufferMode.END,
    accumulating all events in memory without emitting them anywhere. After
    execution completes, the accumulated events can be retrieved via get_events()
    and included in the RunResult.

    This enables full run replay and debugging without requiring an external
    observability backend.
    """

    def __init__(self) -> None:
        """Initialize an empty trace collector."""
        self._events: list[TraceEvent] = []

    @property
    def buffering(self) -> BufferMode:
        """Return END buffer mode (accumulate all events until run completion)."""
        return BufferMode.END

    async def emit(self, event: TraceEvent) -> None:
        """Store a trace event.

        Args:
            event: Event to store
        """
        self._events.append(event)

    async def flush(self) -> None:
        """No-op for END mode collector."""
        pass

    async def close(self) -> None:
        """No-op cleanup."""
        pass

    def get_events(self) -> list[TraceEvent]:
        """Get all collected events.

        Returns:
            Copy of the events list (to prevent external mutation)
        """
        return self._events.copy()

    def clear(self) -> None:
        """Clear all collected events."""
        self._events.clear()


__all__ = ["TraceCollector"]
