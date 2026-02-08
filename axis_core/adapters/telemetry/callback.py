"""Callback telemetry sink for axis-core.

Provides CallbackSink adapter that forwards events to a user-supplied handler.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from axis_core.protocols.telemetry import BufferMode, TraceEvent
from axis_core.redaction import redact_sensitive_data

TelemetryHandler = Callable[[TraceEvent], Any | Awaitable[Any]]


class CallbackSink:
    """Telemetry sink that invokes a callback for each emitted event.

    Args:
        handler: Callable invoked for every event. Can be sync or async.
        buffering: Buffering strategy exposed to the engine.
        redact: Whether to redact sensitive keys from event data.
    """

    def __init__(
        self,
        handler: TelemetryHandler,
        *,
        buffering: BufferMode = BufferMode.IMMEDIATE,
        redact: bool = True,
    ) -> None:
        if not callable(handler):
            raise TypeError("handler must be callable")

        self._handler = handler
        self._buffering = buffering
        self._redact = redact
        self._closed = False

    @property
    def buffering(self) -> BufferMode:
        """Return configured buffering mode."""
        return self._buffering

    async def emit(self, event: TraceEvent) -> None:
        """Invoke callback with the event."""
        if self._closed:
            raise RuntimeError("CallbackSink is closed")

        payload = redact_sensitive_data(event.data) if self._redact else event.data
        event_to_emit = TraceEvent(
            type=event.type,
            timestamp=event.timestamp,
            run_id=event.run_id,
            phase=event.phase,
            cycle=event.cycle,
            step_id=event.step_id,
            data=payload if isinstance(payload, dict) else {"value": payload},
            duration_ms=event.duration_ms,
        )

        result = self._handler(event_to_emit)
        if inspect.isawaitable(result):
            await result

    async def flush(self) -> None:
        """No-op; callback sink does not buffer internally by default."""
        return

    async def close(self) -> None:
        """Mark sink as closed."""
        self._closed = True


__all__ = ["CallbackSink", "TelemetryHandler"]
