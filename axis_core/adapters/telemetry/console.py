"""Console telemetry sink for axis-core.

Provides ConsoleSink adapter that pretty-prints telemetry events to stdout or
a specified output stream. Uses IMMEDIATE buffering mode.
"""

from __future__ import annotations

import sys
from typing import Any, TextIO

from axis_core.protocols.telemetry import BufferMode, TraceEvent

_REDACTED_VALUE = "[REDACTED]"
_SENSITIVE_KEY_FRAGMENTS = (
    "api_key",
    "apikey",
    "secret",
    "token",
    "password",
    "authorization",
    "bearer",
    "access_key",
    "private_key",
)


def _should_redact_key(key: str) -> bool:
    """Check if a key should be redacted based on sensitive fragments.

    Args:
        key: The dictionary key to check

    Returns:
        True if the key contains a sensitive fragment (case-insensitive)
    """
    key_lower = key.lower()
    return any(fragment in key_lower for fragment in _SENSITIVE_KEY_FRAGMENTS)


def _redact_data(value: Any) -> Any:
    """Recursively redact sensitive data from event data structures.

    Args:
        value: The value to redact (can be dict, list, tuple, or scalar)

    Returns:
        The same structure with sensitive values replaced with [REDACTED]
    """
    if isinstance(value, dict):
        return {
            k: _REDACTED_VALUE
            if isinstance(k, str) and _should_redact_key(k)
            else _redact_data(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_data(item) for item in value)
    return value


class ConsoleSink:
    """Telemetry sink that pretty-prints events to console.

    Implements TelemetrySink protocol with BufferMode.IMMEDIATE, printing each
    event immediately as it's emitted. Useful for development and debugging.

    Args:
        output: Output stream to write to (defaults to sys.stdout)
        compact: If True, use compact single-line format. If False, use pretty multi-line format.
    """

    def __init__(
        self,
        output: TextIO | None = None,
        compact: bool = False,
        redact: bool = True,
    ) -> None:
        """Initialize console sink.

        Args:
            output: Output stream (defaults to sys.stdout)
            compact: Use compact format if True
            redact: If True, redact sensitive keys from event data (default: True)
        """
        self._output = output or sys.stdout
        self._compact = compact
        self._redact = redact

    @property
    def buffering(self) -> BufferMode:
        """Return IMMEDIATE buffer mode."""
        return BufferMode.IMMEDIATE

    async def emit(self, event: TraceEvent) -> None:
        """Emit a trace event to the console.

        Args:
            event: Event to emit
        """
        if self._compact:
            self._emit_compact(event)
        else:
            self._emit_pretty(event)

    def _emit_compact(self, event: TraceEvent) -> None:
        """Emit event in compact single-line format."""
        parts = [
            f"[{event.type}]",
            f"run={event.run_id[:8]}",
        ]

        if event.phase:
            parts.append(f"phase={event.phase}")
        if event.cycle is not None:
            parts.append(f"cycle={event.cycle}")
        if event.step_id:
            parts.append(f"step={event.step_id}")
        if event.duration_ms is not None:
            parts.append(f"duration={event.duration_ms:.1f}ms")

        data = _redact_data(event.data) if self._redact else event.data
        if data:
            parts.append(f"data={data}")

        line = " ".join(parts)
        self._output.write(line + "\n")
        self._output.flush()

    def _emit_pretty(self, event: TraceEvent) -> None:
        """Emit event in pretty multi-line format."""
        lines = [
            f"▶ {event.type}",
            f"  run_id: {event.run_id}",
            f"  timestamp: {event.timestamp.isoformat()}",
        ]

        if event.phase:
            lines.append(f"  phase: {event.phase}")
        if event.cycle is not None:
            lines.append(f"  cycle: {event.cycle}")
        if event.step_id:
            lines.append(f"  step_id: {event.step_id}")
        if event.duration_ms is not None:
            lines.append(f"  duration: {event.duration_ms:.1f}ms")

        data = _redact_data(event.data) if self._redact else event.data
        if data:
            lines.append(f"  data: {data}")

        output = "\n".join(lines) + "\n"
        self._output.write(output)
        self._output.flush()

    async def flush(self) -> None:
        """No-op for IMMEDIATE mode."""
        pass

    async def close(self) -> None:
        """No-op cleanup."""
        pass


__all__ = ["ConsoleSink"]
