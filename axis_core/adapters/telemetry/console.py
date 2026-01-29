"""Console telemetry sink for axis-core.

Provides ConsoleSink adapter that pretty-prints telemetry events to stdout or
a specified output stream. Uses IMMEDIATE buffering mode.
"""

from __future__ import annotations

import sys
from typing import TextIO

from axis_core.protocols.telemetry import BufferMode, TraceEvent


class ConsoleSink:
    """Telemetry sink that pretty-prints events to console.

    Implements TelemetrySink protocol with BufferMode.IMMEDIATE, printing each
    event immediately as it's emitted. Useful for development and debugging.

    Args:
        output: Output stream to write to (defaults to sys.stdout)
        compact: If True, use compact single-line format. If False, use pretty multi-line format.
    """

    def __init__(self, output: TextIO | None = None, compact: bool = False) -> None:
        """Initialize console sink.

        Args:
            output: Output stream (defaults to sys.stdout)
            compact: Use compact format if True
        """
        self._output = output or sys.stdout
        self._compact = compact

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
        if event.data:
            parts.append(f"data={event.data}")

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
        if event.data:
            lines.append(f"  data: {event.data}")

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
