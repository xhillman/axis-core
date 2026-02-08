"""File telemetry sink for axis-core.

Provides FileSink adapter that writes telemetry events to JSON Lines (JSONL)
files with configurable buffering behavior.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from axis_core.protocols.telemetry import BufferMode, TraceEvent
from axis_core.redaction import redact_sensitive_data


class FileSink:
    """Telemetry sink that appends trace events to a JSONL file.

    Args:
        path: Output JSONL file path.
        batch_size: Number of buffered events to write per batch when using
            BATCHED mode.
        buffering: Buffering strategy for the sink.
        redact: Whether to redact sensitive keys from event data.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        batch_size: int = 100,
        buffering: BufferMode = BufferMode.BATCHED,
        redact: bool = True,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        self._path = Path(path)
        self._batch_size = batch_size
        self._buffering = buffering
        self._redact = redact
        self._buffer: list[str] = []
        self._closed = False
        self._lock = asyncio.Lock()

    @property
    def buffering(self) -> BufferMode:
        """Return configured buffering mode."""
        return self._buffering

    async def emit(self, event: TraceEvent) -> None:
        """Buffer or write an event depending on buffering mode."""
        line = self._serialize_event(event)

        async with self._lock:
            if self._closed:
                raise RuntimeError("FileSink is closed")

            if self._buffering == BufferMode.IMMEDIATE:
                self._write_lines([line])
                return

            self._buffer.append(line)
            if self._buffering == BufferMode.BATCHED and len(self._buffer) >= self._batch_size:
                pending = list(self._buffer)
                self._buffer.clear()
                self._write_lines(pending)

    async def flush(self) -> None:
        """Flush buffered events to disk."""
        async with self._lock:
            if not self._buffer:
                return
            pending = list(self._buffer)
            self._buffer.clear()
            self._write_lines(pending)

    async def close(self) -> None:
        """Flush remaining events and close the sink."""
        async with self._lock:
            if self._closed:
                return
            pending = list(self._buffer)
            self._buffer.clear()
            if pending:
                self._write_lines(pending)
            self._closed = True

    def _serialize_event(self, event: TraceEvent) -> str:
        data = redact_sensitive_data(event.data) if self._redact else event.data
        payload: dict[str, Any] = {
            "type": event.type,
            "timestamp": event.timestamp.isoformat(),
            "run_id": event.run_id,
            "phase": event.phase,
            "cycle": event.cycle,
            "step_id": event.step_id,
            "data": data,
            "duration_ms": event.duration_ms,
        }
        return json.dumps(payload, separators=(",", ":"), default=str)

    def _write_lines(self, lines: list[str]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
            handle.write("\n")


__all__ = ["FileSink"]
