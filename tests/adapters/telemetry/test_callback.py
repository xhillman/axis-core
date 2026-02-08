"""Tests for CallbackSink telemetry adapter."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from axis_core.adapters.telemetry.callback import CallbackSink
from axis_core.protocols.telemetry import BufferMode, TraceEvent


def _event(data: dict[str, Any] | None = None) -> TraceEvent:
    return TraceEvent(
        type="test_event",
        timestamp=datetime.now(timezone.utc),
        run_id="run-1",
        data=data or {},
    )


class TestCallbackSink:
    """Tests for CallbackSink adapter."""

    def test_buffering_mode_is_immediate_by_default(self) -> None:
        sink = CallbackSink(handler=lambda _: None)
        assert sink.buffering == BufferMode.IMMEDIATE

    @pytest.mark.asyncio
    async def test_emit_calls_sync_handler(self) -> None:
        calls: list[TraceEvent] = []

        def handler(event: TraceEvent) -> None:
            calls.append(event)

        sink = CallbackSink(handler=handler)
        await sink.emit(_event())
        assert len(calls) == 1
        assert calls[0].type == "test_event"

    @pytest.mark.asyncio
    async def test_emit_calls_async_handler(self) -> None:
        calls: list[TraceEvent] = []

        async def handler(event: TraceEvent) -> None:
            calls.append(event)

        sink = CallbackSink(handler=handler)
        await sink.emit(_event())
        assert len(calls) == 1
        assert calls[0].run_id == "run-1"

    @pytest.mark.asyncio
    async def test_redacts_sensitive_fields_by_default(self) -> None:
        seen: list[TraceEvent] = []

        def handler(event: TraceEvent) -> None:
            seen.append(event)

        sink = CallbackSink(handler=handler)
        await sink.emit(_event({"api_key": "secret-123"}))

        assert len(seen) == 1
        assert seen[0].data["api_key"] == "[REDACTED]"

    @pytest.mark.asyncio
    async def test_can_disable_redaction(self) -> None:
        seen: list[TraceEvent] = []

        def handler(event: TraceEvent) -> None:
            seen.append(event)

        sink = CallbackSink(handler=handler, redact=False)
        await sink.emit(_event({"api_key": "secret-123"}))

        assert len(seen) == 1
        assert seen[0].data["api_key"] == "secret-123"

    @pytest.mark.asyncio
    async def test_flush_and_close_are_safe(self) -> None:
        sink = CallbackSink(handler=lambda _: None)
        await sink.flush()
        await sink.close()
        await sink.close()
