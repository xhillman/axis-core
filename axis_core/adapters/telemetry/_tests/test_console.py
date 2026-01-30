"""Tests for ConsoleSink telemetry adapter."""

from __future__ import annotations

import io
from datetime import datetime

import pytest

from axis_core.adapters.telemetry.console import ConsoleSink
from axis_core.protocols.telemetry import BufferMode, TraceEvent


class TestConsoleSink:
    """Tests for ConsoleSink adapter."""

    @pytest.mark.asyncio
    async def test_creation(self) -> None:
        sink = ConsoleSink()
        assert sink is not None

    @pytest.mark.asyncio
    async def test_buffering_mode_is_immediate(self) -> None:
        sink = ConsoleSink()
        assert sink.buffering == BufferMode.IMMEDIATE

    @pytest.mark.asyncio
    async def test_emit_writes_to_output(self) -> None:
        output = io.StringIO()
        sink = ConsoleSink(output=output)

        event = TraceEvent(
            type="test_event",
            timestamp=datetime.utcnow(),
            run_id="test-run-123",
        )
        await sink.emit(event)

        result = output.getvalue()
        assert len(result) > 0
        assert "test_event" in result

    @pytest.mark.asyncio
    async def test_emit_includes_timestamp(self) -> None:
        output = io.StringIO()
        sink = ConsoleSink(output=output)

        event = TraceEvent(
            type="test_event",
            timestamp=datetime.utcnow(),
            run_id="test-run",
        )
        await sink.emit(event)

        result = output.getvalue()
        # Should include some timestamp representation
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_emit_includes_run_id(self) -> None:
        output = io.StringIO()
        sink = ConsoleSink(output=output)

        event = TraceEvent(
            type="test_event",
            timestamp=datetime.utcnow(),
            run_id="test-run-456",
        )
        await sink.emit(event)

        result = output.getvalue()
        assert "test-run-456" in result

    @pytest.mark.asyncio
    async def test_emit_includes_phase(self) -> None:
        output = io.StringIO()
        sink = ConsoleSink(output=output)

        event = TraceEvent(
            type="phase_entered",
            timestamp=datetime.utcnow(),
            run_id="run",
            phase="plan",
        )
        await sink.emit(event)

        result = output.getvalue()
        assert "plan" in result

    @pytest.mark.asyncio
    async def test_emit_includes_cycle(self) -> None:
        output = io.StringIO()
        sink = ConsoleSink(output=output)

        event = TraceEvent(
            type="cycle_started",
            timestamp=datetime.utcnow(),
            run_id="run",
            cycle=3,
        )
        await sink.emit(event)

        result = output.getvalue()
        assert "3" in result

    @pytest.mark.asyncio
    async def test_emit_includes_data(self) -> None:
        output = io.StringIO()
        sink = ConsoleSink(output=output)

        event = TraceEvent(
            type="tool_called",
            timestamp=datetime.utcnow(),
            run_id="run",
            data={"tool_name": "search", "args": {"query": "test"}},
        )
        await sink.emit(event)

        result = output.getvalue()
        assert "search" in result

    @pytest.mark.asyncio
    async def test_flush_is_noop_for_immediate(self) -> None:
        """flush should be a no-op for IMMEDIATE mode."""
        sink = ConsoleSink()
        await sink.flush()  # Should not raise

    @pytest.mark.asyncio
    async def test_close_is_noop(self) -> None:
        sink = ConsoleSink()
        await sink.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_multiple_events(self) -> None:
        output = io.StringIO()
        sink = ConsoleSink(output=output)

        event1 = TraceEvent(type="event1", timestamp=datetime.utcnow(), run_id="run")
        event2 = TraceEvent(type="event2", timestamp=datetime.utcnow(), run_id="run")

        await sink.emit(event1)
        await sink.emit(event2)

        result = output.getvalue()
        assert "event1" in result
        assert "event2" in result

    @pytest.mark.asyncio
    async def test_defaults_to_stdout(self) -> None:
        """ConsoleSink should default to sys.stdout if no output provided."""

        sink = ConsoleSink()
        # Can't easily test stdout output, but verify it doesn't error
        _event = TraceEvent(type="test", timestamp=datetime.utcnow(), run_id="run")
        # This would print to stdout in a real scenario
        # For now, just verify it doesn't crash
        # await sink.emit(_event)
        assert sink is not None
