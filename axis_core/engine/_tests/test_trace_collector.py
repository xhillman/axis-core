"""Tests for TraceCollector class."""

from __future__ import annotations

from datetime import datetime

import pytest

from axis_core.engine.trace_collector import TraceCollector
from axis_core.protocols.telemetry import BufferMode, TraceEvent


class TestTraceCollector:
    """Tests for TraceCollector event accumulator."""

    @pytest.mark.asyncio
    async def test_creation(self) -> None:
        collector = TraceCollector()
        assert collector is not None

    @pytest.mark.asyncio
    async def test_buffering_mode_is_end(self) -> None:
        collector = TraceCollector()
        assert collector.buffering == BufferMode.END

    @pytest.mark.asyncio
    async def test_emit_stores_event(self) -> None:
        collector = TraceCollector()
        event = TraceEvent(
            type="test_event",
            timestamp=datetime.utcnow(),
            run_id="test-run-123",
        )
        await collector.emit(event)
        events = collector.get_events()
        assert len(events) == 1
        assert events[0] is event

    @pytest.mark.asyncio
    async def test_emit_multiple_events(self) -> None:
        collector = TraceCollector()
        event1 = TraceEvent(
            type="event1",
            timestamp=datetime.utcnow(),
            run_id="run1",
        )
        event2 = TraceEvent(
            type="event2",
            timestamp=datetime.utcnow(),
            run_id="run1",
        )
        await collector.emit(event1)
        await collector.emit(event2)
        events = collector.get_events()
        assert len(events) == 2
        assert events[0] is event1
        assert events[1] is event2

    @pytest.mark.asyncio
    async def test_get_events_returns_copy(self) -> None:
        """get_events should return a copy to prevent mutation."""
        collector = TraceCollector()
        event = TraceEvent(
            type="test",
            timestamp=datetime.utcnow(),
            run_id="run",
        )
        await collector.emit(event)

        events1 = collector.get_events()
        events2 = collector.get_events()

        # Should be equal but not the same object
        assert events1 == events2
        assert events1 is not events2

    @pytest.mark.asyncio
    async def test_flush_is_noop(self) -> None:
        """flush should be a no-op for END mode."""
        collector = TraceCollector()
        event = TraceEvent(
            type="test",
            timestamp=datetime.utcnow(),
            run_id="run",
        )
        await collector.emit(event)
        await collector.flush()
        # Events still available
        assert len(collector.get_events()) == 1

    @pytest.mark.asyncio
    async def test_close_is_noop(self) -> None:
        """close should be a no-op."""
        collector = TraceCollector()
        event = TraceEvent(
            type="test",
            timestamp=datetime.utcnow(),
            run_id="run",
        )
        await collector.emit(event)
        await collector.close()
        # Events still available
        assert len(collector.get_events()) == 1

    @pytest.mark.asyncio
    async def test_clear_removes_all_events(self) -> None:
        """clear should remove all collected events."""
        collector = TraceCollector()
        event = TraceEvent(
            type="test",
            timestamp=datetime.utcnow(),
            run_id="run",
        )
        await collector.emit(event)
        assert len(collector.get_events()) == 1

        collector.clear()
        assert len(collector.get_events()) == 0

    @pytest.mark.asyncio
    async def test_events_preserve_order(self) -> None:
        """Events should be returned in the order they were emitted."""
        collector = TraceCollector()
        events_to_emit = [
            TraceEvent(type=f"event{i}", timestamp=datetime.utcnow(), run_id="run")
            for i in range(10)
        ]

        for event in events_to_emit:
            await collector.emit(event)

        collected = collector.get_events()
        for i, event in enumerate(collected):
            assert event.type == f"event{i}"
