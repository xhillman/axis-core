"""Tests for telemetry protocol and dataclasses."""

from datetime import datetime, timezone

import pytest

from axis_core.protocols.telemetry import BufferMode, TelemetrySink, TraceEvent


class TestBufferMode:
    """Tests for BufferMode enum."""

    def test_enum_values(self):
        """Test that enum values match their lowercase names."""
        assert BufferMode.IMMEDIATE.value == "immediate"
        assert BufferMode.BATCHED.value == "batched"
        assert BufferMode.PHASE.value == "phase"
        assert BufferMode.END.value == "end"

    def test_enum_count(self):
        """Test that we have exactly 4 buffer modes."""
        assert len(BufferMode) == 4

    def test_enum_membership(self):
        """Test enum membership checks."""
        assert BufferMode.IMMEDIATE in BufferMode
        assert "invalid" not in [b.value for b in BufferMode]


class TestTraceEvent:
    """Tests for TraceEvent dataclass."""

    def test_minimal(self):
        """Test TraceEvent with only required fields."""
        now = datetime.now(timezone.utc)
        event = TraceEvent(type="test_event", timestamp=now, run_id="run123")
        assert event.type == "test_event"
        assert event.timestamp == now
        assert event.run_id == "run123"
        assert event.phase is None
        assert event.cycle is None
        assert event.step_id is None
        assert event.data == {}
        assert event.duration_ms is None

    def test_with_phase(self):
        """Test TraceEvent with phase."""
        now = datetime.now(timezone.utc)
        event = TraceEvent(
            type="phase_start", timestamp=now, run_id="run123", phase="observe"
        )
        assert event.phase == "observe"

    def test_with_cycle(self):
        """Test TraceEvent with cycle number."""
        now = datetime.now(timezone.utc)
        event = TraceEvent(type="cycle_start", timestamp=now, run_id="run123", cycle=1)
        assert event.cycle == 1

    def test_with_step_id(self):
        """Test TraceEvent with step ID."""
        now = datetime.now(timezone.utc)
        event = TraceEvent(
            type="step_complete", timestamp=now, run_id="run123", step_id="step1"
        )
        assert event.step_id == "step1"

    def test_with_data(self):
        """Test TraceEvent with data payload."""
        now = datetime.now(timezone.utc)
        event = TraceEvent(
            type="tool_call",
            timestamp=now,
            run_id="run123",
            data={"tool": "search", "args": {"query": "test"}},
        )
        assert event.data == {"tool": "search", "args": {"query": "test"}}

    def test_with_duration(self):
        """Test TraceEvent with duration."""
        now = datetime.now(timezone.utc)
        event = TraceEvent(
            type="phase_complete", timestamp=now, run_id="run123", duration_ms=123.45
        )
        assert event.duration_ms == 123.45

    def test_with_all_fields(self):
        """Test TraceEvent with all fields populated."""
        now = datetime.now(timezone.utc)
        event = TraceEvent(
            type="step_complete",
            timestamp=now,
            run_id="run123",
            phase="act",
            cycle=2,
            step_id="step5",
            data={"result": "success"},
            duration_ms=456.78,
        )
        assert event.type == "step_complete"
        assert event.timestamp == now
        assert event.run_id == "run123"
        assert event.phase == "act"
        assert event.cycle == 2
        assert event.step_id == "step5"
        assert event.data == {"result": "success"}
        assert event.duration_ms == 456.78

    def test_immutability(self):
        """Test that TraceEvent is immutable."""
        now = datetime.now(timezone.utc)
        event = TraceEvent(type="test", timestamp=now, run_id="run123")
        with pytest.raises(AttributeError):
            event.type = "new_type"  # type: ignore


class TestTelemetrySink:
    """Tests for TelemetrySink protocol."""

    @pytest.mark.asyncio
    async def test_protocol_implementation(self):
        """Test that a class implementing TelemetrySink conforms to the protocol."""

        class FakeTelemetrySink:
            def __init__(self):
                self.events = []
                self.flushed = False
                self.closed = False

            @property
            def buffering(self) -> BufferMode:
                return BufferMode.BATCHED

            async def emit(self, event: TraceEvent) -> None:
                self.events.append(event)

            async def flush(self) -> None:
                self.flushed = True

            async def close(self) -> None:
                self.closed = True
                await self.flush()

        sink = FakeTelemetrySink()
        assert isinstance(sink, TelemetrySink)

        # Test buffering property
        assert sink.buffering == BufferMode.BATCHED

        # Test emit
        now = datetime.now(timezone.utc)
        event = TraceEvent(type="test", timestamp=now, run_id="run123")
        await sink.emit(event)
        assert len(sink.events) == 1
        assert sink.events[0] == event

        # Test flush
        await sink.flush()
        assert sink.flushed is True

        # Test close
        await sink.close()
        assert sink.closed is True

    @pytest.mark.asyncio
    async def test_protocol_immediate_mode(self):
        """Test TelemetrySink with immediate buffering."""

        class ImmediateSink:
            @property
            def buffering(self) -> BufferMode:
                return BufferMode.IMMEDIATE

            async def emit(self, event: TraceEvent) -> None:
                pass

            async def flush(self) -> None:
                pass

            async def close(self) -> None:
                pass

        sink = ImmediateSink()
        assert isinstance(sink, TelemetrySink)
        assert sink.buffering == BufferMode.IMMEDIATE

    def test_protocol_missing_methods(self):
        """Test that a class missing methods doesn't conform to protocol."""

        class IncompleteSink:
            @property
            def buffering(self) -> BufferMode:
                return BufferMode.IMMEDIATE

            async def emit(self, event: TraceEvent) -> None:
                pass

        sink = IncompleteSink()
        assert not isinstance(sink, TelemetrySink)
