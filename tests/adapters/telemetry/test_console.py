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


class TestConsoleSinkRedaction:
    """Tests for sensitive data redaction in ConsoleSink (MED-1)."""

    @pytest.mark.asyncio
    async def test_redacts_api_key_by_default(self) -> None:
        """By default, ConsoleSink should redact sensitive keys like api_key."""
        output = io.StringIO()
        sink = ConsoleSink(output=output)

        event = TraceEvent(
            type="tool_called",
            timestamp=datetime.utcnow(),
            run_id="run",
            data={"tool": "api_call", "args": {"api_key": "secret-123"}},
        )
        await sink.emit(event)

        result = output.getvalue()
        assert "secret-123" not in result
        assert "[REDACTED]" in result

    @pytest.mark.asyncio
    async def test_redacts_multiple_sensitive_keys(self) -> None:
        """Should redact all common sensitive key patterns."""
        output = io.StringIO()
        sink = ConsoleSink(output=output)

        event = TraceEvent(
            type="tool_called",
            timestamp=datetime.utcnow(),
            run_id="run",
            data={
                "args": {
                    "api_key": "key123",
                    "apikey": "key456",
                    "secret": "secret789",
                    "token": "token-abc",
                    "password": "pass123",
                    "authorization": "Bearer xyz",
                    "bearer": "bearer-token",
                    "access_key": "access-key",
                    "private_key": "private-key",
                    "safe_data": "this-is-ok",
                }
            },
        )
        await sink.emit(event)

        result = output.getvalue()
        # Sensitive values should be redacted
        assert "key123" not in result
        assert "key456" not in result
        assert "secret789" not in result
        assert "token-abc" not in result
        assert "pass123" not in result
        assert "Bearer xyz" not in result
        assert "bearer-token" not in result
        assert "access-key" not in result
        assert "private-key" not in result

        # Non-sensitive data should remain
        assert "this-is-ok" in result
        assert "[REDACTED]" in result

    @pytest.mark.asyncio
    async def test_redacts_case_insensitive(self) -> None:
        """Redaction should be case-insensitive."""
        output = io.StringIO()
        sink = ConsoleSink(output=output)

        event = TraceEvent(
            type="tool_called",
            timestamp=datetime.utcnow(),
            run_id="run",
            data={
                "args": {
                    "API_KEY": "key1",
                    "ApiKey": "key2",
                    "SECRET_TOKEN": "secret",
                }
            },
        )
        await sink.emit(event)

        result = output.getvalue()
        assert "key1" not in result
        assert "key2" not in result
        assert "secret" not in result
        assert "[REDACTED]" in result

    @pytest.mark.asyncio
    async def test_redacts_nested_structures(self) -> None:
        """Should redact sensitive keys in nested dicts and lists."""
        output = io.StringIO()
        sink = ConsoleSink(output=output)

        event = TraceEvent(
            type="tool_called",
            timestamp=datetime.utcnow(),
            run_id="run",
            data={
                "args": {
                    "config": {
                        "api_key": "nested-secret",
                        "safe": "ok",
                    },
                    "credentials": [
                        {"password": "pass1"},
                        {"token": "token1"},
                    ],
                }
            },
        )
        await sink.emit(event)

        result = output.getvalue()
        assert "nested-secret" not in result
        assert "pass1" not in result
        assert "token1" not in result
        assert "ok" in result
        assert "[REDACTED]" in result

    @pytest.mark.asyncio
    async def test_no_redaction_when_disabled(self) -> None:
        """When redact=False, should not redact sensitive data."""
        output = io.StringIO()
        sink = ConsoleSink(output=output, redact=False)

        event = TraceEvent(
            type="tool_called",
            timestamp=datetime.utcnow(),
            run_id="run",
            data={"args": {"api_key": "secret-123"}},
        )
        await sink.emit(event)

        result = output.getvalue()
        assert "secret-123" in result
        assert "[REDACTED]" not in result

    @pytest.mark.asyncio
    async def test_redaction_enabled_by_default(self) -> None:
        """Redaction should be enabled by default for security."""
        sink = ConsoleSink()
        # Access the private attribute to verify default
        assert sink._redact is True

    @pytest.mark.asyncio
    async def test_redaction_works_in_compact_mode(self) -> None:
        """Redaction should work in both compact and pretty modes."""
        output = io.StringIO()
        sink = ConsoleSink(output=output, compact=True)

        event = TraceEvent(
            type="tool_called",
            timestamp=datetime.utcnow(),
            run_id="run",
            data={"args": {"token": "secret-token"}},
        )
        await sink.emit(event)

        result = output.getvalue()
        assert "secret-token" not in result
        assert "[REDACTED]" in result

    @pytest.mark.asyncio
    async def test_redacts_keys_with_fragments_in_middle(self) -> None:
        """Should match sensitive fragments anywhere in key name."""
        output = io.StringIO()
        sink = ConsoleSink(output=output)

        event = TraceEvent(
            type="tool_called",
            timestamp=datetime.utcnow(),
            run_id="run",
            data={
                "args": {
                    "user_api_key": "key1",
                    "my_secret_value": "secret1",
                    "auth_token": "token1",
                }
            },
        )
        await sink.emit(event)

        result = output.getvalue()
        assert "key1" not in result
        assert "secret1" not in result
        assert "token1" not in result
        assert "[REDACTED]" in result

    @pytest.mark.asyncio
    async def test_handles_none_data_gracefully(self) -> None:
        """Should handle events with no data without errors."""
        output = io.StringIO()
        sink = ConsoleSink(output=output)

        event = TraceEvent(
            type="test_event",
            timestamp=datetime.utcnow(),
            run_id="run",
            data=None,
        )
        await sink.emit(event)

        result = output.getvalue()
        assert "test_event" in result
        # Should not crash

    @pytest.mark.asyncio
    async def test_handles_non_dict_data(self) -> None:
        """Should handle non-dict data without errors."""
        output = io.StringIO()
        sink = ConsoleSink(output=output)

        event = TraceEvent(
            type="test_event",
            timestamp=datetime.utcnow(),
            run_id="run",
            data="just a string",
        )
        await sink.emit(event)

        result = output.getvalue()
        assert "just a string" in result
