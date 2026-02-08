"""Tests for FileSink telemetry adapter."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from axis_core.adapters.telemetry.file import FileSink
from axis_core.protocols.telemetry import BufferMode, TraceEvent


def _event(
    *,
    event_type: str = "test_event",
    run_id: str = "run-1",
    data: dict[str, object] | None = None,
) -> TraceEvent:
    return TraceEvent(
        type=event_type,
        timestamp=datetime.now(timezone.utc),
        run_id=run_id,
        data=data or {},
    )


class TestFileSink:
    """Tests for FileSink adapter."""

    @pytest.mark.asyncio
    async def test_buffering_mode_is_batched_by_default(self, tmp_path: Path) -> None:
        sink = FileSink(path=tmp_path / "trace.jsonl")
        assert sink.buffering == BufferMode.BATCHED

    @pytest.mark.asyncio
    async def test_emit_buffers_until_flush(self, tmp_path: Path) -> None:
        path = tmp_path / "trace.jsonl"
        sink = FileSink(path=path, batch_size=10)

        await sink.emit(_event(data={"value": 1}))
        assert not path.exists()

        await sink.flush()
        assert path.exists()
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        payload = json.loads(lines[0])
        assert payload["type"] == "test_event"
        assert payload["run_id"] == "run-1"
        assert payload["data"]["value"] == 1

    @pytest.mark.asyncio
    async def test_emit_flushes_when_batch_size_reached(self, tmp_path: Path) -> None:
        path = tmp_path / "trace.jsonl"
        sink = FileSink(path=path, batch_size=2)

        await sink.emit(_event(data={"value": 1}))
        assert not path.exists()
        await sink.emit(_event(data={"value": 2}))

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2

    @pytest.mark.asyncio
    async def test_close_flushes_pending_events(self, tmp_path: Path) -> None:
        path = tmp_path / "trace.jsonl"
        sink = FileSink(path=path, batch_size=50)

        await sink.emit(_event(data={"value": 7}))
        await sink.close()

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        payload = json.loads(lines[0])
        assert payload["data"]["value"] == 7

    @pytest.mark.asyncio
    async def test_redacts_sensitive_fields_by_default(self, tmp_path: Path) -> None:
        path = tmp_path / "trace.jsonl"
        sink = FileSink(path=path)

        await sink.emit(_event(data={"api_key": "secret-123"}))
        await sink.flush()

        content = path.read_text(encoding="utf-8")
        assert "secret-123" not in content
        assert "[REDACTED]" in content

    @pytest.mark.asyncio
    async def test_can_disable_redaction(self, tmp_path: Path) -> None:
        path = tmp_path / "trace.jsonl"
        sink = FileSink(path=path, redact=False)

        await sink.emit(_event(data={"api_key": "secret-123"}))
        await sink.flush()

        content = path.read_text(encoding="utf-8")
        assert "secret-123" in content
        assert "[REDACTED]" not in content
