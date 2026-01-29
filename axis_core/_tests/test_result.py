"""Tests for RunResult, RunStats, and StreamEvent dataclasses."""

from __future__ import annotations

import dataclasses
from datetime import datetime

import pytest

from axis_core.context import RunState
from axis_core.errors import AxisError, ErrorClass, ErrorRecord
from axis_core.result import RunResult, RunStats, StreamEvent


class TestRunStats:
    """Tests for RunStats frozen dataclass."""

    def test_creation_with_defaults(self) -> None:
        stats = RunStats(
            cycles=0,
            tool_calls=0,
            model_calls=0,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
            duration_ms=0.0,
        )
        assert stats.cycles == 0
        assert stats.cost_usd == 0.0

    def test_creation_with_values(self) -> None:
        stats = RunStats(
            cycles=3,
            tool_calls=5,
            model_calls=4,
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1500,
            cost_usd=0.05,
            duration_ms=1234.5,
        )
        assert stats.cycles == 3
        assert stats.tool_calls == 5
        assert stats.model_calls == 4
        assert stats.input_tokens == 1000
        assert stats.output_tokens == 500
        assert stats.total_tokens == 1500
        assert stats.cost_usd == 0.05
        assert stats.duration_ms == 1234.5

    def test_frozen(self) -> None:
        stats = RunStats(
            cycles=1,
            tool_calls=0,
            model_calls=1,
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost_usd=0.01,
            duration_ms=500.0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            stats.cycles = 2  # type: ignore[misc]


class TestStreamEvent:
    """Tests for StreamEvent dataclass."""

    def test_creation(self) -> None:
        now = datetime.utcnow()
        event = StreamEvent(
            type="model_token",
            timestamp=now,
            data={"token": "hello"},
        )
        assert event.type == "model_token"
        assert event.timestamp == now
        assert event.data == {"token": "hello"}
        assert event.sequence is None

    def test_with_sequence(self) -> None:
        event = StreamEvent(
            type="model_token",
            timestamp=datetime.utcnow(),
            data={"token": "world"},
            sequence=5,
        )
        assert event.sequence == 5

    def test_is_token_true(self) -> None:
        event = StreamEvent(
            type="model_token",
            timestamp=datetime.utcnow(),
            data={"token": "hi"},
        )
        assert event.is_token is True

    def test_is_token_false(self) -> None:
        event = StreamEvent(
            type="phase_entered",
            timestamp=datetime.utcnow(),
            data={"phase": "observe"},
        )
        assert event.is_token is False

    def test_token_property(self) -> None:
        event = StreamEvent(
            type="model_token",
            timestamp=datetime.utcnow(),
            data={"token": "abc"},
        )
        assert event.token == "abc"

    def test_token_property_none_for_non_token(self) -> None:
        event = StreamEvent(
            type="tool_called",
            timestamp=datetime.utcnow(),
            data={},
        )
        assert event.token is None

    def test_is_final_true(self) -> None:
        event = StreamEvent(
            type="run_completed",
            timestamp=datetime.utcnow(),
            data={"success": True},
        )
        assert event.is_final is True

    def test_is_final_false(self) -> None:
        event = StreamEvent(
            type="model_token",
            timestamp=datetime.utcnow(),
            data={"token": "x"},
        )
        assert event.is_final is False

    def test_is_final_for_run_failed(self) -> None:
        event = StreamEvent(
            type="run_failed",
            timestamp=datetime.utcnow(),
            data={"error": "something"},
        )
        assert event.is_final is True

    def test_frozen(self) -> None:
        event = StreamEvent(
            type="model_token",
            timestamp=datetime.utcnow(),
            data={},
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            event.type = "other"  # type: ignore[misc]


class TestRunResult:
    """Tests for RunResult frozen dataclass (AD-036)."""

    def _make_result(self, **overrides: object) -> RunResult:
        """Helper to create a RunResult with defaults."""
        defaults: dict[str, object] = {
            "output": "test output",
            "output_raw": "test output",
            "success": True,
            "error": None,
            "had_recoverable_errors": False,
            "stats": RunStats(
                cycles=1,
                tool_calls=0,
                model_calls=1,
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                cost_usd=0.01,
                duration_ms=500.0,
            ),
            "trace": [],
            "state": RunState(),
            "run_id": "test-run-123",
            "memory_error": None,
        }
        defaults.update(overrides)
        return RunResult(**defaults)  # type: ignore[arg-type]

    def test_creation(self) -> None:
        result = self._make_result()
        assert result.output == "test output"
        assert result.success is True
        assert result.run_id == "test-run-123"

    def test_frozen_immutability(self) -> None:
        result = self._make_result()
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.output = "new"  # type: ignore[misc]

    def test_copy_helper(self) -> None:
        result = self._make_result()
        modified = result.copy(output="modified output")
        assert modified.output == "modified output"
        assert result.output == "test output"  # original unchanged
        assert modified.run_id == result.run_id

    def test_copy_preserves_type(self) -> None:
        result = self._make_result()
        modified = result.copy(success=False)
        assert isinstance(modified, RunResult)
        assert modified.success is False

    def test_success_false_with_error(self) -> None:
        err = AxisError(message="failed", error_class=ErrorClass.RUNTIME)
        result = self._make_result(success=False, error=err)
        assert result.success is False
        assert result.error is err

    def test_had_recoverable_errors(self) -> None:
        result = self._make_result(had_recoverable_errors=True)
        assert result.had_recoverable_errors is True

    def test_memory_error(self) -> None:
        mem_err = AxisError(message="mem fail", error_class=ErrorClass.RUNTIME)
        result = self._make_result(memory_error=mem_err)
        assert result.memory_error is mem_err
