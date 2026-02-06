"""Tests for axis_core context and state management system.

Tests for RunContext, RunState, and supporting dataclasses that provide
the single source of truth for agent execution state.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import pytest

from axis_core.budget import Budget, BudgetState
from axis_core.errors import AxisError, ErrorClass, ErrorRecord
from axis_core.protocols.model import ToolCall
from axis_core.protocols.planner import Plan, PlanStep, StepType
from axis_core.tool import ToolCallRecord

if TYPE_CHECKING:
    from axis_core.context import CycleState, RunContext


class TestNormalizedInput:
    """Tests for NormalizedInput dataclass."""

    def test_creation_basic(self) -> None:
        """Test basic NormalizedInput creation."""
        from axis_core.context import NormalizedInput

        input_ = NormalizedInput(text="hello", original="hello")

        assert input_.text == "hello"
        assert input_.original == "hello"
        assert input_.is_multimodal is False

    def test_creation_multimodal(self) -> None:
        """Test NormalizedInput with multimodal flag."""
        from axis_core.context import NormalizedInput

        original = [{"type": "text", "text": "hello"}, {"type": "image", "data": "..."}]
        input_ = NormalizedInput(text="hello", original=original, is_multimodal=True)

        assert input_.text == "hello"
        assert input_.original == original
        assert input_.is_multimodal is True

    def test_frozen(self) -> None:
        """Test that NormalizedInput is immutable (frozen)."""
        from axis_core.context import NormalizedInput

        input_ = NormalizedInput(text="hello", original="hello")

        with pytest.raises(AttributeError):
            input_.text = "world"  # type: ignore[misc]

    def test_original_can_be_list(self) -> None:
        """Test that original can be a list for multimodal inputs."""
        from axis_core.context import NormalizedInput

        original: list[Any] = [{"type": "text", "text": "hello"}]
        input_ = NormalizedInput(text="hello", original=original)

        assert input_.original == original


class TestObservation:
    """Tests for Observation dataclass (Observe phase output)."""

    def test_creation_minimal(self) -> None:
        """Test Observation creation with minimal fields."""
        from axis_core.context import NormalizedInput, Observation

        input_ = NormalizedInput(text="hello", original="hello")
        obs = Observation(input=input_)

        assert obs.input == input_
        assert obs.memory_context == {}
        assert obs.previous_cycles == ()
        assert obs.tool_requests is None
        assert obs.response is None
        assert obs.goal == ""
        assert isinstance(obs.timestamp, datetime)

    def test_creation_full(self) -> None:
        """Test Observation creation with all fields."""
        from axis_core.context import NormalizedInput, Observation

        input_ = NormalizedInput(text="hello", original="hello")
        tool_call = ToolCall(id="tc-1", name="search", arguments={"query": "test"})
        timestamp = datetime.now(timezone.utc)

        obs = Observation(
            input=input_,
            memory_context={"key": "value"},
            previous_cycles=({"cycle": 0},),
            tool_requests=(tool_call,),
            response="Previous response",
            goal="Find information",
            timestamp=timestamp,
        )

        assert obs.input == input_
        assert obs.memory_context == {"key": "value"}
        assert obs.previous_cycles == ({"cycle": 0},)
        assert obs.tool_requests == (tool_call,)
        assert obs.response == "Previous response"
        assert obs.goal == "Find information"
        assert obs.timestamp == timestamp

    def test_frozen(self) -> None:
        """Test that Observation is immutable (frozen)."""
        from axis_core.context import NormalizedInput, Observation

        input_ = NormalizedInput(text="hello", original="hello")
        obs = Observation(input=input_)

        with pytest.raises(AttributeError):
            obs.goal = "new goal"  # type: ignore[misc]


class TestExecutionResult:
    """Tests for ExecutionResult dataclass (Act phase output)."""

    def test_creation_minimal(self) -> None:
        """Test ExecutionResult creation with defaults."""
        from axis_core.context import ExecutionResult

        result = ExecutionResult()

        assert result.results == {}
        assert result.errors == {}
        assert result.skipped == frozenset()
        assert result.duration_ms == 0.0

    def test_creation_full(self) -> None:
        """Test ExecutionResult creation with all fields."""
        from axis_core.context import ExecutionResult

        error = AxisError(message="tool failed", error_class=ErrorClass.TOOL)
        result = ExecutionResult(
            results={"search": ["result1", "result2"]},
            errors={"calculator": error},
            skipped=frozenset(["slow_tool"]),
            duration_ms=150.5,
        )

        assert result.results == {"search": ["result1", "result2"]}
        assert result.errors == {"calculator": error}
        assert result.skipped == frozenset(["slow_tool"])
        assert result.duration_ms == 150.5

    def test_frozen(self) -> None:
        """Test that ExecutionResult is immutable (frozen)."""
        from axis_core.context import ExecutionResult

        result = ExecutionResult()

        with pytest.raises(AttributeError):
            result.duration_ms = 100.0  # type: ignore[misc]

    def test_skipped_is_frozenset(self) -> None:
        """Test that skipped is a frozenset (immutable set)."""
        from axis_core.context import ExecutionResult

        result = ExecutionResult(skipped=frozenset(["tool1", "tool2"]))

        assert isinstance(result.skipped, frozenset)
        assert "tool1" in result.skipped


class TestEvalDecision:
    """Tests for EvalDecision dataclass (Evaluate phase output)."""

    def test_creation_done(self) -> None:
        """Test EvalDecision when task is complete."""
        from axis_core.context import EvalDecision

        decision = EvalDecision(done=True, reason="Task completed successfully")

        assert decision.done is True
        assert decision.error is None
        assert decision.recoverable is False
        assert decision.reason == "Task completed successfully"

    def test_creation_not_done(self) -> None:
        """Test EvalDecision when task needs more cycles."""
        from axis_core.context import EvalDecision

        decision = EvalDecision(done=False, reason="Need more information")

        assert decision.done is False
        assert decision.error is None
        assert decision.reason == "Need more information"

    def test_creation_with_error(self) -> None:
        """Test EvalDecision with recoverable error."""
        from axis_core.context import EvalDecision

        error = AxisError(message="transient failure", error_class=ErrorClass.MODEL)
        decision = EvalDecision(
            done=False,
            error=error,
            recoverable=True,
            reason="Model call failed, will retry",
        )

        assert decision.done is False
        assert decision.error is error
        assert decision.recoverable is True
        assert decision.reason == "Model call failed, will retry"

    def test_frozen(self) -> None:
        """Test that EvalDecision is immutable (frozen)."""
        from axis_core.context import EvalDecision

        decision = EvalDecision(done=True)

        with pytest.raises(AttributeError):
            decision.done = False  # type: ignore[misc]


class TestModelCallRecord:
    """Tests for ModelCallRecord dataclass."""

    def test_creation(self) -> None:
        """Test ModelCallRecord creation."""
        from axis_core.context import ModelCallRecord

        record = ModelCallRecord(
            model_id="claude-sonnet-4-20250514",
            call_id="call-123",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.0015,
            duration_ms=250.5,
            timestamp=1704067200.0,
        )

        assert record.model_id == "claude-sonnet-4-20250514"
        assert record.call_id == "call-123"
        assert record.input_tokens == 100
        assert record.output_tokens == 50
        assert record.cost_usd == 0.0015
        assert record.duration_ms == 250.5
        assert record.timestamp == 1704067200.0

    def test_frozen(self) -> None:
        """Test that ModelCallRecord is immutable (frozen)."""
        from axis_core.context import ModelCallRecord

        record = ModelCallRecord(
            model_id="gpt-4",
            call_id="call-1",
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.001,
            duration_ms=100.0,
            timestamp=0.0,
        )

        with pytest.raises(AttributeError):
            record.input_tokens = 200  # type: ignore[misc]


class TestCycleState:
    """Tests for CycleState dataclass (complete cycle record)."""

    def _create_sample_cycle_state(self) -> CycleState:
        """Helper to create a sample CycleState for testing."""
        from axis_core.context import (
            CycleState,
            EvalDecision,
            ExecutionResult,
            NormalizedInput,
            Observation,
        )

        input_ = NormalizedInput(text="test", original="test")
        obs = Observation(input=input_, goal="Test goal")
        plan = Plan(id="plan-1", goal="Test", steps=())
        execution = ExecutionResult(results={"tool": "result"})
        evaluation = EvalDecision(done=True, reason="Complete")
        started = datetime.now(timezone.utc)
        ended = datetime.now(timezone.utc)

        return CycleState(
            cycle_number=0,
            observation=obs,
            plan=plan,
            execution=execution,
            evaluation=evaluation,
            started_at=started,
            ended_at=ended,
        )

    def test_creation(self) -> None:
        """Test CycleState creation."""

        cycle = self._create_sample_cycle_state()

        assert cycle.cycle_number == 0
        assert cycle.observation.goal == "Test goal"
        assert cycle.plan.id == "plan-1"
        assert cycle.execution.results == {"tool": "result"}
        assert cycle.evaluation.done is True

    def test_frozen(self) -> None:
        """Test that CycleState is immutable (frozen)."""

        cycle = self._create_sample_cycle_state()

        with pytest.raises(AttributeError):
            cycle.cycle_number = 1  # type: ignore[misc]

    def test_to_dict(self) -> None:
        """Test CycleState serialization to dict."""

        cycle = self._create_sample_cycle_state()
        data = cycle.to_dict()

        assert data["cycle_number"] == 0
        assert "observation" in data
        assert "plan" in data
        assert "execution" in data
        assert "evaluation" in data
        assert "started_at" in data
        assert "ended_at" in data

    def test_from_dict(self) -> None:
        """Test CycleState deserialization from dict."""
        from axis_core.context import CycleState

        cycle = self._create_sample_cycle_state()
        data = cycle.to_dict()
        restored = CycleState.from_dict(data)

        assert restored.cycle_number == cycle.cycle_number
        assert restored.observation.goal == cycle.observation.goal
        assert restored.evaluation.done == cycle.evaluation.done

    def test_serialization_roundtrip(self) -> None:
        """Test that to_dict/from_dict preserves data."""
        from axis_core.context import CycleState

        original = self._create_sample_cycle_state()
        data = original.to_dict()
        restored = CycleState.from_dict(data)

        # Key fields should match
        assert restored.cycle_number == original.cycle_number
        assert restored.observation.input.text == original.observation.input.text
        assert restored.plan.id == original.plan.id
        assert restored.evaluation.done == original.evaluation.done


class TestRunState:
    """Tests for RunState dataclass (mutable state accumulator)."""

    def test_creation(self) -> None:
        """Test RunState creation with defaults."""
        from axis_core.context import RunState

        state = RunState()

        assert state.cycles == ()
        assert state.errors == ()
        assert state.tool_calls == ()
        assert state.model_calls == ()
        assert state.current_observation is None
        assert state.current_plan is None
        assert state.current_execution is None
        assert state.output is None
        assert state.output_raw is None

    def test_append_cycle(self) -> None:
        """Test append-only cycle addition."""
        from axis_core.context import (
            CycleState,
            EvalDecision,
            ExecutionResult,
            NormalizedInput,
            Observation,
            RunState,
        )

        state = RunState()

        # Create a cycle
        input_ = NormalizedInput(text="test", original="test")
        obs = Observation(input=input_)
        plan = Plan(id="plan-1", goal="Test", steps=())
        execution = ExecutionResult()
        evaluation = EvalDecision(done=False)
        started = datetime.now(timezone.utc)
        ended = datetime.now(timezone.utc)

        cycle = CycleState(
            cycle_number=0,
            observation=obs,
            plan=plan,
            execution=execution,
            evaluation=evaluation,
            started_at=started,
            ended_at=ended,
        )

        state.append_cycle(cycle)

        assert len(state.cycles) == 1
        assert state.cycles[0] == cycle

    def test_cycles_returns_tuple(self) -> None:
        """Test that cycles property returns immutable tuple."""
        from axis_core.context import RunState

        state = RunState()

        # Property should return tuple, not list
        assert isinstance(state.cycles, tuple)

    def test_append_error(self) -> None:
        """Test append-only error addition."""
        from axis_core.context import RunState

        state = RunState()

        error = AxisError(message="test error", error_class=ErrorClass.RUNTIME)
        record = ErrorRecord(
            error=error,
            timestamp=datetime.now(timezone.utc),
            phase="act",
            cycle=0,
            recovered=False,
        )

        state.append_error(record)

        assert len(state.errors) == 1
        assert state.errors[0] == record

    def test_errors_returns_tuple(self) -> None:
        """Test that errors property returns immutable tuple."""
        from axis_core.context import RunState

        state = RunState()
        assert isinstance(state.errors, tuple)

    def test_append_tool_call(self) -> None:
        """Test append-only tool call addition."""
        from axis_core.context import RunState

        state = RunState()

        record = ToolCallRecord(
            tool_name="search",
            call_id="call-1",
            args={"query": "test"},
            result="found",
            error=None,
            cached=False,
            duration_ms=100.0,
            timestamp=1704067200.0,
        )

        state.append_tool_call(record)

        assert len(state.tool_calls) == 1
        assert state.tool_calls[0] == record

    def test_tool_calls_returns_tuple(self) -> None:
        """Test that tool_calls property returns immutable tuple."""
        from axis_core.context import RunState

        state = RunState()
        assert isinstance(state.tool_calls, tuple)

    def test_append_model_call(self) -> None:
        """Test append-only model call addition."""
        from axis_core.context import ModelCallRecord, RunState

        state = RunState()

        record = ModelCallRecord(
            model_id="claude-sonnet-4-20250514",
            call_id="call-1",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
            duration_ms=200.0,
            timestamp=1704067200.0,
        )

        state.append_model_call(record)

        assert len(state.model_calls) == 1
        assert state.model_calls[0] == record

    def test_model_calls_returns_tuple(self) -> None:
        """Test that model_calls property returns immutable tuple."""
        from axis_core.context import RunState

        state = RunState()
        assert isinstance(state.model_calls, tuple)

    def test_budget_state_tracking(self) -> None:
        """Test that RunState tracks budget state."""
        from axis_core.context import RunState

        budget_state = BudgetState(cycles=1, cost_usd=0.05)
        state = RunState(budget_state=budget_state)

        assert state.budget_state.cycles == 1
        assert state.budget_state.cost_usd == 0.05

    def test_to_dict(self) -> None:
        """Test RunState serialization."""
        from axis_core.context import RunState

        state = RunState()
        state.output = "final result"
        state.output_raw = "raw response"

        data = state.to_dict()

        assert "cycles" in data
        assert "errors" in data
        assert "tool_calls" in data
        assert "model_calls" in data
        assert data["output"] == "final result"
        assert data["output_raw"] == "raw response"

    def test_to_dict_excludes_retry_state(self) -> None:
        """Test that _retry_state is NOT persisted (AD-014)."""
        from axis_core.context import RunState

        state = RunState()
        # Add some retry state (internal use)
        state._retry_state["step-1"] = {"attempts": 2}

        data = state.to_dict()

        # _retry_state should not be in serialized output
        assert "_retry_state" not in data

    def test_from_dict(self) -> None:
        """Test RunState deserialization."""
        from axis_core.context import RunState

        state = RunState()
        state.output = "test output"
        data = state.to_dict()

        restored = RunState.from_dict(data)

        assert restored.output == "test output"
        assert isinstance(restored.cycles, tuple)

    def test_from_dict_resets_retry_state(self) -> None:
        """Test that deserialization resets retry counters (AD-014)."""
        from axis_core.context import RunState

        state = RunState()
        state._retry_state["step-1"] = {"attempts": 3}
        data = state.to_dict()

        restored = RunState.from_dict(data)

        # Retry state should be reset (empty) on restore
        assert restored._retry_state == {}


class TestRunContext:
    """Tests for RunContext dataclass (single source of truth)."""

    def _create_sample_context(self) -> RunContext:
        """Helper to create a sample RunContext for testing."""
        from axis_core.context import NormalizedInput, RunContext, RunState

        input_ = NormalizedInput(text="test", original="test")
        state = RunState()
        budget = Budget()

        return RunContext(
            run_id="run-123",
            agent_id="agent-456",
            input=input_,
            context={},
            attachments=[],
            config=None,  # Simplified for tests
            budget=budget,
            state=state,
            trace=None,  # Simplified for tests
            started_at=datetime.now(timezone.utc),
            cycle_count=0,
            cancel_token=None,  # Simplified for tests
        )

    def test_creation(self) -> None:
        """Test RunContext creation."""

        ctx = self._create_sample_context()

        assert ctx.run_id == "run-123"
        assert ctx.agent_id == "agent-456"
        assert ctx.input.text == "test"
        assert ctx.cycle_count == 0

    def test_identity_fields_readonly(self) -> None:
        """Test that identity fields are read-only after initialization."""

        ctx = self._create_sample_context()

        # run_id should be read-only
        with pytest.raises(AttributeError):
            ctx.run_id = "new-run-id"

        # agent_id should be read-only
        with pytest.raises(AttributeError):
            ctx.agent_id = "new-agent-id"

    def test_input_readonly(self) -> None:
        """Test that input is read-only after initialization."""
        from axis_core.context import NormalizedInput

        ctx = self._create_sample_context()
        new_input = NormalizedInput(text="new", original="new")

        with pytest.raises(AttributeError):
            ctx.input = new_input

    def test_mutable_fields_can_change(self) -> None:
        """Test that non-identity fields can be modified."""

        ctx = self._create_sample_context()

        # cycle_count should be mutable
        ctx.cycle_count = 1
        assert ctx.cycle_count == 1

        # context dict should be mutable
        ctx.context["key"] = "value"
        assert ctx.context["key"] == "value"

    def test_serialize(self) -> None:
        """Test RunContext serialization."""

        ctx = self._create_sample_context()
        data = ctx.serialize()

        assert data["run_id"] == "run-123"
        assert data["agent_id"] == "agent-456"
        assert "input" in data
        assert "state" in data
        assert "started_at" in data

    def test_serialize_includes_attachment_metadata(self) -> None:
        """RunContext serialization should include attachment metadata only."""
        from axis_core.attachments import Attachment
        from axis_core.context import NormalizedInput, RunContext, RunState

        input_ = NormalizedInput(text="test", original="test")
        state = RunState()
        budget = Budget()
        attachment = Attachment(
            data=b"data",
            mime_type="text/plain",
            filename="note.txt",
        )

        ctx = RunContext(
            run_id="run-123",
            agent_id="agent-456",
            input=input_,
            context={},
            attachments=[attachment],
            config=None,
            budget=budget,
            state=state,
            trace=None,
            started_at=datetime.now(timezone.utc),
            cycle_count=0,
            cancel_token=None,
        )

        data = ctx.serialize()

        assert data["attachments"] == [
            {
                "type": "attachment",
                "mime_type": "text/plain",
                "filename": "note.txt",
                "size_bytes": len(b"data"),
            }
        ]

    def test_deserialize(self) -> None:
        """Test RunContext deserialization."""
        from axis_core.context import RunContext

        ctx = self._create_sample_context()
        data = ctx.serialize()

        restored = RunContext.deserialize(data)

        assert restored.run_id == ctx.run_id
        assert restored.agent_id == ctx.agent_id
        assert restored.input.text == ctx.input.text


class TestRunContextSizeChecking:
    """Tests for RunContext size checking (AD-037)."""

    def test_size_constants(self) -> None:
        """Test that size constants are correctly defined."""
        from axis_core.context import MAX_CONTEXT_SIZE, WARN_CONTEXT_SIZE

        assert WARN_CONTEXT_SIZE == 50 * 1024 * 1024  # 50MB
        assert MAX_CONTEXT_SIZE == 100 * 1024 * 1024  # 100MB

    def test_check_size_small_context(self) -> None:
        """Test check_size with small context (under warning threshold)."""
        from axis_core.context import NormalizedInput, RunContext, RunState

        input_ = NormalizedInput(text="test", original="test")
        state = RunState()
        budget = Budget()

        ctx = RunContext(
            run_id="run-123",
            agent_id="agent-456",
            input=input_,
            context={},
            attachments=[],
            config=None,
            budget=budget,
            state=state,
            trace=None,
            started_at=datetime.now(timezone.utc),
            cycle_count=0,
            cancel_token=None,
        )

        size, should_warn, should_fail = ctx.check_size()

        assert size > 0  # Should have some size
        assert should_warn is False  # Under 50MB
        assert should_fail is False  # Under 100MB

    def test_check_size_returns_tuple(self) -> None:
        """Test that check_size returns (size, warn, fail) tuple."""
        from axis_core.context import NormalizedInput, RunContext, RunState

        input_ = NormalizedInput(text="test", original="test")
        state = RunState()
        budget = Budget()

        ctx = RunContext(
            run_id="run-123",
            agent_id="agent-456",
            input=input_,
            context={},
            attachments=[],
            config=None,
            budget=budget,
            state=state,
            trace=None,
            started_at=datetime.now(timezone.utc),
            cycle_count=0,
            cancel_token=None,
        )

        result = ctx.check_size()

        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], int)  # size
        assert isinstance(result[1], bool)  # should_warn
        assert isinstance(result[2], bool)  # should_fail


class TestSerializationRoundTrip:
    """Integration tests for full serialization round-trips."""

    def test_full_context_roundtrip(self) -> None:
        """Test complete RunContext serialization and deserialization."""
        from axis_core.context import (
            CycleState,
            EvalDecision,
            ExecutionResult,
            ModelCallRecord,
            NormalizedInput,
            Observation,
            RunContext,
            RunState,
        )

        # Build a complete context with state
        input_ = NormalizedInput(text="What is 2+2?", original="What is 2+2?")
        state = RunState()

        # Add a cycle
        obs = Observation(input=input_, goal="Calculate 2+2")
        plan = Plan(
            id="plan-1",
            goal="Use calculator",
            steps=(
                PlanStep(
                    id="step-1",
                    type=StepType.TOOL,
                    payload={"tool": "calculator", "args": {"a": 2, "b": 2}},
                ),
            ),
        )
        execution = ExecutionResult(results={"calculator": 4})
        evaluation = EvalDecision(done=True, reason="Got answer")
        started = datetime.now(timezone.utc)
        ended = datetime.now(timezone.utc)

        cycle = CycleState(
            cycle_number=0,
            observation=obs,
            plan=plan,
            execution=execution,
            evaluation=evaluation,
            started_at=started,
            ended_at=ended,
        )
        state.append_cycle(cycle)

        # Add a model call
        model_call = ModelCallRecord(
            model_id="claude-sonnet-4-20250514",
            call_id="call-1",
            input_tokens=50,
            output_tokens=20,
            cost_usd=0.0007,
            duration_ms=150.0,
            timestamp=1704067200.0,
        )
        state.append_model_call(model_call)

        # Add an error
        error = AxisError(message="minor warning", error_class=ErrorClass.MODEL)
        error_record = ErrorRecord(
            error=error,
            timestamp=datetime.now(timezone.utc),
            phase="plan",
            cycle=0,
            recovered=True,
        )
        state.append_error(error_record)

        state.output = 4
        state.output_raw = "The answer is 4"

        ctx = RunContext(
            run_id="run-test-123",
            agent_id="test-agent",
            input=input_,
            context={"user_id": "user-1"},
            attachments=[],
            config=None,
            budget=Budget(max_cycles=5),
            state=state,
            trace=None,
            started_at=started,
            cycle_count=1,
            cancel_token=None,
        )

        # Serialize and deserialize
        data = ctx.serialize()
        restored = RunContext.deserialize(data)

        # Verify restoration
        assert restored.run_id == "run-test-123"
        assert restored.agent_id == "test-agent"
        assert restored.input.text == "What is 2+2?"
        assert restored.cycle_count == 1
        assert restored.context == {"user_id": "user-1"}

        # Verify state restoration
        assert len(restored.state.cycles) == 1
        assert restored.state.cycles[0].cycle_number == 0
        assert restored.state.cycles[0].evaluation.done is True

        assert len(restored.state.model_calls) == 1
        assert restored.state.model_calls[0].model_id == "claude-sonnet-4-20250514"

        assert len(restored.state.errors) == 1
        assert restored.state.errors[0].recovered is True

        assert restored.state.output == 4
        assert restored.state.output_raw == "The answer is 4"

    def test_multiple_cycles_roundtrip(self) -> None:
        """Test serialization with multiple cycles."""
        from axis_core.context import (
            CycleState,
            EvalDecision,
            ExecutionResult,
            NormalizedInput,
            Observation,
            RunState,
        )

        state = RunState()

        # Add multiple cycles
        for i in range(3):
            input_ = NormalizedInput(text=f"cycle {i}", original=f"cycle {i}")
            obs = Observation(input=input_, goal=f"Goal {i}")
            plan = Plan(id=f"plan-{i}", goal=f"Plan goal {i}", steps=())
            execution = ExecutionResult(results={f"tool_{i}": f"result_{i}"})
            evaluation = EvalDecision(done=(i == 2), reason=f"Reason {i}")
            started = datetime.now(timezone.utc)
            ended = datetime.now(timezone.utc)

            cycle = CycleState(
                cycle_number=i,
                observation=obs,
                plan=plan,
                execution=execution,
                evaluation=evaluation,
                started_at=started,
                ended_at=ended,
            )
            state.append_cycle(cycle)

        data = state.to_dict()
        restored = RunState.from_dict(data)

        assert len(restored.cycles) == 3
        assert restored.cycles[0].cycle_number == 0
        assert restored.cycles[1].cycle_number == 1
        assert restored.cycles[2].cycle_number == 2
        assert restored.cycles[2].evaluation.done is True

    def test_error_serialization(self) -> None:
        """Test that errors are serialized as string representations."""
        from axis_core.context import RunState

        state = RunState()

        # Add an error with complex cause
        cause = ValueError("inner error")
        error = AxisError(
            message="outer error",
            error_class=ErrorClass.TOOL,
            cause=cause,
            details={"key": "value"},
        )
        record = ErrorRecord(
            error=error,
            timestamp=datetime.now(timezone.utc),
            phase="act",
            cycle=1,
            recovered=False,
        )
        state.append_error(record)

        data = state.to_dict()
        restored = RunState.from_dict(data)

        # Error should be restored (though cause may be string representation)
        assert len(restored.errors) == 1
        assert restored.errors[0].phase == "act"
        assert restored.errors[0].cycle == 1
        assert restored.errors[0].recovered is False


# =============================================================================
# Message Building Tests (First-Cycle Model Calling)
# =============================================================================


class TestMessageBuilding:
    """Tests for build_messages() context management."""

    def test_build_messages_first_cycle(self) -> None:
        """First cycle should return just user input."""
        from axis_core.context import NormalizedInput, RunContext, RunState

        state = RunState()
        ctx = RunContext(
            run_id="test-run",
            agent_id="test-agent",
            input=NormalizedInput(text="Hello world", original="Hello world"),
            context={},
            attachments=[],
            config=None,  # type: ignore[arg-type]
            budget=Budget(),
            state=state,
            trace=None,  # type: ignore[arg-type]
            started_at=datetime.now(timezone.utc),
            cycle_count=0,
            cancel_token=None,
        )

        messages = state.build_messages(ctx, strategy="smart", max_cycles=5)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello world"

    def test_build_messages_with_memory_context(self) -> None:
        """First message should include memory context if available."""
        from axis_core.context import (
            NormalizedInput,
            Observation,
            RunContext,
            RunState,
        )

        state = RunState()
        state.current_observation = Observation(
            input=NormalizedInput(text="test", original="test"),
            memory_context={
                "relevant_memories": [
                    {"key": "fact1", "value": "The sky is blue"},
                    {"key": "fact2", "value": "Water is wet"},
                ]
            },
            previous_cycles=(),
            timestamp=datetime.now(timezone.utc),
        )

        ctx = RunContext(
            run_id="test-run",
            agent_id="test-agent",
            input=NormalizedInput(text="What color is the sky?", original="What color is the sky?"),
            context={},
            attachments=[],
            config=None,  # type: ignore[arg-type]
            budget=Budget(),
            state=state,
            trace=None,  # type: ignore[arg-type]
            started_at=datetime.now(timezone.utc),
            cycle_count=0,
            cancel_token=None,
        )

        messages = state.build_messages(ctx, strategy="smart")

        assert len(messages) == 1
        assert "<relevant_context>" in messages[0]["content"]
        assert "The sky is blue" in messages[0]["content"]
        assert "What color is the sky?" in messages[0]["content"]

    def test_build_messages_smart_strategy(self) -> None:
        """Smart strategy should include last N cycles only."""
        from axis_core.context import (
            CycleState,
            EvalDecision,
            ExecutionResult,
            NormalizedInput,
            Observation,
            RunContext,
            RunState,
        )

        state = RunState()

        # Create 10 cycles
        for i in range(10):
            cycle = CycleState(
                cycle_number=i,
                observation=Observation(
                    input=NormalizedInput(text=f"cycle {i}", original=f"cycle {i}"),
                    response=f"response {i}",
                    tool_requests=None,
                    previous_cycles=(),
                    timestamp=datetime.now(timezone.utc),
                ),
                plan=Plan(
                    id=f"plan-{i}",
                    goal=f"goal {i}",
                    steps=(),
                ),
                execution=ExecutionResult(),
                evaluation=EvalDecision(done=False),
                started_at=datetime.now(timezone.utc),
                ended_at=datetime.now(timezone.utc),
            )
            state.append_cycle(cycle)

        ctx = RunContext(
            run_id="test-run",
            agent_id="test-agent",
            input=NormalizedInput(text="original", original="original"),
            context={},
            attachments=[],
            config=None,  # type: ignore[arg-type]
            budget=Budget(),
            state=state,
            trace=None,  # type: ignore[arg-type]
            started_at=datetime.now(timezone.utc),
            cycle_count=10,
            cancel_token=None,
        )

        messages = state.build_messages(ctx, strategy="smart", max_cycles=3)

        # Should have: 1 user message + (last 3 cycles * 1 assistant message each) = 4 messages
        assert len(messages) == 4
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "original"

        # Check that we have responses from cycles 7, 8, 9 (last 3)
        assert "response 7" in messages[1]["content"]
        assert "response 8" in messages[2]["content"]
        assert "response 9" in messages[3]["content"]

    def test_build_messages_with_tool_calls(self) -> None:
        """Messages should include tool calls and results."""
        from axis_core.context import (
            CycleState,
            EvalDecision,
            ExecutionResult,
            NormalizedInput,
            Observation,
            RunContext,
            RunState,
        )
        from axis_core.protocols.model import ToolCall

        state = RunState()

        # Cycle with tool call
        cycle = CycleState(
            cycle_number=0,
            observation=Observation(
                input=NormalizedInput(text="search", original="search"),
                response="Let me search for that",
                tool_requests=(
                    ToolCall(id="call_1", name="search", arguments={"q": "test"}),
                ),
                previous_cycles=(),
                timestamp=datetime.now(timezone.utc),
            ),
            plan=Plan(
                id="plan-1",
                goal="search",
                steps=(
                    PlanStep(
                        id="step_1",
                        type=StepType.TOOL,
                        payload={"tool": "search", "tool_call_id": "call_1", "args": {"q": "test"}},
                    ),
                ),
            ),
            execution=ExecutionResult(results={"step_1": "search results"}),
            evaluation=EvalDecision(done=True),
            started_at=datetime.now(timezone.utc),
            ended_at=datetime.now(timezone.utc),
        )
        state.append_cycle(cycle)

        ctx = RunContext(
            run_id="test-run",
            agent_id="test-agent",
            input=NormalizedInput(text="search", original="search"),
            context={},
            attachments=[],
            config=None,  # type: ignore[arg-type]
            budget=Budget(),
            state=state,
            trace=None,  # type: ignore[arg-type]
            started_at=datetime.now(timezone.utc),
            cycle_count=1,
            cancel_token=None,
        )

        messages = state.build_messages(ctx, strategy="smart")

        # Should have: user message, assistant with tool_calls, tool result
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert "tool_calls" in messages[1]
        assert messages[1]["tool_calls"][0]["id"] == "call_1"
        assert messages[2]["role"] == "tool"
        assert messages[2]["tool_call_id"] == "call_1"
        assert "search results" in messages[2]["content"]

    def test_build_messages_minimal_strategy(self) -> None:
        """Minimal strategy should only return first user message."""
        from axis_core.context import (
            CycleState,
            EvalDecision,
            ExecutionResult,
            NormalizedInput,
            Observation,
            RunContext,
            RunState,
        )

        state = RunState()

        # Add some cycles
        for i in range(5):
            cycle = CycleState(
                cycle_number=i,
                observation=Observation(
                    input=NormalizedInput(text=f"cycle {i}", original=f"cycle {i}"),
                    response=f"response {i}",
                    tool_requests=None,
                    previous_cycles=(),
                    timestamp=datetime.now(timezone.utc),
                ),
                plan=Plan(id=f"plan-{i}", goal=f"goal {i}", steps=()),
                execution=ExecutionResult(),
                evaluation=EvalDecision(done=False),
                started_at=datetime.now(timezone.utc),
                ended_at=datetime.now(timezone.utc),
            )
            state.append_cycle(cycle)

        ctx = RunContext(
            run_id="test-run",
            agent_id="test-agent",
            input=NormalizedInput(text="original", original="original"),
            context={},
            attachments=[],
            config=None,  # type: ignore[arg-type]
            budget=Budget(),
            state=state,
            trace=None,  # type: ignore[arg-type]
            started_at=datetime.now(timezone.utc),
            cycle_count=5,
            cancel_token=None,
        )

        messages = state.build_messages(ctx, strategy="minimal")

        # Should only have original user message
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "original"

    def test_build_messages_full_strategy(self) -> None:
        """Full strategy should include all cycles."""
        from axis_core.context import (
            CycleState,
            EvalDecision,
            ExecutionResult,
            NormalizedInput,
            Observation,
            RunContext,
            RunState,
        )

        state = RunState()

        # Add 3 cycles
        for i in range(3):
            cycle = CycleState(
                cycle_number=i,
                observation=Observation(
                    input=NormalizedInput(text=f"cycle {i}", original=f"cycle {i}"),
                    response=f"response {i}",
                    tool_requests=None,
                    previous_cycles=(),
                    timestamp=datetime.now(timezone.utc),
                ),
                plan=Plan(id=f"plan-{i}", goal=f"goal {i}", steps=()),
                execution=ExecutionResult(),
                evaluation=EvalDecision(done=False),
                started_at=datetime.now(timezone.utc),
                ended_at=datetime.now(timezone.utc),
            )
            state.append_cycle(cycle)

        ctx = RunContext(
            run_id="test-run",
            agent_id="test-agent",
            input=NormalizedInput(text="original", original="original"),
            context={},
            attachments=[],
            config=None,  # type: ignore[arg-type]
            budget=Budget(),
            state=state,
            trace=None,  # type: ignore[arg-type]
            started_at=datetime.now(timezone.utc),
            cycle_count=3,
            cancel_token=None,
        )

        messages = state.build_messages(ctx, strategy="full")

        # Should have: 1 user + 3 assistant messages = 4
        assert len(messages) == 4
        assert messages[0]["role"] == "user"
        assert "response 0" in messages[1]["content"]
        assert "response 1" in messages[2]["content"]
        assert "response 2" in messages[3]["content"]
