"""Tests for SequentialPlanner adapter."""

from unittest.mock import Mock

import pytest

from axis_core.adapters.planners.sequential import SequentialPlanner
from axis_core.context import NormalizedInput, Observation
from axis_core.protocols.model import ToolCall
from axis_core.protocols.planner import Plan, StepType


@pytest.mark.unit
class TestSequentialPlanner:
    """Test suite for SequentialPlanner adapter."""

    @pytest.mark.asyncio
    async def test_plan_with_tool_requests(self) -> None:
        """Test planning when observation contains tool requests."""
        planner = SequentialPlanner()

        # Create observation with tool requests
        tool_requests = (
            ToolCall(id="call_1", name="search", arguments={"query": "test"}),
            ToolCall(id="call_2", name="calculate", arguments={"expr": "2+2"}),
        )

        observation = Observation(
            input=NormalizedInput(text="Test input", original="Test input"),
            tool_requests=tool_requests,
            response="I'll search and calculate.",
            goal="Process user request",
        )

        ctx = Mock()  # Mock RunContext
        ctx.run_id = "run_123"

        plan = await planner.plan(observation, ctx)

        # Verify plan structure
        assert isinstance(plan, Plan)
        assert plan.goal == "Process user request"
        assert len(plan.steps) == 2  # 2 tool steps (no terminal until tools complete)

        # Verify tool steps
        assert plan.steps[0].type == StepType.TOOL
        assert plan.steps[0].payload["tool"] == "search"
        assert plan.steps[0].payload["tool_call_id"] == "call_1"
        assert plan.steps[0].payload["args"] == {"query": "test"}

        assert plan.steps[1].type == StepType.TOOL
        assert plan.steps[1].payload["tool"] == "calculate"
        assert plan.steps[1].payload["tool_call_id"] == "call_2"
        assert plan.steps[1].payload["args"] == {"expr": "2+2"}

    @pytest.mark.asyncio
    async def test_plan_without_tool_requests_first_cycle(self) -> None:
        """Test planning on first cycle (no response yet)."""
        planner = SequentialPlanner()

        observation = Observation(
            input=NormalizedInput(text="Hello", original="Hello"),
            tool_requests=None,
            response=None,  # First cycle - no response yet
            goal="Respond to greeting",
        )

        ctx = Mock()
        ctx.run_id = "run_456"

        plan = await planner.plan(observation, ctx)

        # Should have MODEL step only (need to call model first)
        assert isinstance(plan, Plan)
        assert len(plan.steps) == 1
        assert plan.steps[0].type == StepType.MODEL

    @pytest.mark.asyncio
    async def test_plan_with_final_response(self) -> None:
        """Test planning when model provided final response (no more tools)."""
        planner = SequentialPlanner()

        observation = Observation(
            input=NormalizedInput(text="Hello", original="Hello"),
            tool_requests=None,
            response="Hello! How can I help you today?",  # Final response from previous cycle
            goal="Respond to greeting",
        )

        ctx = Mock()
        ctx.run_id = "run_456"

        plan = await planner.plan(observation, ctx)

        # Should have just TERMINAL (model already responded in previous cycle)
        assert isinstance(plan, Plan)
        assert len(plan.steps) == 1
        assert plan.steps[0].type == StepType.TERMINAL
        assert plan.steps[0].payload["output"] == "Hello! How can I help you today?"

    @pytest.mark.asyncio
    async def test_plan_with_empty_tool_requests(self) -> None:
        """Test planning when tool_requests is empty tuple."""
        planner = SequentialPlanner()

        observation = Observation(
            input=NormalizedInput(text="Test", original="Test"),
            tool_requests=(),
            response="Done",
            goal="Complete task",
        )

        ctx = Mock()
        ctx.run_id = "run_789"

        plan = await planner.plan(observation, ctx)

        # Empty tuple should behave like no tools - just TERMINAL (response already present)
        assert len(plan.steps) == 1
        assert plan.steps[0].type == StepType.TERMINAL

    @pytest.mark.asyncio
    async def test_plan_ids_are_unique(self) -> None:
        """Test that plan and step IDs are unique."""
        planner = SequentialPlanner()

        tool_requests = (
            ToolCall(id="call_1", name="tool_a", arguments={}),
            ToolCall(id="call_2", name="tool_b", arguments={}),
        )

        observation = Observation(
            input=NormalizedInput(text="Test", original="Test"),
            tool_requests=tool_requests,
            goal="Test",
        )

        ctx = Mock()
        ctx.run_id = "run_test"

        plan = await planner.plan(observation, ctx)

        # Check plan ID exists
        assert plan.id
        assert isinstance(plan.id, str)

        # Check all step IDs are unique
        step_ids = [step.id for step in plan.steps]
        assert len(step_ids) == len(set(step_ids))

    @pytest.mark.asyncio
    async def test_plan_metadata(self) -> None:
        """Test that plan includes metadata."""
        planner = SequentialPlanner()

        observation = Observation(
            input=NormalizedInput(text="Test", original="Test"),
            tool_requests=None,
            response="Done",
            goal="Test goal",
        )

        ctx = Mock()
        ctx.run_id = "run_meta"

        plan = await planner.plan(observation, ctx)

        # Check metadata
        assert plan.metadata is not None
        assert isinstance(plan.metadata, dict)
        assert plan.metadata.get("planner") == "sequential"

    @pytest.mark.asyncio
    async def test_plan_reasoning(self) -> None:
        """Test that plan includes reasoning explanation."""
        planner = SequentialPlanner()

        observation = Observation(
            input=NormalizedInput(text="Test", original="Test"),
            tool_requests=(
                ToolCall(id="call_1", name="search", arguments={"q": "test"}),
            ),
            goal="Search for test",
        )

        ctx = Mock()
        ctx.run_id = "run_reason"

        plan = await planner.plan(observation, ctx)

        # Sequential planner should include reasoning
        assert plan.reasoning is not None
        assert isinstance(plan.reasoning, str)
        assert len(plan.reasoning) > 0

    @pytest.mark.asyncio
    async def test_single_tool_request(self) -> None:
        """Test planning with single tool request."""
        planner = SequentialPlanner()

        observation = Observation(
            input=NormalizedInput(text="Test", original="Test"),
            tool_requests=(
                ToolCall(id="call_1", name="single_tool", arguments={"key": "value"}),
            ),
            goal="Run single tool",
        )

        ctx = Mock()
        ctx.run_id = "run_single"

        plan = await planner.plan(observation, ctx)

        # 1 tool step only (no terminal until tools complete)
        assert len(plan.steps) == 1
        assert plan.steps[0].type == StepType.TOOL
        assert plan.steps[0].payload["tool"] == "single_tool"

    @pytest.mark.asyncio
    async def test_step_dependencies(self) -> None:
        """Test that tool steps have no dependencies (sequential execution)."""
        planner = SequentialPlanner()

        observation = Observation(
            input=NormalizedInput(text="Test", original="Test"),
            tool_requests=(
                ToolCall(id="call_1", name="tool_a", arguments={}),
                ToolCall(id="call_2", name="tool_b", arguments={}),
            ),
            goal="Test dependencies",
        )

        ctx = Mock()
        ctx.run_id = "run_deps"

        plan = await planner.plan(observation, ctx)

        # In sequential planner, tool steps should have no explicit dependencies
        # (they execute in order implicitly)
        for step in plan.steps[:-1]:  # All but terminal
            assert step.type == StepType.TOOL
            assert step.dependencies is None or step.dependencies == ()

    @pytest.mark.asyncio
    async def test_plan_with_complex_arguments(self) -> None:
        """Test planning with complex nested arguments."""
        planner = SequentialPlanner()

        complex_args = {
            "nested": {"key": "value"},
            "list": [1, 2, 3],
            "string": "test",
            "number": 42,
        }

        observation = Observation(
            input=NormalizedInput(text="Test", original="Test"),
            tool_requests=(ToolCall(id="call_1", name="complex", arguments=complex_args),),
            goal="Test complex args",
        )

        ctx = Mock()
        ctx.run_id = "run_complex"

        plan = await planner.plan(observation, ctx)

        # Arguments should be preserved exactly
        assert plan.steps[0].payload["args"] == complex_args

    @pytest.mark.asyncio
    async def test_confidence_score(self) -> None:
        """Test that sequential planner has high confidence (simple strategy)."""
        planner = SequentialPlanner()

        observation = Observation(
            input=NormalizedInput(text="Test", original="Test"),
            tool_requests=None,
            response="Done",
            goal="Test confidence",
        )

        ctx = Mock()
        ctx.run_id = "run_confidence"

        plan = await planner.plan(observation, ctx)

        # Sequential planner should be confident (it's deterministic)
        assert plan.confidence is not None
        assert plan.confidence >= 0.9  # High confidence
        assert plan.confidence <= 1.0
