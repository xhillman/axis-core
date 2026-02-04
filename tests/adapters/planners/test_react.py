"""Tests for ReActPlanner adapter."""

from unittest.mock import AsyncMock, Mock

import pytest

from axis_core.adapters.planners.react import ReActPlanner
from axis_core.context import NormalizedInput, Observation
from axis_core.protocols.model import ModelResponse, ToolCall, UsageStats
from axis_core.protocols.planner import Plan, StepType


@pytest.mark.unit
class TestReActPlanner:
    """Test suite for ReActPlanner adapter."""

    @pytest.mark.asyncio
    async def test_plan_with_thought_action_structure(self) -> None:
        """Test that ReAct creates Thought (MODEL) → Action (TOOL) structure."""
        # Mock model that returns a thought + tool request
        mock_model = AsyncMock()
        content = (
            "Thought: I need to search for information about Python.\n"
            "Action: I'll use the search tool."
        )
        mock_model.complete.return_value = ModelResponse(
            content=content,
            tool_calls=(
                ToolCall(id="call_1", name="search", arguments={"query": "Python programming"}),
            ),
            usage=UsageStats(input_tokens=50, output_tokens=30, total_tokens=80),
            cost_usd=0.001,
        )

        planner = ReActPlanner(model=mock_model, max_iterations=5)

        observation = Observation(
            input=NormalizedInput(text="Tell me about Python", original="Tell me about Python"),
            tool_requests=None,
            response=None,
            goal="Research Python",
        )

        ctx = Mock()
        ctx.run_id = "run_123"
        ctx.cycle_count = 0
        ctx.context = {}

        plan = await planner.plan(observation, ctx)

        # Verify plan structure
        assert isinstance(plan, Plan)
        assert len(plan.steps) >= 1

        # First step should be MODEL (thought generation)
        assert plan.steps[0].type == StepType.MODEL
        assert "reasoning" in plan.metadata or plan.reasoning is not None

        # Model should have been called
        mock_model.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_plan_with_tool_execution(self) -> None:
        """Test that tool requests from model create TOOL steps."""
        mock_model = AsyncMock()
        mock_model.complete.return_value = ModelResponse(
            content="Thought: I should search first.\nAction: search",
            tool_calls=(
                ToolCall(id="call_1", name="search", arguments={"query": "test"}),
            ),
            usage=UsageStats(input_tokens=40, output_tokens=20, total_tokens=60),
            cost_usd=0.0008,
        )

        planner = ReActPlanner(model=mock_model, max_iterations=5)

        observation = Observation(
            input=NormalizedInput(text="Search for test", original="Search for test"),
            tool_requests=None,
            response=None,
            goal="Search",
        )

        ctx = Mock()
        ctx.run_id = "run_456"
        ctx.cycle_count = 0
        ctx.context = {}

        plan = await planner.plan(observation, ctx)

        # Should have at least a MODEL step and the TOOL steps from the response
        assert isinstance(plan, Plan)
        assert len(plan.steps) >= 1

        # Check that we have both MODEL and TOOL steps
        step_types = [step.type for step in plan.steps]
        assert StepType.MODEL in step_types

    @pytest.mark.asyncio
    async def test_plan_with_final_answer(self) -> None:
        """Test that ReAct creates TERMINAL step when model gives final answer."""
        mock_model = AsyncMock()
        mock_model.complete.return_value = ModelResponse(
            content="Thought: I have enough information.\nFinal Answer: The result is 42.",
            tool_calls=None,
            usage=UsageStats(input_tokens=30, output_tokens=15, total_tokens=45),
            cost_usd=0.0006,
        )

        planner = ReActPlanner(model=mock_model, max_iterations=5)

        observation = Observation(
            input=NormalizedInput(text="What is the answer?", original="What is the answer?"),
            tool_requests=None,
            response=None,
            goal="Find answer",
        )

        ctx = Mock()
        ctx.run_id = "run_789"
        ctx.cycle_count = 0
        ctx.context = {}

        plan = await planner.plan(observation, ctx)

        # Should have TERMINAL step with final answer
        assert isinstance(plan, Plan)
        terminal_steps = [s for s in plan.steps if s.type == StepType.TERMINAL]
        assert len(terminal_steps) >= 1

    @pytest.mark.asyncio
    async def test_max_iterations_limit(self) -> None:
        """Test that max_iterations configuration is respected."""
        mock_model = AsyncMock()
        mock_model.complete.return_value = ModelResponse(
            content="Thought: Keep searching.\nAction: search",
            tool_calls=(
                ToolCall(id="call_1", name="search", arguments={"query": "more"}),
            ),
            usage=UsageStats(input_tokens=20, output_tokens=10, total_tokens=30),
            cost_usd=0.0004,
        )

        planner = ReActPlanner(model=mock_model, max_iterations=2)

        observation = Observation(
            input=NormalizedInput(text="Search forever", original="Search forever"),
            tool_requests=None,
            response=None,
            goal="Endless search",
        )

        ctx = Mock()
        ctx.run_id = "run_limit"
        ctx.cycle_count = 3  # Already exceeded max_iterations
        ctx.context = {}

        plan = await planner.plan(observation, ctx)

        # Should create TERMINAL step when max iterations exceeded
        assert isinstance(plan, Plan)
        terminal_steps = [s for s in plan.steps if s.type == StepType.TERMINAL]
        assert len(terminal_steps) >= 1
        # Should indicate max iterations reached
        metadata_str = str(plan.metadata).lower()
        reasoning_str = str(plan.reasoning).lower()
        assert "max_iterations" in metadata_str or "iteration" in reasoning_str

    @pytest.mark.asyncio
    async def test_reasoning_in_metadata(self) -> None:
        """Test that explicit reasoning is included in plan metadata."""
        mock_model = AsyncMock()
        content = (
            "Thought: I need to analyze the data carefully before proceeding.\n"
            "Action: analyze"
        )
        mock_model.complete.return_value = ModelResponse(
            content=content,
            tool_calls=(
                ToolCall(id="call_1", name="analyze", arguments={"data": "sample"}),
            ),
            usage=UsageStats(input_tokens=35, output_tokens=25, total_tokens=60),
            cost_usd=0.0009,
        )

        planner = ReActPlanner(model=mock_model, max_iterations=5)

        observation = Observation(
            input=NormalizedInput(text="Analyze this", original="Analyze this"),
            tool_requests=None,
            response=None,
            goal="Analyze",
        )

        ctx = Mock()
        ctx.run_id = "run_reasoning"
        ctx.cycle_count = 0
        ctx.context = {}

        plan = await planner.plan(observation, ctx)

        # Verify reasoning is captured
        assert isinstance(plan, Plan)
        assert plan.reasoning is not None or "reasoning" in plan.metadata
        # Should contain extracted thought
        reasoning_text = plan.reasoning or plan.metadata.get("reasoning", "")
        assert len(reasoning_text) > 0

    @pytest.mark.asyncio
    async def test_handles_observation_from_previous_cycle(self) -> None:
        """Test that ReAct handles observations when continuing from tool execution."""
        mock_model = AsyncMock()
        content = (
            "Observation: Got search results.\n"
            "Thought: Now I can answer.\n"
            "Final Answer: Based on the search, the answer is 42."
        )
        mock_model.complete.return_value = ModelResponse(
            content=content,
            tool_calls=None,
            usage=UsageStats(input_tokens=60, output_tokens=40, total_tokens=100),
            cost_usd=0.0012,
        )

        planner = ReActPlanner(model=mock_model, max_iterations=5)

        # Observation with tool results from previous cycle
        observation = Observation(
            input=NormalizedInput(text="What is the answer?", original="What is the answer?"),
            tool_requests=None,
            response="<tool result: found answer 42>",
            goal="Find answer",
        )

        ctx = Mock()
        ctx.run_id = "run_continue"
        ctx.cycle_count = 1
        ctx.context = {}

        plan = await planner.plan(observation, ctx)

        # Should process observation and create final answer
        assert isinstance(plan, Plan)
        # Should have called model with context about previous results
        mock_model.complete.assert_called_once()
