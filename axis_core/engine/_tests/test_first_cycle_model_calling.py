"""Tests for first-cycle model calling implementation.

Tests that planners create MODEL steps when needed and that model responses
are properly stored and used in subsequent cycles.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from axis_core.budget import Budget
from axis_core.context import (
    CycleState,
    EvalDecision,
    ExecutionResult,
    NormalizedInput,
    Observation,
    RunContext,
    RunState,
)
from axis_core.engine.lifecycle import LifecycleEngine
from axis_core.protocols.model import ModelResponse, ToolCall, UsageStats
from axis_core.protocols.planner import Plan, PlanStep, StepType


# =============================================================================
# Mock adapters
# =============================================================================


class MockModel:
    """Mock model that returns configurable responses."""

    def __init__(self, responses: list[ModelResponse] | None = None):
        self.responses = responses or []
        self._call_count = 0
        self.calls: list[dict[str, Any]] = []

    @property
    def model_id(self) -> str:
        return "mock-model"

    async def complete(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        self.calls.append({"messages": messages, "system": system, "tools": tools})

        if self._call_count < len(self.responses):
            response = self.responses[self._call_count]
            self._call_count += 1
            return response

        # Default response
        return ModelResponse(
            content="default response",
            tool_calls=None,
            usage=UsageStats(input_tokens=10, output_tokens=5, total_tokens=15),
            cost_usd=0.001,
        )


class MockPlanner:
    """Mock planner for basic testing."""

    def __init__(self, plans: list[Plan] | None = None):
        self.plans = plans or []
        self._plan_index = 0

    async def plan(self, observation: Observation, ctx: RunContext) -> Plan:
        if self._plan_index < len(self.plans):
            plan = self.plans[self._plan_index]
            self._plan_index += 1
            return plan

        # Default terminal plan
        return Plan(
            id="default-plan",
            goal="default",
            steps=(
                PlanStep(
                    id="terminal",
                    type=StepType.TERMINAL,
                    payload={"output": "done"},
                ),
            ),
        )


# =============================================================================
# Lifecycle tests for model response storage
# =============================================================================


class TestModelResponseStorage:
    """Test that _execute_model_step stores last_model_response."""

    @pytest.mark.asyncio
    async def test_execute_model_step_stores_response(self) -> None:
        """_execute_model_step should store full response in context."""
        from axis_core.engine.lifecycle import LifecycleEngine

        # Create model with specific response
        model_response = ModelResponse(
            content="test response",
            tool_calls=(
                ToolCall(id="call_1", name="search", arguments={"q": "test"}),
            ),
            usage=UsageStats(input_tokens=10, output_tokens=20, total_tokens=30),
            cost_usd=0.01,
        )
        mock_model = MockModel(responses=[model_response])

        engine = LifecycleEngine(
            model=mock_model,
            planner=MockPlanner(),
        )

        ctx = await engine._initialize(
            input_text="test input",
            agent_id="test-agent",
            budget=Budget(),
        )

        # Execute a MODEL step
        step = PlanStep(
            id="model-step",
            type=StepType.MODEL,
            payload={},
        )

        result = await engine._execute_model_step(ctx, step)

        # Check that response was stored
        assert ctx.state.last_model_response is not None
        assert ctx.state.last_model_response.content == "test response"
        assert ctx.state.last_model_response.tool_calls is not None
        assert len(ctx.state.last_model_response.tool_calls) == 1
        assert ctx.state.last_model_response.tool_calls[0].id == "call_1"

    @pytest.mark.asyncio
    async def test_execute_model_step_builds_messages_from_context(self) -> None:
        """_execute_model_step should build messages if not provided."""
        mock_model = MockModel()
        engine = LifecycleEngine(model=mock_model, planner=MockPlanner())

        ctx = await engine._initialize(
            input_text="Hello",
            agent_id="test-agent",
            budget=Budget(),
        )

        # Execute MODEL step without explicit messages
        step = PlanStep(id="model-step", type=StepType.MODEL, payload={})

        await engine._execute_model_step(ctx, step)

        # Check that model was called with built messages
        assert len(mock_model.calls) == 1
        messages = mock_model.calls[0]["messages"]
        assert len(messages) >= 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"


# =============================================================================
# Observe phase tests
# =============================================================================


class TestObserveWithLastResponse:
    """Test that _observe uses last_model_response."""

    @pytest.mark.asyncio
    async def test_observe_populates_from_last_response(self) -> None:
        """_observe should populate observation from last_model_response."""
        from axis_core.engine.lifecycle import LifecycleEngine

        mock_model = MockModel()
        engine = LifecycleEngine(model=mock_model, planner=MockPlanner())

        ctx = await engine._initialize(
            input_text="test",
            agent_id="test-agent",
            budget=Budget(),
        )

        # Set last_model_response in state
        ctx.state.last_model_response = ModelResponse(
            content="previous response",
            tool_calls=(
                ToolCall(id="call_1", name="search", arguments={"q": "data"}),
            ),
            usage=UsageStats(input_tokens=10, output_tokens=20, total_tokens=30),
            cost_usd=0.01,
        )

        # Call observe
        observation = await engine._observe(ctx)

        # Check that tool_requests and response are populated
        assert observation.response == "previous response"
        assert observation.tool_requests is not None
        assert len(observation.tool_requests) == 1
        assert observation.tool_requests[0].id == "call_1"

    @pytest.mark.asyncio
    async def test_observe_no_last_response(self) -> None:
        """_observe should handle case with no last_model_response."""
        from axis_core.engine.lifecycle import LifecycleEngine

        mock_model = MockModel()
        engine = LifecycleEngine(model=mock_model, planner=MockPlanner())

        ctx = await engine._initialize(
            input_text="test",
            agent_id="test-agent",
            budget=Budget(),
        )

        # Don't set last_model_response (first cycle)
        observation = await engine._observe(ctx)

        # Should have None for response and tool_requests
        assert observation.response is None
        assert observation.tool_requests is None


# =============================================================================
# SequentialPlanner tests
# =============================================================================


class TestSequentialPlannerModelSteps:
    """Test that SequentialPlanner creates MODEL steps on first cycle."""

    @pytest.mark.asyncio
    async def test_first_cycle_creates_model_step(self) -> None:
        """SequentialPlanner should create MODEL step when no tool_requests."""
        from axis_core.adapters.planners.sequential import SequentialPlanner

        planner = SequentialPlanner()

        # First cycle observation (no tool_requests)
        observation = Observation(
            input=NormalizedInput(text="Hello", original="Hello"),
            memory_context={},
            previous_cycles=(),
            tool_requests=None,
            response=None,
            timestamp=datetime.utcnow(),
        )

        ctx = RunContext(
            run_id="test-run",
            agent_id="test-agent",
            input=observation.input,
            context={},
            attachments=[],
            config=None,  # type: ignore[arg-type]
            budget=Budget(),
            state=RunState(),
            trace=None,  # type: ignore[arg-type]
            started_at=datetime.utcnow(),
            cycle_count=0,
            cancel_token=None,
        )

        plan = await planner.plan(observation, ctx)

        # Should have MODEL step only (no response yet)
        assert len(plan.steps) == 1
        assert plan.steps[0].type == StepType.MODEL

    @pytest.mark.asyncio
    async def test_second_cycle_creates_tool_steps(self) -> None:
        """SequentialPlanner should create TOOL steps when tool_requests present."""
        from axis_core.adapters.planners.sequential import SequentialPlanner

        planner = SequentialPlanner()

        # Second cycle observation (with tool_requests)
        observation = Observation(
            input=NormalizedInput(text="Hello", original="Hello"),
            memory_context={},
            previous_cycles=(),
            tool_requests=(
                ToolCall(id="call_1", name="search", arguments={"q": "test"}),
                ToolCall(id="call_2", name="summarize", arguments={}),
            ),
            response="Let me search and summarize",
            timestamp=datetime.utcnow(),
        )

        ctx = RunContext(
            run_id="test-run",
            agent_id="test-agent",
            input=observation.input,
            context={},
            attachments=[],
            config=None,  # type: ignore[arg-type]
            budget=Budget(),
            state=RunState(),
            trace=None,  # type: ignore[arg-type]
            started_at=datetime.utcnow(),
            cycle_count=1,
            cancel_token=None,
        )

        plan = await planner.plan(observation, ctx)

        # Should have 2 TOOL steps only (no TERMINAL until tools complete)
        assert len(plan.steps) == 2
        assert plan.steps[0].type == StepType.TOOL
        assert plan.steps[0].payload["tool"] == "search"
        assert plan.steps[0].payload["tool_call_id"] == "call_1"
        assert plan.steps[1].type == StepType.TOOL
        assert plan.steps[1].payload["tool"] == "summarize"
        assert plan.steps[1].payload["tool_call_id"] == "call_2"
