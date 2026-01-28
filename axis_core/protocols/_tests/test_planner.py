"""Tests for planner protocol and dataclasses."""

import pytest

from axis_core.config import RetryPolicy
from axis_core.protocols.planner import Plan, Planner, PlanStep, StepType


class TestStepType:
    """Tests for StepType enum."""

    def test_enum_values(self):
        """Test that enum values match their lowercase names."""
        assert StepType.MODEL.value == "model"
        assert StepType.TOOL.value == "tool"
        assert StepType.TRANSFORM.value == "transform"
        assert StepType.TERMINAL.value == "terminal"

    def test_enum_count(self):
        """Test that we have exactly 4 step types."""
        assert len(StepType) == 4

    def test_enum_membership(self):
        """Test enum membership checks."""
        assert StepType.MODEL in StepType
        assert "invalid" not in [s.value for s in StepType]


class TestPlanStep:
    """Tests for PlanStep dataclass."""

    def test_minimal(self):
        """Test PlanStep with only required fields."""
        step = PlanStep(id="step1", type=StepType.MODEL)
        assert step.id == "step1"
        assert step.type == StepType.MODEL
        assert step.payload == {}
        assert step.dependencies is None
        assert step.retry_policy is None

    def test_with_payload(self):
        """Test PlanStep with payload."""
        step = PlanStep(
            id="step1",
            type=StepType.TOOL,
            payload={"tool": "search", "args": {"query": "test"}},
        )
        assert step.payload == {"tool": "search", "args": {"query": "test"}}

    def test_with_dependencies(self):
        """Test PlanStep with dependencies."""
        step = PlanStep(
            id="step3",
            type=StepType.TOOL,
            dependencies=("step1", "step2"),
        )
        assert step.dependencies == ("step1", "step2")

    def test_with_retry_policy(self):
        """Test PlanStep with custom retry policy."""
        retry = RetryPolicy(max_attempts=5, backoff="linear")
        step = PlanStep(id="step1", type=StepType.TOOL, retry_policy=retry)
        assert step.retry_policy == retry
        assert step.retry_policy.max_attempts == 5

    def test_all_step_types(self):
        """Test creating steps of all types."""
        model_step = PlanStep(id="s1", type=StepType.MODEL)
        tool_step = PlanStep(id="s2", type=StepType.TOOL)
        transform_step = PlanStep(id="s3", type=StepType.TRANSFORM)
        terminal_step = PlanStep(id="s4", type=StepType.TERMINAL)

        assert model_step.type == StepType.MODEL
        assert tool_step.type == StepType.TOOL
        assert transform_step.type == StepType.TRANSFORM
        assert terminal_step.type == StepType.TERMINAL

    def test_immutability(self):
        """Test that PlanStep is immutable."""
        step = PlanStep(id="step1", type=StepType.MODEL)
        with pytest.raises(AttributeError):
            step.id = "step2"  # type: ignore


class TestPlan:
    """Tests for Plan dataclass."""

    def test_minimal(self):
        """Test Plan with only required fields."""
        steps = (PlanStep(id="step1", type=StepType.MODEL),)
        plan = Plan(id="plan1", goal="Test goal", steps=steps)
        assert plan.id == "plan1"
        assert plan.goal == "Test goal"
        assert plan.steps == steps
        assert plan.reasoning is None
        assert plan.confidence is None
        assert plan.metadata == {}

    def test_with_reasoning(self):
        """Test Plan with reasoning."""
        steps = (PlanStep(id="step1", type=StepType.MODEL),)
        plan = Plan(
            id="plan1",
            goal="Test goal",
            steps=steps,
            reasoning="I chose this approach because...",
        )
        assert plan.reasoning == "I chose this approach because..."

    def test_with_confidence(self):
        """Test Plan with confidence score."""
        steps = (PlanStep(id="step1", type=StepType.MODEL),)
        plan = Plan(id="plan1", goal="Test goal", steps=steps, confidence=0.95)
        assert plan.confidence == 0.95

    def test_with_metadata(self):
        """Test Plan with metadata."""
        steps = (PlanStep(id="step1", type=StepType.MODEL),)
        plan = Plan(
            id="plan1",
            goal="Test goal",
            steps=steps,
            metadata={"planner": "react", "version": "1.0"},
        )
        assert plan.metadata == {"planner": "react", "version": "1.0"}

    def test_multiple_steps(self):
        """Test Plan with multiple steps."""
        steps = (
            PlanStep(id="step1", type=StepType.MODEL),
            PlanStep(id="step2", type=StepType.TOOL, dependencies=("step1",)),
            PlanStep(id="step3", type=StepType.TERMINAL, dependencies=("step2",)),
        )
        plan = Plan(id="plan1", goal="Multi-step goal", steps=steps)
        assert len(plan.steps) == 3
        assert plan.steps[1].dependencies == ("step1",)

    def test_immutability(self):
        """Test that Plan is immutable."""
        steps = (PlanStep(id="step1", type=StepType.MODEL),)
        plan = Plan(id="plan1", goal="Test goal", steps=steps)
        with pytest.raises(AttributeError):
            plan.id = "plan2"  # type: ignore


class TestPlanner:
    """Tests for Planner protocol."""

    @pytest.mark.asyncio
    async def test_protocol_implementation(self):
        """Test that a class implementing Planner conforms to the protocol."""

        class FakePlanner:
            async def plan(self, observation, ctx):
                steps = (
                    PlanStep(id="step1", type=StepType.MODEL),
                    PlanStep(id="step2", type=StepType.TOOL),
                    PlanStep(id="step3", type=StepType.TERMINAL),
                )
                return Plan(
                    id="plan1",
                    goal="Test goal",
                    steps=steps,
                    reasoning="Simple sequential plan",
                    confidence=0.9,
                )

        planner = FakePlanner()
        assert isinstance(planner, Planner)

        # Test plan generation
        plan = await planner.plan(None, None)
        assert plan.id == "plan1"
        assert plan.goal == "Test goal"
        assert len(plan.steps) == 3
        assert plan.reasoning == "Simple sequential plan"
        assert plan.confidence == 0.9

    def test_protocol_missing_methods(self):
        """Test that a class missing methods doesn't conform to protocol."""

        class IncompletePlanner:
            pass

        planner = IncompletePlanner()
        assert not isinstance(planner, Planner)
