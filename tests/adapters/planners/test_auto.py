"""Tests for AutoPlanner adapter."""

import json
from unittest.mock import AsyncMock, Mock

import pytest

from axis_core.adapters.planners.auto import AutoPlanner
from axis_core.context import NormalizedInput, Observation
from axis_core.protocols.model import ModelResponse, ToolCall, UsageStats
from axis_core.protocols.planner import Plan, StepType


def _make_observation(
    *,
    tool_requests: tuple[ToolCall, ...] | None = None,
    response: str | None = None,
    goal: str = "Answer user question",
) -> Observation:
    """Helper to create test observations."""
    return Observation(
        input=NormalizedInput(text="What is 2+2?", original="What is 2+2?"),
        tool_requests=tool_requests,
        response=response,
        goal=goal,
    )


def _make_ctx(*, tools: dict[str, str] | None = None) -> Mock:
    """Helper to create a mock RunContext."""
    ctx = Mock()
    ctx.run_id = "run_test_123"
    ctx.cycle_count = 0
    ctx.context = {
        "__tools__": tools or {
            "search": "Search the web for information",
            "calculate": "Perform mathematical calculations",
        }
    }
    return ctx


def _make_model_response(content: str) -> ModelResponse:
    """Helper to create a model response."""
    return ModelResponse(
        content=content,
        tool_calls=None,
        usage=UsageStats(input_tokens=100, output_tokens=50, total_tokens=150),
        cost_usd=0.001,
    )


def _valid_plan_json(
    *,
    steps: list[dict[str, object]] | None = None,
    reasoning: str = "Need to search first, then calculate",
    confidence: float = 0.85,
) -> str:
    """Helper to produce valid plan JSON."""
    if steps is None:
        steps = [
            {
                "type": "tool",
                "tool": "search",
                "args": {"query": "2+2"},
                "reason": "Search for the answer",
            },
            {
                "type": "tool",
                "tool": "calculate",
                "args": {"expr": "2+2"},
                "reason": "Calculate the result",
            },
        ]
    plan = {
        "reasoning": reasoning,
        "confidence": confidence,
        "steps": steps,
    }
    return json.dumps(plan)


@pytest.mark.unit
class TestAutoPlanner:
    """Test suite for AutoPlanner adapter."""

    @pytest.mark.asyncio
    async def test_basic_plan_generation(self) -> None:
        """Test that AutoPlanner generates a plan from model response."""
        mock_model = AsyncMock()
        mock_model.complete = AsyncMock(
            return_value=_make_model_response(_valid_plan_json())
        )

        planner = AutoPlanner(model=mock_model)
        observation = _make_observation(
            tool_requests=(
                ToolCall(id="call_1", name="search", arguments={"query": "test"}),
            ),
            goal="Answer the question",
        )
        ctx = _make_ctx()

        plan = await planner.plan(observation, ctx)

        assert isinstance(plan, Plan)
        assert plan.goal == "Answer the question"
        assert len(plan.steps) == 2
        assert plan.steps[0].type == StepType.TOOL
        assert plan.steps[0].payload["tool"] == "search"
        assert plan.steps[1].type == StepType.TOOL
        assert plan.steps[1].payload["tool"] == "calculate"

    @pytest.mark.asyncio
    async def test_model_called_with_planning_prompt(self) -> None:
        """Test that the model is called with a structured planning prompt."""
        mock_model = AsyncMock()
        mock_model.complete = AsyncMock(
            return_value=_make_model_response(_valid_plan_json())
        )

        planner = AutoPlanner(model=mock_model)
        observation = _make_observation(
            tool_requests=(
                ToolCall(id="call_1", name="search", arguments={"query": "test"}),
            ),
            goal="Test goal",
        )
        ctx = _make_ctx()

        await planner.plan(observation, ctx)

        # Model should have been called
        mock_model.complete.assert_called_once()
        call_args = mock_model.complete.call_args
        system = call_args.kwargs.get("system", "")

        # System prompt should mention planning
        assert system is not None
        assert "plan" in system.lower()

    @pytest.mark.asyncio
    async def test_plan_with_dependencies(self) -> None:
        """Test parsing plan with step dependencies."""
        plan_json = _valid_plan_json(
            steps=[
                {
                    "type": "tool",
                    "tool": "search",
                    "args": {"query": "test"},
                    "reason": "Search first",
                },
                {
                    "type": "tool",
                    "tool": "calculate",
                    "args": {"expr": "2+2"},
                    "reason": "Calculate using search results",
                    "depends_on": ["step_0"],
                },
            ]
        )

        mock_model = AsyncMock()
        mock_model.complete = AsyncMock(
            return_value=_make_model_response(plan_json)
        )

        planner = AutoPlanner(model=mock_model)
        observation = _make_observation(
            tool_requests=(
                ToolCall(id="call_1", name="search", arguments={"query": "test"}),
                ToolCall(id="call_2", name="calculate", arguments={"expr": "2+2"}),
            ),
        )
        ctx = _make_ctx()

        plan = await planner.plan(observation, ctx)

        # Second step should depend on first
        assert plan.steps[1].dependencies is not None
        assert len(plan.steps[1].dependencies) > 0

    @pytest.mark.asyncio
    async def test_terminal_step_when_no_tools(self) -> None:
        """Test that a TERMINAL step is generated when model says done."""
        plan_json = _valid_plan_json(
            steps=[
                {
                    "type": "terminal",
                    "output": "The answer is 4",
                    "reason": "Direct answer",
                },
            ]
        )

        mock_model = AsyncMock()
        mock_model.complete = AsyncMock(
            return_value=_make_model_response(plan_json)
        )

        planner = AutoPlanner(model=mock_model)
        observation = _make_observation(response="The answer is 4")
        ctx = _make_ctx()

        plan = await planner.plan(observation, ctx)

        assert len(plan.steps) == 1
        assert plan.steps[0].type == StepType.TERMINAL
        assert plan.steps[0].payload["output"] == "The answer is 4"

    @pytest.mark.asyncio
    async def test_fallback_on_model_error(self) -> None:
        """Test fallback to SequentialPlanner when model call fails (AD-016)."""
        mock_model = AsyncMock()
        mock_model.complete = AsyncMock(side_effect=Exception("API error"))

        planner = AutoPlanner(model=mock_model)
        observation = _make_observation(
            tool_requests=(
                ToolCall(id="call_1", name="search", arguments={"query": "test"}),
            ),
            goal="Search and answer",
        )
        ctx = _make_ctx()

        plan = await planner.plan(observation, ctx)

        # Should still get a valid plan (from SequentialPlanner fallback)
        assert isinstance(plan, Plan)
        assert plan.metadata.get("planner") == "sequential"
        assert plan.metadata.get("fallback") is True
        assert plan.metadata.get("fallback_reason") is not None

    @pytest.mark.asyncio
    async def test_fallback_on_parse_error(self) -> None:
        """Test fallback when model returns invalid JSON."""
        mock_model = AsyncMock()
        mock_model.complete = AsyncMock(
            return_value=_make_model_response("This is not JSON at all")
        )

        planner = AutoPlanner(model=mock_model)
        observation = _make_observation(
            tool_requests=(
                ToolCall(id="call_1", name="search", arguments={"query": "test"}),
            ),
        )
        ctx = _make_ctx()

        plan = await planner.plan(observation, ctx)

        # Should fall back to sequential
        assert plan.metadata.get("planner") == "sequential"
        assert plan.metadata.get("fallback") is True

    @pytest.mark.asyncio
    async def test_fallback_on_invalid_plan_structure(self) -> None:
        """Test fallback when model returns JSON but not a valid plan."""
        invalid_plan = json.dumps({"not_a_plan": True})

        mock_model = AsyncMock()
        mock_model.complete = AsyncMock(
            return_value=_make_model_response(invalid_plan)
        )

        planner = AutoPlanner(model=mock_model)
        observation = _make_observation(
            tool_requests=(
                ToolCall(id="call_1", name="search", arguments={"query": "test"}),
            ),
        )
        ctx = _make_ctx()

        plan = await planner.plan(observation, ctx)

        # Should fall back to sequential
        assert plan.metadata.get("planner") == "sequential"
        assert plan.metadata.get("fallback") is True

    @pytest.mark.asyncio
    async def test_confidence_scoring_from_model(self) -> None:
        """Test that confidence is extracted from model response."""
        plan_json = _valid_plan_json(confidence=0.92)

        mock_model = AsyncMock()
        mock_model.complete = AsyncMock(
            return_value=_make_model_response(plan_json)
        )

        planner = AutoPlanner(model=mock_model)
        observation = _make_observation(
            tool_requests=(
                ToolCall(id="call_1", name="search", arguments={"query": "test"}),
            ),
        )
        ctx = _make_ctx()

        plan = await planner.plan(observation, ctx)

        assert plan.confidence is not None
        assert plan.confidence == pytest.approx(0.92)

    @pytest.mark.asyncio
    async def test_confidence_clamped_to_range(self) -> None:
        """Test that confidence values are clamped to [0.0, 1.0]."""
        plan_json = _valid_plan_json(confidence=1.5)

        mock_model = AsyncMock()
        mock_model.complete = AsyncMock(
            return_value=_make_model_response(plan_json)
        )

        planner = AutoPlanner(model=mock_model)
        observation = _make_observation(
            tool_requests=(
                ToolCall(id="call_1", name="search", arguments={"query": "test"}),
            ),
        )
        ctx = _make_ctx()

        plan = await planner.plan(observation, ctx)

        assert plan.confidence is not None
        assert 0.0 <= plan.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_confidence_default_when_missing(self) -> None:
        """Test default confidence when model doesn't provide one."""
        plan_data = {
            "reasoning": "Simple plan",
            "steps": [
                {"type": "tool", "tool": "search", "args": {"query": "test"}},
            ],
        }

        mock_model = AsyncMock()
        mock_model.complete = AsyncMock(
            return_value=_make_model_response(json.dumps(plan_data))
        )

        planner = AutoPlanner(model=mock_model)
        observation = _make_observation(
            tool_requests=(
                ToolCall(id="call_1", name="search", arguments={"query": "test"}),
            ),
        )
        ctx = _make_ctx()

        plan = await planner.plan(observation, ctx)

        # Should have a default confidence
        assert plan.confidence is not None
        assert 0.0 <= plan.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_fallback_confidence_is_lower(self) -> None:
        """Test that fallback plans have reduced confidence."""
        mock_model = AsyncMock()
        mock_model.complete = AsyncMock(side_effect=Exception("API error"))

        planner = AutoPlanner(model=mock_model)
        observation = _make_observation(
            tool_requests=(
                ToolCall(id="call_1", name="search", arguments={"query": "test"}),
            ),
        )
        ctx = _make_ctx()

        plan = await planner.plan(observation, ctx)

        # Fallback plan should have lower confidence than SequentialPlanner's 1.0
        assert plan.confidence is not None
        assert plan.confidence < 1.0

    @pytest.mark.asyncio
    async def test_plan_metadata_includes_planner_name(self) -> None:
        """Test that plan metadata identifies the planner."""
        mock_model = AsyncMock()
        mock_model.complete = AsyncMock(
            return_value=_make_model_response(_valid_plan_json())
        )

        planner = AutoPlanner(model=mock_model)
        observation = _make_observation(
            tool_requests=(
                ToolCall(id="call_1", name="search", arguments={"query": "test"}),
            ),
        )
        ctx = _make_ctx()

        plan = await planner.plan(observation, ctx)

        assert plan.metadata.get("planner") == "auto"

    @pytest.mark.asyncio
    async def test_plan_reasoning_from_model(self) -> None:
        """Test that reasoning is extracted from model response."""
        plan_json = _valid_plan_json(reasoning="We should search first then calculate")

        mock_model = AsyncMock()
        mock_model.complete = AsyncMock(
            return_value=_make_model_response(plan_json)
        )

        planner = AutoPlanner(model=mock_model)
        observation = _make_observation(
            tool_requests=(
                ToolCall(id="call_1", name="search", arguments={"query": "test"}),
            ),
        )
        ctx = _make_ctx()

        plan = await planner.plan(observation, ctx)

        assert plan.reasoning == "We should search first then calculate"

    @pytest.mark.asyncio
    async def test_plan_ids_are_unique(self) -> None:
        """Test that plan and step IDs are unique."""
        mock_model = AsyncMock()
        mock_model.complete = AsyncMock(
            return_value=_make_model_response(_valid_plan_json())
        )

        planner = AutoPlanner(model=mock_model)
        observation = _make_observation(
            tool_requests=(
                ToolCall(id="call_1", name="search", arguments={"query": "test"}),
                ToolCall(id="call_2", name="calculate", arguments={"expr": "2+2"}),
            ),
        )
        ctx = _make_ctx()

        plan = await planner.plan(observation, ctx)

        assert plan.id
        step_ids = [step.id for step in plan.steps]
        assert len(step_ids) == len(set(step_ids))

    @pytest.mark.asyncio
    async def test_model_step_generated(self) -> None:
        """Test MODEL step is generated when model specifies it."""
        plan_json = _valid_plan_json(
            steps=[
                {"type": "model", "reason": "Need to think about this"},
            ]
        )

        mock_model = AsyncMock()
        mock_model.complete = AsyncMock(
            return_value=_make_model_response(plan_json)
        )

        planner = AutoPlanner(model=mock_model)
        observation = _make_observation()
        ctx = _make_ctx()

        plan = await planner.plan(observation, ctx)

        assert len(plan.steps) == 1
        assert plan.steps[0].type == StepType.MODEL

    @pytest.mark.asyncio
    async def test_json_in_markdown_code_block(self) -> None:
        """Test that JSON wrapped in markdown code blocks is extracted."""
        plan_content = _valid_plan_json()
        wrapped = f"Here's my plan:\n```json\n{plan_content}\n```"

        mock_model = AsyncMock()
        mock_model.complete = AsyncMock(
            return_value=_make_model_response(wrapped)
        )

        planner = AutoPlanner(model=mock_model)
        observation = _make_observation(
            tool_requests=(
                ToolCall(id="call_1", name="search", arguments={"query": "test"}),
            ),
        )
        ctx = _make_ctx()

        plan = await planner.plan(observation, ctx)

        # Should successfully parse the JSON from the code block
        assert plan.metadata.get("planner") == "auto"
        assert len(plan.steps) == 2

    @pytest.mark.asyncio
    async def test_empty_steps_triggers_fallback(self) -> None:
        """Test that empty steps array triggers fallback."""
        plan_json = _valid_plan_json(steps=[])

        mock_model = AsyncMock()
        mock_model.complete = AsyncMock(
            return_value=_make_model_response(plan_json)
        )

        planner = AutoPlanner(model=mock_model)
        observation = _make_observation(
            tool_requests=(
                ToolCall(id="call_1", name="search", arguments={"query": "test"}),
            ),
        )
        ctx = _make_ctx()

        plan = await planner.plan(observation, ctx)

        # Empty steps should trigger fallback
        assert plan.metadata.get("fallback") is True

    @pytest.mark.asyncio
    async def test_planner_protocol_compliance(self) -> None:
        """Test that AutoPlanner satisfies the Planner protocol."""
        from axis_core.protocols.planner import Planner

        mock_model = AsyncMock()
        planner = AutoPlanner(model=mock_model)

        assert isinstance(planner, Planner)
