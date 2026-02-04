"""End-to-end integration tests for tool usage in agent lifecycle.

These tests verify that tools are properly discovered, schemas are generated,
and the complete agent → model → tools → result flow works correctly.
"""

from __future__ import annotations

import pytest

from axis_core.budget import Budget
from axis_core.engine.lifecycle import LifecycleEngine
from axis_core.protocols.model import ModelResponse, ToolCall, UsageStats
from axis_core.protocols.planner import Plan, PlanStep, StepType
from axis_core.tool import ToolContext, tool

# =============================================================================
# Mock adapters
# =============================================================================


class MockModelWithTools:
    """Mock model that simulates tool-using behavior.

    This mock handles ToolManifest objects like a real adapter would.
    """

    def __init__(self) -> None:
        self.call_count = 0
        self.last_tools = None
        self.last_tools_type = None

    @property
    def model_id(self) -> str:
        return "mock-model"

    async def complete(self, messages: list, system: str | None = None, tools=None, **kwargs):
        """Simulate model completions with tool calls."""
        self.call_count += 1
        self.last_tools = tools

        # Track what type of tools we received
        if tools:
            if isinstance(tools, list) and tools:
                self.last_tools_type = type(tools[0]).__name__
            else:
                self.last_tools_type = type(tools).__name__

        # First call: request tool usage
        if self.call_count == 1:
            return ModelResponse(
                content="I'll check the weather for you.",
                tool_calls=(
                    ToolCall(
                        id="call_1",
                        name="get_weather",
                        arguments={"city": "San Francisco"},
                    ),
                ),
                usage=UsageStats(input_tokens=50, output_tokens=30, total_tokens=80),
                cost_usd=0.002,
            )

        # Second call: final response after tool execution
        return ModelResponse(
            content="The weather in San Francisco is sunny!",
            tool_calls=None,
            usage=UsageStats(input_tokens=100, output_tokens=50, total_tokens=150),
            cost_usd=0.004,
        )


class SimplePlanner:
    """Simple planner for testing."""

    async def plan(self, observation, ctx):
        """Generate simple plans based on observation."""
        steps = []

        if observation.tool_requests:
            # Execute tools
            for i, tc in enumerate(observation.tool_requests):
                steps.append(
                    PlanStep(
                        id=f"tool_{i}",
                        type=StepType.TOOL,
                        payload={
                            "tool": tc.name,
                            "tool_call_id": tc.id,
                            "args": tc.arguments,
                        },
                        dependencies=None,
                        retry_policy=None,
                    )
                )
        elif observation.response:
            # Terminal step
            steps.append(
                PlanStep(
                    id="terminal",
                    type=StepType.TERMINAL,
                    payload={"output": observation.response},
                    dependencies=None,
                    retry_policy=None,
                )
            )
        else:
            # Call model
            steps.append(
                PlanStep(
                    id="model",
                    type=StepType.MODEL,
                    payload={},
                    dependencies=None,
                    retry_policy=None,
                )
            )

        return Plan(
            id="plan_1",
            goal="Test plan",
            steps=tuple(steps),
            reasoning="Simple test plan",
            confidence=1.0,
            metadata={},
        )


# =============================================================================
# Integration tests
# =============================================================================


class TestToolIntegration:
    """End-to-end tests for tool integration."""

    @pytest.mark.asyncio
    async def test_tool_manifests_passed_to_model(self) -> None:
        """Verify ToolManifest objects (protocol layer) are passed to model."""
        from axis_core.tool import ToolManifest

        @tool
        def get_weather(city: str) -> str:
            """Get the weather for a city."""
            return f"Weather in {city} is sunny"

        model = MockModelWithTools()
        planner = SimplePlanner()

        engine = LifecycleEngine(
            model=model,
            planner=planner,
            tools={"get_weather": get_weather},
        )

        # Execute one cycle with a MODEL step
        ctx = await engine._initialize(
            input_text="What's the weather?",
            agent_id="test",
            budget=Budget(),
        )

        observation = await engine._observe(ctx)
        plan = await engine._plan(ctx, observation)
        await engine._act(ctx, plan)

        # Verify model was called with ToolManifest objects (protocol layer)
        assert model.call_count == 1
        assert model.last_tools is not None
        assert len(model.last_tools) == 1
        # Should be ToolManifest, not dict
        assert isinstance(model.last_tools[0], ToolManifest)
        assert model.last_tools[0].name == "get_weather"
        assert model.last_tools[0].input_schema["type"] == "object"

    @pytest.mark.asyncio
    async def test_complete_tool_execution_flow(self) -> None:
        """Verify complete flow: model requests tool → tool executes → model responds."""

        @tool
        def get_weather(city: str) -> str:
            """Get the weather for a city."""
            return f"Sunny and 72°F in {city}"

        model = MockModelWithTools()
        planner = SimplePlanner()

        engine = LifecycleEngine(
            model=model,
            planner=planner,
            tools={"get_weather": get_weather},
        )

        result = await engine.execute(
            input_text="What's the weather in San Francisco?",
            agent_id="test",
            budget=Budget(max_cycles=10),
        )

        # Verify success
        assert result["success"] is True
        assert result["output"] == "The weather in San Francisco is sunny!"

        # Flow: Cycle 1: call model → Cycle 2: exec tool → Cycle 3: call model → Cycle 4: terminal
        assert result["cycles_completed"] == 4

        # Verify model was called twice (once to request tool, once after tool execution)
        assert model.call_count == 2

        # Verify tool was called with correct arguments
        budget_state = result["budget_state"]
        assert budget_state.tool_calls == 1
        assert budget_state.model_calls == 2

    @pytest.mark.asyncio
    async def test_tool_receives_ctx(self) -> None:
        """Verify ToolContext is injected when tool declares ctx parameter."""
        received: dict[str, str] = {}

        @tool
        def tool_with_ctx(ctx: ToolContext, name: str) -> str:
            received["run_id"] = ctx.run_id
            received["agent_id"] = ctx.agent_id
            return f"{ctx.run_id}:{name}"

        class CtxPlanner:
            async def plan(self, observation, ctx):
                steps = [
                    PlanStep(
                        id="tool_0",
                        type=StepType.TOOL,
                        payload={
                            "tool": "tool_with_ctx",
                            "tool_call_id": "call_0",
                            "args": {"name": "test"},
                        },
                        dependencies=None,
                        retry_policy=None,
                    ),
                    PlanStep(
                        id="terminal",
                        type=StepType.TERMINAL,
                        payload={"output": "done"},
                        dependencies=None,
                        retry_policy=None,
                    ),
                ]
                return Plan(
                    id="plan_ctx",
                    goal="ctx test",
                    steps=tuple(steps),
                    reasoning="ctx test",
                    confidence=1.0,
                    metadata={},
                )

        engine = LifecycleEngine(
            model=MockModelWithTools(),
            planner=CtxPlanner(),
            tools={"tool_with_ctx": tool_with_ctx},
        )

        result = await engine.execute(
            input_text="ctx test",
            agent_id="agent-123",
            budget=Budget(max_cycles=3),
        )

        assert result["success"] is True
        assert received["agent_id"] == "agent-123"
        assert received["run_id"] == result["run_id"]

    @pytest.mark.asyncio
    async def test_tool_manifest_with_multiple_params(self) -> None:
        """Verify manifests preserve multiple parameter info correctly."""
        from axis_core.tool import ToolManifest

        @tool
        def search(query: str, limit: int = 10, sort: str = "relevance") -> str:
            """Search for items."""
            return f"Found {limit} results for '{query}' sorted by {sort}"

        model = MockModelWithTools()
        planner = SimplePlanner()

        engine = LifecycleEngine(
            model=model,
            planner=planner,
            tools={"search": search},
        )

        # Get manifests
        manifests = engine._get_tool_manifests()

        assert len(manifests) == 1
        manifest = manifests[0]
        assert isinstance(manifest, ToolManifest)
        assert manifest.name == "search"
        assert "query" in manifest.input_schema["required"]
        assert "limit" not in manifest.input_schema["required"]
        assert "sort" not in manifest.input_schema["required"]
        assert "limit" in manifest.input_schema["properties"]
        assert "sort" in manifest.input_schema["properties"]

    @pytest.mark.asyncio
    async def test_no_tools_passes_none_to_model(self) -> None:
        """Verify when no tools exist, None is passed to model."""

        model = MockModelWithTools()
        planner = SimplePlanner()

        engine = LifecycleEngine(
            model=model,
            planner=planner,
            tools={},  # No tools
        )

        ctx = await engine._initialize(
            input_text="Hello",
            agent_id="test",
            budget=Budget(),
        )

        observation = await engine._observe(ctx)
        plan = await engine._plan(ctx, observation)
        await engine._act(ctx, plan)

        # Verify model was called without tools
        assert model.call_count == 1
        assert model.last_tools is None

    @pytest.mark.asyncio
    async def test_tool_manifest_missing_handled_gracefully(self) -> None:
        """Verify tools without manifests are skipped with warning."""

        # Regular function without @tool decorator
        def bad_tool(x: int) -> int:
            return x * 2

        model = MockModelWithTools()
        planner = SimplePlanner()

        engine = LifecycleEngine(
            model=model,
            planner=planner,
            tools={"bad_tool": bad_tool},  # Missing _axis_manifest
        )

        # Should not raise, just skip the tool
        manifests = engine._get_tool_manifests()
        assert len(manifests) == 0
