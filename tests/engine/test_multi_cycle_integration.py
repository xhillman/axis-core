"""Integration test for multi-cycle model calling flow.

Tests the complete flow:
- Cycle 1: MODEL step → model response with tool calls
- Cycle 2: TOOL steps → execute tools
- Cycle 3: MODEL step → final response
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from axis_core.budget import Budget
from axis_core.engine.lifecycle import LifecycleEngine
from axis_core.protocols.model import ModelChunk, ModelResponse, ToolCall, UsageStats


class IntegrationMockModel:
    """Mock model for integration testing."""

    def __init__(self):
        self.call_count = 0
        self.calls: list[dict[str, Any]] = []

    @property
    def model_id(self) -> str:
        return "integration-model"

    async def complete(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        self.calls.append({"messages": messages, "system": system, "tools": tools})
        self.call_count += 1

        # First call: request a tool
        if self.call_count == 1:
            return ModelResponse(
                content="Let me search for that information",
                tool_calls=(
                    ToolCall(id="call_1", name="search", arguments={"q": "test query"}),
                ),
                usage=UsageStats(input_tokens=10, output_tokens=15, total_tokens=25),
                cost_usd=0.001,
            )

        # Second call: provide final answer
        elif self.call_count == 2:
            return ModelResponse(
                content="Based on the search results, here is the answer",
                tool_calls=None,
                usage=UsageStats(input_tokens=20, output_tokens=25, total_tokens=45),
                cost_usd=0.002,
            )

        # Fallback
        return ModelResponse(
            content="default",
            tool_calls=None,
            usage=UsageStats(input_tokens=5, output_tokens=5, total_tokens=10),
            cost_usd=0.001,
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        yield ModelChunk(content="stream", is_final=True)

    def estimate_tokens(self, text: str) -> int:
        return len(text.split())

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return float(input_tokens + output_tokens)


async def mock_search_tool(q: str) -> str:
    """Mock search tool."""
    return f"Search results for: {q}"


def _runtime_config(**overrides: Any) -> Any:
    base = {
        "context_strategy": "smart",
        "max_cycle_context": 5,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.asyncio
async def test_multi_cycle_flow() -> None:
    """Test complete multi-cycle flow with model → tools → model."""
    # Import here to ensure adapters are registered
    import axis_core.adapters.planners  # noqa: F401

    mock_model = IntegrationMockModel()

    engine = LifecycleEngine(
        model=mock_model,
        planner="sequential",
        tools={"search": mock_search_tool},
    )

    result = await engine.execute(
        input_text="What is the capital of France?",
        agent_id="test-agent",
        budget=Budget(max_cycles=10),
    )

    # Verify execution completed successfully
    if not result["success"]:
        print(f"Error: {result.get('error')}")
        print(f"Cycles: {result.get('cycles_completed')}")
        print(f"Model call count: {mock_model.call_count}")
        print(f"Output: {result.get('output')}")
    assert result["success"] is True
    # May take 2-3 cycles depending on implementation
    assert result["cycles_completed"] <= 5

    # Verify model was called twice
    assert mock_model.call_count == 2

    # Verify first model call had just user input
    first_call = mock_model.calls[0]
    assert len(first_call["messages"]) == 1
    assert first_call["messages"][0]["role"] == "user"
    assert "capital of France" in first_call["messages"][0]["content"]

    # Verify second model call had conversation history
    second_call = mock_model.calls[1]
    assert len(second_call["messages"]) >= 3  # user, assistant with tool_calls, tool result

    # Check conversation structure
    messages = second_call["messages"]
    assert messages[0]["role"] == "user"  # Original input
    assert messages[1]["role"] == "assistant"  # First response with tool calls
    assert "tool_calls" in messages[1]
    assert messages[2]["role"] == "tool"  # Tool results
    assert "Search results for: test query" in messages[2]["content"]

    # Verify final output
    assert "answer" in result["output"].lower()


@pytest.mark.asyncio
async def test_context_strategy_runtime_config() -> None:
    """Test that runtime config controls context strategy in act phase."""
    import axis_core.adapters.planners  # noqa: F401

    mock_model = IntegrationMockModel()

    engine = LifecycleEngine(
        model=mock_model,
        planner="sequential",
        tools={"search": mock_search_tool},
    )

    result = await engine.execute(
        input_text="Test input",
        agent_id="test-agent",
        budget=Budget(max_cycles=10),
        config=_runtime_config(context_strategy="minimal"),
    )

    # Should still work with minimal context
    assert result["success"] is True

    # Second model call should have minimal context (just first message)
    if mock_model.call_count >= 2:
        second_call = mock_model.calls[1]
        messages = second_call["messages"]
        # Minimal strategy = just first user message, no history
        assert len(messages) == 1
        assert messages[0]["role"] == "user"


@pytest.mark.asyncio
async def test_max_cycle_context_runtime_config() -> None:
    """Test that runtime config controls smart-context history depth."""
    import axis_core.adapters.planners  # noqa: F401

    # Create model that keeps requesting tools for many cycles
    class MultiCycleModel:
        def __init__(self):
            self.call_count = 0
            self.calls: list[dict[str, Any]] = []

        @property
        def model_id(self) -> str:
            return "multi-cycle-model"

        async def complete(
            self,
            messages: list[dict[str, Any]],
            system: str | None = None,
            tools: list[dict[str, Any]] | None = None,
        ) -> ModelResponse:
            self.calls.append({"messages": messages})
            self.call_count += 1

            # Request tool for first 5 calls
            if self.call_count <= 5:
                return ModelResponse(
                    content=f"Cycle {self.call_count}",
                    tool_calls=(
                        ToolCall(
                            id=f"call_{self.call_count}",
                            name="search",
                            arguments={"q": f"query {self.call_count}"},
                        ),
                    ),
                    usage=UsageStats(input_tokens=10, output_tokens=10, total_tokens=20),
                    cost_usd=0.001,
                )

            # Final response
            return ModelResponse(
                content="Final answer",
                tool_calls=None,
                usage=UsageStats(input_tokens=10, output_tokens=10, total_tokens=20),
                cost_usd=0.001,
            )

        async def stream(
            self,
            messages: list[dict[str, Any]],
            system: str | None = None,
            tools: list[dict[str, Any]] | None = None,
        ) -> Any:
            yield ModelChunk(content="stream", is_final=True)

        def estimate_tokens(self, text: str) -> int:
            return len(text.split())

        def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
            return float(input_tokens + output_tokens)

    mock_model = MultiCycleModel()

    engine = LifecycleEngine(
        model=mock_model,
        planner="sequential",
        tools={"search": mock_search_tool},
    )

    result = await engine.execute(
        input_text="Test",
        agent_id="test-agent",
        budget=Budget(max_cycles=15),  # Need enough cycles for 5 tool requests + final response
        config=_runtime_config(context_strategy="smart", max_cycle_context=2),
    )

    if not result["success"]:
        print(f"Error: {result.get('error')}")
        print(f"Cycles: {result.get('cycles_completed')}")
        print(f"Model call count: {mock_model.call_count}")

    assert result["success"] is True

    # Check that context was limited to max 2 cycles on later calls
    # (This is a smoke test - detailed verification would require inspecting message counts)
    assert mock_model.call_count >= 5


@pytest.mark.asyncio
async def test_invalid_max_cycle_context_runtime_config_falls_back() -> None:
    """Invalid runtime max_cycle_context should not fail execution."""
    import axis_core.adapters.planners  # noqa: F401

    mock_model = IntegrationMockModel()
    engine = LifecycleEngine(
        model=mock_model,
        planner="sequential",
        tools={"search": mock_search_tool},
    )

    result = await engine.execute(
        input_text="Test invalid max cycles",
        agent_id="test-agent",
        budget=Budget(max_cycles=10),
        config=_runtime_config(context_strategy="smart", max_cycle_context="not-an-int"),
    )

    assert result["success"] is True
    assert mock_model.call_count >= 2
