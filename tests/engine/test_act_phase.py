"""Focused act-phase behavior tests for context-window guard and pruning."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from axis_core.budget import Budget
from axis_core.engine.lifecycle import LifecycleEngine
from axis_core.protocols.model import ModelResponse, UsageStats
from axis_core.protocols.planner import Plan, PlanStep, StepType
from axis_core.protocols.telemetry import BufferMode, TraceEvent


class MockModelAdapter:
    """Minimal model adapter for act-phase context guard tests."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    @property
    def model_id(self) -> str:
        return "mock-model"

    async def complete(
        self,
        messages: Any,
        system: str | None = None,
        tools: Any | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ModelResponse:
        self.calls.append({"messages": messages, "system": system, "tools": tools})
        return ModelResponse(
            content="ok",
            tool_calls=None,
            usage=UsageStats(input_tokens=5, output_tokens=5, total_tokens=10),
            cost_usd=0.0005,
        )

    async def stream(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Streaming is not needed for these tests")

    def estimate_tokens(self, text: str) -> int:
        return len(text) // 4

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0


class MockPlanner:
    """Planner that returns predefined plans in order."""

    def __init__(self, plans: list[Plan]) -> None:
        self._plans = plans
        self._index = 0

    async def plan(self, observation: Any, ctx: Any) -> Plan:
        if self._index < len(self._plans):
            plan = self._plans[self._index]
            self._index += 1
            return plan
        return Plan(
            id="terminal-default",
            goal="done",
            steps=(PlanStep(id="terminal", type=StepType.TERMINAL, payload={"output": "done"}),),
        )


class MockTelemetrySink:
    """Collects emitted telemetry events."""

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    @property
    def buffering(self) -> BufferMode:
        return BufferMode.IMMEDIATE

    async def emit(self, event: TraceEvent) -> None:
        self.events.append(event)

    async def flush(self) -> None:
        pass

    async def close(self) -> None:
        pass


def _planner_with_messages(messages: list[dict[str, Any]]) -> MockPlanner:
    return MockPlanner(
        plans=[
            Plan(
                id="plan-model",
                goal="guard check",
                steps=(
                    PlanStep(
                        id="model-1",
                        type=StepType.MODEL,
                        payload={"messages": messages},
                    ),
                ),
            ),
            Plan(
                id="plan-terminal",
                goal="done",
                steps=(
                    PlanStep(
                        id="terminal-1",
                        type=StepType.TERMINAL,
                        payload={"output": "done"},
                    ),
                ),
            ),
        ]
    )


def _runtime_config(**overrides: Any) -> Any:
    base = {
        "timeouts": None,
        "retry": None,
        "rate_limits": None,
        "cache": None,
        "confirmation_handler": None,
        "context_window_guard_enabled": True,
        "context_window_tokens": 100,
        "context_window_warn_tokens": 32,
        "context_window_block_tokens": 16,
        "context_pruning_enabled": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class TestActPhaseContextWindowGuard:
    """Integration checks for context-window behavior before model calls."""

    @pytest.mark.asyncio
    async def test_guard_blocks_model_call_when_block_threshold_exceeded(self) -> None:
        model = MockModelAdapter()
        messages = [{"role": "user", "content": "x" * 360}]
        engine = LifecycleEngine(
            model=model,
            planner=_planner_with_messages(messages),
        )

        result = await engine.execute(
            input_text="test",
            agent_id="act-guard-block",
            budget=Budget(max_cycles=2),
            config=_runtime_config(context_window_block_tokens=20),
        )

        assert result["success"] is False
        assert len(model.calls) == 0
        assert any(
            "Context window guard blocked model call" in e.error.message
            for e in result["errors"]
        )

    @pytest.mark.asyncio
    async def test_guard_warns_and_allows_model_call_above_block_threshold(self) -> None:
        model = MockModelAdapter()
        telemetry = MockTelemetrySink()
        messages = [{"role": "user", "content": "x" * 280}]
        engine = LifecycleEngine(
            model=model,
            planner=_planner_with_messages(messages),
            telemetry=[telemetry],
        )

        result = await engine.execute(
            input_text="test",
            agent_id="act-guard-warn",
            budget=Budget(max_cycles=3),
            config=_runtime_config(context_window_warn_tokens=35, context_window_block_tokens=20),
        )

        assert result["success"] is True
        assert len(model.calls) == 1
        assert any(event.type == "context_window_warning" for event in telemetry.events)

    @pytest.mark.asyncio
    async def test_guard_prunes_tool_results_before_blocking(self) -> None:
        model = MockModelAdapter()
        telemetry = MockTelemetrySink()
        messages = [
            {"role": "user", "content": "request"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_1", "name": "lookup", "arguments": {"q": "x"}}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "x" * 800},
            {"role": "user", "content": "follow-up"},
        ]

        engine = LifecycleEngine(
            model=model,
            planner=_planner_with_messages(messages),
            telemetry=[telemetry],
        )

        result = await engine.execute(
            input_text="test",
            agent_id="act-guard-prune",
            budget=Budget(max_cycles=3),
            config=_runtime_config(
                context_window_tokens=120,
                context_window_warn_tokens=40,
                context_window_block_tokens=20,
                context_pruning_enabled=True,
            ),
        )

        assert result["success"] is True
        assert len(model.calls) == 1
        sent_messages = model.calls[0]["messages"]
        assert all(msg.get("role") != "tool" for msg in sent_messages)
        assert any(event.type == "context_window_pruned" for event in telemetry.events)
