"""Tests for Agent class (core API)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from axis_core.agent import Agent
from axis_core.budget import Budget
from axis_core.config import Timeouts
from axis_core.errors import BudgetError
from axis_core.errors import TimeoutError as AxisTimeoutError
from axis_core.protocols.model import ModelChunk, ModelResponse, UsageStats
from axis_core.protocols.planner import Plan, PlanStep, StepType
from axis_core.result import RunResult, RunStats, StreamEvent
from axis_core.tool import tool

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class MockModel:
    """Minimal mock model for testing Agent without real LLM."""

    def __init__(
        self,
        response_text: str = "mock response",
        stream_chunks: list[str] | None = None,
    ) -> None:
        self._response_text = response_text
        self._stream_chunks = stream_chunks or [response_text]
        self.complete_calls: list[dict[str, Any]] = []

    @property
    def model_id(self) -> str:
        return "mock-model"

    async def complete(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        self.complete_calls.append({"messages": messages, "system": system})
        return ModelResponse(
            content=self._response_text,
            tool_calls=[],
            usage=UsageStats(input_tokens=10, output_tokens=5, total_tokens=15),
            cost_usd=0.001,
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        for chunk in self._stream_chunks:
            yield ModelChunk(content=chunk, tool_call_delta=None, is_final=False)
        yield ModelChunk(content="", tool_call_delta=None, is_final=True)

    def estimate_tokens(self, text: str) -> int:
        return len(text) // 4

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens + output_tokens) * 0.00001


class MockPlanner:
    """Minimal mock planner that creates a terminal plan."""

    async def plan(self, observation: Any, ctx: Any) -> Plan:
        return Plan(
            id="plan-1",
            goal="respond to user",
            steps=[
                PlanStep(
                    id="step-terminal",
                    type=StepType.TERMINAL,
                    payload={"output": "mock response"},
                ),
            ],
        )


class SlowPlanner:
    """Planner that sleeps before returning a terminal plan."""

    def __init__(self, delay_seconds: float) -> None:
        self._delay_seconds = delay_seconds

    async def plan(self, observation: Any, ctx: Any) -> Plan:
        await asyncio.sleep(self._delay_seconds)
        return Plan(
            id="plan-slow",
            goal="respond slowly",
            steps=[
                PlanStep(
                    id="step-terminal",
                    type=StepType.TERMINAL,
                    payload={"output": "slow response"},
                ),
            ],
        )


class MockMemory:
    """Minimal mock memory adapter."""

    @property
    def capabilities(self) -> set[str]:
        return set()

    async def store(self, key: str, value: Any, metadata: dict[str, Any] | None = None) -> None:
        pass

    async def retrieve(self, key: str) -> Any:
        return None

    async def search(self, query: str, limit: int = 10) -> list[Any]:
        return []

    async def delete(self, key: str) -> bool:
        return False

    async def clear(self) -> None:
        pass


class MockTelemetrySink:
    """Minimal telemetry sink that records events."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    @property
    def buffering(self) -> str:
        return "immediate"

    async def emit(self, event: Any) -> None:
        self.events.append(event)

    async def flush(self) -> None:
        pass

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Constructor tests (8.1)
# ---------------------------------------------------------------------------


class TestAgentConstructor:
    """Tests for Agent class constructor."""

    def test_minimal_construction(self) -> None:
        agent = Agent(model=MockModel(), planner=MockPlanner(), memory=None)
        assert agent is not None

    def test_construction_with_tools(self) -> None:
        @tool
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}!"

        agent = Agent(tools=[greet], model=MockModel(), planner=MockPlanner())
        assert "greet" in agent._tools

    def test_construction_with_system_prompt(self) -> None:
        agent = Agent(
            model=MockModel(),
            planner=MockPlanner(),
            system="You are a helpful assistant.",
        )
        assert agent._system == "You are a helpful assistant."

    def test_construction_with_budget(self) -> None:
        budget = Budget(max_cycles=5, max_cost_usd=0.50)
        agent = Agent(model=MockModel(), planner=MockPlanner(), budget=budget)
        assert agent._budget.max_cycles == 5

    def test_construction_with_budget_dict(self) -> None:
        agent = Agent(
            model=MockModel(),
            planner=MockPlanner(),
            budget={"max_cycles": 3, "max_cost_usd": 0.25},
        )
        assert agent._budget.max_cycles == 3
        assert agent._budget.max_cost_usd == 0.25

    def test_construction_with_timeouts(self) -> None:
        timeouts = Timeouts(total=60.0)
        agent = Agent(model=MockModel(), planner=MockPlanner(), timeouts=timeouts)
        assert agent._timeouts.total == 60.0

    def test_construction_with_timeouts_dict(self) -> None:
        agent = Agent(
            model=MockModel(),
            planner=MockPlanner(),
            timeouts={"total": 120.0, "act": 30.0},
        )
        assert agent._timeouts.total == 120.0
        assert agent._timeouts.act == 30.0

    def test_default_budget(self) -> None:
        agent = Agent(model=MockModel(), planner=MockPlanner(), memory=None)
        assert agent._budget.max_cycles == 10  # Budget default

    def test_construction_with_memory(self) -> None:
        agent = Agent(
            model=MockModel(), planner=MockPlanner(), memory=MockMemory()
        )
        assert agent._memory is not None

    def test_construction_with_telemetry_sinks(self) -> None:
        sink = MockTelemetrySink()
        agent = Agent(
            model=MockModel(), planner=MockPlanner(), telemetry=[sink]
        )
        assert len(agent._telemetry_sinks) == 1

    def test_construction_telemetry_true(self) -> None:
        """Test that telemetry=True resolves sinks from AXIS_TELEMETRY_SINK env var."""
        import os
        from unittest.mock import patch

        # Test with AXIS_TELEMETRY_SINK=none (empty sinks)
        with patch.dict(os.environ, {"AXIS_TELEMETRY_SINK": "none"}):
            agent = Agent(model=MockModel(), planner=MockPlanner(), telemetry=True)
            assert agent._telemetry_sinks == []
            assert agent._telemetry_enabled is True

    def test_construction_telemetry_true_with_console_sink(self) -> None:
        """Test that telemetry=True with AXIS_TELEMETRY_SINK=console creates ConsoleSink."""
        import os
        from unittest.mock import patch

        from axis_core.adapters.telemetry.console import ConsoleSink

        with patch.dict(os.environ, {"AXIS_TELEMETRY_SINK": "console"}):
            agent = Agent(model=MockModel(), planner=MockPlanner(), telemetry=True)
            assert len(agent._telemetry_sinks) == 1
            assert isinstance(agent._telemetry_sinks[0], ConsoleSink)
            assert agent._telemetry_enabled is True

    def test_telemetry_redaction_enabled_by_default(self) -> None:
        """Test that AXIS_TELEMETRY_REDACT defaults to true (MED-1 security fix)."""
        import os
        from unittest.mock import patch

        from axis_core.adapters.telemetry.console import ConsoleSink

        with patch.dict(os.environ, {"AXIS_TELEMETRY_SINK": "console"}):
            agent = Agent(model=MockModel(), planner=MockPlanner(), telemetry=True)
            sink = agent._telemetry_sinks[0]
            assert isinstance(sink, ConsoleSink)
            assert sink._redact is True

    def test_telemetry_redaction_respects_env_var_true(self) -> None:
        """Test that AXIS_TELEMETRY_REDACT=true enables redaction."""
        import os
        from unittest.mock import patch

        from axis_core.adapters.telemetry.console import ConsoleSink

        with patch.dict(
            os.environ,
            {"AXIS_TELEMETRY_SINK": "console", "AXIS_TELEMETRY_REDACT": "true"},
        ):
            agent = Agent(model=MockModel(), planner=MockPlanner(), telemetry=True)
            sink = agent._telemetry_sinks[0]
            assert isinstance(sink, ConsoleSink)
            assert sink._redact is True

    def test_telemetry_redaction_respects_env_var_false(self) -> None:
        """Test that AXIS_TELEMETRY_REDACT=false disables redaction."""
        import os
        from unittest.mock import patch

        from axis_core.adapters.telemetry.console import ConsoleSink

        with patch.dict(
            os.environ,
            {"AXIS_TELEMETRY_SINK": "console", "AXIS_TELEMETRY_REDACT": "false"},
        ):
            agent = Agent(model=MockModel(), planner=MockPlanner(), telemetry=True)
            sink = agent._telemetry_sinks[0]
            assert isinstance(sink, ConsoleSink)
            assert sink._redact is False

    def test_telemetry_compact_mode_works_with_redaction(self) -> None:
        """Test that compact mode and redaction can be used together."""
        import os
        from unittest.mock import patch

        from axis_core.adapters.telemetry.console import ConsoleSink

        with patch.dict(
            os.environ,
            {
                "AXIS_TELEMETRY_SINK": "console",
                "AXIS_TELEMETRY_COMPACT": "true",
                "AXIS_TELEMETRY_REDACT": "true",
            },
        ):
            agent = Agent(model=MockModel(), planner=MockPlanner(), telemetry=True)
            sink = agent._telemetry_sinks[0]
            assert isinstance(sink, ConsoleSink)
            assert sink._compact is True
            assert sink._redact is True

    def test_construction_telemetry_false(self) -> None:
        agent = Agent(model=MockModel(), planner=MockPlanner(), telemetry=False)
        assert agent._telemetry_enabled is False

    def test_construction_with_fallback(self) -> None:
        fallback = MockModel("fallback response")
        agent = Agent(
            model=MockModel(),
            planner=MockPlanner(),
            fallback=[fallback],
        )
        assert len(agent._fallback) == 1

    def test_agent_id_generated(self) -> None:
        agent = Agent(model=MockModel(), planner=MockPlanner(), memory=None)
        assert agent._agent_id is not None
        assert len(agent._agent_id) > 0


# ---------------------------------------------------------------------------
# Type validation tests (8.2)
# ---------------------------------------------------------------------------


class TestAgentTypeValidation:
    """Tests for runtime type validation on public APIs (AD-034)."""

    def test_tools_must_be_list(self) -> None:
        with pytest.raises(TypeError, match="tools"):
            Agent(tools="not a list", model=MockModel(), planner=MockPlanner())  # type: ignore[arg-type]

    def test_system_must_be_str(self) -> None:
        with pytest.raises(TypeError, match="system"):
            Agent(model=MockModel(), planner=MockPlanner(), system=123)  # type: ignore[arg-type]

    def test_budget_must_be_budget_or_dict(self) -> None:
        with pytest.raises(TypeError, match="budget"):
            Agent(model=MockModel(), planner=MockPlanner(), budget="invalid")  # type: ignore[arg-type]

    def test_verbose_must_be_bool(self) -> None:
        with pytest.raises(TypeError, match="verbose"):
            Agent(model=MockModel(), planner=MockPlanner(), verbose="yes")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# run_async tests (8.3)
# ---------------------------------------------------------------------------


class TestRunAsync:
    """Tests for run_async() — native async implementation."""

    @pytest.mark.asyncio
    async def test_basic_run(self) -> None:
        agent = Agent(model=MockModel(), planner=MockPlanner(), memory=None)
        result = await agent.run_async("Hello")
        assert isinstance(result, RunResult)
        assert result.success is True
        assert result.run_id is not None

    @pytest.mark.asyncio
    async def test_run_returns_output(self) -> None:
        agent = Agent(model=MockModel("test output"), planner=MockPlanner(), memory=None)
        result = await agent.run_async("Hello")
        assert result.output == "mock response"

    @pytest.mark.asyncio
    async def test_run_with_context(self) -> None:
        agent = Agent(model=MockModel(), planner=MockPlanner(), memory=None)
        result = await agent.run_async("Hello", context={"user_id": "123"})
        assert result.success is True

    @pytest.mark.asyncio
    async def test_run_stats_populated(self) -> None:
        agent = Agent(model=MockModel(), planner=MockPlanner(), memory=None)
        result = await agent.run_async("Hello")
        assert isinstance(result.stats, RunStats)
        assert result.stats.cycles >= 0

    @pytest.mark.asyncio
    async def test_run_empty_input_raises(self) -> None:
        agent = Agent(model=MockModel(), planner=MockPlanner(), memory=None)
        result = await agent.run_async("")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_run_input_type_validated(self) -> None:
        agent = Agent(model=MockModel(), planner=MockPlanner(), memory=None)
        with pytest.raises(TypeError, match="input"):
            await agent.run_async(123)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_run_async_honors_explicit_timeout(self) -> None:
        agent = Agent(model=MockModel(), planner=SlowPlanner(delay_seconds=0.2), memory=None)
        result = await agent.run_async("Hello", timeout=0.05)
        assert result.success is False
        assert isinstance(result.error, AxisTimeoutError)

    @pytest.mark.asyncio
    async def test_run_async_uses_default_total_timeout(self) -> None:
        agent = Agent(
            model=MockModel(),
            planner=SlowPlanner(delay_seconds=0.2),
            memory=None,
            timeouts=Timeouts(total=0.05),
        )
        result = await agent.run_async("Hello")
        assert result.success is False
        assert isinstance(result.error, AxisTimeoutError)

    @pytest.mark.asyncio
    async def test_run_async_respects_wall_time_budget(self) -> None:
        agent = Agent(
            model=MockModel(),
            planner=SlowPlanner(delay_seconds=0.2),
            memory=None,
            budget=Budget(max_wall_time_seconds=0.05),
        )
        result = await agent.run_async("Hello", timeout=1.0)
        assert result.success is False
        assert isinstance(result.error, BudgetError)
        assert result.error.resource == "wall_time"


# ---------------------------------------------------------------------------
# run (sync) tests (8.4)
# ---------------------------------------------------------------------------


class TestRunSync:
    """Tests for run() — sync wrapper (AD-027)."""

    def test_sync_run_returns_result(self) -> None:
        agent = Agent(model=MockModel(), planner=MockPlanner(), memory=None)
        result = agent.run("Hello")
        assert isinstance(result, RunResult)
        assert result.success is True

    def test_sync_run_with_context(self) -> None:
        agent = Agent(model=MockModel(), planner=MockPlanner(), memory=None)
        result = agent.run("Hello", context={"key": "value"})
        assert result.success is True

    def test_sync_run_honors_timeout(self) -> None:
        agent = Agent(model=MockModel(), planner=SlowPlanner(delay_seconds=0.2), memory=None)
        result = agent.run("Hello", timeout=0.05)
        assert result.success is False
        assert isinstance(result.error, AxisTimeoutError)

    def test_sync_run_respects_wall_time_budget(self) -> None:
        agent = Agent(
            model=MockModel(),
            planner=SlowPlanner(delay_seconds=0.2),
            memory=None,
            budget=Budget(max_wall_time_seconds=0.05),
        )
        result = agent.run("Hello", timeout=1.0)
        assert result.success is False
        assert isinstance(result.error, BudgetError)
        assert result.error.resource == "wall_time"


# ---------------------------------------------------------------------------
# stream_async tests (8.5)
# ---------------------------------------------------------------------------


class TestStreamAsync:
    """Tests for stream_async() — async streaming (AD-010)."""

    @pytest.mark.asyncio
    async def test_stream_yields_events(self) -> None:
        agent = Agent(model=MockModel(), planner=MockPlanner(), memory=None)
        events: list[StreamEvent] = []
        async for event in agent.stream_async("Hello"):
            events.append(event)
        assert len(events) >= 2  # at least start and final

    @pytest.mark.asyncio
    async def test_stream_starts_with_run_started(self) -> None:
        agent = Agent(model=MockModel(), planner=MockPlanner(), memory=None)
        events: list[StreamEvent] = []
        async for event in agent.stream_async("Hello"):
            events.append(event)
        assert events[0].type == "run_started"

    @pytest.mark.asyncio
    async def test_stream_ends_with_final_event(self) -> None:
        agent = Agent(model=MockModel(), planner=MockPlanner(), memory=None)
        events: list[StreamEvent] = []
        async for event in agent.stream_async("Hello"):
            events.append(event)
        assert events[-1].is_final is True

    @pytest.mark.asyncio
    async def test_stream_emits_model_tokens(self) -> None:
        class ModelPlanner:
            async def plan(self, observation: Any, ctx: Any) -> Plan:
                return Plan(
                    id="plan-model",
                    goal="stream tokens",
                    steps=[
                        PlanStep(id="model", type=StepType.MODEL, payload={}),
                        PlanStep(
                            id="terminal",
                            type=StepType.TERMINAL,
                            payload={"output": "done"},
                        ),
                    ],
                )

        model = MockModel(stream_chunks=["Hello", " ", "world"])
        agent = Agent(model=model, planner=ModelPlanner(), memory=None)
        tokens: list[str] = []
        async for event in agent.stream_async("Hello"):
            if event.is_token and event.token:
                tokens.append(event.token)
        assert "".join(tokens) == "Hello world"

    @pytest.mark.asyncio
    async def test_stream_telemetry_opt_in(self) -> None:
        agent = Agent(model=MockModel(), planner=MockPlanner(), memory=None)
        telemetry_events: list[StreamEvent] = []
        async for event in agent.stream_async("Hello", stream_telemetry=True):
            if event.type == "telemetry":
                telemetry_events.append(event)
        assert telemetry_events

    @pytest.mark.asyncio
    async def test_stream_telemetry_events_are_redacted_by_default(self) -> None:
        @tool
        def secret_tool(payload: dict[str, Any]) -> str:
            return "ok"

        class ToolPlanner:
            async def plan(self, observation: Any, ctx: Any) -> Plan:
                return Plan(
                    id="plan-tool",
                    goal="call tool",
                    steps=[
                        PlanStep(
                            id="tool-step",
                            type=StepType.TOOL,
                            payload={
                                "tool": "secret_tool",
                                "args": {
                                    "payload": {
                                        "api_key": "sk-secret-123",
                                        "nested": {"private_key": "key-456"},
                                        "safe": "visible",
                                    }
                                },
                            },
                        ),
                        PlanStep(
                            id="terminal",
                            type=StepType.TERMINAL,
                            payload={"output": "done"},
                            dependencies=("tool-step",),
                        ),
                    ],
                )

        agent = Agent(tools=[secret_tool], model=MockModel(), planner=ToolPlanner(), memory=None)

        telemetry_events: list[StreamEvent] = []
        async for event in agent.stream_async("Hello", stream_telemetry=True):
            if event.type == "telemetry":
                telemetry_events.append(event)

        tool_called = next(
            e for e in telemetry_events
            if e.data.get("event", {}).get("type") == "tool_called"
        )
        payload = tool_called.data["event"]["data"]["args"]["payload"]
        assert payload["api_key"] == "[REDACTED]"
        assert payload["nested"]["private_key"] == "[REDACTED]"
        assert payload["safe"] == "visible"

    @pytest.mark.asyncio
    async def test_stream_final_event_includes_stats(self) -> None:
        agent = Agent(model=MockModel(), planner=MockPlanner(), memory=None)
        last_event: StreamEvent | None = None
        async for event in agent.stream_async("Hello"):
            last_event = event
        assert last_event is not None
        assert last_event.is_final
        assert "stats" in last_event.data

    @pytest.mark.asyncio
    async def test_stream_input_type_validated(self) -> None:
        agent = Agent(model=MockModel(), planner=MockPlanner(), memory=None)
        with pytest.raises(TypeError, match="input"):
            async for _ in agent.stream_async(42):  # type: ignore[arg-type]
                pass

    @pytest.mark.asyncio
    async def test_stream_async_honors_explicit_timeout(self) -> None:
        agent = Agent(model=MockModel(), planner=SlowPlanner(delay_seconds=0.2), memory=None)
        events: list[StreamEvent] = []
        async for event in agent.stream_async("Hello", timeout=0.05):
            events.append(event)
        assert events[-1].type == "run_failed"

    @pytest.mark.asyncio
    async def test_stream_async_uses_default_total_timeout(self) -> None:
        agent = Agent(
            model=MockModel(),
            planner=SlowPlanner(delay_seconds=0.2),
            memory=None,
            timeouts=Timeouts(total=0.05),
        )
        events: list[StreamEvent] = []
        async for event in agent.stream_async("Hello"):
            events.append(event)
        assert events[-1].type == "run_failed"


# ---------------------------------------------------------------------------
# stream (sync) tests (8.6)
# ---------------------------------------------------------------------------


class TestStreamSync:
    """Tests for stream() — sync streaming wrapper (AD-027)."""

    def test_sync_stream_yields_events(self) -> None:
        agent = Agent(model=MockModel(), planner=MockPlanner(), memory=None)
        events = list(agent.stream("Hello"))
        assert len(events) >= 2

    def test_sync_stream_has_final_event(self) -> None:
        agent = Agent(model=MockModel(), planner=MockPlanner(), memory=None)
        events = list(agent.stream("Hello"))
        assert events[-1].is_final is True

    def test_sync_stream_emits_model_tokens(self) -> None:
        class ModelPlanner:
            async def plan(self, observation: Any, ctx: Any) -> Plan:
                return Plan(
                    id="plan-model",
                    goal="stream tokens",
                    steps=[
                        PlanStep(id="model", type=StepType.MODEL, payload={}),
                        PlanStep(
                            id="terminal",
                            type=StepType.TERMINAL,
                            payload={"output": "done"},
                        ),
                    ],
                )

        model = MockModel(stream_chunks=["Hello", " ", "sync"])
        agent = Agent(model=model, planner=ModelPlanner(), memory=None)
        tokens = [event.token for event in agent.stream("Hello") if event.is_token]
        assert "".join(t for t in tokens if t) == "Hello sync"

    def test_sync_stream_honors_timeout(self) -> None:
        agent = Agent(model=MockModel(), planner=SlowPlanner(delay_seconds=0.2), memory=None)
        events = list(agent.stream("Hello", timeout=0.05))
        assert events[-1].type == "run_failed"


class TestTraceCollection:
    """Tests for trace collection."""

    def test_run_collects_trace_when_telemetry_enabled(self) -> None:
        agent = Agent(model=MockModel(), planner=MockPlanner(), memory=None)
        result = agent.run("Hello")
        assert result.trace

    def test_run_trace_events_are_redacted_by_default(self) -> None:
        @tool
        def secret_tool(payload: dict[str, Any]) -> str:
            return "ok"

        class ToolPlanner:
            async def plan(self, observation: Any, ctx: Any) -> Plan:
                return Plan(
                    id="plan-tool",
                    goal="call tool",
                    steps=[
                        PlanStep(
                            id="tool-step",
                            type=StepType.TOOL,
                            payload={
                                "tool": "secret_tool",
                                "args": {
                                    "payload": {
                                        "api_key": "sk-secret-123",
                                        "nested": {"private_key": "key-456"},
                                        "safe": "visible",
                                    }
                                },
                            },
                        ),
                        PlanStep(
                            id="terminal",
                            type=StepType.TERMINAL,
                            payload={"output": "done"},
                            dependencies=("tool-step",),
                        ),
                    ],
                )

        agent = Agent(tools=[secret_tool], model=MockModel(), planner=ToolPlanner(), memory=None)
        result = agent.run("Hello")

        tool_called = next(event for event in result.trace if event.type == "tool_called")
        payload = tool_called.data["args"]["payload"]
        assert payload["api_key"] == "[REDACTED]"
        assert payload["nested"]["private_key"] == "[REDACTED]"
        assert payload["safe"] == "visible"


# ---------------------------------------------------------------------------
# Single-execution constraint tests (8.7 / AD-008)
# ---------------------------------------------------------------------------


class TestSingleExecution:
    """Tests for single-execution constraint (AD-008)."""

    @pytest.mark.asyncio
    async def test_concurrent_run_raises(self) -> None:
        """Two concurrent run_async calls should fail."""

        class SlowPlanner:
            async def plan(self, observation: Any, ctx: Any) -> Plan:
                await asyncio.sleep(0.5)
                return Plan(
                    id="slow-plan",
                    goal="slow",
                    steps=[
                        PlanStep(
                            id="term",
                            type=StepType.TERMINAL,
                            payload={"output": "done"},
                        ),
                    ],
                )

        agent = Agent(model=MockModel(), planner=SlowPlanner(), memory=None)

        # Start first run
        task1 = asyncio.create_task(agent.run_async("first"))

        # Give it a moment to acquire the lock
        await asyncio.sleep(0.05)

        # Second run should fail
        with pytest.raises(RuntimeError, match="already executing"):
            await agent.run_async("second")

        # Clean up first task
        await task1


# ---------------------------------------------------------------------------
# String-based adapter resolution tests (Task 16.2)
# ---------------------------------------------------------------------------


class TestStringAdapterResolution:
    """Tests for string-based adapter construction through Agent."""

    def test_model_string_resolves_via_registry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Model string identifiers should resolve to correct adapter instances."""
        try:
            from axis_core.adapters.models import AnthropicModel
            from axis_core.engine.registry import model_registry
            from axis_core.engine.resolver import resolve_adapter
        except ImportError:
            pytest.skip("Anthropic package not installed")

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        model = resolve_adapter("claude-haiku", model_registry)
        assert isinstance(model, AnthropicModel)
        assert model.model_id == "claude-haiku-4-5-20251001"

    def test_all_string_adapters_resolve_via_registry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All adapter strings should resolve to correct instances via registry."""
        try:
            from axis_core.adapters.memory import EphemeralMemory
            from axis_core.adapters.models import AnthropicModel
            from axis_core.adapters.planners import SequentialPlanner
            from axis_core.engine.registry import (
                memory_registry,
                model_registry,
                planner_registry,
            )
            from axis_core.engine.resolver import resolve_adapter
        except ImportError:
            pytest.skip("Required packages not installed")

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        model = resolve_adapter("claude-sonnet", model_registry)
        planner = resolve_adapter("sequential", planner_registry)
        memory = resolve_adapter("ephemeral", memory_registry)

        assert isinstance(model, AnthropicModel)
        assert isinstance(planner, SequentialPlanner)
        assert isinstance(memory, EphemeralMemory)


class TestConfigDefaults:
    """Tests for config default fallback behavior."""

    def test_agent_uses_config_defaults(self) -> None:
        """Agent should fall back to config defaults when params not provided."""
        from axis_core.config import config

        # Temporarily override config defaults
        original_model = config.default_model
        original_planner = config.default_planner
        original_memory = config.default_memory

        try:
            config.default_model = "test-model"
            config.default_planner = "test-planner"
            config.default_memory = "test-memory"

            # Create agent without specifying adapters
            agent = Agent()

            # Should have picked up config defaults
            assert agent._model == "test-model"
            assert agent._planner == "test-planner"
            assert agent._memory == "test-memory"
        finally:
            # Restore original config
            config.default_model = original_model
            config.default_planner = original_planner
            config.default_memory = original_memory

    def test_explicit_params_override_defaults(self) -> None:
        """Explicit parameters should override config defaults."""
        from axis_core.config import config

        # Temporarily set config defaults
        original_model = config.default_model
        original_planner = config.default_planner

        try:
            config.default_model = "default-model"
            config.default_planner = "default-planner"

            # Create agent with explicit params
            agent = Agent(model="explicit-model", planner="explicit-planner")

            # Should use explicit values, not defaults
            assert agent._model == "explicit-model"
            assert agent._planner == "explicit-planner"
        finally:
            config.default_model = original_model
            config.default_planner = original_planner

    def test_none_param_overrides_default(self) -> None:
        """Explicitly passing None should override config default."""
        from axis_core.config import config

        original_model = config.default_model

        try:
            config.default_model = "default-model"

            # Explicitly pass None (user wants no model, not the default)
            agent = Agent(model=None, planner=None)

            # Should use None, not config default
            assert agent._model is None
            assert agent._planner is None
        finally:
            config.default_model = original_model
