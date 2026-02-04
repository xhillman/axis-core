"""Tests for lifecycle engine.

Tests cover:
- LifecycleEngine class instantiation
- Phase execution (_initialize, _observe, _plan, _act, _evaluate, _finalize)
- Main execution loop with cycle counting and budget checks
- Phase boundary telemetry emission
- Plan validation (AD-006)
- Step failure handling (AD-042)
- Memory persistence failures (AD-007)
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
from axis_core.engine.lifecycle import LifecycleEngine, Phase
from axis_core.errors import (
    BudgetError,
    CancelledError,
    ConfigError,
    PlanError,
)
from axis_core.protocols.model import ModelResponse, UsageStats
from axis_core.protocols.planner import Plan, PlanStep, StepType
from axis_core.protocols.telemetry import BufferMode, TraceEvent

# =============================================================================
# Mock adapters for testing
# =============================================================================


class MockModelAdapter:
    """Mock model adapter for testing."""

    def __init__(
        self,
        responses: list[ModelResponse] | None = None,
        stream_responses: list[list[dict]] | None = None,
    ):
        self.responses = responses or []
        self.stream_responses = stream_responses or []
        self._response_index = 0
        self.calls: list[dict] = []

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
        self.calls.append(
            {
                "messages": messages,
                "system": system,
                "tools": tools,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if self._response_index < len(self.responses):
            response = self.responses[self._response_index]
            self._response_index += 1
            return response
        # Default response
        return ModelResponse(
            content="Default response",
            tool_calls=None,
            usage=UsageStats(input_tokens=10, output_tokens=5, total_tokens=15),
            cost_usd=0.001,
        )

    async def stream(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Streaming not implemented in mock")

    def estimate_tokens(self, text: str) -> int:
        return len(text) // 4

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens + output_tokens) * 0.00001


class MockMemoryAdapter:
    """Mock memory adapter for testing."""

    def __init__(self, fail_on_store: bool = False):
        self._storage: dict[str, Any] = {}
        self._fail_on_store = fail_on_store

    @property
    def capabilities(self) -> set:
        from axis_core.protocols.memory import MemoryCapability

        return {MemoryCapability.KEYWORD_SEARCH}

    async def store(
        self,
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
        ttl: int | None = None,
        namespace: str | None = None,
    ) -> None:
        if self._fail_on_store:
            raise RuntimeError("Memory store failed")
        self._storage[key] = {"value": value, "metadata": metadata}

    async def retrieve(
        self,
        key: str,
        namespace: str | None = None,
    ) -> Any | None:
        item = self._storage.get(key)
        return item["value"] if item else None

    async def search(
        self,
        query: str,
        limit: int = 10,
        namespace: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list:
        return []

    async def delete(self, key: str, namespace: str | None = None) -> bool:
        if key in self._storage:
            del self._storage[key]
            return True
        return False

    async def clear(self, namespace: str | None = None) -> int:
        count = len(self._storage)
        self._storage.clear()
        return count


class MockPlanner:
    """Mock planner for testing."""

    def __init__(self, plans: list[Plan] | None = None, fail: bool = False):
        self.plans = plans or []
        self._plan_index = 0
        self._fail = fail
        self.calls: list[dict] = []

    async def plan(self, observation: Observation, ctx: RunContext) -> Plan:
        self.calls.append({"observation": observation, "ctx": ctx})
        if self._fail:
            raise PlanError(message="Planning failed")
        if self._plan_index < len(self.plans):
            plan = self.plans[self._plan_index]
            self._plan_index += 1
            return plan
        # Default terminal plan
        return Plan(
            id="default-plan",
            goal="Complete task",
            steps=(
                PlanStep(
                    id="terminal-1",
                    type=StepType.TERMINAL,
                    payload={"output": "Task completed"},
                ),
            ),
        )


class MockTelemetrySink:
    """Mock telemetry sink for testing."""

    def __init__(self):
        self.events: list[TraceEvent] = []
        self._flushed = False
        self._closed = False

    @property
    def buffering(self) -> BufferMode:
        return BufferMode.IMMEDIATE

    async def emit(self, event: TraceEvent) -> None:
        self.events.append(event)

    async def flush(self) -> None:
        self._flushed = True

    async def close(self) -> None:
        self._closed = True


class MockCancelToken:
    """Mock cancellation token for testing."""

    def __init__(self, cancelled: bool = False, reason: str | None = None):
        self._cancelled = cancelled
        self._reason = reason

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def cancel(self, reason: str | None = None) -> None:
        self._cancelled = True
        self._reason = reason


# =============================================================================
# Test fixtures
# =============================================================================


@pytest.fixture
def mock_model() -> MockModelAdapter:
    return MockModelAdapter()


@pytest.fixture
def mock_memory() -> MockMemoryAdapter:
    return MockMemoryAdapter()


@pytest.fixture
def mock_planner() -> MockPlanner:
    return MockPlanner()


@pytest.fixture
def mock_telemetry() -> MockTelemetrySink:
    return MockTelemetrySink()


@pytest.fixture
def tools() -> dict[str, Any]:
    """Sample tools for testing."""

    async def search(query: str) -> str:
        return f"Search results for: {query}"

    async def calculate(expression: str) -> float:
        return eval(expression)  # noqa: S307 - test only

    return {"search": search, "calculate": calculate}


# =============================================================================
# Phase enum tests
# =============================================================================


class TestPhaseEnum:
    """Tests for Phase enum."""

    def test_phase_values(self) -> None:
        """Phase enum should have all expected phases."""
        assert Phase.INITIALIZE.value == "initialize"
        assert Phase.OBSERVE.value == "observe"
        assert Phase.PLAN.value == "plan"
        assert Phase.ACT.value == "act"
        assert Phase.EVALUATE.value == "evaluate"
        assert Phase.FINALIZE.value == "finalize"


# =============================================================================
# LifecycleEngine instantiation tests
# =============================================================================


class TestLifecycleEngineInit:
    """Tests for LifecycleEngine initialization."""

    def test_create_engine(
        self,
        mock_model: MockModelAdapter,
        mock_memory: MockMemoryAdapter,
        mock_planner: MockPlanner,
        mock_telemetry: MockTelemetrySink,
        tools: dict[str, Any],
    ) -> None:
        """Engine should be created with required adapters."""
        engine = LifecycleEngine(
            model=mock_model,
            memory=mock_memory,
            planner=mock_planner,
            telemetry=[mock_telemetry],
            tools=tools,
        )
        assert engine is not None
        assert engine.model is mock_model
        assert engine.memory is mock_memory
        assert engine.planner is mock_planner

    def test_create_engine_minimal(
        self,
        mock_model: MockModelAdapter,
        mock_planner: MockPlanner,
    ) -> None:
        """Engine should be created with minimal config (no memory, no telemetry)."""
        engine = LifecycleEngine(
            model=mock_model,
            planner=mock_planner,
        )
        assert engine is not None
        assert engine.memory is None
        assert engine.telemetry == []


# =============================================================================
# Initialize phase tests
# =============================================================================


class TestInitializePhase:
    """Tests for _initialize phase."""

    @pytest.mark.asyncio
    async def test_initialize_creates_context(
        self,
        mock_model: MockModelAdapter,
        mock_planner: MockPlanner,
    ) -> None:
        """Initialize should create RunContext with proper fields."""
        engine = LifecycleEngine(model=mock_model, planner=mock_planner)

        ctx = await engine._initialize(
            input_text="Hello, world!",
            agent_id="test-agent",
            budget=Budget(),
        )

        assert ctx is not None
        assert isinstance(ctx, RunContext)
        assert ctx.run_id is not None
        assert ctx.agent_id == "test-agent"
        assert ctx.input.text == "Hello, world!"
        assert isinstance(ctx.state, RunState)
        assert ctx.cycle_count == 0

    @pytest.mark.asyncio
    async def test_initialize_with_context_dict(
        self,
        mock_model: MockModelAdapter,
        mock_planner: MockPlanner,
    ) -> None:
        """Initialize should accept and store context dict."""
        engine = LifecycleEngine(model=mock_model, planner=mock_planner)

        ctx = await engine._initialize(
            input_text="Test input",
            agent_id="test-agent",
            budget=Budget(),
            context={"user_id": "123", "session": "abc"},
        )

        assert ctx.context == {"user_id": "123", "session": "abc"}

    @pytest.mark.asyncio
    async def test_initialize_validates_empty_input(
        self,
        mock_model: MockModelAdapter,
        mock_planner: MockPlanner,
    ) -> None:
        """Initialize should reject empty input."""
        engine = LifecycleEngine(model=mock_model, planner=mock_planner)

        with pytest.raises(ConfigError, match="empty"):
            await engine._initialize(
                input_text="",
                agent_id="test-agent",
                budget=Budget(),
            )


# =============================================================================
# Observe phase tests
# =============================================================================


class TestObservePhase:
    """Tests for _observe phase."""

    @pytest.mark.asyncio
    async def test_observe_creates_observation(
        self,
        mock_model: MockModelAdapter,
        mock_memory: MockMemoryAdapter,
        mock_planner: MockPlanner,
    ) -> None:
        """Observe should create Observation from input and memory context."""
        engine = LifecycleEngine(
            model=mock_model,
            memory=mock_memory,
            planner=mock_planner,
        )

        ctx = await engine._initialize(
            input_text="What is the weather?",
            agent_id="test-agent",
            budget=Budget(),
        )

        observation = await engine._observe(ctx)

        assert observation is not None
        assert isinstance(observation, Observation)
        assert observation.input.text == "What is the weather?"
        assert observation.timestamp is not None

    @pytest.mark.asyncio
    async def test_observe_includes_memory_context(
        self,
        mock_model: MockModelAdapter,
        mock_memory: MockMemoryAdapter,
        mock_planner: MockPlanner,
    ) -> None:
        """Observe should include relevant memory in observation."""
        # Store some memory first
        await mock_memory.store("user_pref", {"theme": "dark"})

        engine = LifecycleEngine(
            model=mock_model,
            memory=mock_memory,
            planner=mock_planner,
        )

        ctx = await engine._initialize(
            input_text="Show my preferences",
            agent_id="test-agent",
            budget=Budget(),
        )

        observation = await engine._observe(ctx)

        # Memory context should be gathered
        assert isinstance(observation.memory_context, dict)

    @pytest.mark.asyncio
    async def test_observe_includes_previous_cycles(
        self,
        mock_model: MockModelAdapter,
        mock_planner: MockPlanner,
    ) -> None:
        """Observe should include summary of previous cycles."""
        engine = LifecycleEngine(model=mock_model, planner=mock_planner)

        ctx = await engine._initialize(
            input_text="Continue task",
            agent_id="test-agent",
            budget=Budget(),
        )

        # Simulate a previous cycle
        prev_cycle = CycleState(
            cycle_number=0,
            observation=Observation(
                input=NormalizedInput(text="First input", original="First input"),
            ),
            plan=Plan(
                id="plan-1",
                goal="Test",
                steps=(PlanStep(id="s1", type=StepType.TERMINAL, payload={}),),
            ),
            execution=ExecutionResult(),
            evaluation=EvalDecision(done=False, reason="Continue"),
            started_at=datetime.utcnow(),
            ended_at=datetime.utcnow(),
        )
        ctx.state.append_cycle(prev_cycle)
        ctx.cycle_count = 1

        observation = await engine._observe(ctx)

        assert len(observation.previous_cycles) == 1


# =============================================================================
# Plan phase tests (AD-006)
# =============================================================================


class TestPlanPhase:
    """Tests for _plan phase including AD-006 validation."""

    @pytest.mark.asyncio
    async def test_plan_generates_plan(
        self,
        mock_model: MockModelAdapter,
        mock_planner: MockPlanner,
        tools: dict[str, Any],
    ) -> None:
        """Plan should generate a valid Plan object."""
        engine = LifecycleEngine(
            model=mock_model,
            planner=mock_planner,
            tools=tools,
        )

        ctx = await engine._initialize(
            input_text="Search for something",
            agent_id="test-agent",
            budget=Budget(),
        )
        observation = await engine._observe(ctx)

        plan = await engine._plan(ctx, observation)

        assert plan is not None
        assert isinstance(plan, Plan)
        assert len(plan.steps) > 0

    @pytest.mark.asyncio
    async def test_plan_validates_tool_exists(
        self,
        mock_model: MockModelAdapter,
        tools: dict[str, Any],
    ) -> None:
        """Plan validation should fail if tool doesn't exist (AD-006)."""
        # Create plan with non-existent tool
        bad_plan = Plan(
            id="bad-plan",
            goal="Test",
            steps=(
                PlanStep(
                    id="step-1",
                    type=StepType.TOOL,
                    payload={"tool": "nonexistent_tool", "args": {}},
                ),
            ),
        )

        planner = MockPlanner(plans=[bad_plan])
        engine = LifecycleEngine(
            model=mock_model,
            planner=planner,
            tools=tools,
        )

        ctx = await engine._initialize(
            input_text="Test",
            agent_id="test-agent",
            budget=Budget(),
        )
        observation = await engine._observe(ctx)

        with pytest.raises(PlanError, match="tool.*not found|unknown tool"):
            await engine._plan(ctx, observation)

    @pytest.mark.asyncio
    async def test_plan_validates_dependencies(
        self,
        mock_model: MockModelAdapter,
        tools: dict[str, Any],
    ) -> None:
        """Plan validation should fail if dependencies are invalid (AD-006)."""
        # Create plan with invalid dependency
        bad_plan = Plan(
            id="bad-plan",
            goal="Test",
            steps=(
                PlanStep(
                    id="step-1",
                    type=StepType.TOOL,
                    payload={"tool": "search", "args": {"query": "test"}},
                    dependencies=("nonexistent-step",),
                ),
            ),
        )

        planner = MockPlanner(plans=[bad_plan])
        engine = LifecycleEngine(
            model=mock_model,
            planner=planner,
            tools=tools,
        )

        ctx = await engine._initialize(
            input_text="Test",
            agent_id="test-agent",
            budget=Budget(),
        )
        observation = await engine._observe(ctx)

        with pytest.raises(PlanError, match="dependency|invalid"):
            await engine._plan(ctx, observation)


# =============================================================================
# Act phase tests (AD-042)
# =============================================================================


class TestActPhase:
    """Tests for _act phase including AD-042 step failure handling."""

    @pytest.mark.asyncio
    async def test_act_executes_tool_step(
        self,
        mock_model: MockModelAdapter,
        mock_planner: MockPlanner,
        tools: dict[str, Any],
    ) -> None:
        """Act should execute tool steps and return results."""
        plan = Plan(
            id="plan-1",
            goal="Search",
            steps=(
                PlanStep(
                    id="step-1",
                    type=StepType.TOOL,
                    payload={"tool": "search", "args": {"query": "weather"}},
                ),
                PlanStep(
                    id="terminal",
                    type=StepType.TERMINAL,
                    payload={},
                    dependencies=("step-1",),
                ),
            ),
        )

        engine = LifecycleEngine(
            model=mock_model,
            planner=mock_planner,
            tools=tools,
        )

        ctx = await engine._initialize(
            input_text="Search for weather",
            agent_id="test-agent",
            budget=Budget(),
        )
        # Observe phase (populates context state)
        await engine._observe(ctx)

        result = await engine._act(ctx, plan)

        assert isinstance(result, ExecutionResult)
        assert "step-1" in result.results
        assert "weather" in result.results["step-1"]

    @pytest.mark.asyncio
    async def test_act_continues_independent_steps_on_failure(
        self,
        mock_model: MockModelAdapter,
        mock_planner: MockPlanner,
    ) -> None:
        """Act should continue executing independent steps when one fails (AD-042)."""

        async def failing_tool(x: int) -> int:
            raise ValueError("Tool failed")

        async def working_tool(x: int) -> int:
            return x * 2

        tools = {"failing": failing_tool, "working": working_tool}

        plan = Plan(
            id="plan-1",
            goal="Test",
            steps=(
                PlanStep(
                    id="fail-step",
                    type=StepType.TOOL,
                    payload={"tool": "failing", "args": {"x": 1}},
                ),
                PlanStep(
                    id="work-step",
                    type=StepType.TOOL,
                    payload={"tool": "working", "args": {"x": 5}},
                ),
                PlanStep(
                    id="terminal",
                    type=StepType.TERMINAL,
                    payload={},
                ),
            ),
        )

        engine = LifecycleEngine(
            model=mock_model,
            planner=mock_planner,
            tools=tools,
        )

        ctx = await engine._initialize(
            input_text="Test",
            agent_id="test-agent",
            budget=Budget(),
        )

        result = await engine._act(ctx, plan)

        # Both steps should be attempted - failure doesn't stop independent steps
        assert "fail-step" in result.errors
        assert "work-step" in result.results
        assert result.results["work-step"] == 10

    @pytest.mark.asyncio
    async def test_act_skips_dependent_steps_on_failure(
        self,
        mock_model: MockModelAdapter,
        mock_planner: MockPlanner,
    ) -> None:
        """Act should skip dependent steps when their dependency fails (AD-042)."""

        async def failing_tool(x: int) -> int:
            raise ValueError("Tool failed")

        async def dependent_tool(x: int) -> int:
            return x * 2

        tools = {"failing": failing_tool, "dependent": dependent_tool}

        plan = Plan(
            id="plan-1",
            goal="Test",
            steps=(
                PlanStep(
                    id="fail-step",
                    type=StepType.TOOL,
                    payload={"tool": "failing", "args": {"x": 1}},
                ),
                PlanStep(
                    id="dep-step",
                    type=StepType.TOOL,
                    payload={"tool": "dependent", "args": {"x": 5}},
                    dependencies=("fail-step",),
                ),
                PlanStep(
                    id="terminal",
                    type=StepType.TERMINAL,
                    payload={},
                ),
            ),
        )

        engine = LifecycleEngine(
            model=mock_model,
            planner=mock_planner,
            tools=tools,
        )

        ctx = await engine._initialize(
            input_text="Test",
            agent_id="test-agent",
            budget=Budget(),
        )

        result = await engine._act(ctx, plan)

        # Dependent step should be skipped
        assert "fail-step" in result.errors
        assert "dep-step" in result.skipped


# =============================================================================
# Evaluate phase tests
# =============================================================================


class TestEvaluatePhase:
    """Tests for _evaluate phase."""

    @pytest.mark.asyncio
    async def test_evaluate_done_on_terminal_plan(
        self,
        mock_model: MockModelAdapter,
        mock_planner: MockPlanner,
    ) -> None:
        """Evaluate should return done=True when plan has TERMINAL step."""
        plan = Plan(
            id="plan-1",
            goal="Complete",
            steps=(
                PlanStep(
                    id="terminal",
                    type=StepType.TERMINAL,
                    payload={"output": "Done"},
                ),
            ),
        )

        engine = LifecycleEngine(model=mock_model, planner=mock_planner)

        ctx = await engine._initialize(
            input_text="Test",
            agent_id="test-agent",
            budget=Budget(),
        )

        execution = ExecutionResult()
        decision = await engine._evaluate(ctx, plan, execution)

        assert decision.done is True
        assert decision.error is None

    @pytest.mark.asyncio
    async def test_evaluate_done_on_budget_exhausted(
        self,
        mock_model: MockModelAdapter,
        mock_planner: MockPlanner,
    ) -> None:
        """Evaluate should return done=True with error when budget exhausted."""
        plan = Plan(
            id="plan-1",
            goal="Continue",
            steps=(
                PlanStep(id="step-1", type=StepType.MODEL, payload={}),
            ),
        )

        engine = LifecycleEngine(model=mock_model, planner=mock_planner)

        ctx = await engine._initialize(
            input_text="Test",
            agent_id="test-agent",
            budget=Budget(max_cycles=1),
        )
        ctx.state.budget_state.cycles = 1  # Exhaust cycles

        execution = ExecutionResult()
        decision = await engine._evaluate(ctx, plan, execution)

        assert decision.done is True
        assert isinstance(decision.error, BudgetError)

    @pytest.mark.asyncio
    async def test_evaluate_done_on_cancellation(
        self,
        mock_model: MockModelAdapter,
        mock_planner: MockPlanner,
    ) -> None:
        """Evaluate should return done=True when cancelled."""
        plan = Plan(
            id="plan-1",
            goal="Continue",
            steps=(PlanStep(id="step-1", type=StepType.MODEL, payload={}),),
        )

        engine = LifecycleEngine(model=mock_model, planner=mock_planner)

        ctx = await engine._initialize(
            input_text="Test",
            agent_id="test-agent",
            budget=Budget(),
        )
        ctx.cancel_token = MockCancelToken(cancelled=True, reason="User cancelled")

        execution = ExecutionResult()
        decision = await engine._evaluate(ctx, plan, execution)

        assert decision.done is True
        assert isinstance(decision.error, CancelledError)


# =============================================================================
# Finalize phase tests (AD-007)
# =============================================================================


class TestFinalizePhase:
    """Tests for _finalize phase including AD-007 memory failure handling."""

    @pytest.mark.asyncio
    async def test_finalize_returns_result(
        self,
        mock_model: MockModelAdapter,
        mock_memory: MockMemoryAdapter,
        mock_planner: MockPlanner,
    ) -> None:
        """Finalize should return final result."""
        engine = LifecycleEngine(
            model=mock_model,
            memory=mock_memory,
            planner=mock_planner,
        )

        ctx = await engine._initialize(
            input_text="Test",
            agent_id="test-agent",
            budget=Budget(),
        )
        ctx.state.output = "Final output"

        result = await engine._finalize(ctx)

        # Result dict should contain output
        assert result["output"] == "Final output"
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_finalize_handles_memory_failure(
        self,
        mock_model: MockModelAdapter,
        mock_planner: MockPlanner,
    ) -> None:
        """Finalize should succeed but record error on memory failure (AD-007)."""
        failing_memory = MockMemoryAdapter(fail_on_store=True)

        engine = LifecycleEngine(
            model=mock_model,
            memory=failing_memory,
            planner=mock_planner,
        )

        ctx = await engine._initialize(
            input_text="Test",
            agent_id="test-agent",
            budget=Budget(),
        )
        ctx.state.output = "Final output"

        result = await engine._finalize(ctx)

        # Should still succeed
        assert result["success"] is True
        assert result["output"] == "Final output"
        # But memory_error should be recorded
        assert result["memory_error"] is not None

    @pytest.mark.asyncio
    async def test_finalize_flushes_telemetry(
        self,
        mock_model: MockModelAdapter,
        mock_planner: MockPlanner,
        mock_telemetry: MockTelemetrySink,
    ) -> None:
        """Finalize should flush and close telemetry sinks."""
        engine = LifecycleEngine(
            model=mock_model,
            planner=mock_planner,
            telemetry=[mock_telemetry],
        )

        ctx = await engine._initialize(
            input_text="Test",
            agent_id="test-agent",
            budget=Budget(),
        )
        ctx.state.output = "Done"

        await engine._finalize(ctx)

        assert mock_telemetry._flushed is True
        assert mock_telemetry._closed is True


# =============================================================================
# Main execution loop tests
# =============================================================================


class TestExecutionLoop:
    """Tests for main execution loop."""

    @pytest.mark.asyncio
    async def test_execute_single_cycle(
        self,
        mock_model: MockModelAdapter,
        mock_planner: MockPlanner,
    ) -> None:
        """Execute should complete a single cycle and return result."""
        # Plan that terminates immediately
        terminal_plan = Plan(
            id="plan-1",
            goal="Complete",
            steps=(
                PlanStep(
                    id="terminal",
                    type=StepType.TERMINAL,
                    payload={"output": "Task done"},
                ),
            ),
        )
        planner = MockPlanner(plans=[terminal_plan])

        engine = LifecycleEngine(model=mock_model, planner=planner)

        result = await engine.execute(
            input_text="Do something",
            agent_id="test-agent",
            budget=Budget(),
        )

        assert result["success"] is True
        assert result["cycles_completed"] == 1

    @pytest.mark.asyncio
    async def test_execute_multiple_cycles(
        self,
        mock_model: MockModelAdapter,
    ) -> None:
        """Execute should run multiple cycles until terminal."""
        # Plans: first continues, second terminates
        continue_plan = Plan(
            id="plan-1",
            goal="Continue",
            steps=(PlanStep(id="step-1", type=StepType.MODEL, payload={}),),
        )
        terminal_plan = Plan(
            id="plan-2",
            goal="Complete",
            steps=(
                PlanStep(
                    id="terminal",
                    type=StepType.TERMINAL,
                    payload={"output": "Done"},
                ),
            ),
        )

        planner = MockPlanner(plans=[continue_plan, terminal_plan])
        engine = LifecycleEngine(model=mock_model, planner=planner)

        result = await engine.execute(
            input_text="Multi-step task",
            agent_id="test-agent",
            budget=Budget(max_cycles=10),
        )

        assert result["success"] is True
        assert result["cycles_completed"] == 2

    @pytest.mark.asyncio
    async def test_execute_respects_max_cycles(
        self,
        mock_model: MockModelAdapter,
    ) -> None:
        """Execute should stop at max_cycles even if not terminal."""
        # Plan that never terminates
        continue_plan = Plan(
            id="plan-1",
            goal="Continue",
            steps=(PlanStep(id="step-1", type=StepType.MODEL, payload={}),),
        )

        planner = MockPlanner(plans=[continue_plan] * 10)
        engine = LifecycleEngine(model=mock_model, planner=planner)

        result = await engine.execute(
            input_text="Endless task",
            agent_id="test-agent",
            budget=Budget(max_cycles=3),
        )

        assert result["success"] is False
        assert result["cycles_completed"] == 3
        assert isinstance(result["error"], BudgetError)


# =============================================================================
# Telemetry tests
# =============================================================================


class TestTelemetryEmission:
    """Tests for phase boundary telemetry emission."""

    @pytest.mark.asyncio
    async def test_emits_phase_events(
        self,
        mock_model: MockModelAdapter,
        mock_planner: MockPlanner,
        mock_telemetry: MockTelemetrySink,
    ) -> None:
        """Engine should emit phase_entered and phase_exited events."""
        terminal_plan = Plan(
            id="plan-1",
            goal="Complete",
            steps=(
                PlanStep(
                    id="terminal",
                    type=StepType.TERMINAL,
                    payload={"output": "Done"},
                ),
            ),
        )
        planner = MockPlanner(plans=[terminal_plan])

        engine = LifecycleEngine(
            model=mock_model,
            planner=planner,
            telemetry=[mock_telemetry],
        )

        await engine.execute(
            input_text="Test",
            agent_id="test-agent",
            budget=Budget(),
        )

        # Check for phase events
        event_types = [e.type for e in mock_telemetry.events]
        assert "phase_entered" in event_types
        assert "phase_exited" in event_types

    @pytest.mark.asyncio
    async def test_emits_run_events(
        self,
        mock_model: MockModelAdapter,
        mock_planner: MockPlanner,
        mock_telemetry: MockTelemetrySink,
    ) -> None:
        """Engine should emit run_started and run_completed events."""
        terminal_plan = Plan(
            id="plan-1",
            goal="Complete",
            steps=(
                PlanStep(
                    id="terminal",
                    type=StepType.TERMINAL,
                    payload={"output": "Done"},
                ),
            ),
        )
        planner = MockPlanner(plans=[terminal_plan])

        engine = LifecycleEngine(
            model=mock_model,
            planner=planner,
            telemetry=[mock_telemetry],
        )

        await engine.execute(
            input_text="Test",
            agent_id="test-agent",
            budget=Budget(),
        )

        event_types = [e.type for e in mock_telemetry.events]
        assert "run_started" in event_types
        assert "run_completed" in event_types

    @pytest.mark.asyncio
    async def test_emits_cycle_events(
        self,
        mock_model: MockModelAdapter,
        mock_planner: MockPlanner,
        mock_telemetry: MockTelemetrySink,
    ) -> None:
        """Engine should emit cycle_started and cycle_completed events."""
        terminal_plan = Plan(
            id="plan-1",
            goal="Complete",
            steps=(
                PlanStep(
                    id="terminal",
                    type=StepType.TERMINAL,
                    payload={"output": "Done"},
                ),
            ),
        )
        planner = MockPlanner(plans=[terminal_plan])

        engine = LifecycleEngine(
            model=mock_model,
            planner=planner,
            telemetry=[mock_telemetry],
        )

        await engine.execute(
            input_text="Test",
            agent_id="test-agent",
            budget=Budget(),
        )

        event_types = [e.type for e in mock_telemetry.events]
        assert "cycle_started" in event_types
        assert "cycle_completed" in event_types


# =============================================================================
# Adapter resolution tests (Task 16.2, 16.4)
# =============================================================================


class TestAdapterResolution:
    """Tests for string-to-adapter resolution in LifecycleEngine."""

    def test_model_string_resolution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LifecycleEngine should resolve model strings to instances."""
        try:
            from axis_core.adapters.models import AnthropicModel
        except ImportError:
            pytest.skip("Anthropic package not installed")

        # Set dummy API key for test
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        # Pass string identifier
        engine = LifecycleEngine(
            model="claude-haiku",
            planner=MockPlanner(),
        )

        # Should be resolved to AnthropicModel instance
        assert isinstance(engine.model, AnthropicModel)
        assert engine.model.model_id == "claude-haiku-4-5-20251001"

    def test_model_instance_passthrough(self) -> None:
        """LifecycleEngine should accept model instances directly."""
        mock_model = MockModelAdapter()

        engine = LifecycleEngine(
            model=mock_model,
            planner=MockPlanner(),
        )

        # Should keep the same instance
        assert engine.model is mock_model

    def test_planner_string_resolution(self) -> None:
        """LifecycleEngine should resolve planner strings to instances."""
        from axis_core.adapters.planners import SequentialPlanner

        mock_model = MockModelAdapter()

        # Pass string identifier
        engine = LifecycleEngine(
            model=mock_model,
            planner="sequential",
        )

        # Should be resolved to SequentialPlanner instance
        assert isinstance(engine.planner, SequentialPlanner)

    def test_memory_string_resolution(self) -> None:
        """LifecycleEngine should resolve memory strings to instances."""
        from axis_core.adapters.memory import EphemeralMemory

        mock_model = MockModelAdapter()

        # Pass string identifier
        engine = LifecycleEngine(
            model=mock_model,
            planner=MockPlanner(),
            memory="ephemeral",
        )

        # Should be resolved to EphemeralMemory instance
        assert isinstance(engine.memory, EphemeralMemory)

    def test_unknown_model_raises_config_error(self) -> None:
        """LifecycleEngine should raise ConfigError for unknown model strings."""
        with pytest.raises(ConfigError, match="Unknown adapter 'nonexistent-model'"):
            LifecycleEngine(
                model="nonexistent-model",
                planner=MockPlanner(),
            )

    def test_none_values_preserved(self) -> None:
        """LifecycleEngine should preserve None for optional adapters."""
        mock_model = MockModelAdapter()

        engine = LifecycleEngine(
            model=mock_model,
            planner=MockPlanner(),
            memory=None,
        )

        assert engine.memory is None


# =============================================================================
# Tool Schema Tests (TDD for tool integration)
# =============================================================================


class TestToolManifestExtraction:
    """Tests for tool manifest extraction from registered tools."""

    def test_get_tool_manifests_empty_tools(self) -> None:
        """_get_tool_manifests should return empty list when no tools."""
        mock_model = MockModelAdapter()
        planner = MockPlanner()

        engine = LifecycleEngine(
            model=mock_model,
            planner=planner,
            tools={},
        )

        manifests = engine._get_tool_manifests()
        assert manifests == []

    def test_get_tool_manifests_with_single_tool(self) -> None:
        """_get_tool_manifests should return ToolManifest objects."""
        from axis_core.tool import ToolManifest, tool

        @tool
        def get_weather(city: str) -> str:
            """Get the weather for a city."""
            return f"Weather in {city}"

        mock_model = MockModelAdapter()
        planner = MockPlanner()

        engine = LifecycleEngine(
            model=mock_model,
            planner=planner,
            tools={"get_weather": get_weather},
        )

        manifests = engine._get_tool_manifests()

        assert len(manifests) == 1
        manifest = manifests[0]
        assert isinstance(manifest, ToolManifest)
        assert manifest.name == "get_weather"
        assert manifest.description == "Get the weather for a city."
        assert manifest.input_schema["type"] == "object"
        assert "city" in manifest.input_schema["properties"]
        assert manifest.input_schema["properties"]["city"]["type"] == "string"
        assert "city" in manifest.input_schema["required"]

    def test_get_tool_manifests_with_multiple_tools(self) -> None:
        """_get_tool_manifests should return manifests for all tools."""
        from axis_core.tool import ToolManifest, tool

        @tool
        def get_weather(city: str) -> str:
            """Get weather."""
            return "sunny"

        @tool
        def get_time(timezone: str) -> str:
            """Get current time."""
            return "12:00"

        mock_model = MockModelAdapter()
        planner = MockPlanner()

        engine = LifecycleEngine(
            model=mock_model,
            planner=planner,
            tools={"get_weather": get_weather, "get_time": get_time},
        )

        manifests = engine._get_tool_manifests()

        assert len(manifests) == 2
        assert all(isinstance(m, ToolManifest) for m in manifests)
        tool_names = {m.name for m in manifests}
        assert tool_names == {"get_weather", "get_time"}

    def test_get_tool_manifests_with_optional_params(self) -> None:
        """_get_tool_manifests should preserve optional parameter info."""
        from axis_core.tool import ToolManifest, tool

        @tool
        def search(query: str, limit: int = 10) -> str:
            """Search for something."""
            return f"Results for {query}"

        mock_model = MockModelAdapter()
        planner = MockPlanner()

        engine = LifecycleEngine(
            model=mock_model,
            planner=planner,
            tools={"search": search},
        )

        manifests = engine._get_tool_manifests()

        assert len(manifests) == 1
        manifest = manifests[0]
        assert isinstance(manifest, ToolManifest)
        assert "query" in manifest.input_schema["required"]
        assert "limit" not in manifest.input_schema["required"]
        assert "limit" in manifest.input_schema["properties"]

    @pytest.mark.asyncio
    async def test_model_step_passes_tool_manifests(self) -> None:
        """MODEL step execution should pass ToolManifest objects to model.complete()."""
        from axis_core.tool import ToolManifest, tool

        @tool
        def test_tool(arg: str) -> str:
            """Test tool."""
            return "result"

        mock_model = MockModelAdapter(
            responses=[
                ModelResponse(
                    content="Response with tool call",
                    tool_calls=None,
                    usage=UsageStats(input_tokens=10, output_tokens=20, total_tokens=30),
                    cost_usd=0.001,
                )
            ]
        )
        planner = MockPlanner()

        engine = LifecycleEngine(
            model=mock_model,
            planner=planner,
            tools={"test_tool": test_tool},
        )

        ctx = await engine._initialize(
            input_text="test input",
            agent_id="test-agent",
            budget=Budget(),
        )

        # Create a MODEL step
        step = PlanStep(
            id="model_1",
            type=StepType.MODEL,
            payload={},
            dependencies=None,
            retry_policy=None,
        )

        # Execute the step
        await engine._execute_model_step(ctx, step)

        # Verify model.complete() was called with ToolManifest objects (protocol layer)
        assert len(mock_model.calls) == 1
        call = mock_model.calls[0]
        assert "tools" in call
        assert call["tools"] is not None
        assert len(call["tools"]) == 1
        # Should be ToolManifest object, not dict
        assert isinstance(call["tools"][0], ToolManifest)
        assert call["tools"][0].name == "test_tool"

    @pytest.mark.asyncio
    async def test_model_step_with_no_tools(self) -> None:
        """MODEL step execution with no tools should pass None for tools."""
        mock_model = MockModelAdapter(
            responses=[
                ModelResponse(
                    content="Response",
                    tool_calls=None,
                    usage=UsageStats(input_tokens=10, output_tokens=20, total_tokens=30),
                    cost_usd=0.001,
                )
            ]
        )
        planner = MockPlanner()

        engine = LifecycleEngine(
            model=mock_model,
            planner=planner,
            tools={},  # No tools
        )

        ctx = await engine._initialize(
            input_text="test input",
            agent_id="test-agent",
            budget=Budget(),
        )

        step = PlanStep(
            id="model_1",
            type=StepType.MODEL,
            payload={},
            dependencies=None,
            retry_policy=None,
        )

        await engine._execute_model_step(ctx, step)

        # Verify model.complete() was called without tools
        assert len(mock_model.calls) == 1
        call = mock_model.calls[0]
        # Tools should be None or not in call when no tools available
        assert call.get("tools") is None


# =============================================================================
# Model fallback tests (Task 15.0, AD-013)
# =============================================================================


class MockFailingModelAdapter(MockModelAdapter):
    """Mock model adapter that fails with recoverable errors."""

    def __init__(
        self,
        fail_count: int = 1,
        error_type: str = "RateLimitError",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.fail_count = fail_count
        self.error_type = error_type
        self.call_count = 0

    async def complete(self, *args: Any, **kwargs: Any) -> ModelResponse:
        self.call_count += 1
        if self.call_count <= self.fail_count:
            # Simulate recoverable error with proper exception class name
            # ModelError.from_exception() checks exception.__class__.__name__
            if self.error_type == "RateLimitError":
                # Create an exception with the proper class name
                exc_class = type("RateLimitError", (Exception,), {})
                raise exc_class("Rate limit exceeded")
            elif self.error_type == "ConnectionError":
                exc_class = type("ConnectionError", (Exception,), {})
                raise exc_class("Connection failed")
            else:
                exc_class = type(self.error_type, (Exception,), {})
                raise exc_class(f"{self.error_type} occurred")
        return await super().complete(*args, **kwargs)


class TestModelFallback:
    """Tests for model fallback system (AD-013)."""

    @pytest.mark.asyncio
    async def test_fallback_on_recoverable_error(self) -> None:
        """Should fallback to second model on recoverable error."""

        # Primary model fails with recoverable error
        primary = MockFailingModelAdapter(
            fail_count=999,  # Always fails
            error_type="RateLimitError",
        )

        # Fallback model succeeds
        fallback = MockModelAdapter(
            responses=[
                ModelResponse(
                    content="Fallback response",
                    tool_calls=None,
                    usage=UsageStats(input_tokens=10, output_tokens=5, total_tokens=15),
                    cost_usd=0.001,
                )
            ]
        )

        planner = MockPlanner()
        engine = LifecycleEngine(
            model=primary,
            planner=planner,
            fallback=[fallback],
        )

        ctx = await engine._initialize(
            input_text="test input",
            agent_id="test-agent",
            budget=Budget(),
        )

        # Call the fallback method
        response = await engine._call_model_with_fallback(
            ctx=ctx,
            messages=[{"role": "user", "content": "test"}],
            system=None,
            tools=None,
        )

        assert response.content == "Fallback response"
        assert primary.call_count == 1  # Primary tried once
        assert len(fallback.calls) == 1  # Fallback used

    @pytest.mark.asyncio
    async def test_fallback_preserves_request_parameters(self) -> None:
        """Should pass exact same parameters to fallback (AD-013)."""
        primary = MockFailingModelAdapter(fail_count=999)
        fallback = MockModelAdapter(
            responses=[
                ModelResponse(
                    content="OK",
                    tool_calls=None,
                    usage=UsageStats(input_tokens=5, output_tokens=5, total_tokens=10),
                    cost_usd=0.0005,
                )
            ]
        )

        planner = MockPlanner()
        engine = LifecycleEngine(
            model=primary,
            planner=planner,
            fallback=[fallback],
        )

        ctx = await engine._initialize(
            input_text="test",
            agent_id="test-agent",
            budget=Budget(),
        )

        test_messages = [{"role": "user", "content": "original request"}]
        test_system = "You are a helpful assistant"
        test_tools = [{"name": "tool1"}]

        await engine._call_model_with_fallback(
            ctx=ctx,
            messages=test_messages,
            system=test_system,
            tools=test_tools,
        )

        # Verify fallback received exact same parameters
        assert len(fallback.calls) == 1
        call = fallback.calls[0]
        assert call["messages"] == test_messages
        assert call["system"] == test_system
        assert call["tools"] == test_tools

    @pytest.mark.asyncio
    async def test_fallback_chain_multiple_models(self) -> None:
        """Should try multiple fallback models in order."""
        primary = MockFailingModelAdapter(fail_count=999)
        fallback1 = MockFailingModelAdapter(fail_count=999)
        fallback2 = MockModelAdapter(
            responses=[
                ModelResponse(
                    content="Third model works",
                    tool_calls=None,
                    usage=UsageStats(input_tokens=10, output_tokens=5, total_tokens=15),
                    cost_usd=0.001,
                )
            ]
        )

        planner = MockPlanner()
        engine = LifecycleEngine(
            model=primary,
            planner=planner,
            fallback=[fallback1, fallback2],
        )

        ctx = await engine._initialize(
            input_text="test",
            agent_id="test-agent",
            budget=Budget(),
        )

        response = await engine._call_model_with_fallback(
            ctx=ctx,
            messages=[{"role": "user", "content": "test"}],
            system=None,
            tools=None,
        )

        assert response.content == "Third model works"
        assert primary.call_count == 1
        assert fallback1.call_count == 1
        assert len(fallback2.calls) == 1

    @pytest.mark.asyncio
    async def test_fallback_fails_if_all_models_fail(self) -> None:
        """Should raise ModelError with cause chain if all models fail."""
        from axis_core.errors import ModelError

        primary = MockFailingModelAdapter(fail_count=999)
        fallback1 = MockFailingModelAdapter(fail_count=999)
        fallback2 = MockFailingModelAdapter(fail_count=999)

        planner = MockPlanner()
        engine = LifecycleEngine(
            model=primary,
            planner=planner,
            fallback=[fallback1, fallback2],
        )

        ctx = await engine._initialize(
            input_text="test",
            agent_id="test-agent",
            budget=Budget(),
        )

        with pytest.raises(ModelError, match="All models failed"):
            await engine._call_model_with_fallback(
                ctx=ctx,
                messages=[{"role": "user", "content": "test"}],
                system=None,
                tools=None,
            )

        # Verify all models were tried
        assert primary.call_count == 1
        assert fallback1.call_count == 1
        assert fallback2.call_count == 1

    @pytest.mark.asyncio
    async def test_no_fallback_on_non_recoverable_error(self) -> None:
        """Should not fallback on non-recoverable errors like ValidationError."""
        from axis_core.errors import ModelError

        class MockNonRecoverableModel(MockModelAdapter):
            async def complete(self, *args: Any, **kwargs: Any) -> ModelResponse:
                # Simulate non-recoverable error
                raise ValueError("Invalid parameters")

        primary = MockNonRecoverableModel()
        fallback = MockModelAdapter(
            responses=[
                ModelResponse(
                    content="Should not reach here",
                    tool_calls=None,
                    usage=UsageStats(input_tokens=1, output_tokens=1, total_tokens=2),
                    cost_usd=0.0001,
                )
            ]
        )

        planner = MockPlanner()
        engine = LifecycleEngine(
            model=primary,
            planner=planner,
            fallback=[fallback],
        )

        ctx = await engine._initialize(
            input_text="test",
            agent_id="test-agent",
            budget=Budget(),
        )

        with pytest.raises(ModelError):
            await engine._call_model_with_fallback(
                ctx=ctx,
                messages=[{"role": "user", "content": "test"}],
                system=None,
                tools=None,
            )

        # Fallback should not have been called
        assert len(fallback.calls) == 0

    @pytest.mark.asyncio
    async def test_fallback_emits_telemetry_event(self) -> None:
        """Should emit model_fallback telemetry event on fallback."""
        primary = MockFailingModelAdapter(fail_count=999)
        fallback = MockModelAdapter(
            responses=[
                ModelResponse(
                    content="OK",
                    tool_calls=None,
                    usage=UsageStats(input_tokens=5, output_tokens=5, total_tokens=10),
                    cost_usd=0.0005,
                )
            ]
        )

        # Mock telemetry sink
        events: list[TraceEvent] = []

        class MockTelemetrySink:
            @property
            def buffering(self) -> BufferMode:
                return BufferMode.IMMEDIATE

            async def emit(self, event: TraceEvent) -> None:
                events.append(event)

            async def flush(self) -> None:
                pass

            async def close(self) -> None:
                pass

        telemetry_sink = MockTelemetrySink()
        planner = MockPlanner()

        engine = LifecycleEngine(
            model=primary,
            planner=planner,
            fallback=[fallback],
            telemetry=[telemetry_sink],
        )

        ctx = await engine._initialize(
            input_text="test",
            agent_id="test-agent",
            budget=Budget(),
        )

        await engine._call_model_with_fallback(
            ctx=ctx,
            messages=[{"role": "user", "content": "test"}],
            system=None,
            tools=None,
        )

        # Check for model_fallback event
        fallback_events = [e for e in events if e.type == "model_fallback"]
        assert len(fallback_events) == 1
        assert fallback_events[0].data.get("from_model") == "mock-model"
        assert fallback_events[0].data.get("to_model") == "mock-model"

    @pytest.mark.asyncio
    async def test_planner_fallback_emits_telemetry_event(self) -> None:
        """Should emit planner_fallback telemetry event when planner falls back (AD-016)."""
        model = MockModelAdapter(
            responses=[
                ModelResponse(
                    content="Done",
                    tool_calls=None,
                    usage=UsageStats(input_tokens=5, output_tokens=5, total_tokens=10),
                    cost_usd=0.0005,
                )
            ]
        )

        events: list[TraceEvent] = []

        class MockTelemetrySink:
            @property
            def buffering(self) -> BufferMode:
                return BufferMode.IMMEDIATE

            async def emit(self, event: TraceEvent) -> None:
                events.append(event)

            async def flush(self) -> None:
                pass

            async def close(self) -> None:
                pass

        telemetry_sink = MockTelemetrySink()

        # Planner that returns a plan with fallback metadata (simulates AutoPlanner fallback)
        fallback_plan = Plan(
            id="fallback-plan",
            goal="Complete task",
            steps=(
                PlanStep(
                    id="terminal-1",
                    type=StepType.TERMINAL,
                    payload={"output": "Task completed"},
                ),
            ),
            confidence=0.5,
            metadata={
                "planner": "sequential",
                "fallback": True,
                "fallback_reason": "Model call failed: API error",
            },
        )
        planner = MockPlanner(plans=[fallback_plan])

        engine = LifecycleEngine(
            model=model,
            planner=planner,
            telemetry=[telemetry_sink],
        )

        ctx = await engine._initialize(
            input_text="test",
            agent_id="test-agent",
            budget=Budget(),
        )

        observation = Observation(
            input=NormalizedInput(text="test", original="test"),
            response=None,
            goal="Test planner fallback telemetry",
        )

        await engine._plan(ctx, observation)

        # Check for planner_fallback event
        fallback_events = [e for e in events if e.type == "planner_fallback"]
        assert len(fallback_events) == 1
        assert fallback_events[0].data.get("original_planner") == "sequential"
        assert "API error" in fallback_events[0].data.get("reason", "")
