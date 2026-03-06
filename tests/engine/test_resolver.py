"""Tests for engine/resolver.py string-to-adapter resolution."""

from __future__ import annotations

from typing import Any

import pytest

from axis_core.engine.resolver import resolve_adapter
from axis_core.errors import ConfigError
from axis_core.protocols.memory import MemoryCapability, MemoryItem
from axis_core.protocols.model import ModelChunk, ModelResponse, UsageStats
from axis_core.protocols.planner import Plan, PlanStep, StepType

# ---------------------------------------------------------------------------
# Mock adapters for testing
# ---------------------------------------------------------------------------


class FakeModelAdapter:
    """Protocol-shaped model adapter for resolver validation tests."""

    def __init__(self, name: str = "mock-model") -> None:
        self.name = name

    @property
    def model_id(self) -> str:
        return self.name

    async def complete(self, messages: Any, **kwargs: Any) -> ModelResponse:
        return ModelResponse(
            content="ok",
            tool_calls=None,
            usage=UsageStats(input_tokens=1, output_tokens=1, total_tokens=2),
            cost_usd=0.0,
        )

    async def stream(self, messages: Any, **kwargs: Any) -> Any:
        yield ModelChunk(content="ok", is_final=True)

    def estimate_tokens(self, text: str) -> int:
        return len(text.split())

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return float(input_tokens + output_tokens)


class FakeMemoryAdapter:
    """Protocol-shaped memory adapter for resolver validation tests."""

    @property
    def capabilities(self) -> set[MemoryCapability]:
        return {MemoryCapability.KEYWORD_SEARCH}

    async def store(self, key: str, value: Any, **kwargs: Any) -> None:
        return None

    async def retrieve(self, key: str, namespace: str | None = None) -> Any | None:
        return None

    async def search(
        self,
        query: str,
        limit: int = 10,
        namespace: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        return []

    async def delete(self, key: str, namespace: str | None = None) -> bool:
        return False

    async def clear(self, namespace: str | None = None) -> int:
        return 0


class FakePlanner:
    """Protocol-shaped planner for resolver validation tests."""

    async def plan(self, observation: Any, ctx: Any) -> Plan:
        return Plan(
            id="plan-1",
            goal="test",
            steps=(PlanStep(id="terminal", type=StepType.TERMINAL),),
        )


class MockRegistry:
    """Mock registry for testing."""

    def __init__(self, entry_point_group: str | None = None) -> None:
        self._adapters: dict[str, type[Any]] = {}
        self._entry_point_group = entry_point_group

    def register(self, name: str, adapter_class: type[Any]) -> None:
        self._adapters[name] = adapter_class

    def get(self, name: str) -> type[Any] | None:
        return self._adapters.get(name)


# ---------------------------------------------------------------------------
# resolve_adapter tests
# ---------------------------------------------------------------------------


class TestResolveAdapter:
    """Tests for resolve_adapter() function."""

    def test_adapter_instance_passthrough(self) -> None:
        """resolve_adapter should pass through adapter instances."""
        adapter = FakeModelAdapter("test-model")
        result = resolve_adapter(adapter, MockRegistry("axis.models"))
        assert result is adapter

    def test_string_resolution_from_registry(self) -> None:
        """resolve_adapter should resolve strings via registry."""
        registry = MockRegistry()
        registry.register("test-adapter", FakeModelAdapter)

        result = resolve_adapter("test-adapter", registry)
        assert isinstance(result, FakeModelAdapter)

    def test_string_resolution_with_kwargs(self) -> None:
        """resolve_adapter should pass kwargs to adapter constructor."""

        class ConfigurableAdapter:
            def __init__(self, config_value: str = "default") -> None:
                self.config_value = config_value

        registry = MockRegistry()
        registry.register("configurable", ConfigurableAdapter)  # type: ignore[arg-type]

        result = resolve_adapter("configurable", registry, config_value="custom")  # type: ignore[arg-type]
        assert result.config_value == "custom"

    def test_unknown_string_raises_error(self) -> None:
        """resolve_adapter should raise ConfigError for unknown strings."""
        registry = MockRegistry()
        with pytest.raises(ConfigError, match="Unknown adapter"):
            resolve_adapter("unknown", registry)

    def test_none_returns_none(self) -> None:
        """resolve_adapter should return None for None input."""
        result = resolve_adapter(None, MockRegistry())
        assert result is None

    def test_invalid_object_raises_for_model_registry(self) -> None:
        """resolve_adapter should reject arbitrary objects for model registries."""
        with pytest.raises(TypeError, match="Invalid model adapter"):
            resolve_adapter(object(), MockRegistry("axis.models"))  # type: ignore[arg-type]

    def test_invalid_object_raises_for_memory_registry(self) -> None:
        """resolve_adapter should reject arbitrary objects for memory registries."""
        with pytest.raises(TypeError, match="Invalid memory adapter"):
            resolve_adapter(object(), MockRegistry("axis.memory"))  # type: ignore[arg-type]

    def test_invalid_object_raises_for_planner_registry(self) -> None:
        """resolve_adapter should reject arbitrary objects for planner registries."""
        with pytest.raises(TypeError, match="Invalid planner adapter"):
            resolve_adapter(object(), MockRegistry("axis.planners"))  # type: ignore[arg-type]

    def test_memory_adapter_instance_passthrough(self) -> None:
        """resolve_adapter should pass through valid memory adapter instances."""
        adapter = FakeMemoryAdapter()
        result = resolve_adapter(adapter, MockRegistry("axis.memory"))
        assert result is adapter

    def test_planner_instance_passthrough(self) -> None:
        """resolve_adapter should pass through valid planner instances."""
        adapter = FakePlanner()
        result = resolve_adapter(adapter, MockRegistry("axis.planners"))
        assert result is adapter

    def test_invalid_type_raises_error(self) -> None:
        """resolve_adapter should raise TypeError for invalid types."""
        with pytest.raises(TypeError, match="registered adapter name"):
            resolve_adapter(123, MockRegistry("axis.models"))  # type: ignore[arg-type]
