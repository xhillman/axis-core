"""Tests for engine/resolver.py string-to-adapter resolution."""

from __future__ import annotations

from typing import Any

import pytest

from axis_core.engine.resolver import resolve_adapter
from axis_core.errors import ConfigError


# ---------------------------------------------------------------------------
# Mock adapters for testing
# ---------------------------------------------------------------------------


class MockAdapter:
    """Mock adapter for testing."""

    def __init__(self, name: str = "mock") -> None:
        self.name = name


class MockRegistry:
    """Mock registry for testing."""

    def __init__(self) -> None:
        self._adapters: dict[str, type[MockAdapter]] = {}

    def register(self, name: str, adapter_class: type[MockAdapter]) -> None:
        self._adapters[name] = adapter_class

    def get(self, name: str) -> type[MockAdapter] | None:
        return self._adapters.get(name)


# ---------------------------------------------------------------------------
# resolve_adapter tests
# ---------------------------------------------------------------------------


class TestResolveAdapter:
    """Tests for resolve_adapter() function."""

    def test_adapter_instance_passthrough(self) -> None:
        """resolve_adapter should pass through adapter instances."""
        adapter = MockAdapter("test")
        result = resolve_adapter(adapter, MockRegistry())
        assert result is adapter

    def test_string_resolution_from_registry(self) -> None:
        """resolve_adapter should resolve strings via registry."""
        registry = MockRegistry()
        registry.register("test-adapter", MockAdapter)

        result = resolve_adapter("test-adapter", registry)
        assert isinstance(result, MockAdapter)

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

    def test_invalid_type_raises_error(self) -> None:
        """resolve_adapter should raise TypeError for invalid types."""
        with pytest.raises(TypeError, match="must be str, adapter instance, or None"):
            resolve_adapter(123, MockRegistry())  # type: ignore[arg-type]
