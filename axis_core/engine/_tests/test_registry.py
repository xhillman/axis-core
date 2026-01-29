"""Tests for engine/registry.py adapter registries (AD-023)."""

from __future__ import annotations

from typing import Any

import pytest

from axis_core.engine.registry import (
    AdapterRegistry,
    MemoryRegistry,
    ModelRegistry,
    PlannerRegistry,
)
from axis_core.errors import ConfigError


# ---------------------------------------------------------------------------
# Mock adapters
# ---------------------------------------------------------------------------


class MockAdapter:
    """Mock adapter for testing."""

    def __init__(self, config: str = "default") -> None:
        self.config = config


# ---------------------------------------------------------------------------
# AdapterRegistry tests
# ---------------------------------------------------------------------------


class TestAdapterRegistry:
    """Tests for base AdapterRegistry class."""

    def test_register_and_get(self) -> None:
        registry = AdapterRegistry[MockAdapter]()
        registry.register("test-adapter", MockAdapter)
        adapter_class = registry.get("test-adapter")
        assert adapter_class is MockAdapter

    def test_get_unknown_returns_none(self) -> None:
        registry = AdapterRegistry[MockAdapter]()
        result = registry.get("unknown")
        assert result is None

    def test_register_duplicate_replaces(self) -> None:
        """Later registrations replace earlier ones."""
        registry = AdapterRegistry[MockAdapter]()

        class FirstAdapter:
            pass

        class SecondAdapter:
            pass

        registry.register("test", FirstAdapter)  # type: ignore[arg-type]
        registry.register("test", SecondAdapter)  # type: ignore[arg-type]

        result = registry.get("test")
        assert result is SecondAdapter

    def test_list_registered(self) -> None:
        registry = AdapterRegistry[MockAdapter]()
        registry.register("adapter1", MockAdapter)
        registry.register("adapter2", MockAdapter)

        names = registry.list()
        assert "adapter1" in names
        assert "adapter2" in names

    def test_empty_list(self) -> None:
        registry = AdapterRegistry[MockAdapter]()
        names = registry.list()
        assert names == []


# ---------------------------------------------------------------------------
# ModelRegistry tests
# ---------------------------------------------------------------------------


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_creation(self) -> None:
        registry = ModelRegistry()
        assert isinstance(registry, AdapterRegistry)

    def test_built_in_models_not_registered_by_default(self) -> None:
        """Built-in models would be registered by model adapters, not the registry."""
        registry = ModelRegistry()
        # Registry starts empty; models register themselves when imported
        assert registry.list() == []

    def test_can_register_custom_model(self) -> None:
        registry = ModelRegistry()
        registry.register("custom-model", MockAdapter)  # type: ignore[arg-type]
        assert registry.get("custom-model") is MockAdapter


# ---------------------------------------------------------------------------
# MemoryRegistry tests
# ---------------------------------------------------------------------------


class TestMemoryRegistry:
    """Tests for MemoryRegistry."""

    def test_creation(self) -> None:
        registry = MemoryRegistry()
        assert isinstance(registry, AdapterRegistry)

    def test_can_register_custom_memory(self) -> None:
        registry = MemoryRegistry()
        registry.register("custom-memory", MockAdapter)  # type: ignore[arg-type]
        assert registry.get("custom-memory") is MockAdapter


# ---------------------------------------------------------------------------
# PlannerRegistry tests
# ---------------------------------------------------------------------------


class TestPlannerRegistry:
    """Tests for PlannerRegistry."""

    def test_creation(self) -> None:
        registry = PlannerRegistry()
        assert isinstance(registry, AdapterRegistry)

    def test_can_register_custom_planner(self) -> None:
        registry = PlannerRegistry()
        registry.register("custom-planner", MockAdapter)  # type: ignore[arg-type]
        assert registry.get("custom-planner") is MockAdapter
