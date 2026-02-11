"""Tests for engine/registry.py adapter registries (AD-023)."""

from __future__ import annotations

import json
import subprocess
import sys

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

    def test_can_register_custom_model(self) -> None:
        registry = ModelRegistry()
        registry.register("custom-model", MockAdapter)  # type: ignore[arg-type]
        assert registry.get("custom-model") is MockAdapter


# ---------------------------------------------------------------------------
# Built-in model registration tests (Task 16.1)
# ---------------------------------------------------------------------------


class TestBuiltInModelRegistration:
    """Tests for built-in model registration."""

    def test_auto_registration_works_without_manual_adapter_imports(self) -> None:
        """Registry import should eagerly register built-in model aliases."""
        script = """
import json
from axis_core.engine.registry import model_registry
registered = model_registry.list()
print(json.dumps({
    "count": len(registered),
    "has_claude_haiku": "claude-haiku" in registered,
}))
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
        data = json.loads(completed.stdout.strip())

        assert data["count"] > 0
        assert data["has_claude_haiku"] is True

    def test_anthropic_models_registered(self) -> None:
        """Built-in Anthropic models should be registered in global model_registry."""
        from axis_core.engine.registry import model_registry

        # Import models to trigger registration
        try:
            import axis_core.adapters.models  # noqa: F401
        except ImportError:
            pytest.skip("Anthropic package not installed")

        # Check that all built-in Anthropic models are registered
        registered = model_registry.list()
        expected_models = [
            "claude-3-haiku-20240307",
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514",
            "claude-opus-4-1-20250805",
            "claude-sonnet-4-5-20250929",
            "claude-haiku-4-5-20251001",
            "claude-opus-4-5-20251101",
            "claude-opus-4-6",
        ]
        for model in expected_models:
            assert model in registered, f"Expected model {model} to be registered"

    def test_string_resolution_creates_anthropic_instance(self) -> None:
        """Resolving a Claude model string should create AnthropicModel instance."""
        from axis_core.engine.registry import model_registry
        from axis_core.engine.resolver import resolve_adapter

        try:
            from axis_core.adapters.models import AnthropicModel
        except ImportError:
            pytest.skip("Anthropic package not installed")

        # Resolve string to instance
        model = resolve_adapter(
            "claude-sonnet-4-20250514",
            model_registry,
            api_key="test-key",
        )

        # Should get an AnthropicModel instance
        assert isinstance(model, AnthropicModel)
        assert model.model_id == "claude-sonnet-4-20250514"

    def test_string_resolution_passes_kwargs(self) -> None:
        """String resolution should pass constructor kwargs to model."""
        from axis_core.engine.registry import model_registry
        from axis_core.engine.resolver import resolve_adapter

        try:
            from axis_core.adapters.models import AnthropicModel
        except ImportError:
            pytest.skip("Anthropic package not installed")

        model = resolve_adapter(
            "claude-sonnet-4-20250514",
            model_registry,
            api_key="custom-key",
            temperature=0.8,
            max_tokens=500,
        )

        assert isinstance(model, AnthropicModel)
        # Check private attributes (AnthropicModel stores these as private)
        assert model._api_key == "custom-key"
        assert model._temperature == 0.8
        assert model._max_tokens == 500

    def test_unknown_model_raises_config_error(self) -> None:
        """Resolving unknown model string should raise ConfigError."""
        from axis_core.engine.registry import model_registry
        from axis_core.engine.resolver import resolve_adapter

        with pytest.raises(ConfigError, match="Unknown adapter 'nonexistent-model'"):
            resolve_adapter("nonexistent-model", model_registry)

    def test_convenience_aliases_registered(self) -> None:
        """Convenience aliases should map to latest model versions."""
        from axis_core.engine.registry import model_registry
        from axis_core.engine.resolver import resolve_adapter

        try:
            from axis_core.adapters.models import AnthropicModel
        except ImportError:
            pytest.skip("Anthropic package not installed")

        # Test that convenience aliases work
        haiku = resolve_adapter("claude-haiku", model_registry, api_key="test")
        sonnet = resolve_adapter("claude-sonnet", model_registry, api_key="test")
        opus = resolve_adapter("claude-opus", model_registry, api_key="test")

        # Verify they create instances with correct model IDs
        assert isinstance(haiku, AnthropicModel)
        assert haiku.model_id == "claude-haiku-4-5-20251001"

        assert isinstance(sonnet, AnthropicModel)
        assert sonnet.model_id == "claude-sonnet-4-5-20250929"

        assert isinstance(opus, AnthropicModel)
        assert opus.model_id == "claude-opus-4-6"

    def test_openai_responses_models_registered(self) -> None:
        """Responses API model IDs should be available in built-in model registry."""
        from axis_core.engine.registry import model_registry

        try:
            import axis_core.adapters.models  # noqa: F401
        except ImportError:
            pytest.skip("OpenAI package not installed")

        registered = model_registry.list()
        expected_models = [
            "gpt-5-codex",
            "codex-mini-latest",
            "gpt-5-search-api",
            "gpt-4o-search-preview",
            "gpt-4o-mini-search-preview",
            "o3-deep-research",
            "computer-use-preview",
        ]
        for model in expected_models:
            assert model in registered, f"Expected Responses model {model} to be registered"


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
