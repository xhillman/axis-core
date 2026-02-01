"""Adapter registries for axis-core (AD-023).

This module provides registry classes for models, memory, and planners, with support
for entry point discovery and explicit registration.

Architecture Decisions:
- AD-023: Hybrid entry points + explicit registration for adapter discovery
"""

from __future__ import annotations

import logging
from typing import Any, Generic, TypeVar

logger = logging.getLogger("axis_core.engine.registry")

T = TypeVar("T")


class AdapterRegistry(Generic[T]):
    """Base registry for adapters.

    Supports explicit registration and optional entry point discovery.
    Entry points are discovered from installed packages via importlib.metadata.
    """

    def __init__(self, entry_point_group: str | None = None) -> None:
        """Initialize registry.

        Args:
            entry_point_group: Entry point group name for auto-discovery
                             (e.g., "axis.models"). If None, no auto-discovery.
        """
        self._adapters: dict[str, type[T]] = {}
        self._entry_point_group = entry_point_group

        if entry_point_group:
            self._discover_entry_points()

    def _discover_entry_points(self) -> None:
        """Auto-discover adapters from installed packages via entry points."""
        if not self._entry_point_group:
            return

        try:
            from importlib.metadata import entry_points

            # Get all entry points for this group
            eps = entry_points(group=self._entry_point_group)

            for ep in eps:
                try:
                    adapter_class = ep.load()
                    self.register(ep.name, adapter_class)
                    logger.debug(
                        "Loaded adapter '%s' from entry point %s",
                        ep.name,
                        ep.value,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to load adapter '%s' from entry point: %s",
                        ep.name,
                        e,
                    )
        except ImportError:
            # importlib.metadata not available (shouldn't happen on py3.10+)
            logger.warning(
                "Entry point discovery unavailable (importlib.metadata not found)"
            )
        except Exception as e:
            logger.warning("Entry point discovery failed: %s", e)

    def register(self, name: str, adapter_class: type[T]) -> None:
        """Explicitly register an adapter.

        Args:
            name: Adapter identifier (e.g., "claude-sonnet-4")
            adapter_class: Adapter class to instantiate

        Examples:
            >>> registry.register("my-model", MyModelAdapter)
        """
        self._adapters[name] = adapter_class
        logger.debug("Registered adapter: %s -> %s", name, adapter_class.__name__)

    def get(self, name: str) -> type[T] | None:
        """Get adapter class by name.

        Args:
            name: Adapter identifier

        Returns:
            Adapter class or None if not found
        """
        return self._adapters.get(name)

    def list(self) -> list[str]:
        """List all registered adapter names.

        Returns:
            List of adapter identifiers
        """
        return list(self._adapters.keys())


class ModelRegistry(AdapterRegistry[Any]):
    """Registry for model adapters.

    Auto-discovers models from entry point group "axis.models".
    """

    def __init__(self) -> None:
        super().__init__(entry_point_group="axis.models")


class MemoryRegistry(AdapterRegistry[Any]):
    """Registry for memory adapters.

    Auto-discovers memory adapters from entry point group "axis.memory".
    """

    def __init__(self) -> None:
        super().__init__(entry_point_group="axis.memory")


class PlannerRegistry(AdapterRegistry[Any]):
    """Registry for planner adapters.

    Auto-discovers planners from entry point group "axis.planners".
    """

    def __init__(self) -> None:
        super().__init__(entry_point_group="axis.planners")


# Global registry instances
model_registry = ModelRegistry()
memory_registry = MemoryRegistry()
planner_registry = PlannerRegistry()


# Trigger built-in adapter registration by importing adapter modules
# This ensures lazy factories are registered when registries are first accessed
def _register_builtin_adapters() -> None:
    """Import adapter modules to trigger lazy registration.

    This is called automatically when the registry module is imported,
    ensuring built-in adapters are always available in the registries.
    """
    try:
        # Import adapter modules to trigger registration
        # These imports have minimal overhead due to lazy loading
        import axis_core.adapters.models  # noqa: F401
        import axis_core.adapters.memory  # noqa: F401
        import axis_core.adapters.planners  # noqa: F401
    except ImportError:
        # If adapters aren't available for some reason, that's fine
        # Registration will happen when they're imported elsewhere
        pass


# Auto-register built-in adapters when this module is loaded
_register_builtin_adapters()


__all__ = [
    "AdapterRegistry",
    "ModelRegistry",
    "MemoryRegistry",
    "PlannerRegistry",
    "model_registry",
    "memory_registry",
    "planner_registry",
]
