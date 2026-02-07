"""String-to-adapter resolution for axis-core.

This module provides the resolve_adapter() function for converting string identifiers
to adapter instances, working with the adapter registries.
"""

from __future__ import annotations

from typing import Any, TypeVar

from axis_core.engine.registry import AdapterRegistry
from axis_core.errors import ConfigError

T = TypeVar("T")


def resolve_adapter(
    value: str | T | None,
    registry: AdapterRegistry[T],
    **kwargs: Any,
) -> T | None:
    """Resolve a string identifier to an adapter instance via registry.

    Args:
        value: String identifier, adapter instance, or None
        registry: Registry object with get(name) method
        **kwargs: Keyword arguments to pass to adapter constructor

    Returns:
        Adapter instance or None

    Raises:
        ConfigError: If string identifier not found in registry
        TypeError: If value type is invalid

    Examples:
        >>> resolve_adapter("claude-sonnet-4", model_registry)
        <AnthropicModel instance>

        >>> resolve_adapter(my_adapter_instance, registry)
        <returns same instance>

        >>> resolve_adapter(None, registry)
        None
    """
    # None passthrough
    if value is None:
        return None

    # String resolution via registry
    if isinstance(value, str):
        adapter_class = registry.get(value)
        if adapter_class is None:
            raise ConfigError(
                f"Unknown adapter '{value}'. "
                f"Check registry or use explicit adapter instance."
            )
        return adapter_class(**kwargs)

    # Instance passthrough - check it's not a primitive type
    if not isinstance(value, (int, float, bool, list, dict, tuple, set)):
        # It's an object instance - pass through
        return value

    raise TypeError(
        f"Adapter value must be str, adapter instance, or None, "
        f"got {type(value).__name__}"
    )


__all__ = ["resolve_adapter"]
