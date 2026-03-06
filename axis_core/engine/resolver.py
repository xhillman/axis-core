"""String-to-adapter resolution for axis-core.

This module provides the resolve_adapter() function for converting string identifiers
to adapter instances, working with the adapter registries.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, TypeVar

from axis_core.engine.registry import AdapterRegistry
from axis_core.errors import ConfigError

T = TypeVar("T")


@dataclass(frozen=True)
class _AdapterValidationSpec:
    """Validation rules for registry-specific adapter instances."""

    category: str
    property_members: tuple[str, ...]
    callable_members: tuple[str, ...]


_KNOWN_ADAPTER_SPECS: dict[str, _AdapterValidationSpec] = {
    "axis.models": _AdapterValidationSpec(
        category="model",
        property_members=("model_id",),
        callable_members=("complete", "stream", "estimate_tokens", "estimate_cost"),
    ),
    "axis.memory": _AdapterValidationSpec(
        category="memory",
        property_members=("capabilities",),
        callable_members=("store", "retrieve", "search", "delete", "clear"),
    ),
    "axis.planners": _AdapterValidationSpec(
        category="planner",
        property_members=(),
        callable_members=("plan",),
    ),
}

_INVALID_NON_ADAPTER_TYPES = (int, float, bool, list, dict, tuple, set)


def _validation_spec_for_registry(
    registry: AdapterRegistry[Any],
) -> _AdapterValidationSpec | None:
    """Return the validation contract for a known registry category."""
    entry_point_group = getattr(registry, "_entry_point_group", None)
    if isinstance(entry_point_group, str):
        return _KNOWN_ADAPTER_SPECS.get(entry_point_group)
    return None


def _has_property_member(value: object, member: str) -> bool:
    """Check for a member without invoking descriptors or properties."""
    try:
        inspect.getattr_static(value, member)
    except AttributeError:
        return False
    return True


def _missing_adapter_members(
    value: object,
    spec: _AdapterValidationSpec,
) -> list[str]:
    """Return missing protocol-like members for a candidate adapter instance."""
    missing: list[str] = []

    for member in spec.property_members:
        if not _has_property_member(value, member):
            missing.append(member)

    for member in spec.callable_members:
        if not callable(getattr(value, member, None)):
            missing.append(member)

    return missing


def _invalid_adapter_message(
    value: object,
    spec: _AdapterValidationSpec,
) -> str:
    """Build an actionable validation error for an invalid adapter instance."""
    missing_members = _missing_adapter_members(value, spec)
    missing_detail = ""
    if missing_members:
        missing_detail = f" Missing required members: {', '.join(missing_members)}."

    return (
        f"Invalid {spec.category} adapter value of type {type(value).__name__}. "
        "Expected None, a registered adapter name (str), or an adapter instance "
        f"implementing the {spec.category} adapter contract."
        f"{missing_detail}"
    )


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

    validation_spec = _validation_spec_for_registry(registry)

    # Instance passthrough - validate against known adapter contracts when possible
    if validation_spec is not None:
        if not _missing_adapter_members(value, validation_spec):
            return value
        raise TypeError(_invalid_adapter_message(value, validation_spec))

    if not isinstance(value, _INVALID_NON_ADAPTER_TYPES):
        return value

    raise TypeError(
        f"Adapter value must be str, adapter instance, or None, "
        f"got {type(value).__name__}"
    )


__all__ = ["resolve_adapter"]
