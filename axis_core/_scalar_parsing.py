"""Internal helpers for scalar coercion and rate parsing."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSY_VALUES = frozenset({"0", "false", "no", "off"})
_STRICT_TRUTHY_VALUES = frozenset({"true"})
_STRICT_FALSY_VALUES = frozenset({"false"})
_RATE_PERIOD_SECONDS = {
    "second": 1.0,
    "minute": 60.0,
    "hour": 3600.0,
}


def coerce_bool(
    value: Any,
    *,
    default: bool = False,
    truthy_values: Collection[str] = _TRUTHY_VALUES,
    falsy_values: Collection[str] = _FALSY_VALUES,
) -> bool:
    """Coerce a bool from bools or env-style strings."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if not isinstance(value, str):
        return default

    normalized = value.strip().lower()
    if normalized in truthy_values:
        return True
    if normalized in falsy_values:
        return False
    return default


def coerce_env_flag(value: Any, *, default: bool = False) -> bool:
    """Coerce strict env flags where only literal true/false are accepted."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        return False
    return value.strip().lower() in _STRICT_TRUTHY_VALUES


def coerce_positive_int(value: Any) -> int | None:
    """Coerce positive integer config values."""
    if isinstance(value, int):
        return value if value > 0 else None
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    try:
        parsed = int(value.strip())
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def coerce_non_negative_int(value: Any) -> int | None:
    """Coerce non-negative integer config values."""
    if isinstance(value, int):
        return value if value >= 0 else None
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    try:
        parsed = int(value.strip())
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def parse_rate_limit(rate_spec: str, field_name: str) -> tuple[int, float]:
    """Parse a rate string like ``10/minute`` into count/period seconds."""
    if "/" not in rate_spec:
        raise ValueError(
            f"Invalid rate format for {field_name}: '{rate_spec}'. "
            "Expected format: 'count/period' (e.g., '60/minute')"
        )

    count_raw, period_raw = rate_spec.split("/", 1)
    try:
        count = int(count_raw)
    except ValueError as exc:
        raise ValueError(
            f"Invalid rate format for {field_name}: '{rate_spec}'. "
            "Count must be an integer."
        ) from exc

    period_seconds = _RATE_PERIOD_SECONDS.get(period_raw)
    if period_seconds is None:
        raise ValueError(
            f"Invalid period for {field_name}: '{period_raw}'. "
            "Must be 'second', 'minute', or 'hour'."
        )

    return count, period_seconds
