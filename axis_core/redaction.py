"""Shared sensitive-data redaction utilities."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

REDACTED_VALUE = "[REDACTED]"
_SENSITIVE_KEY_FRAGMENTS = (
    "api_key",
    "apikey",
    "secret",
    "token",
    "password",
    "authorization",
    "bearer",
    "access_key",
    "private_key",
)


def is_sensitive_key(key: str) -> bool:
    """Return True when a key name is likely to contain secrets."""
    key_lower = key.lower()
    return any(fragment in key_lower for fragment in _SENSITIVE_KEY_FRAGMENTS)


def redact_sensitive_data(value: Any) -> Any:
    """Recursively redact values under sensitive keys."""
    if isinstance(value, Mapping):
        return {
            k: REDACTED_VALUE if isinstance(k, str) and is_sensitive_key(k)
            else redact_sensitive_data(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_sensitive_data(item) for item in value)
    return value


def persist_sensitive_tool_data_enabled() -> bool:
    """Whether raw tool args/results can be persisted for debugging."""
    return os.getenv("AXIS_PERSIST_SENSITIVE_TOOL_DATA", "false").lower() == "true"


__all__ = [
    "REDACTED_VALUE",
    "is_sensitive_key",
    "redact_sensitive_data",
    "persist_sensitive_tool_data_enabled",
]
