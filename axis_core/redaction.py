"""Shared sensitive-data redaction utilities."""

from __future__ import annotations

import os
import re
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
_SENSITIVE_INLINE_VALUE_PATTERN = re.compile(
    r"(?i)(?P<key>\b(?:api[-_ ]?key|secret|token|password|authorization|"
    r"access[-_ ]?key|private[-_ ]?key)\b)"
    r"(?P<sep>\s*(?:=|:)\s*)"
    r"(?P<quote>['\"]?)(?P<value>[^\s,;'\"]+)(?P=quote)"
)
_BEARER_TOKEN_PATTERN = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+\b")
_OPENAI_STYLE_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")


def _normalize_key_name(key: str) -> str:
    """Normalize key names for robust fragment matching."""
    return key.lower().replace("-", "_")


def is_sensitive_key(key: str) -> bool:
    """Return True when a key name is likely to contain secrets."""
    key_lower = _normalize_key_name(key)
    return any(fragment in key_lower for fragment in _SENSITIVE_KEY_FRAGMENTS)


def redact_sensitive_string(value: str) -> str:
    """Redact inline secret-like values from free-form strings."""

    def _replace_inline(match: re.Match[str]) -> str:
        key = match.group("key")
        sep = match.group("sep")
        quote = match.group("quote") or ""
        return f"{key}{sep}{quote}{REDACTED_VALUE}{quote}"

    redacted = _BEARER_TOKEN_PATTERN.sub(f"Bearer {REDACTED_VALUE}", value)
    redacted = _SENSITIVE_INLINE_VALUE_PATTERN.sub(_replace_inline, redacted)
    return _OPENAI_STYLE_KEY_PATTERN.sub(REDACTED_VALUE, redacted)


def redact_sensitive_data(value: Any) -> Any:
    """Recursively redact values under sensitive keys."""
    if isinstance(value, str):
        return redact_sensitive_string(value)
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
    "redact_sensitive_string",
    "redact_sensitive_data",
    "persist_sensitive_tool_data_enabled",
]
