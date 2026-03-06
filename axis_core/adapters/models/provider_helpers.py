"""Shared helper utilities for provider model adapters."""

from __future__ import annotations

import re
from typing import Any

DEFAULT_SCHEMA_FIELDS_TO_STRIP = frozenset(
    {
        "$schema",
        "$id",
        "title",
        "default",
        "examples",
    }
)

DEFAULT_TOOL_CALL_ID_INVALID_CHARS = re.compile(r"[^A-Za-z0-9_-]+")


def sanitize_schema_node(value: Any, *, fields_to_strip: frozenset[str]) -> Any:
    """Recursively strip provider-problematic schema fields."""
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, raw in value.items():
            if key in fields_to_strip or raw is None:
                continue
            sanitized[key] = sanitize_schema_node(raw, fields_to_strip=fields_to_strip)

        properties = sanitized.get("properties")
        required = sanitized.get("required")
        if isinstance(properties, dict) and isinstance(required, list):
            allowed = set(properties.keys())
            sanitized["required"] = [
                item for item in required if isinstance(item, str) and item in allowed
            ]

        return sanitized

    if isinstance(value, list):
        return [sanitize_schema_node(item, fields_to_strip=fields_to_strip) for item in value]

    return value


def sanitize_tool_schema_for_provider(
    schema: Any,
    *,
    fields_to_strip: frozenset[str],
) -> dict[str, Any]:
    """Return a provider-safe tool schema."""
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}

    sanitized = sanitize_schema_node(schema, fields_to_strip=fields_to_strip)
    if not isinstance(sanitized, dict):
        return {"type": "object", "properties": {}}

    if "type" not in sanitized:
        sanitized["type"] = "object"

    properties = sanitized.get("properties")
    if not isinstance(properties, dict):
        sanitized["properties"] = {}

    required = sanitized.get("required")
    if not isinstance(required, list):
        sanitized.pop("required", None)

    return sanitized


def normalize_tool_call_id(
    raw_id: Any,
    *,
    next_index: int,
    id_map: dict[str, str],
    used_ids: set[str],
    invalid_chars: re.Pattern[str],
    prefix: str,
    max_len: int,
) -> str:
    """Normalize tool call IDs to provider-safe values with stable mapping."""
    raw_text = str(raw_id).strip() if raw_id is not None else ""
    if raw_text and raw_text in id_map:
        return id_map[raw_text]

    if raw_text and not invalid_chars.search(raw_text) and len(raw_text) <= max_len:
        candidate = raw_text
    else:
        base = invalid_chars.sub("_", raw_text).strip("_")
        if not base:
            base = f"{prefix}_{next_index}"
        elif not base.startswith(f"{prefix}_"):
            base = f"{prefix}_{base}"

        candidate = base[:max_len].rstrip("_")
        if not candidate:
            candidate = f"{prefix}_{next_index}"

    suffix = 1
    unique_candidate = candidate
    while unique_candidate in used_ids:
        suffix_text = f"_{suffix}"
        max_base_len = max_len - len(suffix_text)
        unique_candidate = f"{candidate[:max_base_len]}{suffix_text}"
        suffix += 1

    used_ids.add(unique_candidate)
    if raw_text:
        id_map[raw_text] = unique_candidate
    return unique_candidate
