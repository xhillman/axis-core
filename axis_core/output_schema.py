"""Structured-output coercion helpers for Agent output_schema enforcement."""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Any

from axis_core.errors import AxisError, ErrorClass

_JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def schema_name(schema: type[Any]) -> str:
    """Return a human-readable schema type name."""
    return getattr(schema, "__name__", repr(schema))


def parse_stream_partial(text: str, schema: type[Any]) -> Any | None:
    """Best-effort partial structured output for streaming events.

    Partial parsing is intentionally loose and only emits values that are broadly
    compatible with the requested output schema.
    """
    parsed = _parse_first_json_value(text)
    if parsed is None:
        return None

    if schema is dict and isinstance(parsed, dict):
        return parsed
    if schema is list and isinstance(parsed, list):
        return parsed
    if dataclasses.is_dataclass(schema) and isinstance(parsed, dict):
        return parsed
    if _is_pydantic_model(schema) and isinstance(parsed, dict):
        return parsed
    if schema not in (dict, list) and isinstance(parsed, dict):
        return parsed
    return None


def coerce_to_output_schema(
    *,
    output: Any,
    output_raw: str | None,
    schema: type[Any],
) -> Any:
    """Coerce runtime output into the declared output schema.

    Raises:
        AxisError: When no coercion path can satisfy the schema.
    """
    if not isinstance(schema, type):
        raise AxisError(
            message=f"Invalid output_schema: expected type, got {type(schema).__name__}",
            error_class=ErrorClass.RUNTIME,
        )

    errors: list[str] = []
    for candidate in _iter_candidates(output=output, output_raw=output_raw):
        try:
            return _coerce_candidate(candidate, schema)
        except Exception as e:  # pragma: no cover - message aggregation path
            errors.append(str(e))

    raise AxisError(
        message=(
            "output_schema validation failed for "
            f"{schema_name(schema)}. No compatible structured output was produced."
        ),
        error_class=ErrorClass.RUNTIME,
        details={
            "schema": schema_name(schema),
            "attempt_errors": errors[-3:],
        },
    )


def _iter_candidates(*, output: Any, output_raw: str | None) -> list[Any]:
    candidates: list[Any] = []

    if output is not None:
        candidates.append(output)
        if isinstance(output, str):
            parsed_from_output = _parse_first_json_value(output)
            if parsed_from_output is not None:
                candidates.append(parsed_from_output)

    if output_raw:
        if output_raw not in candidates:
            candidates.append(output_raw)
        parsed_from_raw = _parse_first_json_value(output_raw)
        if parsed_from_raw is not None and parsed_from_raw not in candidates:
            candidates.append(parsed_from_raw)

    return candidates


def _coerce_candidate(value: Any, schema: type[Any]) -> Any:
    # Pydantic v2
    model_validate = getattr(schema, "model_validate", None)
    if callable(model_validate):
        return model_validate(value)

    # Pydantic v1
    parse_obj = getattr(schema, "parse_obj", None)
    if callable(parse_obj):
        return parse_obj(value)

    if schema is str:
        if isinstance(value, str):
            return value
        raise TypeError("expected string output")

    if schema is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "on"}:
                return True
            if normalized in {"false", "0", "no", "off"}:
                return False
        raise TypeError("expected boolean output")

    if schema is int:
        if isinstance(value, bool):
            raise TypeError("expected integer output")
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            return int(value.strip())
        raise TypeError("expected integer output")

    if schema is float:
        if isinstance(value, bool):
            raise TypeError("expected float output")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            return float(value.strip())
        raise TypeError("expected float output")

    if schema is dict:
        if isinstance(value, dict):
            return value
        raise TypeError("expected JSON object output")

    if schema is list:
        if isinstance(value, list):
            return value
        raise TypeError("expected JSON array output")

    if dataclasses.is_dataclass(schema):
        if isinstance(value, schema):
            return value
        if not isinstance(value, dict):
            raise TypeError("expected JSON object for dataclass output_schema")
        return schema(**value)

    if isinstance(value, schema):
        return value

    if isinstance(value, dict):
        try:
            return schema(**value)
        except TypeError:
            pass

    return schema(value)


def _parse_first_json_value(text: str) -> Any | None:
    payload = text.strip()
    if not payload:
        return None

    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        pass

    for match in _JSON_FENCE_PATTERN.finditer(text):
        block = match.group(1).strip()
        if not block:
            continue
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            continue

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
            return parsed
        except json.JSONDecodeError:
            continue

    return None


def _is_pydantic_model(schema: type[Any]) -> bool:
    return callable(getattr(schema, "model_validate", None)) or callable(
        getattr(schema, "parse_obj", None)
    )

