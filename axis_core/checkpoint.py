"""Checkpoint format helpers for lifecycle phase-boundary persistence."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from axis_core.context import RunContext
from axis_core.errors import ConfigError

CHECKPOINT_VERSION = 1


def create_checkpoint(
    ctx: RunContext,
    *,
    phase: str,
    next_phase: str | None = None,
) -> dict[str, Any]:
    """Build a versioned checkpoint envelope from RunContext serialization."""
    return {
        "version": CHECKPOINT_VERSION,
        "phase": phase,
        "next_phase": next_phase,
        "saved_at": datetime.utcnow().isoformat(),
        "context": ctx.serialize(),
    }


def parse_checkpoint(data: dict[str, Any]) -> tuple[RunContext, str, str | None]:
    """Validate and deserialize a checkpoint envelope."""
    version = data.get("version")
    if version != CHECKPOINT_VERSION:
        raise ConfigError(
            message=(
                f"Unsupported checkpoint version '{version}'. "
                f"Expected '{CHECKPOINT_VERSION}'."
            )
        )

    phase = data.get("phase")
    if not isinstance(phase, str) or not phase:
        raise ConfigError(message="Checkpoint phase is missing or invalid.")

    next_phase = data.get("next_phase")
    if next_phase is not None and not isinstance(next_phase, str):
        raise ConfigError(message="Checkpoint next_phase must be a string when provided.")

    context_data = data.get("context")
    if not isinstance(context_data, dict):
        raise ConfigError(message="Checkpoint context is missing or corrupt.")

    try:
        ctx = RunContext.deserialize(context_data)
    except Exception as e:  # pragma: no cover - defensive envelope parsing
        raise ConfigError(message="Checkpoint context failed to deserialize.", cause=e) from e

    return ctx, phase, next_phase

