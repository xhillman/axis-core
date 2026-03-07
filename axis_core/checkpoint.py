"""Checkpoint format helpers for lifecycle phase-boundary persistence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from axis_core.cancel import CancelToken
from axis_core.context import ExecutionResult, Observation, RunContext
from axis_core.errors import ConfigError
from axis_core.protocols.planner import Plan

CHECKPOINT_VERSION = 1
_CHECKPOINT_PHASES = frozenset(
    {
        "initialize",
        "observe",
        "plan",
        "act",
        "evaluate",
        "finalize",
    }
)
_DEFAULT_NEXT_PHASE_BY_PHASE = {
    "initialize": "observe",
    "observe": "plan",
    "plan": "act",
    "act": "evaluate",
    "evaluate": "observe",
    "finalize": None,
}
_ALLOWED_NEXT_PHASES_BY_PHASE = {
    "initialize": frozenset({"observe"}),
    "observe": frozenset({"plan"}),
    "plan": frozenset({"act"}),
    "act": frozenset({"evaluate"}),
    "evaluate": frozenset({"observe", "finalize"}),
    "finalize": frozenset(),
}


@dataclass(frozen=True)
class CheckpointResumeState:
    """Checkpointed cycle values that allow resume to skip earlier phases."""

    observation: Observation | None = None
    plan: Plan | None = None
    execution: ExecutionResult | None = None


@dataclass(frozen=True)
class PreparedCheckpointResume:
    """Validated checkpoint data ready for lifecycle resume execution."""

    context: RunContext
    next_phase: str
    resume_state: CheckpointResumeState


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


def _coerce_phase(raw_phase: str, *, field_name: str) -> str:
    """Validate a checkpoint phase field against known lifecycle phases."""
    if raw_phase not in _CHECKPOINT_PHASES:
        raise ConfigError(message=f"Checkpoint {field_name} '{raw_phase}' is invalid.")
    return raw_phase


def _validate_checkpoint_boundary_state(ctx: RunContext, phase: str) -> None:
    """Validate that checkpoint state matches the declared phase boundary."""
    if phase in {"observe", "plan", "act", "evaluate"}:
        if ctx.state.current_observation is None:
            raise ConfigError(
                message=(
                    "Checkpoint is incompatible with phase boundary: "
                    "current_observation is required."
                )
            )
    if phase in {"plan", "act", "evaluate"}:
        if ctx.state.current_plan is None:
            raise ConfigError(
                message=(
                    "Checkpoint is incompatible with phase boundary: "
                    "current_plan is required."
                )
            )
    if phase in {"act", "evaluate"} and ctx.state.current_execution is None:
        raise ConfigError(
            message=(
                "Checkpoint is incompatible with phase boundary: "
                "current_execution is required."
            )
        )


def _require_resume_state(
    value: Any | None,
    *,
    phase: str,
    state_field: str,
) -> Any:
    """Require checkpoint state needed to resume from a given phase."""
    if value is None:
        raise ConfigError(
            message=(
                "Checkpoint is incompatible with resume phase: "
                f"{state_field} is required for {phase}."
            )
        )
    return value


def _prepare_cycle_resume_state(
    ctx: RunContext,
    *,
    start_phase: str,
) -> CheckpointResumeState:
    """Materialize checkpointed cycle values needed before the first resumed phase."""
    if start_phase == "observe":
        return CheckpointResumeState()

    observation = _require_resume_state(
        ctx.state.current_observation,
        phase=start_phase,
        state_field="current_observation",
    )
    if start_phase == "plan":
        return CheckpointResumeState(observation=observation)

    plan = _require_resume_state(
        ctx.state.current_plan,
        phase=start_phase,
        state_field="current_plan",
    )
    if start_phase == "act":
        return CheckpointResumeState(observation=observation, plan=plan)

    execution = _require_resume_state(
        ctx.state.current_execution,
        phase=start_phase,
        state_field="current_execution",
    )
    return CheckpointResumeState(
        observation=observation,
        plan=plan,
        execution=execution,
    )


def prepare_checkpoint_resume(
    checkpoint: dict[str, Any],
    *,
    cancel_token: CancelToken | None = None,
    config: Any | None = None,
) -> PreparedCheckpointResume:
    """Parse, validate, and materialize checkpoint state for lifecycle resume."""
    ctx, phase_raw, next_phase_raw = parse_checkpoint(checkpoint)
    phase = _coerce_phase(phase_raw, field_name="phase")
    _validate_checkpoint_boundary_state(ctx, phase)

    allowed_next_phases = _ALLOWED_NEXT_PHASES_BY_PHASE[phase]
    if not allowed_next_phases:
        raise ConfigError(
            message=(
                "Checkpoint phase boundary is not resumable. "
                "Only pre-finalize boundaries are supported."
            )
        )

    if next_phase_raw is None:
        next_phase = _DEFAULT_NEXT_PHASE_BY_PHASE[phase]
    else:
        next_phase = _coerce_phase(next_phase_raw, field_name="next_phase")
        if next_phase not in allowed_next_phases:
            raise ConfigError(
                message=(
                    f"Checkpoint next_phase '{next_phase}' is incompatible "
                    f"with checkpoint phase '{phase}'."
                )
            )

    if next_phase is None:
        raise ConfigError(
            message=(
                "Checkpoint phase boundary is not resumable. "
                "Only pre-finalize boundaries are supported."
            )
        )

    if config is not None:
        ctx.config = config
    if cancel_token is not None:
        ctx.cancel_token = cancel_token

    return PreparedCheckpointResume(
        context=ctx,
        next_phase=next_phase,
        resume_state=_prepare_cycle_resume_state(ctx, start_phase=next_phase),
    )
