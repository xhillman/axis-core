"""Event type constants for telemetry.

This module defines all event types emitted by the axis-core execution engine,
organized by category as specified in the PRD.
"""

from __future__ import annotations


class EventTypes:
    """Constants for all telemetry event types."""

    # ===== Lifecycle Events (10.2) =====
    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"

    # ===== Phase Events (10.3) =====
    PHASE_ENTERED = "phase_entered"
    PHASE_EXITED = "phase_exited"
    PHASE_FAILED = "phase_failed"

    # ===== Cycle Events (10.4) =====
    CYCLE_STARTED = "cycle_started"
    CYCLE_COMPLETED = "cycle_completed"

    # ===== Plan Events =====
    PLAN_CREATED = "plan_created"
    PLAN_VALIDATED = "plan_validated"

    # ===== Step Events =====
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"
    STEP_RETRIED = "step_retried"

    # ===== Tool Events (10.5) =====
    TOOL_CALLED = "tool_called"
    TOOL_RETURNED = "tool_returned"
    TOOL_FAILED = "tool_failed"
    TOOL_CACHED = "tool_cached"

    # ===== Model Events (10.6) =====
    MODEL_CALLED = "model_called"
    MODEL_RETURNED = "model_returned"
    MODEL_FAILED = "model_failed"
    MODEL_TOKEN = "model_token"

    # ===== Budget Events (10.7) =====
    BUDGET_WARNING = "budget_warning"
    BUDGET_EXCEEDED = "budget_exceeded"


__all__ = ["EventTypes"]
