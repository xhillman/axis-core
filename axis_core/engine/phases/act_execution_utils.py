"""Shared act-phase retry helpers used by tool and model execution paths."""

from __future__ import annotations

import asyncio
import random

from axis_core.config import RetryPolicy
from axis_core.context import RunContext
from axis_core.errors import AxisError, ModelError
from axis_core.protocols.planner import PlanStep


def resolve_retry_policy(
    ctx: RunContext,
    step: PlanStep,
    tool_retry: RetryPolicy | None = None,
) -> RetryPolicy:
    """Resolve effective retry policy for a step."""
    if step.retry_policy is not None:
        return step.retry_policy
    if tool_retry is not None:
        return tool_retry

    config_retry = getattr(getattr(ctx, "config", None), "retry", None)
    if isinstance(config_retry, RetryPolicy):
        return config_retry

    return RetryPolicy(max_attempts=1, jitter=False, initial_delay=0.0, max_delay=0.0)


def matches_retry_filter(error: Exception, retry_policy: RetryPolicy) -> bool:
    """Return True when retry_on filter allows retry for this error."""
    if retry_policy.retry_on is None:
        return True

    filters = {entry.lower() for entry in retry_policy.retry_on}
    error_name = type(error).__name__.lower()
    return any(token in error_name for token in filters)


def is_retryable_tool_error(error: Exception, retry_policy: RetryPolicy) -> bool:
    """Determine whether a tool failure should be retried."""
    if not matches_retry_filter(error, retry_policy):
        return False

    if isinstance(error, AxisError):
        return error.recoverable

    if isinstance(error, (TypeError, ValueError, KeyError)):
        return False

    return True


def is_retryable_model_error(error: ModelError, retry_policy: RetryPolicy) -> bool:
    """Determine whether a model failure should be retried."""
    if not ModelError.is_reason_recoverable(error.reason):
        return False
    return matches_retry_filter(error.cause or error, retry_policy)


def retry_delay_seconds(retry_policy: RetryPolicy, attempt: int) -> float:
    """Compute retry delay before the next attempt."""
    if retry_policy.backoff == "fixed":
        delay = retry_policy.initial_delay
    elif retry_policy.backoff == "linear":
        delay = retry_policy.initial_delay * attempt
    else:
        delay = retry_policy.initial_delay * (2 ** max(0, attempt - 1))

    delay = min(delay, retry_policy.max_delay)
    if retry_policy.jitter and delay > 0:
        delay *= 0.5 + random.random()
    return max(0.0, delay)


async def sleep_for_retry(retry_policy: RetryPolicy, attempt: int) -> None:
    """Sleep based on retry policy for an attempt number."""
    delay = retry_delay_seconds(retry_policy, attempt)
    if delay > 0:
        await asyncio.sleep(delay)


def record_retry_attempt(ctx: RunContext, step: PlanStep) -> None:
    """Track retry attempts in run state (not persisted by design)."""
    retry_state = ctx.state._retry_state.get(step.id, {"attempts": 0})
    retry_state["attempts"] = int(retry_state.get("attempts", 0)) + 1
    ctx.state._retry_state[step.id] = retry_state


__all__ = [
    "is_retryable_model_error",
    "is_retryable_tool_error",
    "matches_retry_filter",
    "record_retry_attempt",
    "resolve_retry_policy",
    "retry_delay_seconds",
    "sleep_for_retry",
]
