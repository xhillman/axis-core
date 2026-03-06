"""Serialization helpers for context checkpointing."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from axis_core.budget import Budget, BudgetState
from axis_core.config import RetryPolicy
from axis_core.context.types import (
    EvalDecision,
    ExecutionResult,
    ModelCallRecord,
    NormalizedInput,
    Observation,
)
from axis_core.errors import AxisError, ErrorClass, ErrorRecord
from axis_core.redaction import (
    persist_sensitive_tool_data_enabled,
    redact_sensitive_data,
)


def _serialize_normalized_input(input_: NormalizedInput) -> dict[str, Any]:
    """Serialize NormalizedInput to dict."""
    return {
        "text": input_.text,
        "original": input_.original,
        "is_multimodal": input_.is_multimodal,
    }


def _deserialize_normalized_input(data: dict[str, Any]) -> NormalizedInput:
    """Deserialize NormalizedInput from dict."""
    return NormalizedInput(
        text=data["text"],
        original=data["original"],
        is_multimodal=data.get("is_multimodal", False),
    )


def _serialize_observation(obs: Observation) -> dict[str, Any]:
    """Serialize Observation to dict."""
    return {
        "input": _serialize_normalized_input(obs.input),
        "memory_context": obs.memory_context,
        "previous_cycles": list(obs.previous_cycles),
        "tool_requests": (
            [_serialize_tool_call(tool_call) for tool_call in obs.tool_requests]
            if obs.tool_requests
            else None
        ),
        "response": obs.response,
        "goal": obs.goal,
        "timestamp": obs.timestamp.isoformat(),
    }


def _deserialize_observation(data: dict[str, Any]) -> Observation:
    """Deserialize Observation from dict."""
    from axis_core.protocols.model import ToolCall

    tool_requests = None
    if data.get("tool_requests"):
        tool_requests = tuple(
            ToolCall(
                id=tool_call["id"],
                name=tool_call["name"],
                arguments=tool_call.get("arguments", {}),
            )
            for tool_call in data["tool_requests"]
        )

    return Observation(
        input=_deserialize_normalized_input(data["input"]),
        memory_context=data.get("memory_context", {}),
        previous_cycles=tuple(data.get("previous_cycles", [])),
        tool_requests=tool_requests,
        response=data.get("response"),
        goal=data.get("goal", ""),
        timestamp=datetime.fromisoformat(data["timestamp"]),
    )


def _serialize_tool_call(tc: Any) -> dict[str, Any]:
    """Serialize ToolCall to dict."""
    return {
        "id": tc.id,
        "name": tc.name,
        "arguments": tc.arguments,
    }


def _serialize_execution_result(result: ExecutionResult) -> dict[str, Any]:
    """Serialize ExecutionResult to dict."""
    return {
        "results": result.results,
        "errors": {
            name: _serialize_axis_error(error)
            for name, error in result.errors.items()
        },
        "skipped": list(result.skipped),
        "duration_ms": result.duration_ms,
    }


def _deserialize_execution_result(data: dict[str, Any]) -> ExecutionResult:
    """Deserialize ExecutionResult from dict."""
    return ExecutionResult(
        results=data.get("results", {}),
        errors={
            name: _deserialize_axis_error(error)
            for name, error in data.get("errors", {}).items()
        },
        skipped=frozenset(data.get("skipped", [])),
        duration_ms=data.get("duration_ms", 0.0),
    )


def _serialize_eval_decision(decision: EvalDecision) -> dict[str, Any]:
    """Serialize EvalDecision to dict."""
    return {
        "done": decision.done,
        "error": _serialize_axis_error(decision.error) if decision.error else None,
        "recoverable": decision.recoverable,
        "reason": decision.reason,
    }


def _deserialize_eval_decision(data: dict[str, Any]) -> EvalDecision:
    """Deserialize EvalDecision from dict."""
    return EvalDecision(
        done=data["done"],
        error=_deserialize_axis_error(data["error"]) if data.get("error") else None,
        recoverable=data.get("recoverable", False),
        reason=data.get("reason", ""),
    )


def _serialize_axis_error(error: AxisError) -> dict[str, Any]:
    """Serialize AxisError to dict."""
    return {
        "message": redact_sensitive_data(error.message),
        "error_class": error.error_class.value,
        "phase": error.phase,
        "cycle": error.cycle,
        "step_id": error.step_id,
        "recoverable": error.recoverable,
        "retry_after": error.retry_after,
        "details": redact_sensitive_data(error.details),
        "cause": redact_sensitive_data(str(error.cause)) if error.cause else None,
    }


def _deserialize_axis_error(data: dict[str, Any]) -> AxisError:
    """Deserialize AxisError from dict."""
    return AxisError(
        message=data["message"],
        error_class=ErrorClass(data["error_class"]),
        phase=data.get("phase"),
        cycle=data.get("cycle"),
        step_id=data.get("step_id"),
        recoverable=data.get("recoverable", False),
        retry_after=data.get("retry_after"),
        details=data.get("details", {}),
        cause=None,
    )


def _serialize_retry_policy(policy: RetryPolicy | None) -> dict[str, Any] | None:
    """Serialize RetryPolicy to dict."""
    if policy is None:
        return None
    return {
        "max_attempts": policy.max_attempts,
        "backoff": policy.backoff,
        "initial_delay": policy.initial_delay,
        "max_delay": policy.max_delay,
        "jitter": policy.jitter,
        "retry_on": list(policy.retry_on) if policy.retry_on is not None else None,
    }


def _deserialize_retry_policy(data: dict[str, Any] | None) -> RetryPolicy | None:
    """Deserialize RetryPolicy from dict."""
    if data is None:
        return None

    default_policy = RetryPolicy()
    retry_on_raw = data.get("retry_on", default_policy.retry_on)
    retry_on = None if retry_on_raw is None else [str(item) for item in retry_on_raw]

    return RetryPolicy(
        max_attempts=int(data.get("max_attempts", default_policy.max_attempts)),
        backoff=str(data.get("backoff", default_policy.backoff)),
        initial_delay=float(data.get("initial_delay", default_policy.initial_delay)),
        max_delay=float(data.get("max_delay", default_policy.max_delay)),
        jitter=bool(data.get("jitter", default_policy.jitter)),
        retry_on=retry_on,
    )


def _serialize_plan(plan: Any) -> dict[str, Any]:
    """Serialize Plan to dict."""
    return {
        "id": plan.id,
        "goal": plan.goal,
        "steps": [
            {
                "id": step.id,
                "type": step.type.value,
                "payload": step.payload,
                "dependencies": list(step.dependencies) if step.dependencies else None,
                "retry_policy": _serialize_retry_policy(step.retry_policy),
            }
            for step in plan.steps
        ],
        "reasoning": plan.reasoning,
        "confidence": plan.confidence,
        "metadata": plan.metadata,
    }


def _deserialize_plan(data: dict[str, Any]) -> Any:
    """Deserialize Plan from dict."""
    from axis_core.protocols.planner import Plan, PlanStep, StepType

    steps = tuple(
        PlanStep(
            id=step["id"],
            type=StepType(step["type"]),
            payload=step.get("payload", {}),
            dependencies=tuple(step["dependencies"]) if step.get("dependencies") else None,
            retry_policy=_deserialize_retry_policy(step.get("retry_policy")),
        )
        for step in data.get("steps", [])
    )

    return Plan(
        id=data["id"],
        goal=data["goal"],
        steps=steps,
        reasoning=data.get("reasoning"),
        confidence=data.get("confidence"),
        metadata=data.get("metadata", {}),
    )


def _serialize_model_call_record(record: ModelCallRecord) -> dict[str, Any]:
    """Serialize ModelCallRecord to dict."""
    return {
        "model_id": record.model_id,
        "call_id": record.call_id,
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
        "cost_usd": record.cost_usd,
        "duration_ms": record.duration_ms,
        "timestamp": record.timestamp,
    }


def _deserialize_model_call_record(data: dict[str, Any]) -> ModelCallRecord:
    """Deserialize ModelCallRecord from dict."""
    return ModelCallRecord(
        model_id=data["model_id"],
        call_id=data["call_id"],
        input_tokens=data["input_tokens"],
        output_tokens=data["output_tokens"],
        cost_usd=data["cost_usd"],
        duration_ms=data["duration_ms"],
        timestamp=data["timestamp"],
    )


def _serialize_tool_call_record(record: Any) -> dict[str, Any]:
    """Serialize ToolCallRecord to dict."""
    include_sensitive = persist_sensitive_tool_data_enabled()
    args = record.args if include_sensitive else redact_sensitive_data(record.args)
    result = record.result if include_sensitive else redact_sensitive_data(record.result)
    error = record.error if include_sensitive else redact_sensitive_data(record.error)
    return {
        "tool_name": record.tool_name,
        "call_id": record.call_id,
        "args": args,
        "result": result,
        "error": error,
        "cached": record.cached,
        "duration_ms": record.duration_ms,
        "timestamp": record.timestamp,
    }


def _deserialize_tool_call_record(data: dict[str, Any]) -> Any:
    """Deserialize ToolCallRecord from dict."""
    from axis_core.tool import ToolCallRecord

    return ToolCallRecord(
        tool_name=data["tool_name"],
        call_id=data["call_id"],
        args=data["args"],
        result=data["result"],
        error=data.get("error"),
        cached=data.get("cached", False),
        duration_ms=data.get("duration_ms", 0.0),
        timestamp=data.get("timestamp", 0.0),
    )


def _serialize_error_record(record: ErrorRecord) -> dict[str, Any]:
    """Serialize ErrorRecord to dict."""
    return {
        "error": _serialize_axis_error(record.error),
        "timestamp": record.timestamp.isoformat(),
        "phase": record.phase,
        "cycle": record.cycle,
        "recovered": record.recovered,
    }


def _deserialize_error_record(data: dict[str, Any]) -> ErrorRecord:
    """Deserialize ErrorRecord from dict."""
    return ErrorRecord(
        error=_deserialize_axis_error(data["error"]),
        timestamp=datetime.fromisoformat(data["timestamp"]),
        phase=data["phase"],
        cycle=data["cycle"],
        recovered=data["recovered"],
    )


def _serialize_budget(budget: Budget) -> dict[str, Any]:
    """Serialize Budget to dict."""
    return {
        "max_cycles": budget.max_cycles,
        "max_tool_calls": budget.max_tool_calls,
        "max_model_calls": budget.max_model_calls,
        "max_cost_usd": budget.max_cost_usd,
        "max_wall_time_seconds": budget.max_wall_time_seconds,
        "max_input_tokens": budget.max_input_tokens,
        "max_output_tokens": budget.max_output_tokens,
        "warn_at_cost_usd": budget.warn_at_cost_usd,
    }


def _deserialize_budget(data: dict[str, Any]) -> Budget:
    """Deserialize Budget from dict."""
    if not data:
        return Budget()
    return Budget(
        max_cycles=data.get("max_cycles", 10),
        max_tool_calls=data.get("max_tool_calls", 50),
        max_model_calls=data.get("max_model_calls", 20),
        max_cost_usd=data.get("max_cost_usd", 1.00),
        max_wall_time_seconds=data.get("max_wall_time_seconds", 300.0),
        max_input_tokens=data.get("max_input_tokens"),
        max_output_tokens=data.get("max_output_tokens"),
        warn_at_cost_usd=data.get("warn_at_cost_usd", 0.80),
    )


def _serialize_budget_state(state: BudgetState) -> dict[str, Any]:
    """Serialize BudgetState to dict."""
    return {
        "cycles": state.cycles,
        "tool_calls": state.tool_calls,
        "model_calls": state.model_calls,
        "input_tokens": state.input_tokens,
        "output_tokens": state.output_tokens,
        "cost_usd": state.cost_usd,
        "wall_time_seconds": state.wall_time_seconds,
    }


def _deserialize_budget_state(data: dict[str, Any]) -> BudgetState:
    """Deserialize BudgetState from dict."""
    if not data:
        return BudgetState()
    return BudgetState(
        cycles=data.get("cycles", 0),
        tool_calls=data.get("tool_calls", 0),
        model_calls=data.get("model_calls", 0),
        input_tokens=data.get("input_tokens", 0),
        output_tokens=data.get("output_tokens", 0),
        cost_usd=data.get("cost_usd", 0.0),
        wall_time_seconds=data.get("wall_time_seconds", 0.0),
    )
