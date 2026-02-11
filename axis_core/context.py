"""Context and state management for axis-core agent execution.

This module provides the core state management system including:
- NormalizedInput: Normalized user input (text + original)
- Observation: Output from the Observe phase
- ExecutionResult: Output from the Act phase
- EvalDecision: Output from the Evaluate phase
- ModelCallRecord: Record of a single LLM call
- CycleState: Complete record of one observe-plan-act-evaluate cycle
- RunState: Mutable state accumulator with append-only semantics
- RunContext: Single source of truth for an agent run

Architecture Decisions:
- AD-005: Checkpoint at phase boundaries; serialize()/deserialize() methods
- AD-014: Persist error history; reset retry counters on resume
- AD-037: Warn at 50MB context size, fail at 100MB
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from axis_core.protocols.model import ModelResponse, ToolCall
    from axis_core.protocols.planner import Plan

from axis_core.attachments import AttachmentLike, serialize_attachments
from axis_core.budget import Budget, BudgetState
from axis_core.errors import AxisError, ErrorClass, ErrorRecord
from axis_core.redaction import (
    persist_sensitive_tool_data_enabled,
    redact_sensitive_data,
)

# Size limits for context (AD-037)
WARN_CONTEXT_SIZE = 50 * 1024 * 1024  # 50MB
MAX_CONTEXT_SIZE = 100 * 1024 * 1024  # 100MB


@dataclass(frozen=True)
class NormalizedInput:
    """Normalized representation of user input.

    Stores both the normalized text form and the original input, which may be
    a string or a list of multimodal content blocks.

    Attributes:
        text: Normalized text representation of the input
        original: Original input (str or list for multimodal)
        is_multimodal: Whether input contains non-text content (images, etc.)
    """

    text: str
    original: str | list[Any]
    is_multimodal: bool = False


@dataclass(frozen=True)
class Observation:
    """Output from the Observe phase of execution.

    Captures the current state of the world as seen by the agent, including
    the user input, relevant memory context, and any pending tool requests
    from a previous model response.

    Attributes:
        input: Normalized user input
        memory_context: Relevant context retrieved from memory
        previous_cycles: Summary of prior cycles in this run
        tool_requests: Tool calls requested by model (if continuing)
        response: Previous model response (if continuing)
        goal: Extracted or inferred goal for this run
        timestamp: When this observation was created
    """

    input: NormalizedInput
    memory_context: dict[str, Any] = field(default_factory=dict)
    previous_cycles: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    tool_requests: tuple[ToolCall, ...] | None = None
    response: str | None = None
    goal: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class ExecutionResult:
    """Output from the Act phase of execution.

    Contains results from tool executions, any errors encountered, and
    tools that were skipped (e.g., due to dependencies or rate limits).

    Attributes:
        results: Map of tool name to execution result
        errors: Map of tool name to error encountered
        skipped: Set of tool names that were skipped
        duration_ms: Total execution time in milliseconds
    """

    results: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, AxisError] = field(default_factory=dict)
    skipped: frozenset[str] = field(default_factory=frozenset)
    duration_ms: float = 0.0


@dataclass(frozen=True)
class EvalDecision:
    """Output from the Evaluate phase of execution.

    Determines whether the agent should continue cycling or has completed
    its task, and captures any errors that occurred.

    Attributes:
        done: Whether the task is complete
        error: Error that occurred (if any)
        recoverable: Whether the error is recoverable (can retry)
        reason: Human-readable explanation of the decision
    """

    done: bool
    error: AxisError | None = None
    recoverable: bool = False
    reason: str = ""


@dataclass(frozen=True)
class ModelCallRecord:
    """Immutable record of a single LLM API call.

    Captures all information about a model invocation for observability,
    debugging, and cost tracking. Similar to ToolCallRecord.

    Attributes:
        model_id: Identifier of the model called
        call_id: Unique identifier for this specific call
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cost_usd: Cost in USD for this call
        duration_ms: Execution time in milliseconds
        timestamp: Unix timestamp when call started
    """

    model_id: str
    call_id: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    duration_ms: float
    timestamp: float


@dataclass(frozen=True)
class CycleState:
    """Complete record of one observe-plan-act-evaluate cycle.

    Immutable snapshot of a completed cycle, used for history tracking
    and checkpointing.

    Attributes:
        cycle_number: Zero-indexed cycle number
        observation: Observation from this cycle
        plan: Plan generated for this cycle
        execution: Results from executing the plan
        evaluation: Decision made after execution
        started_at: When this cycle started
        ended_at: When this cycle ended
    """

    cycle_number: int
    observation: Observation
    plan: Plan
    execution: ExecutionResult
    evaluation: EvalDecision
    started_at: datetime
    ended_at: datetime

    def to_dict(self) -> dict[str, Any]:
        """Serialize CycleState to a dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            "cycle_number": self.cycle_number,
            "observation": _serialize_observation(self.observation),
            "plan": _serialize_plan(self.plan),
            "execution": _serialize_execution_result(self.execution),
            "evaluation": _serialize_eval_decision(self.evaluation),
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CycleState:
        """Deserialize CycleState from a dictionary.

        Args:
            data: Dictionary from to_dict()

        Returns:
            Restored CycleState instance
        """
        return cls(
            cycle_number=data["cycle_number"],
            observation=_deserialize_observation(data["observation"]),
            plan=_deserialize_plan(data["plan"]),
            execution=_deserialize_execution_result(data["execution"]),
            evaluation=_deserialize_eval_decision(data["evaluation"]),
            started_at=datetime.fromisoformat(data["started_at"]),
            ended_at=datetime.fromisoformat(data["ended_at"]),
        )


@dataclass
class RunState:
    """Mutable state accumulator for an agent run.

    Provides append-only semantics for tracking cycles, errors, and calls.
    Properties return immutable tuples to prevent external mutation.

    Per AD-014, retry state is NOT persisted and is reset on resume.

    Attributes:
        current_observation: Observation for current (incomplete) cycle
        current_plan: Plan for current cycle
        current_execution: Execution result for current cycle
        budget_state: Current budget consumption
        output: Final output value
        output_raw: Raw string output from model
    """

    # Private lists for append-only semantics
    _cycles: list[CycleState] = field(default_factory=list, repr=False)
    _errors: list[ErrorRecord] = field(default_factory=list, repr=False)
    _tool_calls: list[Any] = field(default_factory=list, repr=False)  # ToolCallRecord
    _model_calls: list[ModelCallRecord] = field(default_factory=list, repr=False)

    # Current cycle state (write-once per cycle)
    current_observation: Observation | None = None
    current_plan: Plan | None = None
    current_execution: ExecutionResult | None = None

    # Last model response for next Observe phase
    last_model_response: ModelResponse | None = None

    # Budget tracking
    budget_state: BudgetState = field(default_factory=BudgetState)

    # Final output
    output: Any = None
    output_raw: str | None = None

    # NOT persisted (AD-014) - reset on resume
    _retry_state: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def cycles(self) -> tuple[CycleState, ...]:
        """Return immutable view of completed cycles."""
        return tuple(self._cycles)

    @property
    def errors(self) -> tuple[ErrorRecord, ...]:
        """Return immutable view of error history."""
        return tuple(self._errors)

    @property
    def tool_calls(self) -> tuple[Any, ...]:
        """Return immutable view of tool call records."""
        return tuple(self._tool_calls)

    @property
    def model_calls(self) -> tuple[ModelCallRecord, ...]:
        """Return immutable view of model call records."""
        return tuple(self._model_calls)

    def append_cycle(self, cycle: CycleState) -> None:
        """Append a completed cycle to history.

        Args:
            cycle: Completed cycle to add
        """
        self._cycles.append(cycle)

    def append_error(self, error: ErrorRecord) -> None:
        """Append an error record to history.

        Args:
            error: Error record to add
        """
        self._errors.append(error)

    def append_tool_call(self, record: Any) -> None:
        """Append a tool call record to history.

        Args:
            record: ToolCallRecord to add
        """
        self._tool_calls.append(record)

    def append_model_call(self, record: ModelCallRecord) -> None:
        """Append a model call record to history.

        Args:
            record: ModelCallRecord to add
        """
        self._model_calls.append(record)

    def build_messages(
        self,
        ctx: RunContext,
        strategy: str = "smart",
        max_cycles: int = 5,
    ) -> list[dict[str, Any]]:
        """Build message array for next model call.

        Reconstructs conversation history from completed cycles using
        the specified strategy to control token usage.

        Args:
            ctx: Run context with input and cycle history
            strategy: Context strategy ("smart", "full", "minimal")
            max_cycles: For "smart" strategy, number of recent cycles to include

        Returns:
            List of messages in standard internal format

        Strategies:
            - "smart": First message with memory context + last N cycles
            - "full": All cycles (expensive, use for debugging)
            - "minimal": Just first message (most token-efficient)
        """
        messages: list[dict[str, Any]] = []

        # Prepend session history if provided via context
        session_history = ctx.context.get("__session_history__")
        if isinstance(session_history, list):
            for session_msg in session_history:
                if isinstance(session_msg, dict):
                    messages.append(dict(session_msg))

        # Start with original user input (always include)
        first_message_content = ctx.input.text

        # Inject memory context if available (only on first message)
        if self.current_observation and self.current_observation.memory_context:
            mem_ctx = self.current_observation.memory_context
            if mem_ctx.get("relevant_memories"):
                context_parts = ["<relevant_context>"]
                for mem in mem_ctx["relevant_memories"]:
                    context_parts.append(f"- {mem.get('key', '')}: {mem.get('value', '')}")
                context_parts.append("</relevant_context>")
                context_str = "\n".join(context_parts)
                first_message_content = f"{context_str}\n\n{first_message_content}"

        messages.append({"role": "user", "content": first_message_content})

        # Add conversation history based on strategy
        if strategy == "full":
            cycles_to_include = self._cycles
        elif strategy == "smart":
            cycles_to_include = self._cycles[-max_cycles:] if self._cycles else []
        elif strategy == "minimal":
            cycles_to_include = []
        else:
            # Default to smart
            cycles_to_include = self._cycles[-max_cycles:] if self._cycles else []

        # Reconstruct messages from cycles
        for cycle in cycles_to_include:
            # Add assistant message if there's a response OR tool requests
            # (tool-only responses may have empty content but still need the message)
            if cycle.observation.response or cycle.observation.tool_requests:
                msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": cycle.observation.response or "",
                }

                # Add tool calls if present
                if cycle.observation.tool_requests:
                    msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "name": tc.name,
                            "arguments": tc.arguments,
                        }
                        for tc in cycle.observation.tool_requests
                    ]

                messages.append(msg)

            # Add tool results if any tools were executed
            if cycle.execution and cycle.execution.results:
                # Extract tool results from execution
                for step in cycle.plan.steps if cycle.plan else []:
                    if step.id in cycle.execution.results:
                        tool_call_id = step.payload.get("tool_call_id")
                        if tool_call_id:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": str(cycle.execution.results[step.id]),
                            })

        return messages

    def to_dict(self) -> dict[str, Any]:
        """Serialize RunState to a dictionary.

        Note: _retry_state is NOT included (AD-014).

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            "cycles": [c.to_dict() for c in self._cycles],
            "errors": [_serialize_error_record(e) for e in self._errors],
            "tool_calls": [_serialize_tool_call_record(t) for t in self._tool_calls],
            "model_calls": [_serialize_model_call_record(m) for m in self._model_calls],
            "current_observation": (
                _serialize_observation(self.current_observation)
                if self.current_observation
                else None
            ),
            "current_plan": (
                _serialize_plan(self.current_plan) if self.current_plan else None
            ),
            "current_execution": (
                _serialize_execution_result(self.current_execution)
                if self.current_execution
                else None
            ),
            "budget_state": _serialize_budget_state(self.budget_state),
            "output": self.output,
            "output_raw": self.output_raw,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunState:
        """Deserialize RunState from a dictionary.

        Note: _retry_state is reset to empty (AD-014).

        Args:
            data: Dictionary from to_dict()

        Returns:
            Restored RunState instance
        """
        state = cls()
        state._cycles = [CycleState.from_dict(c) for c in data.get("cycles", [])]
        state._errors = [
            _deserialize_error_record(e) for e in data.get("errors", [])
        ]
        state._tool_calls = [
            _deserialize_tool_call_record(t) for t in data.get("tool_calls", [])
        ]
        state._model_calls = [
            _deserialize_model_call_record(m) for m in data.get("model_calls", [])
        ]

        if data.get("current_observation"):
            state.current_observation = _deserialize_observation(
                data["current_observation"]
            )
        if data.get("current_plan"):
            state.current_plan = _deserialize_plan(data["current_plan"])
        if data.get("current_execution"):
            state.current_execution = _deserialize_execution_result(
                data["current_execution"]
            )

        state.budget_state = _deserialize_budget_state(data.get("budget_state", {}))
        state.output = data.get("output")
        state.output_raw = data.get("output_raw")

        # Reset retry state (AD-014)
        state._retry_state = {}

        return state


@dataclass
class RunContext:
    """Single source of truth for an agent run.

    Contains all state needed for agent execution, including identity,
    input, configuration, and mutable state. Identity fields are
    read-only after initialization.

    Per AD-005, supports serialize()/deserialize() for checkpointing.
    Per AD-037, check_size() warns at 50MB and fails at 100MB.

    Attributes:
        run_id: Unique identifier for this run (read-only)
        agent_id: Identifier for the agent (read-only)
        input: Normalized user input (read-only)
        context: Mutable context dict for sharing state
        attachments: List of attachments (metadata-only in serialization)
        config: Resolved configuration
        budget: Budget limits
        state: Mutable run state
        trace: Telemetry trace collector
        started_at: When this run started
        cycle_count: Current cycle number
        cancel_token: Cancellation token
    """

    # Identity (read-only after init)
    run_id: str
    agent_id: str

    # Input (read-only)
    input: NormalizedInput

    # Context and attachments
    context: dict[str, Any]
    attachments: list[AttachmentLike]

    # Configuration
    config: Any  # ResolvedConfig
    budget: Budget

    # State
    state: RunState

    # Telemetry
    trace: Any  # TraceCollector

    # Timing
    started_at: datetime
    cycle_count: int

    # Cancellation
    cancel_token: Any  # CancelToken

    # Read-only protection flag
    _initialized: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Mark context as initialized to enable read-only protection."""
        object.__setattr__(self, "_initialized", True)

    def __setattr__(self, name: str, value: Any) -> None:
        """Enforce read-only fields after initialization.

        Prevents modification of identity fields (run_id, agent_id, input)
        while allowing other fields to be modified.
        """
        if getattr(self, "_initialized", False):
            if name in ("run_id", "agent_id", "input"):
                raise AttributeError(f"RunContext.{name} is read-only")
        object.__setattr__(self, name, value)

    def serialize(self) -> dict[str, Any]:
        """Serialize RunContext to a dictionary for checkpointing.

        Per AD-005, this is used at phase boundaries.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            "run_id": self.run_id,
            "agent_id": self.agent_id,
            "input": _serialize_normalized_input(self.input),
            "context": self.context,
            "attachments": serialize_attachments(self.attachments),
            "config": None,  # Config re-resolved on restore
            "budget": _serialize_budget(self.budget),
            "state": self.state.to_dict(),
            "trace": None,  # Trace not persisted
            "started_at": self.started_at.isoformat(),
            "cycle_count": self.cycle_count,
            "cancel_token": None,  # Token not persisted
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> RunContext:
        """Deserialize RunContext from a dictionary.

        Args:
            data: Dictionary from serialize()

        Returns:
            Restored RunContext instance
        """
        return cls(
            run_id=data["run_id"],
            agent_id=data["agent_id"],
            input=_deserialize_normalized_input(data["input"]),
            context=data.get("context", {}),
            attachments=list(data.get("attachments", [])),
            config=data.get("config"),
            budget=_deserialize_budget(data.get("budget", {})),
            state=RunState.from_dict(data.get("state", {})),
            trace=data.get("trace"),
            started_at=datetime.fromisoformat(data["started_at"]),
            cycle_count=data.get("cycle_count", 0),
            cancel_token=data.get("cancel_token"),
        )

    def check_size(self) -> tuple[int, bool, bool]:
        """Check the serialized size of this context.

        Per AD-037, warns at 50MB and fails at 100MB.

        Returns:
            Tuple of (size_bytes, should_warn, should_fail)
        """
        serialized = json.dumps(self.serialize(), default=str)
        size = len(serialized.encode("utf-8"))
        should_warn = size >= WARN_CONTEXT_SIZE
        should_fail = size >= MAX_CONTEXT_SIZE
        return (size, should_warn, should_fail)


# ============================================================================
# Serialization helpers
# ============================================================================


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
            [_serialize_tool_call(tc) for tc in obs.tool_requests]
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
                id=tc["id"],
                name=tc["name"],
                arguments=tc.get("arguments", {}),
            )
            for tc in data["tool_requests"]
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


def _serialize_tool_call(tc: ToolCall) -> dict[str, Any]:
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
            name: _serialize_axis_error(err) for name, err in result.errors.items()
        },
        "skipped": list(result.skipped),
        "duration_ms": result.duration_ms,
    }


def _deserialize_execution_result(data: dict[str, Any]) -> ExecutionResult:
    """Deserialize ExecutionResult from dict."""
    return ExecutionResult(
        results=data.get("results", {}),
        errors={
            name: _deserialize_axis_error(err) for name, err in data.get("errors", {}).items()
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
        "cause": (
            redact_sensitive_data(str(error.cause))
            if error.cause
            else None
        ),
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
        cause=None,  # Cannot fully restore exception
    )


def _serialize_plan(plan: Plan) -> dict[str, Any]:
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
                "retry_policy": None,  # RetryPolicy serialization simplified
            }
            for step in plan.steps
        ],
        "reasoning": plan.reasoning,
        "confidence": plan.confidence,
        "metadata": plan.metadata,
    }


def _deserialize_plan(data: dict[str, Any]) -> Plan:
    """Deserialize Plan from dict."""
    from axis_core.protocols.planner import Plan, PlanStep, StepType

    steps = tuple(
        PlanStep(
            id=s["id"],
            type=StepType(s["type"]),
            payload=s.get("payload", {}),
            dependencies=tuple(s["dependencies"]) if s.get("dependencies") else None,
            retry_policy=None,  # RetryPolicy deserialization simplified
        )
        for s in data.get("steps", [])
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


__all__ = [
    "NormalizedInput",
    "Observation",
    "ExecutionResult",
    "EvalDecision",
    "ModelCallRecord",
    "CycleState",
    "RunState",
    "RunContext",
    "WARN_CONTEXT_SIZE",
    "MAX_CONTEXT_SIZE",
]
