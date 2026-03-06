"""Core context dataclasses and run-state types."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from axis_core.attachments import AttachmentLike, serialize_attachments
from axis_core.budget import Budget, BudgetState
from axis_core.errors import ErrorRecord

if TYPE_CHECKING:
    from axis_core.protocols.model import ModelResponse, ToolCall
    from axis_core.protocols.planner import Plan

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
    errors: dict[str, Any] = field(default_factory=dict)
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
    error: Any = None
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
        """Serialize CycleState to a dictionary."""
        from axis_core.context import codec

        return {
            "cycle_number": self.cycle_number,
            "observation": codec._serialize_observation(self.observation),
            "plan": codec._serialize_plan(self.plan),
            "execution": codec._serialize_execution_result(self.execution),
            "evaluation": codec._serialize_eval_decision(self.evaluation),
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CycleState:
        """Deserialize CycleState from a dictionary."""
        from axis_core.context import codec

        return cls(
            cycle_number=data["cycle_number"],
            observation=codec._deserialize_observation(data["observation"]),
            plan=codec._deserialize_plan(data["plan"]),
            execution=codec._deserialize_execution_result(data["execution"]),
            evaluation=codec._deserialize_eval_decision(data["evaluation"]),
            started_at=datetime.fromisoformat(data["started_at"]),
            ended_at=datetime.fromisoformat(data["ended_at"]),
        )


@dataclass
class RunState:
    """Mutable state accumulator for an agent run.

    Provides append-only semantics for tracking cycles, errors, and calls.
    Properties return immutable tuples to prevent external mutation.

    Per AD-014, retry state is NOT persisted and is reset on resume.
    """

    _cycles: list[CycleState] = field(default_factory=list, repr=False)
    _errors: list[ErrorRecord] = field(default_factory=list, repr=False)
    _tool_calls: list[Any] = field(default_factory=list, repr=False)
    _model_calls: list[ModelCallRecord] = field(default_factory=list, repr=False)

    current_observation: Observation | None = None
    current_plan: Plan | None = None
    current_execution: ExecutionResult | None = None

    last_model_response: ModelResponse | None = None

    budget_state: BudgetState = field(default_factory=BudgetState)

    output: Any = None
    output_raw: str | None = None

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
        """Append a completed cycle to history."""
        self._cycles.append(cycle)

    def append_error(self, error: ErrorRecord) -> None:
        """Append an error record to history."""
        self._errors.append(error)

    def append_tool_call(self, record: Any) -> None:
        """Append a tool call record to history."""
        self._tool_calls.append(record)

    def append_model_call(self, record: ModelCallRecord) -> None:
        """Append a model call record to history."""
        self._model_calls.append(record)

    def build_messages(
        self,
        ctx: RunContext,
        strategy: str = "smart",
        max_cycles: int = 5,
    ) -> list[dict[str, Any]]:
        """Build the message array for the next model call."""
        messages: list[dict[str, Any]] = []

        session_history = ctx.context.get("__session_history__")
        if isinstance(session_history, list):
            for session_msg in session_history:
                if isinstance(session_msg, dict):
                    messages.append(dict(session_msg))

        first_message_content = ctx.input.text

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

        if strategy == "full":
            cycles_to_include = self._cycles
        elif strategy == "smart":
            cycles_to_include = self._cycles[-max_cycles:] if self._cycles else []
        elif strategy == "minimal":
            cycles_to_include = []
        else:
            cycles_to_include = self._cycles[-max_cycles:] if self._cycles else []

        for cycle in cycles_to_include:
            if cycle.observation.response or cycle.observation.tool_requests:
                msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": cycle.observation.response or "",
                }

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

            if cycle.execution and cycle.execution.results:
                for step in cycle.plan.steps if cycle.plan else []:
                    if step.id in cycle.execution.results:
                        tool_call_id = step.payload.get("tool_call_id")
                        if tool_call_id:
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "content": str(cycle.execution.results[step.id]),
                                }
                            )

        return messages

    def to_dict(self) -> dict[str, Any]:
        """Serialize RunState to a dictionary."""
        from axis_core.context import codec

        return {
            "cycles": [cycle.to_dict() for cycle in self._cycles],
            "errors": [codec._serialize_error_record(error) for error in self._errors],
            "tool_calls": [codec._serialize_tool_call_record(call) for call in self._tool_calls],
            "model_calls": [codec._serialize_model_call_record(call) for call in self._model_calls],
            "current_observation": (
                codec._serialize_observation(self.current_observation)
                if self.current_observation
                else None
            ),
            "current_plan": codec._serialize_plan(self.current_plan) if self.current_plan else None,
            "current_execution": (
                codec._serialize_execution_result(self.current_execution)
                if self.current_execution
                else None
            ),
            "budget_state": codec._serialize_budget_state(self.budget_state),
            "output": self.output,
            "output_raw": self.output_raw,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunState:
        """Deserialize RunState from a dictionary."""
        from axis_core.context import codec

        state = cls()
        state._cycles = [CycleState.from_dict(cycle) for cycle in data.get("cycles", [])]
        state._errors = [codec._deserialize_error_record(error) for error in data.get("errors", [])]
        state._tool_calls = [
            codec._deserialize_tool_call_record(call) for call in data.get("tool_calls", [])
        ]
        state._model_calls = [
            codec._deserialize_model_call_record(call) for call in data.get("model_calls", [])
        ]

        if data.get("current_observation"):
            state.current_observation = codec._deserialize_observation(data["current_observation"])
        if data.get("current_plan"):
            state.current_plan = codec._deserialize_plan(data["current_plan"])
        if data.get("current_execution"):
            state.current_execution = codec._deserialize_execution_result(data["current_execution"])

        state.budget_state = codec._deserialize_budget_state(data.get("budget_state", {}))
        state.output = data.get("output")
        state.output_raw = data.get("output_raw")
        state._retry_state = {}

        return state


@dataclass
class RunContext:
    """Single source of truth for an agent run."""

    run_id: str
    agent_id: str
    input: NormalizedInput
    context: dict[str, Any]
    attachments: list[AttachmentLike]
    config: Any
    budget: Budget
    state: RunState
    trace: Any
    started_at: datetime
    cycle_count: int
    cancel_token: Any
    _initialized: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Mark context as initialized to enable read-only protection."""
        object.__setattr__(self, "_initialized", True)

    def __setattr__(self, name: str, value: Any) -> None:
        """Enforce read-only identity fields after initialization."""
        if getattr(self, "_initialized", False) and name in ("run_id", "agent_id", "input"):
            raise AttributeError(f"RunContext.{name} is read-only")
        object.__setattr__(self, name, value)

    def serialize(self) -> dict[str, Any]:
        """Serialize RunContext to a dictionary for checkpointing."""
        from axis_core.context import codec

        return {
            "run_id": self.run_id,
            "agent_id": self.agent_id,
            "input": codec._serialize_normalized_input(self.input),
            "context": self.context,
            "attachments": serialize_attachments(self.attachments),
            "config": None,
            "budget": codec._serialize_budget(self.budget),
            "state": self.state.to_dict(),
            "trace": None,
            "started_at": self.started_at.isoformat(),
            "cycle_count": self.cycle_count,
            "cancel_token": None,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> RunContext:
        """Deserialize RunContext from a dictionary."""
        from axis_core.context import codec

        return cls(
            run_id=data["run_id"],
            agent_id=data["agent_id"],
            input=codec._deserialize_normalized_input(data["input"]),
            context=data.get("context", {}),
            attachments=list(data.get("attachments", [])),
            config=data.get("config"),
            budget=codec._deserialize_budget(data.get("budget", {})),
            state=RunState.from_dict(data.get("state", {})),
            trace=data.get("trace"),
            started_at=datetime.fromisoformat(data["started_at"]),
            cycle_count=data.get("cycle_count", 0),
            cancel_token=data.get("cancel_token"),
        )

    def check_size(self) -> tuple[int, bool, bool]:
        """Check the serialized size of this context."""
        serialized = json.dumps(self.serialize(), default=str)
        size = len(serialized.encode("utf-8"))
        should_warn = size >= WARN_CONTEXT_SIZE
        should_fail = size >= MAX_CONTEXT_SIZE
        return (size, should_warn, should_fail)
