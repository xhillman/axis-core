"""Context and state management for axis-core agent execution.

This package provides the core state management system including:
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

from axis_core.context.transcript import (
    ContextWindowAssessment,
    ContextWindowGuard,
    estimate_transcript_tokens,
    normalize_transcript_messages,
    prune_messages_for_context_window,
)
from axis_core.context.types import (
    MAX_CONTEXT_SIZE,
    WARN_CONTEXT_SIZE,
    CycleState,
    EvalDecision,
    ExecutionResult,
    ModelCallRecord,
    NormalizedInput,
    Observation,
    RunContext,
    RunState,
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
    "ContextWindowAssessment",
    "ContextWindowGuard",
    "WARN_CONTEXT_SIZE",
    "MAX_CONTEXT_SIZE",
    "estimate_transcript_tokens",
    "normalize_transcript_messages",
    "prune_messages_for_context_window",
]
