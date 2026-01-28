"""Adapter protocol interfaces for axis-core.

This module defines Protocol interfaces for pluggable adapters:
- ModelAdapter: LLM providers (Anthropic, OpenAI, Ollama)
- MemoryAdapter: State persistence (Ephemeral, SQLite, Redis)
- Planner: Planning strategies (ReAct, Sequential, Auto)
- TelemetrySink: Observability backends (stdout, OpenTelemetry, LangSmith)

Usage:
    from axis_core.protocols import ModelAdapter, MemoryAdapter
    from axis_core.protocols import UsageStats, ModelResponse, MemoryItem

All protocols use structural subtyping via typing.Protocol - implementations
don't need to explicitly inherit from these interfaces.
"""

__all__ = [
    # Model protocol
    "ModelAdapter",
    "ModelResponse",
    "ModelChunk",
    "ToolCall",
    "UsageStats",
    # Memory protocol
    "MemoryAdapter",
    "MemoryItem",
    "MemoryCapability",
    # Planner protocol
    "Planner",
    "Plan",
    "PlanStep",
    "StepType",
    # Telemetry protocol
    "TelemetrySink",
    "TraceEvent",
    "BufferMode",
]


def __getattr__(name: str) -> object:
    """Lazy loading of protocol classes to avoid circular imports."""
    # Model protocol
    if name == "ModelAdapter":
        from axis_core.protocols.model import ModelAdapter

        return ModelAdapter
    if name == "ModelResponse":
        from axis_core.protocols.model import ModelResponse

        return ModelResponse
    if name == "ModelChunk":
        from axis_core.protocols.model import ModelChunk

        return ModelChunk
    if name == "ToolCall":
        from axis_core.protocols.model import ToolCall

        return ToolCall
    if name == "UsageStats":
        from axis_core.protocols.model import UsageStats

        return UsageStats

    # Memory protocol
    if name == "MemoryAdapter":
        from axis_core.protocols.memory import MemoryAdapter

        return MemoryAdapter
    if name == "MemoryItem":
        from axis_core.protocols.memory import MemoryItem

        return MemoryItem
    if name == "MemoryCapability":
        from axis_core.protocols.memory import MemoryCapability

        return MemoryCapability

    # Planner protocol
    if name == "Planner":
        from axis_core.protocols.planner import Planner

        return Planner
    if name == "Plan":
        from axis_core.protocols.planner import Plan

        return Plan
    if name == "PlanStep":
        from axis_core.protocols.planner import PlanStep

        return PlanStep
    if name == "StepType":
        from axis_core.protocols.planner import StepType

        return StepType

    # Telemetry protocol
    if name == "TelemetrySink":
        from axis_core.protocols.telemetry import TelemetrySink

        return TelemetrySink
    if name == "TraceEvent":
        from axis_core.protocols.telemetry import TraceEvent

        return TraceEvent
    if name == "BufferMode":
        from axis_core.protocols.telemetry import BufferMode

        return BufferMode

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
