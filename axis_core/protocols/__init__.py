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

import importlib

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Model protocol
    "ModelAdapter": ("axis_core.protocols.model", "ModelAdapter"),
    "ModelResponse": ("axis_core.protocols.model", "ModelResponse"),
    "ModelChunk": ("axis_core.protocols.model", "ModelChunk"),
    "ToolCall": ("axis_core.protocols.model", "ToolCall"),
    "UsageStats": ("axis_core.protocols.model", "UsageStats"),
    # Memory protocol
    "MemoryAdapter": ("axis_core.protocols.memory", "MemoryAdapter"),
    "MemoryItem": ("axis_core.protocols.memory", "MemoryItem"),
    "MemoryCapability": ("axis_core.protocols.memory", "MemoryCapability"),
    "SessionStore": ("axis_core.protocols.memory", "SessionStore"),
    # Planner protocol
    "Planner": ("axis_core.protocols.planner", "Planner"),
    "Plan": ("axis_core.protocols.planner", "Plan"),
    "PlanStep": ("axis_core.protocols.planner", "PlanStep"),
    "StepType": ("axis_core.protocols.planner", "StepType"),
    # Telemetry protocol
    "TelemetrySink": ("axis_core.protocols.telemetry", "TelemetrySink"),
    "TraceEvent": ("axis_core.protocols.telemetry", "TraceEvent"),
    "BufferMode": ("axis_core.protocols.telemetry", "BufferMode"),
}

__all__ = list(_LAZY_IMPORTS.keys())


def __getattr__(name: str) -> object:
    """Lazy loading of protocol classes to avoid circular imports."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
