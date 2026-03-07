# Protocols & Types Map

> **When to open:** Changing adapter interfaces, error types, data structures, or shared types.

## Protocol Files

| File | Protocol | Key Types | Implementors |
|---|---|---|---|
| `axis_core/protocols/model.py` | `ModelAdapter` | `ModelResponse`, `ModelChunk`, `ToolCall`, `UsageStats` | `AnthropicModel`, `OpenAIModel`, `OpenAIResponsesModel` |
| `axis_core/protocols/memory.py` | `MemoryAdapter`, `SessionStore` | `MemoryItem`, `MemoryCapability` | `EphemeralMemory`, `SQLiteMemory`, `RedisMemory`, `SynapticMemory` |
| `axis_core/protocols/planner.py` | `Planner` | `Plan`, `PlanStep`, `StepType` | `SequentialPlanner`, `AutoPlanner`, `ReActPlanner` |
| `axis_core/protocols/telemetry.py` | `TelemetrySink` | `TraceEvent`, `BufferMode` | `ConsoleSink`, `FileSink`, `CallbackSink`, `TraceCollector` |

**All protocols use `typing.Protocol` (structural typing, no ABCs).**

## Error Hierarchy

```text
axis_core/errors.py

AxisError (base)
├── InputError
├── ConfigError
├── PlanError
├── TimeoutError
├── CancelledError
├── ConcurrencyError
├── ToolError
├── ModelError
└── BudgetError
```

`ErrorClass` and `ErrorRecord` live alongside the concrete error types.

## Core Data Types

| File | Key Types | Frozen? |
|---|---|---|
| `axis_core/context/types.py` | `RunContext`, `RunState`, `CycleState`, `Observation`, `ExecutionResult`, `EvalDecision`, `ModelCallRecord` | Mixed |
| `axis_core/result.py` | `RunResult`, `StreamEvent`, `RunStats` | Yes |
| `axis_core/budget.py` | `Budget`, `BudgetState` | Budget frozen, BudgetState mutable |
| `axis_core/_tool_types.py` | `ToolManifest`, `ToolCallRecord`, `Capability` | Mixed |
| `axis_core/_tool_runtime.py` | `ToolContext`, `RateLimiter` | Mixed |
| `axis_core/tool.py` | Public export facade for tool types/runtime helpers | N/A |
| `axis_core/session.py` | `Session`, `Message`, `ContentPart` | Session mutable |
| `axis_core/attachments.py` | `Attachment`, `Image`, `PDF` | Yes |
| `axis_core/cancel.py` | `CancelToken` | Mutable |
| `axis_core/config.py` | `Config`, `ResolvedConfig`, `RuntimeSettings`, `TelemetrySettings`, `Timeouts`, `RetryPolicy`, `RateLimits`, `CacheConfig`, `ToolPolicy` | Config mutable singleton |

## Ownership Boundaries

- `protocols/` defines interfaces; adapters implement them and the engine consumes them
- `errors.py` is standalone and imported broadly
- `context/types.py` is the state backbone for lifecycle execution
- `config.py` owns env loading, runtime/telemetry setting resolution, and explicit bootstrap

## Common Change Patterns

- **Add a protocol method** → update every implementor and protocol test
- **Add an error type** → update `ErrorClass`, exports, and any classification logic
- **Change RunContext** → review `context/__init__.py`, all phase modules, and checkpoint/serialization paths
- **Change ToolManifest or tool metadata types** → review `_tool_types.py`, `_tool_decorator.py`, `tool.py` exports, and model adapter tool conversion
- **Change ToolContext or runtime helpers** → review `_tool_runtime.py`, act-phase services, and tool tests
- **Change telemetry protocol types** → review `protocols/telemetry.py`, telemetry adapters, and `TraceCollector`

## Sharp Edges

- `context/types.py` uses `TYPE_CHECKING` imports to avoid circular dependencies
- `session.py` uses `TYPE_CHECKING` for memory protocol references
- `RunState` uses append-only lists exposed as tuples
- `config.py` merges defaults → env → constructor → runtime
- `RateLimiter` now lives in `_tool_runtime.py`; review engine integration before changing it
