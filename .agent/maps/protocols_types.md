# Protocols & Types Map

> **When to open:** Changing adapter interfaces, error types, data structures, or shared types.

## Protocol Files

| File | Protocol | Key Types | Implementors |
|---|---|---|---|
| `axis_core/protocols/model.py` (217L) | `ModelAdapter` | `ModelResponse`, `ModelChunk`, `ToolCall`, `UsageStats` | `AnthropicModel`, `OpenAIModel` |
| `axis_core/protocols/memory.py` (188L) | `MemoryAdapter`, `SessionStore` | `MemoryItem`, `MemoryCapability` (Enum) | `EphemeralMemory`, `SQLiteMemory`, `RedisMemory` |
| `axis_core/protocols/planner.py` (109L) | `Planner` | `Plan`, `PlanStep`, `StepType` (Enum) | `SequentialPlanner`, `AutoPlanner`, `ReActPlanner` |
| `axis_core/protocols/telemetry.py` (99L) | `TelemetrySink` | `TraceEvent`, `BufferMode` (Enum) | `ConsoleSink`, `TraceCollector` |

**All protocols use `typing.Protocol` (structural typing, no ABCs).**

## Error Hierarchy

```
axis_core/errors.py (234L)

AxisError (base)
├── InputError        # Bad user input
├── ConfigError       # Invalid configuration
├── PlanError         # Planning failure
├── TimeoutError      # Phase/total timeout exceeded
├── CancelledError    # User-initiated cancellation
├── ConcurrencyError  # Race condition / locking
├── ToolError         # Tool execution failure
├── ModelError        # Model API failure
└── BudgetError       # Budget limit exceeded

ErrorClass (Enum): INPUT, CONFIG, PLAN, TIMEOUT, CANCEL, CONCURRENCY, TOOL, MODEL, BUDGET
ErrorRecord: classification + recovery info for error tracking
```

## Core Data Types

| File | Key Types | Frozen? |
|---|---|---|
| `context.py` (965L) | `RunContext`, `RunState`, `CycleState`, `NormalizedInput`, `Observation`, `ExecutionResult`, `EvalDecision`, `ModelCallRecord` | Mixed (RunContext mutable, most others frozen) |
| `result.py` (119L) | `RunResult`, `StreamEvent`, `RunStats` | Yes |
| `budget.py` (115L) | `Budget`, `BudgetState` | Budget frozen, BudgetState mutable |
| `tool.py` (445L) | `ToolManifest`, `ToolContext`, `ToolCallRecord`, `Capability` (Enum), `RateLimiter` | Manifest frozen |
| `session.py` (345L) | `Session`, `Message`, `ContentPart` | Session mutable |
| `attachments.py` (119L) | `Attachment`, `Image`, `PDF` | Yes |
| `cancel.py` (29L) | `CancelToken` | Mutable (set/check) |
| `config.py` (320L) | `Config`, `ResolvedConfig`, `Timeouts`, `RetryPolicy`, `RateLimits`, `CacheConfig` | Config mutable singleton |

## Ownership Boundaries

- **protocols/** defines interfaces — adapters implement them, engine consumes them
- **errors.py** is standalone — imported everywhere, depends on nothing
- **context.py** is the state backbone — lifecycle phases read/write via RunContext
- **config.py** owns env var loading — all other modules read from `config` singleton

## Common Change Patterns

- **If you add a protocol method** → must update ALL implementors (check adapters map)
- **If you add an error type** → add to `ErrorClass` enum + create subclass + update `__init__.py` exports
- **If you change RunContext** → affects all 6 phase modules + agent.py
- **If you change ToolManifest** → affects `act.py` (schema extraction), model adapters (tool conversion)

## Sharp Edges

- `context.py` uses TYPE_CHECKING imports for `protocols.model` and `protocols.planner` to avoid circular deps
- `session.py` uses TYPE_CHECKING import for `protocols.memory`
- `RunState` uses append-only pattern (private `_list`, public `tuple` property)
- `config.py` has `deep_merge()` (AD-015) for merging config layers — resolution order: defaults → env → constructor → runtime
- `RateLimiter` in tool.py is defined and tested but NOT wired into the lifecycle
