# Engine & Lifecycle Map

> **When to open:** Changing execution logic, lifecycle phases, model fallback, tool execution, or the main agent loop.

## Key Files

| File | Responsibility |
|---|---|
| `axis_core/engine/lifecycle.py` | `LifecycleEngine`, loop orchestration, checkpoint/resume coordination |
| `axis_core/engine/cycle_runner.py` | Dedicated observe/plan/act/evaluate cycle orchestration |
| `axis_core/engine/phases/initialize.py` | Create `RunContext`, validate config, emit telemetry |
| `axis_core/engine/phases/observe.py` | Gather input, memory, and transcript context |
| `axis_core/engine/phases/plan.py` | Planner invocation and plan validation |
| `axis_core/engine/phases/act.py` | Tool execution, model execution, fallback, retry/policy enforcement |
| `axis_core/engine/phases/evaluate.py` | Stop/continue decisions and budget checks |
| `axis_core/engine/phases/finalize.py` | Persist memory, flush telemetry, build `RunResult` |
| `axis_core/engine/registry.py` | Registries, lazy-factory helpers, entry-point loading |
| `axis_core/engine/resolver.py` | String/name → adapter resolution |
| `axis_core/engine/trace_collector.py` | Buffered trace accumulation |
| `axis_core/agent.py` | Public API wrapper over the engine |

## Execution Flow

```text
Agent.run(prompt) / run_async(prompt)
  → Agent._build_engine() → LifecycleEngine
  → LifecycleEngine.execute()
      → initialize → observe → plan → act → evaluate
      → repeat loop until done
      → finalize
```

## Ownership Boundaries

- `lifecycle.py` owns engine composition, public execute/resume entrypoints, and checkpoint/resume coordination
- `cycle_runner.py` owns the steady-state observe/plan/act/evaluate loop and finalize handoff
- `phases/*.py` own individual phase behavior
- `act.py` is the heaviest phase and still carries most execution policy logic
- `agent.py` owns public API surface, not engine internals
- `registry.py` owns adapter factories and plugin discovery

## Common Change Patterns

- **Phase enum change** → update lifecycle dispatch and telemetry/checkpoint references
- **Plan/PlanStep change** → update `protocols/planner.py`, `plan.py`, and `act.py`
- **RunContext change** → review all six phase modules
- **Budget constraint change** → update `evaluate.py` and `budget.py`
- **Model-calling change** → update `act.py` (`try_models_with_fallback`, `_execute_model_step`) and regression tests around transcript/tool-manifest handling

## Sharp Edges

- `act.py` mixes tool execution and model execution and is still the largest lifecycle hotspot
- `lifecycle.py` and `cycle_runner.py` now split execution orchestration, so changes must preserve phase/checkpoint order across both files
- `finalize()` persists memory in a non-fatal try/except path
- Phase functions are standalone functions, not methods
