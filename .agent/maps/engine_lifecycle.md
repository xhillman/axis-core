# Engine & Lifecycle Map

> **When to open:** Changing execution logic, lifecycle phases, model fallback, tool execution, or the main agent loop.

## Key Files

| File | Lines | Responsibility |
|---|---|---|
| `axis_core/engine/lifecycle.py` | 489 | `LifecycleEngine.execute()` — main loop, phase orchestration |
| `axis_core/engine/phases/initialize.py` | 93 | Create RunContext, validate config, emit telemetry |
| `axis_core/engine/phases/observe.py` | 108 | Gather input, load memory, assess state |
| `axis_core/engine/phases/plan.py` | 120 | Call planner, validate plan (AD-006) |
| `axis_core/engine/phases/act.py` | 618 | Execute steps: tool calls + model calls, fallback (AD-013) |
| `axis_core/engine/phases/evaluate.py` | 137 | Check termination: cancel, terminal step, budget, errors |
| `axis_core/engine/phases/finalize.py` | 96 | Persist memory (non-fatal AD-007), emit summary |
| `axis_core/engine/registry.py` | 223 | `AdapterRegistry`, `make_lazy_factory()`, built-in registration |
| `axis_core/engine/resolver.py` | 72 | `resolve_adapter()` — string to adapter instance |
| `axis_core/engine/trace_collector.py` | 63 | Event accumulator (BufferMode.END) |
| `axis_core/agent.py` | 783 | `Agent` class — public API, builds engine, wraps async |

## Execution Flow

```
Agent.run(prompt) / run_async(prompt)
  → Agent._build_engine() → LifecycleEngine
  → LifecycleEngine.execute()
      → phases.initialize()     → RunContext created
      → LOOP:
          → phases.observe()    → Observation
          → phases.plan()       → Plan (via Planner adapter)
          → phases.act()        → ExecutionResult (tool + model steps)
          → phases.evaluate()   → EvalDecision (continue/stop)
      → phases.finalize()       → RunResult
```

## Ownership Boundaries

- **lifecycle.py** owns the loop and phase dispatch — do NOT add execution logic here
- **phases/*.py** own individual phase behavior — each is self-contained
- **act.py** is the largest phase (618L) — owns tool execution, model fallback, step dependencies
- **agent.py** owns public API surface — do NOT put engine logic here
- **registry.py** owns adapter factories — adapters register themselves here

## Common Change Patterns

- **If you change a Phase enum value** → update lifecycle.py phase dispatch + telemetry events
- **If you change Plan/PlanStep** → update `protocols/planner.py`, then `phases/plan.py` and `phases/act.py`
- **If you change RunContext** → check all 6 phase modules (they all receive it)
- **If you add a budget constraint** → update `phases/evaluate.py` and `budget.py`
- **If you change model calling** → update `phases/act.py` (`try_models_with_fallback`, `_execute_model_step`)

## Sharp Edges

- `act.py` has both tool execution AND model execution — they share dependency resolution logic
- `lifecycle.py` was split from 1,400 lines → phases are now separate modules but lifecycle.py still orchestrates
- `finalize()` persists memory in a try/except (AD-007: memory persistence is non-fatal)
- Model fallback in `act.py` (`try_models_with_fallback`) retries with fallback models on failure
- Phase functions are standalone functions, not methods — they receive all deps as arguments
