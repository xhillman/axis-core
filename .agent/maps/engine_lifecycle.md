# Engine & Lifecycle Map

> **When to open:** Changing execution logic, lifecycle phases, model fallback, tool execution, or the main agent loop.

## Key Files

| File | Responsibility |
|---|---|
| `axis_core/engine/lifecycle.py` | `LifecycleEngine`, engine composition, execute/resume entrypoints |
| `axis_core/engine/cycle_runner.py` | Dedicated observe/plan/act/evaluate cycle orchestration |
| `axis_core/checkpoint.py` | Checkpoint envelopes, boundary validation, resume-state materialization |
| `axis_core/engine/phases/initialize.py` | Create `RunContext`, validate config, emit telemetry |
| `axis_core/engine/phases/observe.py` | Gather input, memory, and transcript context |
| `axis_core/engine/phases/plan.py` | Planner invocation and plan validation |
| `axis_core/engine/phases/act.py` | Step orchestration, dependency skipping, error wrapping |
| `axis_core/engine/phases/act_tool_execution.py` | Tool execution policy, confirmation, retry, caching, idempotency |
| `axis_core/engine/phases/act_model_execution.py` | Model execution, transcript normalization, tool manifests, fallback/caching |
| `axis_core/engine/phases/act_model_invocation.py` | Model fallback selection, streaming aggregation, and usage estimation helpers |
| `axis_core/engine/phases/act_runtime_settings.py` | Step → config → default act-phase setting resolution |
| `axis_core/engine/phases/evaluate.py` | Stop/continue decisions and budget checks |
| `axis_core/engine/phases/finalize.py` | Persist memory, flush telemetry, build `RunResult` |
| `axis_core/engine/runtime_policy.py` | Shared timeout, retry, rate-limit, and cache services |
| `axis_core/engine/registry.py` | Registries, lazy-factory helpers, entry-point loading |
| `axis_core/engine/resolver.py` | String/name → adapter resolution |
| `axis_core/engine/trace_collector.py` | Buffered trace accumulation |
| `axis_core/agent.py` | Public API wrapper over the engine |

## Execution Flow

```text
Agent.run(prompt) / run_async(prompt)
  → Agent._build_engine() → LifecycleEngine
  → LifecycleEngine.execute() / resume()
      → initialize or `prepare_checkpoint_resume()`
      → LifecycleCycleRunner.run()
          → observe → plan → act → evaluate
          → repeat loop until done
      → finalize
```

## Ownership Boundaries

- `lifecycle.py` owns engine composition, public execute/resume entrypoints, and phase delegation
- `cycle_runner.py` owns the steady-state observe/plan/act/evaluate loop and phase-boundary checkpoint persistence
- `checkpoint.py` owns checkpoint envelope validation and resume-state materialization
- `phases/*.py` own individual phase behavior
- `act.py` owns step orchestration only; tool/model execution policy now lives in dedicated act services
- `act_tool_execution.py` owns tool policy, destructive confirmation, retry, caching, and idempotency
- `act_model_execution.py` owns transcript normalization, context-window policy, request assembly, caching, and telemetry updates around model steps
- `act_model_invocation.py` owns fallback retries, streaming response aggregation, and token/cost estimation helpers shared by model execution
- `act_runtime_settings.py` owns step payload → resolved config → default precedence for act-phase knobs
- `agent.py` owns public API surface, not engine internals
- `runtime_policy.py` owns shared timeout, rate-limit, retry, and cache helpers used across the loop
- `registry.py` owns adapter factories and plugin discovery

## Common Change Patterns

- **Phase enum or checkpoint boundary change** → update `lifecycle.py`, `cycle_runner.py`, `checkpoint.py`, and telemetry/checkpoint references
- **Plan/PlanStep change** → update `protocols/planner.py`, `plan.py`, `act.py`, and any affected act-phase service
- **RunContext change** → review all six phase modules
- **Budget constraint change** → update `evaluate.py` and `budget.py`
- **Tool execution policy change** → update `act_tool_execution.py`, relevant config/runtime settings, and tool/engine regression tests
- **Model-calling change** → update `act_model_execution.py` and/or `act_model_invocation.py`, `act_runtime_settings.py` when needed, and regression tests around transcript/tool-manifest handling

## Sharp Edges

- Resume correctness depends on `checkpoint.py`, `lifecycle.py`, and `cycle_runner.py` agreeing on phase boundaries and saved state
- `act.py` is now a coordinator over dedicated services; avoid re-introducing tool/model policy logic there
- `act_model_execution.py` and `act_model_invocation.py` split request preparation from model-calling mechanics; keep that boundary intact
- `lifecycle.py` and `cycle_runner.py` now split execution orchestration, so changes must preserve phase/checkpoint order across both files
- `finalize()` persists memory in a non-fatal try/except path
- Phase functions are standalone functions, not methods
