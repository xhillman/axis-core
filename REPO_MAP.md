# Repo Map — axis-core

> **Purpose:** Minimal-context router. Open this first, then only the sub-map(s) needed.
>
> **How to use:** Find your task type in the router table → open the linked map(s) → work.

## Task Router

| Task | Open these maps |
|---|---|
| Add/modify a model adapter (Anthropic, OpenAI) | [adapters.md](.agent/maps/adapters.md) |
| Add/modify a memory adapter (SQLite, Redis, Ephemeral) | [adapters.md](.agent/maps/adapters.md) |
| Add/modify a planner (Sequential, Auto, ReAct) | [adapters.md](.agent/maps/adapters.md) |
| Change lifecycle execution logic | [engine_lifecycle.md](.agent/maps/engine_lifecycle.md) |
| Add a new lifecycle phase or modify phase behavior | [engine_lifecycle.md](.agent/maps/engine_lifecycle.md) |
| Fix a bug in agent.run() / stream() | [engine_lifecycle.md](.agent/maps/engine_lifecycle.md), [domain_core.md](.agent/maps/domain_core.md) |
| Change protocols / adapter interfaces | [protocols_types.md](.agent/maps/protocols_types.md) |
| Modify error handling / error types | [protocols_types.md](.agent/maps/protocols_types.md) |
| Modify context, state, or session management | [domain_core.md](.agent/maps/domain_core.md) |
| Change budget / config / environment loading | [configs_env.md](.agent/maps/configs_env.md) |
| Add/fix tests | [testing_quality.md](.agent/maps/testing_quality.md) |
| Change build, packaging, CI, or release process | [build_release.md](.agent/maps/build_release.md) |
| Update process docs, prompt templates, workflow skills, agent guidance, or execution memory/log docs (`AGENTS.md`, `CLAUDE.md`, `dev/spec-driven.md`, `dev/process-tasks.md`, `dev/memory.md`, `dev/task-summaries.md`, `dev/production-safety-gate.md`, `dev/skills/*`) | [meta_process.md](.agent/maps/meta_process.md) |
| Add telemetry / observability features | [adapters.md](.agent/maps/adapters.md) |
| Add a new tool or modify tool system | [domain_core.md](.agent/maps/domain_core.md) |
| Work on examples | [adapters.md](.agent/maps/adapters.md) (for planner/model context) |

## Architecture (3 layers)

```
Agent API (agent.py)
    ↓ builds
Execution Engine (engine/lifecycle.py → engine/phases/*.py)
    ↓ calls via Protocols
Adapters (adapters/models/, adapters/memory/, adapters/planners/, adapters/telemetry/)
```

**Lifecycle loop:** Initialize → [Observe → Plan → Act → Evaluate]* → Finalize

## Directory Tree (source only)

```
axis_core/
├── __init__.py          # Lazy-loading public API (Agent, tool, Budget, errors, etc.)
├── agent.py             # Agent class — public entry point (783 lines)
├── context.py           # RunContext, RunState, CycleState (965 lines)
├── result.py            # RunResult, StreamEvent, RunStats
├── tool.py              # @tool decorator, ToolManifest, ToolContext, RateLimiter
├── budget.py            # Budget limits + BudgetState tracking
├── config.py            # Config singleton, Timeouts, RetryPolicy, env loading
├── errors.py            # Error hierarchy (AxisError → 8 subclasses)
├── session.py           # Multi-turn Session, Message, optimistic locking
├── attachments.py       # Image, PDF, Attachment (eager-load, 10MB limit)
├── cancel.py            # CancelToken (cooperative cancellation)
├── redaction.py         # Sensitive data redaction for telemetry
├── engine/
│   ├── lifecycle.py     # LifecycleEngine.execute() — main loop (489 lines)
│   ├── phases/          # One module per phase: initialize, observe, plan, act, evaluate, finalize
│   ├── registry.py      # AdapterRegistry + make_lazy_factory() (223 lines)
│   ├── resolver.py      # resolve_adapter() string → instance
│   └── trace_collector.py
├── protocols/           # Protocol interfaces (structural typing)
│   ├── model.py         # ModelAdapter, ModelResponse, ToolCall, UsageStats
│   ├── memory.py        # MemoryAdapter, SessionStore, MemoryItem
│   ├── planner.py       # Planner, Plan, PlanStep, StepType
│   └── telemetry.py     # TelemetrySink, TraceEvent, BufferMode
├── adapters/
│   ├── models/          # AnthropicModel (618L), OpenAIModel (642L)
│   ├── memory/          # EphemeralMemory (220L), SQLiteMemory (306L), RedisMemory (302L)
│   ├── planners/        # SequentialPlanner (129L), AutoPlanner (410L), ReActPlanner (469L)
│   └── telemetry/       # ConsoleSink (119L)
├── loadouts/            # Empty — pre-configured templates (not yet implemented)
└── testing/             # Empty — test utilities (not yet implemented)

tests/                   # Mirrors axis_core/ structure, 633 tests
examples/                # simple_tool_agent.py, autoplanner_example.py, react_planner_example.py
scripts/                 # bump_version.sh, publish.sh, test_install.sh
dev/                     # SPEC.md, PRD, task list, process docs
```

## Key Conventions

- **Python 3.10+**, strict mypy, ruff (100 char lines)
- **Immutable dataclasses** for all data types (frozen=True)
- **Protocols** for adapter interfaces (structural typing, no ABC)
- **Lazy loading** via `__getattr__` in `__init__.py` files
- **Lazy factories** via `make_lazy_factory()` for adapter registration
- **Async-first**: all core is async, sync methods use `asyncio.run()`
- **Append-only state**: RunState uses private lists, exposed as tuples
- **Public-contract testing**: test only public API + documented extension points

## Golden Paths

**Run tests:** `pytest` (all) · `pytest -m "not slow"` (fast) · `pytest tests/engine/test_lifecycle.py` (single)

**Add a new adapter:**

1. Create `axis_core/adapters/{category}/new_adapter.py` implementing the Protocol
2. Add lazy factory registration in `axis_core/adapters/{category}/__init__.py`
3. Add tests in `tests/adapters/{category}/test_new_adapter.py`
4. If optional dep: add to `pyproject.toml` extras, wrap import in try/except

**Fix a bug:**

1. Identify the layer (Agent API → Engine → Adapter)
2. Read the relevant phase module in `engine/phases/` or adapter file
3. Write a failing test first (public-contract only)
4. Fix, run `pytest && ruff check axis_core --fix && mypy axis_core --strict`

**Add a feature:**

1. Check `dev/tasks-axis-core-prd.md` for task context
2. Check `dev/SPEC.md` for ADR constraints
3. Follow TDD flow from `dev/process-tasks.md`
