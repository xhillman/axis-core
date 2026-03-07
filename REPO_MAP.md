# Repo Map — axis-core

> **Purpose:** Minimal-context router. Open this first, then only the sub-map(s) needed.
>
> **How to use:** Find your task type in the router table → open the linked map(s) → work.

## Task Router

| Task | Open these maps |
|---|---|
| Add/modify a model adapter (Anthropic, OpenAI, Responses API) | [adapters.md](.agent/maps/adapters.md) |
| Add/modify a memory adapter (SQLite, Redis, Ephemeral, Synaptic) | [adapters.md](.agent/maps/adapters.md) |
| Add/modify a planner | [adapters.md](.agent/maps/adapters.md) |
| Add/modify a telemetry sink or provider helper | [adapters.md](.agent/maps/adapters.md) |
| Change lifecycle execution logic | [engine_lifecycle.md](.agent/maps/engine_lifecycle.md) |
| Add a new lifecycle phase or modify phase behavior | [engine_lifecycle.md](.agent/maps/engine_lifecycle.md) |
| Fix a bug in `Agent.run()` / `run_async()` / `stream()` / `stream_async()` | [engine_lifecycle.md](.agent/maps/engine_lifecycle.md), [domain_core.md](.agent/maps/domain_core.md) |
| Change protocols / adapter interfaces | [protocols_types.md](.agent/maps/protocols_types.md) |
| Modify error handling / shared types | [protocols_types.md](.agent/maps/protocols_types.md) |
| Modify context, state, session, checkpoint, output schema, CLI, or tool behavior | [domain_core.md](.agent/maps/domain_core.md) |
| Change budget / config / environment loading | [configs_env.md](.agent/maps/configs_env.md) |
| Add/fix tests or run quality gates | [testing_quality.md](.agent/maps/testing_quality.md) |
| Change build, packaging, CI, or release process | [build_release.md](.agent/maps/build_release.md) |
| Update process docs, contracts, repo routing, agent guidance, or doc-policy checkers (`AGENTS.md`, `CLAUDE.md`, `REPO_MAP.md`, `.agent/maps/*.md`, `dev/process-tasks.md`, `dev/spec-driven.md`, `dev/contracts/*`, `scripts/check_*`) | [meta_process.md](.agent/maps/meta_process.md) |
| Add telemetry / observability features | [adapters.md](.agent/maps/adapters.md), [engine_lifecycle.md](.agent/maps/engine_lifecycle.md) |
| Add a new tool or modify tool system | [domain_core.md](.agent/maps/domain_core.md), [engine_lifecycle.md](.agent/maps/engine_lifecycle.md) |
| Work on examples or docs that depend on adapter/runtime behavior | [adapters.md](.agent/maps/adapters.md), [domain_core.md](.agent/maps/domain_core.md) |

## Architecture (current topology)

```text
Public entrypoints (`agent.py`, `cli.py`)
    ↓ build and drive
Execution engine (`engine/lifecycle.py` + `engine/phases/*.py`)
    ↓ reads/writes shared runtime state
Context/state package (`context/`, `session.py`, `checkpoint.py`, `result.py`)
    ↓ calls interfaces from
Protocols (`protocols/*.py`)
    ↓ implemented by
Adapters (`adapters/models|memory|planners|telemetry`)
```

**Lifecycle loop:** Initialize → [Observe → Plan → Act → Evaluate]* → Finalize

## Directory Tree (source only)

```text
axis_core/
├── __init__.py          # Lazy-loaded public exports
├── agent.py             # Primary public API surface
├── cli.py               # CLI entrypoint / command handling
├── budget.py            # Budget limits and mutable usage tracking
├── checkpoint.py        # Checkpoint serialization helpers
├── config.py            # Config/env loading and runtime defaults
├── output_schema.py     # Output-schema normalization + validation helpers
├── result.py            # RunResult, StreamEvent, RunStats
├── session.py           # Multi-turn session model + optimistic locking
├── tool.py              # @tool decorator, manifests, schemas, policies
├── attachments.py       # Image/PDF attachment helpers
├── cancel.py            # Cooperative cancellation token
├── errors.py            # Error hierarchy
├── redaction.py         # Sensitive-data redaction helpers
├── context/
│   ├── __init__.py      # Public context exports
│   ├── types.py         # RunContext, RunState, CycleState, phase result types
│   ├── transcript.py    # Transcript repair / pruning helpers
│   └── codec.py         # Context serialization helpers
├── engine/
│   ├── cycle_runner.py  # Steady-state observe / plan / act / evaluate orchestration
│   ├── lifecycle.py     # LifecycleEngine + orchestration helpers
│   ├── phases/          # initialize / observe / plan / act / evaluate / finalize
│   ├── registry.py      # Adapter registries + lazy factory helpers
│   ├── resolver.py      # String/name → adapter resolution
│   └── trace_collector.py
├── protocols/
│   ├── model.py
│   ├── memory.py
│   ├── planner.py
│   └── telemetry.py
├── adapters/
│   ├── models/          # Anthropic, OpenAI chat, OpenAI Responses, provider helpers
│   ├── memory/          # Ephemeral, SQLite, Redis, Synaptic
│   ├── planners/        # Sequential, Auto, ReAct
│   └── telemetry/       # Console, callback, file sinks
├── loadouts/__init__.py # Loadout package placeholder
└── testing/__init__.py  # Shared testing package placeholder

tests/                   # Mirrors runtime areas: adapters, engine, protocols, context, tool, budget
                         # Current suite collects 831 items / 3 skipped
examples/                # Simple tool, planner, and synaptic-session examples
docs/                    # Getting-started and examples docs
scripts/                 # Acceptance/doc-policy/memory/safety checkers + release scripts
dev/
├── contracts/           # Active implementation contracts
├── archive/             # Historical task lists, summaries, and release/safety records
├── process-tasks.md     # Canonical execution process
├── spec-driven.md       # Prompt/execution template
├── SPEC.md              # Technical specification / ADR source
└── axis-core-prd.md     # Product intent / requirements
```

## Key Conventions

- Python 3.10+, strict mypy, ruff (100-char line length)
- Lazy-loading public API via `__getattr__` exports
- Lazy adapter registration via `make_lazy_factory()`
- Async-first runtime with sync wrappers on top
- Public-contract testing: test public APIs and documented extension points, not private helpers
- Active maintainability work is tracked in `dev/contracts/`; historical execution artifacts live in `dev/archive/`

## Golden Paths

**Run tests:** `./scripts/test.sh` (all) · `./scripts/test.sh -m "not slow"` (faster subset) · `./scripts/test.sh tests/engine/test_lifecycle.py` (single file)

**Add a new adapter:**

1. Create `axis_core/adapters/{category}/new_adapter.py` implementing the relevant protocol
2. Register it in `axis_core/adapters/{category}/__init__.py`
3. Add tests in `tests/adapters/{category}/test_new_adapter.py`
4. If it has an optional dependency: add the extra in `pyproject.toml` and guard imports clearly

**Fix a bug:**

1. Identify the owning layer (public API, engine, shared state, or adapter)
2. Read the routed map, then the owning runtime file(s)
3. Write a failing public-contract test first unless the task is explicitly `[NO-TEST]`
4. Fix, then run the touched checks from `dev/process-tasks.md`

**Add a feature:**

1. Check `dev/contracts/README.md` for the active contract sequence
2. Open the specific `dev/contracts/<id>-*.md` file for invariants and acceptance criteria
3. Use `dev/process-tasks.md` for execution mechanics and required gates
