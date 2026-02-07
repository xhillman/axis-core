# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

axis-core is a Python AI agent framework providing a modular, observable execution kernel for building production-ready agents. Currently in early development (v0.1.0 alpha) with project skeleton in place.

## Build & Development Commands

```bash
# Install for development
pip install -e ".[dev]"

# Install with specific adapters
pip install -e ".[anthropic,openai,redis,full]"

# Run all tests
pytest

# Run tests with coverage
pytest --cov=axis_core

# Skip slow tests
pytest -m "not slow"

# Run single test file
pytest tests/engine/test_lifecycle.py

# Lint
ruff check axis_core --fix

# Type check (strict mode enforced)
mypy axis_core --strict

# Dependency management (supply chain security)
# Generate lockfile after updating dependencies in pyproject.toml
uv pip compile pyproject.toml -o requirements.lock

# Install from lockfile for reproducible builds
pip install -r requirements.lock

# Verify lockfile is valid
pytest tests/test_lockfile.py
```

## Dependency Management & Supply Chain Security

**Lockfile:** `requirements.lock` pins exact versions of all dependencies (direct + transitive) for reproducible builds.

**Update cadence:**

- **Monthly:** Review and update lockfile with latest security patches
- **On-demand:** Regenerate lockfile when adding/removing dependencies in `pyproject.toml`
- **Security alerts:** Regenerate immediately if critical CVE affects locked dependencies

**Commands:**

```bash
# After modifying pyproject.toml dependencies
uv pip compile pyproject.toml -o requirements.lock

# Audit for known vulnerabilities (recommended weekly)
pip install pip-audit
pip-audit -r requirements.lock
```

## Architecture

**Three-layer design:**

1. **Agent API** (`axis_core/agent.py`) — Public `Agent` class with `run()`, `stream()`, `run_async()`, `stream_async()` methods
2. **Execution Engine** (`axis_core/engine/`) — Lifecycle phases: Initialize → Observe → Plan → Act → Evaluate → Finalize (cycles)
3. **Adapters** (`axis_core/adapters/`) — Pluggable implementations for models, memory, planners, telemetry

**Key directories:**

- `axis_core/protocols/` — Adapter Protocol interfaces (ModelAdapter, MemoryAdapter, Planner, TelemetrySink)
- `axis_core/adapters/models/` — LLM providers (Anthropic, OpenAI, Ollama)
- `axis_core/adapters/memory/` — State persistence (Ephemeral, SQLite, Redis)
- `axis_core/adapters/planners/` — Planning strategies (Auto, ReAct, Sequential)
- `axis_core/loadouts/` — Pre-configured agent templates
- `dev/` — Design docs including full PRD and technical spec

## Key Design Patterns

- **String-based adapter resolution:** `"claude-sonnet-4-20250514"` auto-resolves to `AnthropicModel`
- **Lazy loading:** `__init__.py` uses `__getattr__` to avoid circular imports
- **Async native:** Sync methods wrap async core
- **Environment-first config:** Resolution order: defaults → env vars → constructor args → runtime args

### Adapter Auto-Registration

Built-in adapters are automatically registered via a lazy factory pattern:

1. **Registry System** (`axis_core/engine/registry.py`): Global registries (`model_registry`, `memory_registry`, `planner_registry`) store adapter factories
2. **Lazy Factories** (each `axis_core/adapters/{category}/__init__.py`): Create wrapper classes that defer importing actual implementations until instantiation
3. **Automatic Registration**: When `axis_core.engine.registry` is imported, it triggers all adapter `__init__.py` files which register their factories
4. **Optional Dependencies**: Adapters with optional deps (e.g., Anthropic) only import the implementation package when first used, providing helpful error messages if dependencies are missing

**Adding new adapters:**

- Follow the lazy factory pattern in existing `__init__.py` files (see `axis_core/adapters/models/__init__.py` for reference)
- Register via `{category}_registry.register(name, factory_class)`
- For adapters with optional dependencies, wrap imports in try/except and raise `ConfigError` with installation instructions
- **IMPORTANT**: When implementing adapters for specific products (Anthropic, OpenAI, Redis, etc.), always reference the official product documentation to ensure correct usage of their APIs and best practices

## Testing Conventions

- Tests live in top-level `/tests` directory, mirroring `axis_core/` structure
- Markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`
- Use `@pytest.mark.asyncio` for async tests (asyncio_mode=auto in pytest.ini)
- Pattern: `test_*.py` files with `Test*` classes
- All imports from tests use absolute imports: `from axis_core.* import ...`

### Public-Contract Testing Policy

Tests must exercise behavior through **public API or documented extension points**, not private/internal methods. This prevents test brittleness when implementation details change.

**Allowed test surfaces:**

- `Agent.run()`, `run_async()`, `stream()`, `stream_async()` — agent-level public API
- `LifecycleEngine.execute()` — engine-level public API
- Lifecycle phase methods (`_initialize`, `_observe`, `_plan`, `_act`, `_evaluate`, `_finalize`) — documented architectural extension points
- Adapter protocol methods (`complete`, `stream`, `store`, `retrieve`, `plan`) — protocol contracts
- `resolve_adapter()` and registry APIs — public adapter resolution

**Do not test directly:**

- Internal helpers like `_try_models_with_fallback`, `_execute_model_step`, `_get_tool_manifests`, `_identify_exhausted_resource`
- Agent internals like `_build_engine`
- Any method prefixed with `_` that is not a lifecycle phase method

## Code Style

- Python 3.10+
- Type hints required everywhere (mypy strict mode)
- Ruff for linting (line length: 100)
- Dataclasses for immutable structures
- Protocols for adapter interfaces

## Configuration

See `.env.example` for all environment variables. Key ones:

- `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` — API keys
- `AXIS_DEFAULT_MODEL` — Default model (claude-sonnet-4-20250514)
- `AXIS_DEFAULT_MEMORY` — Default memory adapter (ephemeral)
- `AXIS_DEFAULT_PLANNER` — Default planner (sequential)
- `AXIS_MAX_CYCLES`, `AXIS_MAX_COST_USD` — Budget constraints

## Documentation

README.md should be kept up to date to reflect the current state of the project. A section that includes future enhancements and planned features is encouraged to give users insight into the roadmap.

After completing significant features or changes, ensure that the README.md is updated accordingly.

## Project Authority Map

- Task List: tasks-axis-core-prd.md (source of truth for task IDs, sub-tasks, file targets)
- Architecture & Constraints: SPEC.md (ADRs, invariants, design rules)
- Execution Process: process-tasks.md (TDD flow, pacing, completion protocol)
- Product Intent (only if needed): axis-core-prd.md
