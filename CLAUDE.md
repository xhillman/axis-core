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
pytest axis_core/engine/_tests/test_lifecycle.py

# Lint
ruff check axis_core --fix

# Type check (strict mode enforced)
mypy axis_core --strict
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

## Testing Conventions

- Tests live in `_tests/` subdirectories within each module
- Markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`
- Use `@pytest.mark.asyncio` for async tests (asyncio_mode=auto in pytest.ini)
- Pattern: `test_*.py` files with `Test*` classes

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

## Project Authority Map:

- Task List: tasks-axis-core-prd.md (source of truth for task IDs, sub-tasks, file targets)
- Architecture & Constraints: SPEC.md (ADRs, invariants, design rules)
- Execution Process: process-tasks.md (TDD flow, pacing, completion protocol)
- Product Intent (only if needed): axis-core-prd.md
