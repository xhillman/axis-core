# axis-core

A modular, observable AI agent framework for building production-ready agents in Python.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)

## Features

- **Lifecycle-based execution** — Observe → Plan → Act → Evaluate loop with built-in cycle management
- **Protocol-based adapters** — Pluggable models, memory, planners, and telemetry
- **Model fallback** — Automatic fallback to secondary models on recoverable errors
- **Tool system** — Simple `@tool` decorator with automatic schema generation
- **Destructive tool safeguards** — `Capability.DESTRUCTIVE` tools require explicit confirmation
- **Runtime policy enforcement** — Per-phase timeouts, retries, rate limits, and in-memory caching
- **Checkpoint/resume** — Versioned phase-boundary checkpoints with public resume APIs
- **Session support** — Multi-turn conversations with optimistic locking (persistence via memory adapters)
- **Budget tracking** — Cost, token, and cycle limits with real-time tracking
- **Attachments & cancellation** — Image/PDF attachments (10MB limit) and cooperative cancellation
- **OpenAI Responses routing** — Codex/search/deep-research/computer-use model IDs auto-route
- **Built-in observability** — Phase-level telemetry, trace collection, and pluggable sinks
- **Security redaction hardening** — Hyphenated secret keys and free-form error strings are
  redacted in telemetry and persisted run-state records
- **Type-safe** — Full type hints with mypy strict mode
- **Production-ready** — Async-native with comprehensive error handling and recovery

## Installation

```bash
# Basic installation (no model providers)
pip install axis-core

# With Anthropic (recommended for production)
pip install axis-core[anthropic]

# With OpenAI
pip install axis-core[openai]

# With OpenRouter (OpenAI-compatible endpoint)
pip install axis-core[openrouter]

# With both providers
pip install axis-core[anthropic,openai,openrouter]

# With Redis memory adapter
pip install axis-core[redis]

# With SQLite memory adapter
pip install axis-core[sqlite]

# Full installation (all optional dependencies)
pip install axis-core[full]

# Development installation
pip install -e ".[dev,anthropic,openai]"
```

**Note:** axis-core requires Python 3.10 or higher.

## Quick Start

### Basic Example

```python
import asyncio
from axis_core import Agent, tool, Budget

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Weather in {city}: Sunny, 72°F"

@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    # NOTE: eval is unsafe with untrusted input; replace with a safe parser in production.
    return str(eval(expression))

async def main():
    agent = Agent(
        tools=[get_weather, calculate],
        model="claude-sonnet-4-20250514",
        fallback=["gpt-4o"],  # Fallback to GPT-4o if Claude fails
        planner="sequential",
        system="You are a helpful assistant with access to tools.",
        budget=Budget(max_cost_usd=0.50),
    )

    result = await agent.run_async("What's the weather in Tokyo?")
    print(result.output)
    print(f"Cost: ${result.stats.cost_usd:.4f}")
    print(f"Cycles: {result.stats.cycles}")

asyncio.run(main())
```

### With Streaming

```python
for event in agent.stream("Solve 42 * 137"):
    if event.is_token:
        print(event.token, end="", flush=True)
    elif event.is_final:
        stats = event.data.get("stats")
        if stats:
            print(f"\n\nTotal cost: ${stats['cost_usd']:.4f}")
```

### Error Handling & Fallback

```python
# Primary model might hit rate limits, fallback chain handles gracefully
agent = Agent(
    model="claude-opus-4-6",          # Primary (expensive, might hit limits)
    fallback=[
        "claude-sonnet-4-20250514",   # First fallback
        "gpt-4o",                      # Second fallback
        "claude-haiku",                # Final fallback (fast, cheap)
    ],
    budget=Budget(max_cost_usd=2.00),
)

result = agent.run("Complex task...")
# Automatically retries with fallback models on rate limits or connection errors
```

## Documentation

- [Getting Started](docs/getting-started.md)
- [Tools Guide](docs/guides/tools.md)
- [Models Guide](docs/guides/models.md)
- [Runtime Operations](docs/guides/runtime-operations.md)
- [Budget and Limits](docs/guides/budget-and-limits.md)
- [Agent Reference](docs/reference/agent.md)
- [Config Reference](docs/reference/config.md)
- [Errors Reference](docs/reference/errors.md)
- [Environment Variables](docs/reference/env-vars.md)
- [Examples Index](docs/examples.md)
- [Contributing](CONTRIBUTING.md)

## Architecture

axis-core uses a three-layer architecture:

```markdown
┌─────────────────────────────────────────┐
│              Agent API                  │  ← run(), stream(), run_async()
├─────────────────────────────────────────┤
│          Lifecycle Engine               │  ← Observe → Plan → Act → Evaluate
├─────────────────────────────────────────┤
│    Models  │  Memory  │  Planners  │ ...│  ← Pluggable adapters
└─────────────────────────────────────────┘
```

### Execution Lifecycle

Each agent run follows this cycle:

1. **Initialize** — Create context, validate config
2. **Observe** — Gather input, load memory, assess state
3. **Plan** — Generate execution plan (tool calls, model calls)
4. **Act** — Execute plan steps with dependency handling
5. **Evaluate** — Check termination conditions
6. **Finalize** — Persist memory, emit summary

### Key Design Decisions

**Why axis-core is different:**

- **Budget-first**: Hard limits on cost, tokens, and cycles prevent runaway expenses
- **Observable by default**: Full telemetry without code changes (phase events, tool calls, model usage)
- **Error recovery**: Distinguishes transient failures (retry) from permanent errors (fail fast)
- **Fallback chains**: Automatic model failover on rate limits/timeouts preserves availability
- **Protocol-based**: No inheritance hierarchies—adapters implement simple Protocols
- **Async-native**: All I/O is truly async (sync methods are thin wrappers)
- **Type-safe**: mypy --strict enforced across entire codebase
- **Supply chain security**: Lockfile-based dependencies with regular vulnerability audits

## API Reference

The full API is documented in:

- [Agent Reference](docs/reference/agent.md)
- [Config Reference](docs/reference/config.md)
- [Errors Reference](docs/reference/errors.md)

For practical workflows, start with:

- [Tools Guide](docs/guides/tools.md)
- [Models Guide](docs/guides/models.md)
- [Runtime Operations](docs/guides/runtime-operations.md)
- [Budget and Limits](docs/guides/budget-and-limits.md)

## Environment Variables

Use the complete environment variable reference:

- [Environment Variables](docs/reference/env-vars.md)
- [.env Example](.env.example)

## Supported Models

| Provider | Models | Status | Installation |
| -------- | ------ | ------ | ------------ |
| Anthropic | Claude Opus 4.6, Sonnet 4.5, Haiku 4.5 | ✅ Stable | `pip install axis-core[anthropic]` |
| OpenAI | GPT-5, GPT-4.1, GPT-4o, o1/o3/o4 + Responses (codex/search/deep-research/computer-use/search-preview) | ✅ Stable | `pip install axis-core[openai]` |
| OpenRouter | OpenAI-compatible hosted models | ✅ Supported via OpenAI adapter | `pip install axis-core[openrouter]` |

OpenRouter uses the OpenAI-compatible adapter path. Configure:

```bash
OPENAI_API_KEY=<openrouter-api-key>
OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

Responses API models are selected automatically by model ID (no extra flags required). Examples:
`gpt-5-codex`, `gpt-5-search-api`, `gpt-4o-search-preview`, `o3-deep-research`, `computer-use-preview`.

**Model Fallback**: Automatically fallback to secondary models on recoverable errors (rate limits, connection issues):

```python
agent = Agent(
    model="claude-sonnet-4-20250514",
    fallback=["gpt-4o", "claude-haiku"],  # Try these if primary fails
)
```

## Status

**v0.6.0 (Alpha)** — Core features with sessions, attachments, persistent memory, and hardened adapter auto-registration:

### ✅ Completed

**Core Engine:**

- Lifecycle engine with full Observe → Plan → Act → Evaluate cycle
- Agent API with sync/async methods (run, run_async, stream, stream_async)
- Configuration system with environment variable support
- Budget tracking (cycles, tokens, cost, wall time)
- Hard timeout enforcement (`timeout` arg and `Timeouts.total`) with wall-time budget cutoffs
- Comprehensive error handling and recovery
- Type-safe with mypy strict mode
- Session support with optimistic locking (agent.session / session.run)
- Attachments (Image/PDF) with 10MB size limits and metadata serialization
- Cooperative cancellation via CancelToken

**Model Adapters:**

- Anthropic (Claude Opus 4, Sonnet 4, Haiku)
- OpenAI (GPT-4, GPT-4o, GPT-5, o1/o3/o4)
- OpenAI Responses model routing (codex/search/deep-research/computer-use IDs)
- OpenRouter via OpenAI-compatible endpoint (`OPENAI_BASE_URL`)
- Model fallback system (automatic retry with secondary models)
- String-based model resolution (`"claude-sonnet-4-20250514"` → adapter)
- Built-in adapter auto-registration during standard `from axis_core import Agent` imports

**Tool System:**

- `@tool` decorator with automatic JSON schema generation
- Capability declarations (NETWORK, FILESYSTEM, DESTRUCTIVE, etc.)
- Runtime enforcement for global and tool-level timeout/retry/rate-limit/cache policies
- Confirmation handler enforcement for `Capability.DESTRUCTIVE` tools (`confirmation_handler`/`on_confirm`)
- Tool context with budget access

**Memory Adapters:**

- EphemeralMemory — In-memory storage with keyword search (no dependencies)
- SQLiteMemory — Persistent local storage with FTS5 keyword search (`pip install axis-core[sqlite]`)
- RedisMemory — Distributed storage with TTL and namespace support (`pip install axis-core[redis]`)
- All adapters support session persistence with optimistic concurrency locking

**Planners:**

- SequentialPlanner — Executes tool requests in order
- AutoPlanner — LLM-based planning that selects and orders tools
- ReActPlanner — Reasoning + Acting loop with explicit thought steps
- Adapter registry with plugin discovery

**Observability:**

- Phase-level telemetry (`ConsoleSink`, `FileSink`, `CallbackSink`)
- Trace event collection
- Budget warnings and exceeded events

### 🚧 In Progress / Planned

See [Roadmap](#roadmap) below for upcoming features.

## Roadmap

axis-core is under active development. This roadmap separates committed work from exploratory ideas.

### Planned (Committed)

These are scoped targets tied to the current task list and near-term releases.
Checkpoint/resume is already shipped and documented in Features/Status sections above.

**Telemetry & Memory:**

- Semantic search capabilities (vector-based)
- Memory adapter URL resolution (`sqlite:///path/to/db`, `redis://host:port`)

**Platform & API:**

- `research_agent()`, `support_agent()`, and `code_agent()` loadouts
- Planner fallback and plan confidence scoring

### Exploring (Not Committed)

These are directional ideas. They are not scheduled and may change or be dropped.

**Potential Integrations:**

- Additional model providers beyond current built-ins (for example: Gemini)
- Deeper model-specific optimizations (parallel tool calls, cache tuning)
- Framework integrations (LangChain, FastAPI, Gradio/Streamlit, Jupyter)

**Potential Product Surface:**

- Extended developer tooling (`axis_core.testing`, REPL/debug UX, editor integrations)
- Enterprise capabilities (distributed execution, A/B testing, compliance)
- Advanced multi-agent capabilities

---

**Want to contribute?** Check out our [Contributing Guide](CONTRIBUTING.md) or join the discussion in [GitHub Issues](https://github.com/yourusername/axis-core/issues).

## Development

### Setup

```bash
# Install dev dependencies
pip install -e ".[dev,anthropic,openai]"

# Run tests
pytest

# Run with coverage
pytest --cov=axis_core

# Skip slow integration tests
pytest -m "not slow"

# Run the live Anthropic integration test (skips automatically if key is missing)
ANTHROPIC_API_KEY=sk-ant-... pytest tests/engine/test_real_llm_integration.py

# Type checking (strict mode enforced)
mypy axis_core --strict

# Linting
ruff check axis_core --fix
```

### Supply Chain Security

axis-core uses a lockfile-based dependency management for reproducible builds:

```bash
# Update dependencies after modifying pyproject.toml
uv pip compile pyproject.toml -o requirements.lock

# Install from lockfile (reproducible builds)
pip install -r requirements.lock

# Audit for vulnerabilities (recommended weekly)
pip install pip-audit
pip-audit -r requirements.lock
```

### Architecture Principles

- **Async-native**: All I/O operations are async, sync methods are thin wrappers
- **Protocol-based**: Adapters implement Protocols, not base classes
- **Append-only state**: RunState uses immutable dataclasses with append methods
- **Error recovery**: Distinguishes recoverable (retry) vs permanent (fail fast) errors
- **Budget enforcement**: Hard limits on cycles, tokens, and cost with graceful degradation
- **Observable by default**: Telemetry at phase boundaries, not scattered throughout code

See [SPEC.md](dev/SPEC.md) for full architectural decision records (ADRs).

## Contributing

See the full contributor workflow in [CONTRIBUTING.md](CONTRIBUTING.md), including:

- development setup
- quality gates (`pytest`, `ruff`, `mypy`)
- public-contract testing policy
- PR requirements and documentation update rules

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
