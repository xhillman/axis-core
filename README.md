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
- **Session support** — Multi-turn conversations with optimistic locking (persistence via memory adapters)
- **Budget tracking** — Cost, token, and cycle limits with real-time tracking
- **Attachments & cancellation** — Image/PDF attachments (10MB limit) and cooperative cancellation
- **Built-in observability** — Phase-level telemetry, trace collection, and pluggable sinks
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

# With both providers
pip install axis-core[anthropic,openai]

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
    model="claude-opus-4-20250514",   # Primary (expensive, might hit limits)
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

### Agent

```python
from axis_core import Agent

agent = Agent(
    tools=[...],              # List of @tool-decorated functions
    model="claude-sonnet-4-20250514",    # Model adapter or string identifier
    fallback=["gpt-4o", "claude-haiku"], # Fallback models on error
    planner="sequential",     # Planner adapter or string identifier
    system="...",             # System prompt
    budget=Budget(            # Resource constraints
        max_cycles=10,
        max_cost_usd=1.00,
    ),
)

# Synchronous
result = agent.run("Your prompt here")

# Asynchronous  
result = await agent.run_async("Your prompt here")

# Streaming
for event in agent.stream("Your prompt here"):
    print(event)
```

### Tools

```python
from axis_core import tool, Capability

@tool
def simple_tool(arg: str) -> str:
    """A simple tool."""
    return f"Result: {arg}"

@tool(
    capabilities=[Capability.NETWORK],
    timeout=30.0,
    rate_limit="10/minute",
)
async def advanced_tool(url: str, max_retries: int = 3) -> str:
    """An advanced tool with capabilities."""
    # Implementation
    pass
```

### Budget

```python
from axis_core import Budget

budget = Budget(
    max_cycles=10,           # Maximum observe-plan-act-evaluate cycles
    max_tool_calls=50,       # Maximum tool invocations
    max_model_calls=20,      # Maximum LLM calls
    max_tokens=100_000,      # Maximum total tokens
    max_cost_usd=5.00,       # Maximum cost in USD
    max_wall_time_seconds=300,  # Maximum wall-clock time
)
```

### Sessions

```python
from axis_core import Agent

agent = Agent(model="claude-sonnet-4-20250514", planner="sequential")

# Create a new session
session = agent.session(id="user-123")
result = session.run("What's the weather in Tokyo?")

# Resume later (history persists via the configured memory adapter)
session = agent.session(id="user-123")
result = session.run("What about Osaka?")

# For durable persistence across restarts, use SQLite or Redis memory:
# agent = Agent(model="claude-sonnet-4-20250514", memory="sqlite")
# agent = Agent(model="claude-sonnet-4-20250514", memory="redis")
```

### Attachments & Cancellation

Attachments are loaded eagerly and limited to 10MB each.

```python
from axis_core import Agent, CancelToken, Image, PDF

agent = Agent(model="claude-sonnet-4-20250514", planner="sequential")

attachments = [
    Image.from_file("diagram.png"),
    PDF.from_file("spec.pdf"),
]

cancel_token = CancelToken()

result = agent.run(
    "Summarize the PDF and describe the diagram.",
    attachments=attachments,
    cancel_token=cancel_token,
)
```

### Results

```python
result = agent.run("...")

result.output        # Parsed output
result.output_raw    # Raw string output
result.success       # Whether run succeeded
result.error         # Error if failed
result.stats.cycles  # Number of cycles executed
result.stats.cost_usd  # Total cost
result.stats.tool_calls  # Number of tool calls
```

### Credential Handling

`Agent(auth=...)` is deprecated and ignored.

Tool credentials should be loaded by tools directly (for example, from environment variables
or a secret manager), not passed through agent constructor/runtime context objects.

## Environment Variables

```bash
# API Keys
ANTHROPIC_API_KEY=sk-ant-...        # Required for Anthropic models
OPENAI_API_KEY=sk-...               # Required for OpenAI models

# Default Configuration
AXIS_DEFAULT_MODEL=claude-sonnet-4-20250514
AXIS_DEFAULT_MEMORY=ephemeral
AXIS_DEFAULT_PLANNER=sequential

# Budget Defaults
AXIS_MAX_CYCLES=10
AXIS_MAX_COST_USD=1.00
AXIS_MAX_TOOL_CALLS=50
AXIS_MAX_MODEL_CALLS=20

# Telemetry
AXIS_TELEMETRY_SINK=console         # console, file, callback, none
AXIS_TELEMETRY_COMPACT=false        # Compact console output
AXIS_TELEMETRY_REDACT=true          # Redact sensitive data
AXIS_TELEMETRY_FILE=./axis_trace.jsonl  # JSONL output path (file sink)
AXIS_TELEMETRY_CALLBACK=module:function # Callback import path (callback sink)
AXIS_TELEMETRY_BUFFER_MODE=batched  # immediate, batched, phase, end (file sink)
AXIS_TELEMETRY_BATCH_SIZE=100       # Flush threshold in batched mode (file sink)
AXIS_PERSIST_SENSITIVE_TOOL_DATA=false  # Debug-only raw tool args/results in RunState

# Advanced
AXIS_CONTEXT_STRATEGY=smart         # Context building strategy
AXIS_MAX_CYCLE_CONTEXT=5            # Max cycles to include in context
```

## Supported Models

| Provider | Models | Status | Installation |
| -------- | ------ | ------ | ------------ |
| Anthropic | Claude Opus 4, Sonnet 4, Haiku | ✅ Stable | `pip install axis-core[anthropic]` |
| OpenAI | GPT-5, GPT-4.1, GPT-4o, o1/o3/o4 | ✅ Stable | `pip install axis-core[openai]` |
| Ollama | Local models | 🚧 Planned | `pip install axis-core[ollama]` |

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
- Model fallback system (automatic retry with secondary models)
- String-based model resolution (`"claude-sonnet-4-20250514"` → adapter)
- Built-in adapter auto-registration during standard `from axis_core import Agent` imports

**Tool System:**

- `@tool` decorator with automatic JSON schema generation
- Capability declarations (NETWORK, FILESYSTEM, DESTRUCTIVE, etc.)
- Tool metadata for rate limits/timeouts (runtime enforcement in progress)
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

**Execution & Safety:**

- **Confirmation Handler** — User approval for destructive operations (AD-002)
- **Runtime Enforcement** — Timeouts, rate limits, retries, and cache behavior
- **Checkpoint/Resume** — Persist and resume agent runs at phase boundaries (AD-005)

**Telemetry & Memory:**

- Semantic search capabilities (vector-based)
- Memory adapter URL resolution (`sqlite:///path/to/db`, `redis://host:port`)

**Platform & API:**

- OpenAI Responses API routing support
- `research_agent()`, `support_agent()`, and `code_agent()` loadouts
- Planner fallback and plan confidence scoring

### Exploring (Not Committed)

These are directional ideas. They are not scheduled and may change or be dropped.

**Potential Integrations:**

- Additional model providers beyond current built-ins (for example: OpenRouter, Gemini)
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

We welcome contributions! Here's how to get started:

1. **Check the roadmap** — See if your idea aligns with planned features
2. **Open an issue** — Discuss your proposal before writing code
3. **Follow TDD** — Write tests first (see [process-tasks.md](dev/process-tasks.md))
4. **Match the style** — Use ruff, mypy strict mode, Python 3.10+ typing
5. **Update docs** — Keep README and docstrings current

### Guidelines

- **Tests required** — All PRs must include tests (we enforce TDD)
- **Type hints required** — Full type coverage with mypy --strict
- **No breaking changes** — Maintain backward compatibility within major versions
- **Security first** — Never commit API keys, use environment variables
- **Clean commits** — Squash before merging, write clear commit messages

### Project Structure

```
axis_core/
├── __init__.py          # Public API exports
├── agent.py             # Agent class (main entry point)
├── engine/              # Lifecycle engine internals
│   ├── lifecycle.py     # Phase execution
│   └── registry.py      # Adapter registration
├── adapters/            # Pluggable implementations
│   ├── models/          # LLM providers (Anthropic, OpenAI)
│   ├── memory/          # Storage backends
│   ├── planners/        # Planning strategies
│   └── telemetry/       # Observability sinks
├── protocols/           # Adapter Protocol definitions
└── testing/             # Test utilities (coming soon)

tests/                   # Test suite (mirrors axis_core structure)
dev/                     # Design docs, specs, task lists
```

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
