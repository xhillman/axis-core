# axis-core

A modular, observable AI agent framework for building production-ready agents in Python.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)]

## Features

- **Lifecycle-based execution** — Observe → Plan → Act → Evaluate loop with built-in cycle management
- **Protocol-based adapters** — Pluggable models, memory, planners, and telemetry
- **Tool system** — Simple `@tool` decorator with automatic schema generation
- **Budget tracking** — Cost, token, and cycle limits with real-time tracking
- **Built-in observability** — Phase-level telemetry and trace collection
- **Type-safe** — Full type hints with mypy strict mode

## Installation

```bash
# Basic installation
pip install axis-core

# With Anthropic/Claude support
pip install axis-core[anthropic]

# Development installation
pip install -e ".[dev,anthropic]"
```

## Quick Start

```python
import asyncio
from axis_core import Agent, tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Weather in {city}: Sunny, 72°F"

@tool  
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    return f"{expression} = {eval(expression)}"

async def main():
    agent = Agent(
        tools=[get_weather, calculate],
        model="claude-haiku",
        planner="sequential",
        system="You are a helpful assistant with access to tools.",
    )
    
    result = await agent.run_async("What's the weather in Tokyo?")
    print(result.output)
    print(f"Cost: ${result.stats.cost_usd:.4f}")

asyncio.run(main())
```

## Architecture

axis-core uses a three-layer architecture:

```mermaid
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

## API Reference

### Agent

```python
from axis_core import Agent

agent = Agent(
    tools=[...],              # List of @tool-decorated functions
    model="claude-sonnet",    # Model adapter or string identifier
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

## Environment Variables

```bash
# Required for Anthropic models
ANTHROPIC_API_KEY=sk-ant-...

# Optional configuration
AXIS_DEFAULT_MODEL=claude-sonnet-4-20250514
AXIS_DEFAULT_PLANNER=sequential
AXIS_MAX_CYCLES=10
AXIS_MAX_COST_USD=1.00
AXIS_TELEMETRY_SINK=console  # console, file, none
```

## Supported Models

| Provider | Models | Installation |
| -------- | ------ | ------------ |
| Anthropic | Claude Opus, Sonnet, Haiku | `pip install axis-core[anthropic]` |
| OpenAI | GPT-4, GPT-4o (planned) | `pip install axis-core[openai]` |
| Ollama | Local models (planned) | `pip install axis-core[ollama]` |

## Status

**v0.1.0 (Alpha)** — Core functionality is working:

- ✅ Lifecycle engine with full cycle support
- ✅ Agent API (run, run_async, stream, stream_async)
- ✅ Tool system with schema generation
- ✅ Anthropic/Claude model adapter
- ✅ Sequential planner
- ✅ Budget tracking and telemetry
- 🚧 OpenAI adapter (planned)
- 🚧 Sessions/multi-turn (planned)
- 🚧 Advanced planners (planned)

## Development

```bash
# Install dev dependencies
pip install -e ".[dev,anthropic]"

# Run tests
pytest

# Run with coverage
pytest --cov=axis_core

# Type checking
mypy axis_core --strict

# Linting
ruff check axis_core --fix
```

## License

MIT License — see [LICENSE](LICENSE) for details.
