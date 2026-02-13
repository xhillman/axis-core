# axis-core

A modular, observable AI agent framework for building production-ready agents in Python.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Status: Beta](https://img.shields.io/badge/status-beta-yellow.svg)

## Highlights

- Lifecycle execution engine: Initialize -> Observe -> Plan -> Act -> Evaluate -> Finalize
- Pluggable adapters for models, memory, planners, and telemetry
- Built-in model fallback for recoverable provider errors
- Tool system with `@tool`, schema generation, and destructive-action confirmation hooks
- Runtime policy enforcement (timeouts, retries, rate limits, cache)
- Checkpoint/resume support for phase-boundary recovery
- Budget controls for cost, token, and cycle limits
- Type hints with strict mypy coverage

## Installation

```bash
# Core package
pip install axis-core

# Provider and adapter extras
pip install axis-core[anthropic]
pip install axis-core[openai]
pip install axis-core[openrouter]
pip install axis-core[redis]
pip install axis-core[sqlite]

# Everything
pip install axis-core[full]
```

Requires Python 3.10+.

## Quick Start

```python
import asyncio
from axis_core import Agent, Budget, tool


@tool
def get_weather(city: str) -> str:
    return f"Weather in {city}: Sunny, 72F"


async def main() -> None:
    agent = Agent(
        tools=[get_weather],
        model="claude-sonnet-4-20250514",
        fallback=["gpt-4o"],
        planner="sequential",
        budget=Budget(max_cost_usd=0.50),
        system="You are a concise assistant.",
    )

    result = await agent.run_async("What is the weather in Tokyo?")
    print(result.output)
    print(result.stats)


asyncio.run(main())
```

## Streaming

```python
for event in agent.stream("Solve 42 * 137"):
    if event.is_token:
        print(event.token, end="", flush=True)
    elif event.is_final:
        print("\nDone")
```

## Documentation

- [Getting Started](docs/getting-started.md)
- [Examples](docs/examples.md)
- [Tools Guide](docs/guides/tools.md)
- [Models Guide](docs/guides/models.md)
- [Runtime Operations](docs/guides/runtime-operations.md)
- [Budget and Limits](docs/guides/budget-and-limits.md)
- [Agent Reference](docs/reference/agent.md)
- [Config Reference](docs/reference/config.md)
- [Errors Reference](docs/reference/errors.md)
- [Environment Variables](docs/reference/env-vars.md)
- [Contributing](CONTRIBUTING.md)

## Supported Providers

| Provider | Installation | Notes |
| -------- | ------------ | ----- |
| Anthropic | `pip install axis-core[anthropic]` | Claude models via Anthropic adapter |
| OpenAI | `pip install axis-core[openai]` | Chat Completions + Responses routing |
| OpenRouter | `pip install axis-core[openrouter]` | OpenAI-compatible base URL path |

For OpenRouter:

```bash
OPENAI_API_KEY=<openrouter-api-key>
OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

## Status

`v0.8.0b` (Beta pre-release)

- Beta means APIs are stabilizing, but breaking changes are still possible before `1.0.0`.
- See [CHANGELOG.md](CHANGELOG.md) for release notes.

## Roadmap

### Planned (Committed)

- Semantic memory search capabilities
- Memory adapter URL-style resolution
- Preconfigured loadouts (`research_agent`, `support_agent`, `code_agent`)
- Planner fallback and plan confidence scoring

### Exploring (Not Committed)

- Additional provider integrations
- Deeper model-specific optimizations
- Extended developer tooling and enterprise workflow support

## Development

```bash
pip install -e ".[dev,anthropic,openai]"
pytest
ruff check axis_core tests --fix
mypy axis_core --strict
```

For full contributor workflow and quality gates, see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache License 2.0. See [LICENSE](LICENSE).
