# Getting Started

This guide gets a first agent running with tools, streaming, and sessions.

## Prerequisites

- Python `3.10+`
- `pip` (or `uv`)
- One provider API key:
  - `ANTHROPIC_API_KEY` for Claude models
  - `OPENAI_API_KEY` for OpenAI/OpenRouter-compatible models

## Installation

```bash
# Base package
pip install axis-core

# Common provider extras
pip install axis-core[anthropic]
pip install axis-core[openai]

# Optional adapters
pip install axis-core[redis]
pip install axis-core[sqlite]

# Everything
pip install axis-core[full]
```

## Environment Setup

Create a `.env` file (or export vars in your shell):

```bash
ANTHROPIC_API_KEY=your-key
# or
OPENAI_API_KEY=your-key
```

axis-core will load `.env` automatically when `python-dotenv` is available.

## First Agent

```python
from axis_core import Agent, Budget, tool

@tool
def get_weather(city: str) -> str:
    """Return weather text for a city."""
    return f"Weather in {city}: sunny, 72F"

agent = Agent(
    tools=[get_weather],
    model="claude-sonnet-4-20250514",
    planner="sequential",
    budget=Budget(max_cost_usd=0.25),
    system="You are a concise assistant.",
)

result = agent.run("What is the weather in Tokyo?")
print(result.output)
print(result.success)
print(result.stats.cost_usd)
```

## Async Run

```python
import asyncio

async def main() -> None:
    result = await agent.run_async("Summarize the weather.")
    print(result.output)

asyncio.run(main())
```

## First Streaming Run

```python
for event in agent.stream("Compute 42 * 137"):
    if event.is_token:
        print(event.token, end="", flush=True)
    elif event.is_final:
        print("\nDone:", event.type)
```

Common stream event types:

- `run_started`
- `model_token`
- `telemetry` (only when `stream_telemetry=True`)
- `run_completed`
- `run_failed`

## First Multi-Turn Session

```python
session = agent.session(id="demo-chat", max_history=50)

r1 = session.run("My name is Alex.")
r2 = session.run("What is my name?")

print(r1.output)
print(r2.output)
```

If memory persistence is configured, session state can be resumed by id.

## Next Steps

- Tool design and safety: `docs/guides/tools.md`
- Provider setup and fallback: `docs/guides/models.md`
- Runtime operations (streaming/sessions/checkpoints/cancellation): `docs/guides/runtime-operations.md`
- Budget/timeouts/retry/rate limits/cache: `docs/guides/budget-and-limits.md`
- API details: `docs/reference/agent.md`, `docs/reference/config.md`, `docs/reference/errors.md`, `docs/reference/env-vars.md`
