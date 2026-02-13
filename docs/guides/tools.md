# Tools Guide

Tools are regular Python callables decorated with `@tool`. The decorator adds a manifest,
JSON schema, and async wrapper so the runtime can call tools safely.

## `@tool` Basics

```python
from axis_core import tool

@tool
def get_time(city: str) -> str:
    """Return local time for a city."""
    return f"{city}: 10:30"
```

- Name defaults to function name.
- Description defaults to docstring.
- Input schema is inferred from type hints.

## Decorator Options

```python
from axis_core import Capability, tool
from axis_core import RetryPolicy

@tool(
    name="fetch_url",
    capabilities=[Capability.NETWORK],
    timeout=20.0,
    rate_limit="10/second",
    cache_ttl=300,
    retry=RetryPolicy(max_attempts=3, backoff="exponential"),
)
async def fetch(url: str) -> str:
    """Fetch URL content."""
    return "ok"
```

Supported options:

- `name`
- `description`
- `capabilities`
- `cache_ttl`
- `rate_limit`
- `timeout`
- `retry`

## Type Hints and JSON Schema

Schema generation supports:

- Primitive types: `str`, `int`, `float`, `bool`
- Containers: `list`, `dict`
- `Optional[T]` (`T | None`)
- Pydantic models (`model_json_schema()`)

Note: unions with multiple non-`None` types are not supported in tool params.

## Capabilities and Safety

`Capability` enum members:

- `NETWORK`
- `FILESYSTEM`
- `DATABASE`
- `EMAIL`
- `PAYMENT`
- `DESTRUCTIVE`
- `SUBPROCESS`
- `SECRETS`

Use these as declarative safety metadata for auditing and approvals.

## Confirmation Flow (`on_confirm`)

`Capability.DESTRUCTIVE` tools should be guarded with `Agent.on_confirm()`:

```python
from axis_core import Agent, Capability, tool

@tool(capabilities=[Capability.DESTRUCTIVE])
def delete_item(item_id: str) -> str:
    return f"deleted {item_id}"

agent = Agent(model="claude-sonnet-4-20250514", tools=[delete_item])
agent.on_confirm(lambda tool_name, args: tool_name == "delete_item")
```

The confirmation handler can be sync or async and returns `bool`.

## `ToolContext` Usage

If a tool declares `ctx`, the runtime injects a `ToolContext` instance:

```python
from axis_core import ToolContext, tool

@tool
def annotate(note: str, ctx: ToolContext) -> str:
    run_id = ctx.run_id
    cycle = ctx.cycle
    ctx.context["last_note"] = note
    return f"[{run_id}:{cycle}] {note}"
```

`ToolContext` fields:

- Read-only: `run_id`, `agent_id`, `cycle`, `budget`, `budget_state`
- Mutable: `context` dict

## Manifest Introspection

Each tool wrapper carries metadata in `_axis_manifest`:

```python
manifest = get_time._axis_manifest
print(manifest.name)
print(manifest.input_schema)
```

## Best Practices

- Keep tools narrow and deterministic.
- Validate external inputs before side effects.
- Prefer idempotent behavior when possible.
- Return concise, structured strings for model consumption.
- Use capability declarations consistently for auditability.
