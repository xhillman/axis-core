# Runtime Operations Guide

This guide covers day-2 runtime workflows: streaming, sessions, checkpoint/resume,
cancellation/timeouts, and failure handling patterns.

## Streaming

### Sync

```python
for event in agent.stream("Explain retries"):
    if event.is_token:
        print(event.token, end="", flush=True)
    elif event.is_final:
        print("\nFinal event:", event.type)
```

### Async

```python
async for event in agent.stream_async("Explain retries"):
    ...
```

`stream_telemetry=True` includes telemetry events in the stream.

## Sessions

Create/resume a session:

```python
session = agent.session(id="support-chat", max_history=100)
result = session.run("User message")
```

Session behavior:

- Tracks history as `Message` records
- Truncates to `max_history`
- Attaches session history into run context
- Persists through memory adapters when available
- Uses optimistic concurrency (raises `ConcurrencyError` on version mismatch)

## Checkpoint and Resume

Enable checkpoint persistence on the agent:

```python
agent = Agent(
    model="claude-sonnet-4-20250514",
    checkpoint=True,
    checkpoint_dir="./checkpoints",
)
```

Resume from path or payload:

```python
resumed = agent.resume("./checkpoints/<run_id>.json")
# or
resumed = agent.resume({"version": 1, "phase": "act", "context": {...}})
```

Checkpoint envelope includes:

- `version`
- `phase`
- `next_phase` (optional)
- `saved_at`
- serialized `context`

## Cancellation and Timeouts

Cooperative cancellation:

```python
from axis_core import CancelToken

token = CancelToken()
# token.cancel("User requested stop")
result = agent.run("long task", cancel_token=token)
```

`CancelToken` semantics:

- One-shot cancellation
- First reason wins

Timeout layers:

- Per-run timeout argument: `timeout=...`
- Config timeout policy (`Timeouts.total` and phase-specific values)

## Runtime Error Handling Patterns

Pattern for resilient callers:

```python
result = agent.run("task")
if result.success:
    print(result.output)
else:
    print(result.error)
    # Inspect result.error.error_class, recoverable, details, etc.
```

For long-running workflows:

- Always set explicit budgets and timeouts
- Use checkpoints for restartability
- Stream output for user-visible progress
- Capture and persist run ids for trace correlation
