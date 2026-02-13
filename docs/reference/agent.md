# Agent API Reference

`Agent` is the primary public entry point.

## Constructor

```python
def __init__(
    self,
    tools: list[Callable[..., Any]] | None = None,
    *,
    system: str | None = None,
    persona: str | None = None,
    model: Any = _UNSET,
    fallback: list[Any] | None = None,
    memory: Any = _UNSET,
    planner: Any = _UNSET,
    budget: dict[str, Any] | Budget | None = None,
    timeouts: dict[str, Any] | Timeouts | None = None,
    rate_limits: dict[str, Any] | RateLimits | None = None,
    retry: dict[str, Any] | RetryPolicy | None = None,
    cache: dict[str, Any] | CacheConfig | None = None,
    telemetry: bool | list[Any] = True,
    verbose: bool = False,
    auth: dict[str, dict[str, Any]] | None = None,
    confirmation_handler: ConfirmationHandler | None = None,
    checkpoint: bool = False,
    checkpoint_dir: str = "./checkpoints",
) -> None
```

Notes:

- `auth` is deprecated and ignored.
- `model`, `planner`, and `memory` can be strings or adapter instances.
- `telemetry=True` resolves sinks from environment.

## `run_async()`

```python
async def run_async(
    self,
    input: str | list[Any],
    *,
    context: dict[str, Any] | None = None,
    attachments: list[AttachmentLike] | None = None,
    output_schema: type | None = None,
    timeout: float | None = None,
    cancel_token: CancelToken | None = None,
) -> RunResult
```

Executes one run asynchronously.

## `run()`

```python
def run(
    self,
    input: str | list[Any],
    *,
    context: dict[str, Any] | None = None,
    attachments: list[AttachmentLike] | None = None,
    output_schema: type | None = None,
    timeout: float | None = None,
    cancel_token: CancelToken | None = None,
) -> RunResult
```

Sync wrapper over `run_async()`.

Important: cannot be called from an async context.

## `stream_async()`

```python
async def stream_async(
    self,
    input: str | list[Any],
    *,
    context: dict[str, Any] | None = None,
    attachments: list[AttachmentLike] | None = None,
    output_schema: type | None = None,
    timeout: float | None = None,
    cancel_token: CancelToken | None = None,
    stream_telemetry: bool = False,
) -> AsyncIterator[StreamEvent]
```

Event types emitted:

- `run_started`
- `model_token`
- `telemetry` (optional)
- `run_completed`
- `run_failed`

## `stream()`

```python
def stream(
    self,
    input: str | list[Any],
    *,
    context: dict[str, Any] | None = None,
    attachments: list[AttachmentLike] | None = None,
    output_schema: type | None = None,
    timeout: float | None = None,
    cancel_token: CancelToken | None = None,
    stream_telemetry: bool = False,
) -> Iterator[StreamEvent]
```

Sync wrapper over `stream_async()`.

## `resume_async()`

```python
async def resume_async(
    self,
    checkpoint: str | dict[str, Any],
    *,
    timeout: float | None = None,
    cancel_token: CancelToken | None = None,
) -> RunResult
```

Resumes execution from a checkpoint path or payload.

## `resume()`

```python
def resume(
    self,
    checkpoint: str | dict[str, Any],
    *,
    timeout: float | None = None,
    cancel_token: CancelToken | None = None,
) -> RunResult
```

Sync wrapper over `resume_async()`.

## `session_async()` and `session()`

```python
async def session_async(self, id: str | None = None, *, max_history: int = 100) -> Session

def session(self, id: str | None = None, *, max_history: int = 100) -> Session
```

Create or resume session objects.

## `on_confirm()`

```python
def on_confirm(self, handler: ConfirmationHandler) -> Agent
```

Registers destructive tool approval callback and returns `self`.

## Common Exceptions and Failures

- Parameter type validation errors raise `TypeError`.
- Concurrent execution on one `Agent` instance raises `RuntimeError`.
- Run-time failures are returned in `RunResult.error` when possible.
