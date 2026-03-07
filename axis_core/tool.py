"""Tool system for axis-core agents.

This module provides the complete tool system including:
- Capability enum for security declarations
- ToolManifest for tool metadata
- ToolContext for runtime context with read-only budget access
- Idempotency helpers for safe side-effect dedupe across retries
- ToolCallRecord for execution tracking
- RateLimiter for rate limiting with token bucket algorithm
- generate_tool_schema() for automatic JSON schema generation
- @tool decorator for registering functions as tools
"""

import asyncio
import functools
import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from types import UnionType
from typing import (
    Annotated,
    Any,
    Literal,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
)

from typing_extensions import NotRequired, Required

from axis_core.budget import Budget, BudgetState
from axis_core.config import RetryPolicy


class Capability(Enum):
    """Security capabilities that tools can declare.

    These capabilities help agents and users understand what resources a tool can access.
    Tools should declare all capabilities they use for transparency and security auditing.
    """

    NETWORK = "network"
    FILESYSTEM = "filesystem"
    DATABASE = "database"
    EMAIL = "email"
    PAYMENT = "payment"
    DESTRUCTIVE = "destructive"
    SUBPROCESS = "subprocess"
    SECRETS = "secrets"


@dataclass(frozen=True)
class ToolManifest:
    """Metadata describing a tool's interface and behavior.

    This immutable manifest is generated from function signatures and decorator arguments.
    It provides all information needed by the execution engine and LLM to use the tool.

    Attributes:
        name: Tool name (defaults to function name)
        description: Human-readable description (from docstring)
        input_schema: JSON schema for parameters
        output_schema: Descriptive JSON schema for return value when known (`{}` = unconstrained)
        capabilities: Security capabilities the tool requires
        cache_ttl: Cache time-to-live in seconds (None = no caching)
        rate_limit: Rate limit string like "10/second" (None = no limit)
        timeout: Timeout in seconds (None = no timeout)
        retry: Retry policy for failures (None = no retries)
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    capabilities: tuple[Capability, ...]
    cache_ttl: int | None = None
    rate_limit: str | None = None
    timeout: float | None = None
    retry: RetryPolicy | None = None


@dataclass
class ToolContext:
    """Runtime context passed to tool functions.

    Provides read-only access to execution state and budget information. Tools can read
    budget limits to make informed decisions but cannot modify them. The context dict
    is mutable to allow tools to share state within a run.

    Attributes:
        run_id: Unique identifier for this agent run
        agent_id: Identifier for the agent instance
        cycle: Current cycle number (0-indexed)
        context: Mutable dict for sharing state between tools
        budget: Budget configuration (read-only)
        budget_state: Current budget consumption (read-only)
        idempotency_key: Optional stable key for dedupe-safe side effects
        retry_attempt: Current retry attempt number (1-indexed)
    """

    run_id: str
    agent_id: str
    cycle: int
    context: dict[str, object]
    budget: Budget
    budget_state: BudgetState
    idempotency_key: str | None = None
    retry_attempt: int = 1
    _initialized: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Mark context as initialized to enable read-only protection."""
        object.__setattr__(self, "_initialized", True)

    def __setattr__(self, name: str, value: object) -> None:
        """Enforce read-only fields after initialization.

        Prevents modification of run_id, agent_id, cycle, budget, and budget_state
        while allowing context dict reassignment and normal initialization.
        """
        if getattr(self, "_initialized", False):
            if name in (
                "run_id",
                "agent_id",
                "cycle",
                "budget",
                "budget_state",
                "idempotency_key",
                "retry_attempt",
            ):
                raise AttributeError(f"ToolContext.{name} is read-only")
        object.__setattr__(self, name, value)


_IDEMPOTENCY_RESULTS_CONTEXT_KEY = "__axis_idempotency_results__"


def build_idempotency_key(*, run_id: str, cycle: int, step_id: str, tool_name: str) -> str:
    """Build a stable tool idempotency key for a specific run/cycle/step."""
    return f"axis:{run_id}:{cycle}:{step_id}:{tool_name}"


def _get_idempotency_store(ctx: ToolContext) -> dict[str, Any]:
    raw_store = ctx.context.get(_IDEMPOTENCY_RESULTS_CONTEXT_KEY)
    if isinstance(raw_store, dict):
        return cast(dict[str, Any], raw_store)

    store: dict[str, Any] = {}
    ctx.context[_IDEMPOTENCY_RESULTS_CONTEXT_KEY] = store
    return store


def get_idempotent_result(
    ctx: ToolContext,
    *,
    key: str | None = None,
) -> tuple[bool, Any]:
    """Return cached idempotent result for key or context key."""
    effective_key = key if key is not None else ctx.idempotency_key
    if effective_key is None:
        return (False, None)

    store = _get_idempotency_store(ctx)
    if effective_key not in store:
        return (False, None)
    return (True, store[effective_key])


def set_idempotent_result(
    ctx: ToolContext,
    result: Any,
    *,
    key: str | None = None,
) -> None:
    """Persist idempotent result in tool context using key or context key."""
    effective_key = key if key is not None else ctx.idempotency_key
    if effective_key is None:
        return

    store = _get_idempotency_store(ctx)
    store[effective_key] = result


async def run_idempotent(
    ctx: ToolContext,
    operation: Callable[[], Any],
    *,
    key: str | None = None,
) -> Any:
    """Run operation and memoize result by idempotency key when available."""
    found, cached_result = get_idempotent_result(ctx, key=key)
    if found:
        return cached_result

    result = operation()
    if inspect.isawaitable(result):
        result = await result
    set_idempotent_result(ctx, result, key=key)
    return result


@dataclass(frozen=True)
class ToolCallRecord:
    """Immutable record of a single tool execution.

    Captures all information about a tool invocation for observability, debugging,
    and telemetry. Used internally by the execution engine.

    Attributes:
        tool_name: Name of the tool that was called
        call_id: Unique identifier for this specific call
        args: Arguments passed to the tool
        result: Return value from the tool (None if error)
        error: Error message if tool failed (None if success)
        cached: Whether result was served from cache
        duration_ms: Execution time in milliseconds
        timestamp: Unix timestamp when call started
    """

    tool_name: str
    call_id: str
    args: dict[str, Any]
    result: Any
    error: str | None
    cached: bool
    duration_ms: float
    timestamp: float


class RateLimiter:
    """Token bucket rate limiter for controlling request rates.

    Implements the token bucket algorithm: tokens are added at a constant rate up to
    a maximum capacity. Each request consumes one token. When tokens are exhausted,
    requests must wait for refill.

    This is thread-safe for asyncio but not for multi-threading.

    Attributes:
        count: Maximum number of tokens (bucket capacity)
        period_seconds: Time period over which tokens refill
    """

    def __init__(self, count: int, period_seconds: float) -> None:
        """Initialize rate limiter with token bucket parameters.

        Args:
            count: Maximum tokens (bucket capacity)
            period_seconds: Period for token refill (tokens_per_second = count/period)
        """
        self._count = count
        self._period_seconds = period_seconds
        self._tokens = float(count)
        self._last_refill = time.time()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time since last refill."""
        now = time.time()
        elapsed = now - self._last_refill

        if elapsed > 0 and self._period_seconds > 0:
            # Calculate tokens to add based on elapsed time
            tokens_to_add = (elapsed / self._period_seconds) * self._count
            self._tokens = min(self._count, self._tokens + tokens_to_add)
            self._last_refill = now

    def try_acquire(self) -> bool:
        """Try to acquire a token without waiting.

        Returns:
            True if token acquired, False if no tokens available
        """
        self._refill()

        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True

        return False

    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary.

        This will wait until a token becomes available through refill.
        """
        async with self._lock:
            while True:
                self._refill()

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return

                # Calculate how long to wait for next token
                if self._count > 0 and self._period_seconds > 0:
                    time_per_token = self._period_seconds / self._count
                    await asyncio.sleep(time_per_token / 2)  # Check twice per token period
                else:
                    # If count is 0, wait a bit and retry
                    await asyncio.sleep(0.1)


def generate_tool_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """Generate JSON schema from function signature.

    Inspects the function's parameters and type hints to create a JSON schema
    compatible with LLM tool use APIs. Supports basic Python types, optionals,
    and Pydantic models.

    Type mapping:
        - str → "string"
        - int → "integer"
        - float → "number"
        - bool → "boolean"
        - list → {"type": "array"}
        - dict → {"type": "object"}
        - T | None → Same as T, but not required
        - Pydantic models → model_json_schema()

    Args:
        func: Function to generate schema for

    Returns:
        JSON schema dict with "properties" and "required" keys

    Raises:
        TypeError: If function uses unsupported Union types (multiple non-None types)
    """
    sig = inspect.signature(func)
    type_hints = get_type_hints(func, include_extras=True)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        # Skip 'ctx' parameter - it's injected by the kernel
        if param_name == "ctx":
            continue

        # Get type hint
        param_type = type_hints.get(param_name, Any)

        # Check if parameter is optional (has default or is T | None)
        param_type, type_is_optional = _unwrap_optional_type(param_type)
        is_optional = param.default != inspect.Parameter.empty or type_is_optional

        # Generate JSON schema for the type
        json_type = _python_type_to_json_schema(param_type, path=param_name)
        properties[param_name] = json_type

        # Add to required if not optional
        if not is_optional:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _generate_tool_output_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """Infer descriptive output metadata from the function return annotation.

    The output schema is advisory metadata only. If the return annotation is absent or cannot be
    represented honestly with the supported JSON schema subset, this returns `{}` to indicate that
    the tool output is unconstrained.
    """
    type_hints = get_type_hints(func, include_extras=True)
    if "return" not in type_hints:
        return {}

    return_type = _unwrap_annotated_type(type_hints["return"])
    if return_type is Any:
        return {}

    if _is_union_type(return_type):
        union_members = tuple(_unwrap_annotated_type(arg) for arg in get_args(return_type))
        non_none_members = tuple(member for member in union_members if member is not type(None))
        if len(non_none_members) == 1 and len(non_none_members) != len(union_members):
            try:
                return {
                    "anyOf": [
                        _python_type_to_json_schema(non_none_members[0], path="return"),
                        {"type": "null"},
                    ]
                }
            except TypeError:
                return {}
        return {}

    try:
        return _python_type_to_json_schema(return_type, path="return")
    except TypeError:
        return {}


def _python_type_to_json_schema(python_type: Any, *, path: str) -> dict[str, Any]:
    """Convert a Python type to JSON schema type.

    Args:
        python_type: Python type annotation

    Returns:
        JSON schema dict
    """
    python_type = _unwrap_annotated_type(python_type)
    python_type, _ = _unwrap_optional_type(python_type)

    if _is_union_type(python_type):
        raise TypeError(
            f"Unsupported Union type for {_describe_schema_path(path)}: {python_type}. "
            "Only Optional[T] (T | None) is supported."
        )

    # Handle basic types
    if python_type is Any:
        return {"type": "object"}
    if python_type is type(None):
        return {"type": "null"}
    if python_type is str:
        return {"type": "string"}
    if python_type is int:
        return {"type": "integer"}
    if python_type is float:
        return {"type": "number"}
    if python_type is bool:
        return {"type": "boolean"}
    if python_type is list:
        return {"type": "array"}
    if python_type is dict:
        return {"type": "object"}

    # Check if it's a TypedDict before generic dict handling.
    if is_typeddict(python_type):
        return _typed_dict_to_json_schema(python_type, path=path)

    # Handle generic types (List[T], Dict[K, V], etc.)
    origin = get_origin(python_type)
    if origin is list:
        args = get_args(python_type)
        if len(args) != 1 or args[0] is Any:
            return {"type": "array"}
        return {
            "type": "array",
            "items": _python_type_to_json_schema(args[0], path=f"{path}[]"),
        }
    if origin is dict:
        args = get_args(python_type)
        if len(args) != 2:
            return {"type": "object"}

        key_type, value_type = args
        key_type = _unwrap_annotated_type(key_type)
        if key_type not in (str, Any):
            raise TypeError(
                f"Dictionary types for {_describe_schema_path(path)} must use string keys "
                f"for JSON schema compatibility: {python_type}."
            )

        schema: dict[str, Any] = {"type": "object"}
        if value_type is not Any:
            schema["additionalProperties"] = _python_type_to_json_schema(
                value_type,
                path=f"{path}.*",
            )
        return schema
    if origin is Literal:
        return _literal_to_json_schema(python_type, path=path)
    if origin in {Required, NotRequired}:
        args = get_args(python_type)
        if len(args) != 1:
            return {"type": "object"}
        return _python_type_to_json_schema(args[0], path=path)

    # Check if it's a Pydantic model
    if hasattr(python_type, "model_json_schema"):
        pydantic_schema: dict[str, Any] = python_type.model_json_schema()
        return pydantic_schema

    # Fallback for unknown types
    return {"type": "object"}


def _unwrap_annotated_type(python_type: Any) -> Any:
    """Strip Annotated metadata while preserving the underlying schema type."""
    while get_origin(python_type) is Annotated:
        args = get_args(python_type)
        if not args:
            return Any
        python_type = args[0]
    return python_type


def _unwrap_optional_type(python_type: Any) -> tuple[Any, bool]:
    """Return the underlying type for Optional[T] and whether it was optional."""
    python_type = _unwrap_annotated_type(python_type)
    origin = get_origin(python_type)
    if origin not in {Union, UnionType}:
        return (python_type, False)

    args = tuple(_unwrap_annotated_type(arg) for arg in get_args(python_type))
    non_none_types = tuple(arg for arg in args if arg is not type(None))
    if len(non_none_types) == 1 and len(non_none_types) != len(args):
        return (non_none_types[0], True)
    return (python_type, False)


def _is_union_type(python_type: Any) -> bool:
    """Return True when the annotation is a non-optional union."""
    return get_origin(_unwrap_annotated_type(python_type)) in {Union, UnionType}


def _literal_to_json_schema(python_type: Any, *, path: str) -> dict[str, Any]:
    """Convert Literal values to a deterministic enum schema."""
    values = get_args(python_type)
    if not values:
        return {"type": "object"}

    json_type = _literal_json_type(values[0])
    for value in values[1:]:
        value_type = _literal_json_type(value)
        if value_type != json_type:
            raise TypeError(
                f"Literal values for {_describe_schema_path(path)} must share the same JSON type: "
                f"{python_type}."
            )

    return {"type": json_type, "enum": list(values)}


def _literal_json_type(value: Any) -> str:
    """Map a literal value to its JSON schema scalar type."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    raise TypeError(
        "Only string, boolean, integer, and number Literal values are supported for "
        f"tool schemas: {value!r}."
    )


def _typed_dict_to_json_schema(python_type: Any, *, path: str) -> dict[str, Any]:
    """Convert a TypedDict class into an object schema."""
    annotations = get_type_hints(python_type, include_extras=True)
    required_keys = cast(set[str], set(getattr(python_type, "__required_keys__", set())))
    optional_keys = cast(set[str], set(getattr(python_type, "__optional_keys__", set())))
    total = bool(getattr(python_type, "__total__", True))

    properties: dict[str, Any] = {}
    required: list[str] = []

    for field_name, field_type in annotations.items():
        is_required = field_name in required_keys or (
            field_name not in optional_keys and total and field_name not in required_keys
        )

        origin = get_origin(field_type)
        if origin in {Required, NotRequired}:
            args = get_args(field_type)
            if len(args) == 1:
                field_type = args[0]
            is_required = origin is Required

        properties[field_name] = _python_type_to_json_schema(
            field_type,
            path=f"{path}.{field_name}",
        )
        if is_required:
            required.append(field_name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _describe_schema_path(path: str) -> str:
    """Return a readable label for error messages."""
    if any(token in path for token in (".", "[", "*")):
        return f"field '{path}'"
    return f"parameter '{path}'"


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    capabilities: list[Capability] | None = None,
    cache_ttl: int | None = None,
    rate_limit: str | None = None,
    timeout: float | None = None,
    retry: RetryPolicy | None = None,
) -> Callable[..., Any]:
    """Decorator to register a function as an axis-core tool.

    Supports two syntaxes:
        @tool                    # No parentheses
        @tool(name="custom")     # With parameters

    The decorator:
    1. Generates a ToolManifest from the function signature
    2. Wraps the function in an async wrapper (always async)
    3. Attaches metadata: _axis_tool, _axis_manifest, __wrapped__

    Args:
        func: Function to decorate (when used without parentheses)
        name: Override tool name (default: function name)
        description: Override description (default: docstring)
        capabilities: Security capabilities this tool requires
        cache_ttl: Cache time-to-live in seconds
        rate_limit: Rate limit like "10/second"
        timeout: Timeout in seconds
        retry: Retry policy for failures

    Returns:
        Decorated async function with tool metadata

    Example:
        @tool
        def greet(name: str) -> str:
            '''Greet a person by name.'''
            return f"Hello, {name}!"

        @tool(capabilities=[Capability.NETWORK], timeout=30.0)
        async def fetch_url(url: str) -> str:
            '''Fetch content from a URL.'''
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                return response.text
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        """Inner decorator that does the actual wrapping."""
        # Generate manifest
        tool_name = name if name is not None else fn.__name__
        tool_description = description if description is not None else (fn.__doc__ or "")
        tool_capabilities = tuple(capabilities) if capabilities is not None else ()

        input_schema = generate_tool_schema(fn)
        output_schema = _generate_tool_output_schema(fn)

        manifest = ToolManifest(
            name=tool_name,
            description=tool_description,
            input_schema=input_schema,
            output_schema=output_schema,
            capabilities=tool_capabilities,
            cache_ttl=cache_ttl,
            rate_limit=rate_limit,
            timeout=timeout,
            retry=retry,
        )

        # Create async wrapper
        if inspect.iscoroutinefunction(fn):
            # Already async - wrap directly
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await fn(*args, **kwargs)
        else:
            # Sync function - make async wrapper
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return fn(*args, **kwargs)

        # Attach metadata
        setattr(async_wrapper, "_axis_tool", True)
        setattr(async_wrapper, "_axis_manifest", manifest)
        setattr(async_wrapper, "__wrapped__", fn)

        return async_wrapper

    # Handle dual syntax: @tool vs @tool(...)
    if func is not None:
        # Called as @tool without parentheses
        return decorator(func)

    # Called as @tool(...) with parentheses
    return decorator


# Internal components not exported in __all__
__all__ = [
    "Capability",
    "ToolManifest",
    "ToolContext",
    "build_idempotency_key",
    "get_idempotent_result",
    "set_idempotent_result",
    "run_idempotent",
    "generate_tool_schema",
    "tool",
]
