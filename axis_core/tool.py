"""Tool system for axis-core agents.

This module provides the complete tool system including:
- Capability enum for security declarations
- ToolManifest for tool metadata
- ToolContext for runtime context with read-only budget access
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
from typing import Any, get_args, get_origin, get_type_hints

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
        output_schema: JSON schema for return value
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
    """

    run_id: str
    agent_id: str
    cycle: int
    context: dict[str, object]
    budget: Budget
    budget_state: BudgetState
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
            if name in ("run_id", "agent_id", "cycle", "budget", "budget_state"):
                raise AttributeError(f"ToolContext.{name} is read-only")
        object.__setattr__(self, name, value)


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
    type_hints = get_type_hints(func)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        # Skip 'ctx' parameter - it's injected by the kernel
        if param_name == "ctx":
            continue

        # Get type hint
        param_type = type_hints.get(param_name, Any)

        # Check if parameter is optional (has default or is T | None)
        is_optional = param.default != inspect.Parameter.empty

        # Handle Optional[T] (which is Union[T, None])
        origin = get_origin(param_type)
        if origin is not None:
            args = get_args(param_type)

            # Handle Union types
            is_union = origin is type(None) or (
                hasattr(origin, "__name__") and origin.__name__ == "UnionType"
            )
            if is_union:
                # This is a Union or |
                non_none_types = [arg for arg in args if arg is not type(None)]

                if len(non_none_types) == 1:
                    # This is Optional[T] - extract T
                    param_type = non_none_types[0]
                    is_optional = True
                elif len(non_none_types) > 1:
                    # Union with multiple non-None types
                    raise TypeError(
                        f"Unsupported Union type for parameter '{param_name}': {param_type}. "
                        "Only Optional[T] (T | None) is supported."
                    )

        # Generate JSON schema for the type
        json_type = _python_type_to_json_schema(param_type)
        properties[param_name] = json_type

        # Add to required if not optional
        if not is_optional:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _python_type_to_json_schema(python_type: Any) -> dict[str, Any]:
    """Convert a Python type to JSON schema type.

    Args:
        python_type: Python type annotation

    Returns:
        JSON schema dict
    """
    # Handle basic types
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

    # Handle generic types (List[T], Dict[K, V], etc.)
    origin = get_origin(python_type)
    if origin is list:
        return {"type": "array"}
    if origin is dict:
        return {"type": "object"}

    # Check if it's a Pydantic model
    if hasattr(python_type, "model_json_schema"):
        schema: dict[str, Any] = python_type.model_json_schema()
        return schema

    # Fallback for unknown types
    return {"type": "object"}


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
        # For now, use simple output schema - can be enhanced later
        output_schema: dict[str, Any] = {"type": "string"}

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
    "generate_tool_schema",
    "tool",
]
