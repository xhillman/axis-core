"""Internal decorator plumbing for the public tool surface."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any

from axis_core._tool_schema import (
    generate_tool_output_schema as _generate_tool_output_schema,
)
from axis_core._tool_schema import generate_tool_schema
from axis_core._tool_types import Capability, ToolManifest
from axis_core.config import RetryPolicy


def _build_manifest(
    fn: Callable[..., Any],
    *,
    name: str | None,
    description: str | None,
    capabilities: list[Capability] | None,
    cache_ttl: int | None,
    rate_limit: str | None,
    timeout: float | None,
    retry: RetryPolicy | None,
) -> ToolManifest:
    tool_name = name if name is not None else fn.__name__
    tool_description = description if description is not None else (fn.__doc__ or "")
    tool_capabilities = tuple(capabilities) if capabilities is not None else ()

    return ToolManifest(
        name=tool_name,
        description=tool_description,
        input_schema=generate_tool_schema(fn),
        output_schema=_generate_tool_output_schema(fn),
        capabilities=tool_capabilities,
        cache_ttl=cache_ttl,
        rate_limit=rate_limit,
        timeout=timeout,
        retry=retry,
    )


def _wrap_tool_function(fn: Callable[..., Any]) -> Callable[..., Any]:
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_coroutine_wrapper(*args: Any, **kwargs: Any) -> Any:
            return await fn(*args, **kwargs)

        return async_coroutine_wrapper

    @functools.wraps(fn)
    async def async_sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs)

    return async_sync_wrapper


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
    """Decorator to register a function as an axis-core tool."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        manifest = _build_manifest(
            fn,
            name=name,
            description=description,
            capabilities=capabilities,
            cache_ttl=cache_ttl,
            rate_limit=rate_limit,
            timeout=timeout,
            retry=retry,
        )
        async_wrapper = _wrap_tool_function(fn)

        setattr(async_wrapper, "_axis_tool", True)
        setattr(async_wrapper, "_axis_manifest", manifest)
        setattr(async_wrapper, "__wrapped__", fn)
        return async_wrapper

    if func is not None:
        return decorator(func)

    return decorator


__all__ = ["tool"]
