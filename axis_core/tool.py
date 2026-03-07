"""Public compatibility facade for axis-core tool APIs."""

from axis_core._tool_decorator import tool
from axis_core._tool_runtime import (
    RateLimiter,
    ToolContext,
    build_idempotency_key,
    get_idempotent_result,
    run_idempotent,
    set_idempotent_result,
)
from axis_core._tool_schema import generate_tool_schema
from axis_core._tool_types import Capability, ToolCallRecord, ToolManifest

__all__ = [
    "Capability",
    "ToolCallRecord",
    "ToolManifest",
    "RateLimiter",
    "ToolContext",
    "build_idempotency_key",
    "get_idempotent_result",
    "set_idempotent_result",
    "run_idempotent",
    "generate_tool_schema",
    "tool",
]
