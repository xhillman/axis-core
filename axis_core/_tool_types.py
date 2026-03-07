"""Internal metadata types for the tool subsystem."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from axis_core.config import RetryPolicy


class Capability(Enum):
    """Security capabilities that tools can declare."""

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
    """Metadata describing a tool's interface and behavior."""

    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    capabilities: tuple[Capability, ...]
    cache_ttl: int | None = None
    rate_limit: str | None = None
    timeout: float | None = None
    retry: RetryPolicy | None = None


@dataclass(frozen=True)
class ToolCallRecord:
    """Immutable record of a single tool execution."""

    tool_name: str
    call_id: str
    args: dict[str, Any]
    result: Any
    error: str | None
    cached: bool
    duration_ms: float
    timestamp: float


__all__ = ["Capability", "ToolCallRecord", "ToolManifest"]
