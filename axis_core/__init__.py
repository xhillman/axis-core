"""
Axis Core - A modular, observable AI agent framework.

Usage:
    import axis_core
    # or
    import axis_core as axis

Example:
    from axis_core import Agent, tool, Budget

    @tool
    def greet(name: str) -> str:
        return f"Hello, {name}!"

    agent = Agent(tools=[greet])
    result = agent.run("Greet the user named Alice")
"""

import importlib
from typing import Any

__version__ = "0.5.1"

# Lazy-loading registry: maps public name → (module_path, attribute_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Core
    "Agent": ("axis_core.agent", "Agent"),
    # Tool system
    "tool": ("axis_core.tool", "tool"),
    "ToolContext": ("axis_core.tool", "ToolContext"),
    "ToolManifest": ("axis_core.tool", "ToolManifest"),
    "Capability": ("axis_core.tool", "Capability"),
    # Budget
    "Budget": ("axis_core.budget", "Budget"),
    "BudgetState": ("axis_core.budget", "BudgetState"),
    # Configuration
    "config": ("axis_core.config", "config"),
    "Timeouts": ("axis_core.config", "Timeouts"),
    "RetryPolicy": ("axis_core.config", "RetryPolicy"),
    "RateLimits": ("axis_core.config", "RateLimits"),
    "CacheConfig": ("axis_core.config", "CacheConfig"),
    # Errors
    "AxisError": ("axis_core.errors", "AxisError"),
    "InputError": ("axis_core.errors", "InputError"),
    "ConfigError": ("axis_core.errors", "ConfigError"),
    "PlanError": ("axis_core.errors", "PlanError"),
    "ToolError": ("axis_core.errors", "ToolError"),
    "ModelError": ("axis_core.errors", "ModelError"),
    "BudgetError": ("axis_core.errors", "BudgetError"),
    "TimeoutError": ("axis_core.errors", "TimeoutError"),
    "CancelledError": ("axis_core.errors", "CancelledError"),
    "ConcurrencyError": ("axis_core.errors", "ConcurrencyError"),
    "ErrorClass": ("axis_core.errors", "ErrorClass"),
    "ErrorRecord": ("axis_core.errors", "ErrorRecord"),
    # Results
    "RunResult": ("axis_core.result", "RunResult"),
    "StreamEvent": ("axis_core.result", "StreamEvent"),
    "RunStats": ("axis_core.result", "RunStats"),
    # Context
    "RunContext": ("axis_core.context", "RunContext"),
    "RunState": ("axis_core.context", "RunState"),
    "Session": ("axis_core.session", "Session"),
    "Message": ("axis_core.session", "Message"),
    "Attachment": ("axis_core.attachments", "Attachment"),
    "Image": ("axis_core.attachments", "Image"),
    "PDF": ("axis_core.attachments", "PDF"),
    "CancelToken": ("axis_core.cancel", "CancelToken"),
}

# Public API
__all__ = ["__version__", *_LAZY_IMPORTS.keys()]


def __getattr__(name: str) -> Any:
    """Lazy loading of submodules to avoid circular imports and missing module errors."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Override module-level imports that conflict with exported names
# This handles cases where submodule names conflict with exported objects
from axis_core.config import config as config  # noqa: F401, E402
from axis_core.tool import tool as tool  # noqa: F401, E402
