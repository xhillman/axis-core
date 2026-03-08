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

__version__ = "0.13.0"

# Public export registry: maps public name -> (module_path, attribute_name).
# Most exports stay lazy; `config` and `tool` are eagerly rebound below to preserve
# `from axis_core import config` / `from axis_core import tool` ergonomics.
_EXPORTS: dict[str, tuple[str, str]] = {
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
    "bootstrap_environment": ("axis_core.config", "bootstrap_environment"),
    "Timeouts": ("axis_core.config", "Timeouts"),
    "RetryPolicy": ("axis_core.config", "RetryPolicy"),
    "RateLimits": ("axis_core.config", "RateLimits"),
    "CacheConfig": ("axis_core.config", "CacheConfig"),
    "ToolPolicy": ("axis_core.config", "ToolPolicy"),
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

_EAGER_EXPORTS = ("config", "tool")

# Public API
__all__ = ["__version__", *_EXPORTS.keys()]


def _load_export(name: str) -> Any:
    module_path, attr_name = _EXPORTS[name]
    module = importlib.import_module(module_path)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __getattr__(name: str) -> Any:
    """Resolve public exports lazily on first attribute access."""
    if name in _EXPORTS:
        return _load_export(name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


# `config` and `tool` share names with importable submodules. If those submodules are
# imported directly, Python can otherwise bind the package attribute to the module object
# instead of the intended public export. Rebinding them here keeps package-level imports
# stable without making the rest of the package bootstrap eager or introducing extra config
# bootstrap side effects.
for _export_name in _EAGER_EXPORTS:
    _load_export(_export_name)

del _export_name
