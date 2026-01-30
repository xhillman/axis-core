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

from typing import Any

__version__ = "0.1.0"

# Public API - will be populated as modules are implemented
__all__ = [
    # Version
    "__version__",
    # Core
    "Agent",
    # Tool system
    "tool",
    "ToolContext",
    "ToolManifest",
    "Capability",
    # Budget
    "Budget",
    "BudgetState",
    # Configuration
    "config",
    "Timeouts",
    "RetryPolicy",
    "RateLimits",
    "CacheConfig",
    # Errors
    "AxisError",
    "InputError",
    "ConfigError",
    "PlanError",
    "ToolError",
    "ModelError",
    "BudgetError",
    "TimeoutError",
    "CancelledError",
    "ErrorClass",
    "ErrorRecord",
    # Results
    "RunResult",
    "StreamEvent",
    "RunStats",
    # Context
    "RunContext",
    "RunState",
]


def __getattr__(name: str) -> Any:
    """Lazy loading of submodules to avoid circular imports and missing module errors."""
    # Core
    if name == "Agent":
        from axis_core.agent import Agent

        return Agent

    # Tool system
    if name == "tool":
        from axis_core.tool import tool

        return tool
    if name == "ToolContext":
        from axis_core.tool import ToolContext

        return ToolContext
    if name == "ToolManifest":
        from axis_core.tool import ToolManifest

        return ToolManifest
    if name == "Capability":
        from axis_core.tool import Capability

        return Capability

    # Budget
    if name == "Budget":
        from axis_core.budget import Budget

        return Budget
    if name == "BudgetState":
        from axis_core.budget import BudgetState

        return BudgetState

    # Configuration
    if name == "config":
        from axis_core import config as config_module

        return config_module
    if name == "Timeouts":
        from axis_core.config import Timeouts

        return Timeouts
    if name == "RetryPolicy":
        from axis_core.config import RetryPolicy

        return RetryPolicy
    if name == "RateLimits":
        from axis_core.config import RateLimits

        return RateLimits
    if name == "CacheConfig":
        from axis_core.config import CacheConfig

        return CacheConfig

    # Errors
    if name == "AxisError":
        from axis_core.errors import AxisError

        return AxisError
    if name == "InputError":
        from axis_core.errors import InputError

        return InputError
    if name == "ConfigError":
        from axis_core.errors import ConfigError

        return ConfigError
    if name == "PlanError":
        from axis_core.errors import PlanError

        return PlanError
    if name == "ToolError":
        from axis_core.errors import ToolError

        return ToolError
    if name == "ModelError":
        from axis_core.errors import ModelError

        return ModelError
    if name == "BudgetError":
        from axis_core.errors import BudgetError

        return BudgetError
    if name == "TimeoutError":
        from axis_core.errors import TimeoutError

        return TimeoutError
    if name == "CancelledError":
        from axis_core.errors import CancelledError

        return CancelledError
    if name == "ErrorClass":
        from axis_core.errors import ErrorClass

        return ErrorClass
    if name == "ErrorRecord":
        from axis_core.errors import ErrorRecord

        return ErrorRecord

    # Results
    if name == "RunResult":
        from axis_core.result import RunResult

        return RunResult
    if name == "StreamEvent":
        from axis_core.result import StreamEvent

        return StreamEvent
    if name == "RunStats":
        from axis_core.result import RunStats

        return RunStats

    # Context
    if name == "RunContext":
        from axis_core.context import RunContext

        return RunContext
    if name == "RunState":
        from axis_core.context import RunState

        return RunState

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Override module-level imports that conflict with exported names
# This handles the case where 'tool' module name conflicts with 'tool' function
from axis_core.tool import tool as tool  # noqa: F401, E402
