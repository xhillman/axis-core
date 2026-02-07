"""Planner adapters for agent execution strategies.

This module provides built-in planner implementations with lazy loading.

Available planners:
- SequentialPlanner: Execute steps in order (no dependencies)
- AutoPlanner: LLM-based planning with fallback to Sequential (AD-016)
- ReActPlanner: Reason-Act loop strategy with explicit reasoning traces
"""

from axis_core.engine.registry import make_lazy_factory, planner_registry

__all__: list[str] = []

_PLANNER_MODULE = "axis_core.adapters.planners"

# Register built-in planner adapters
planner_registry.register(
    "sequential",
    make_lazy_factory(f"{_PLANNER_MODULE}.sequential", "SequentialPlanner"),
)

planner_registry.register(
    "auto",
    make_lazy_factory(f"{_PLANNER_MODULE}.auto", "AutoPlanner"),
)

planner_registry.register(
    "react",
    make_lazy_factory(f"{_PLANNER_MODULE}.react", "ReActPlanner"),
)


# ===========================================================================
# Eager export of planner classes (for direct use)
# ===========================================================================

try:
    from axis_core.adapters.planners.sequential import SequentialPlanner  # noqa: F401

    __all__.extend(["SequentialPlanner"])
except ImportError:
    pass

try:
    from axis_core.adapters.planners.auto import AutoPlanner  # noqa: F401

    __all__.extend(["AutoPlanner"])
except ImportError:
    pass

try:
    from axis_core.adapters.planners.react import ReActPlanner  # noqa: F401

    __all__.extend(["ReActPlanner"])
except ImportError:
    pass
