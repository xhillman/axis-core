"""Planner adapters for agent execution strategies.

This module provides built-in planner implementations with lazy loading.

Available planners:
- SequentialPlanner: Execute steps in order (no dependencies)
- AutoPlanner: Automatically choose planning strategy (future)
- ReActPlanner: Reason-Act loop strategy (future)
"""

from typing import Any

from axis_core.engine.registry import planner_registry

__all__: list[str] = []


# ===========================================================================
# Lazy factory for Sequential planner
# ===========================================================================


def _make_lazy_sequential_factory() -> type[Any]:
    """Create a lazy-loading factory for SequentialPlanner.

    Lazy loading avoids circular imports and defers module loading.
    """

    class LazySequentialFactory:
        """Lazy factory for SequentialPlanner."""

        def __init__(self, **kwargs: Any) -> None:
            from axis_core.adapters.planners.sequential import SequentialPlanner

            instance = SequentialPlanner(**kwargs)
            self.__dict__.update(instance.__dict__)
            self.__class__ = instance.__class__

    return LazySequentialFactory


# Register sequential planner (always available)
planner_registry.register("sequential", _make_lazy_sequential_factory())

# Register "auto" as an alias for sequential (for now)
# In the future, "auto" will choose the best planner based on context
planner_registry.register("auto", _make_lazy_sequential_factory())


# ===========================================================================
# Lazy factories for future planners
# ===========================================================================

# ReAct planner (when implemented):
# def _make_lazy_react_factory() -> type[Any]:
#     class LazyReActFactory:
#         def __init__(self, **kwargs: Any) -> None:
#             from axis_core.adapters.planners.react import ReActPlanner
#             instance = ReActPlanner(**kwargs)
#             self.__dict__.update(instance.__dict__)
#             self.__class__ = instance.__class__
#     return LazyReActFactory
#
# planner_registry.register("react", _make_lazy_react_factory())


# ===========================================================================
# Eager export of planner classes (for direct use)
# ===========================================================================

# Try to export the actual class for users who want to import it directly
try:
    from axis_core.adapters.planners.sequential import SequentialPlanner

    __all__.extend(["SequentialPlanner"])
except ImportError:
    pass
