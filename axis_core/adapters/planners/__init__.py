"""Planner adapters for agent execution strategies.

This module provides built-in planner implementations with lazy loading.

Available planners:
- SequentialPlanner: Execute steps in order (no dependencies)
- AutoPlanner: LLM-based planning with fallback to Sequential (AD-016)
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
            self.__class__ = instance.__class__  # type: ignore[assignment]

    return LazySequentialFactory


# Register sequential planner (always available)
planner_registry.register("sequential", _make_lazy_sequential_factory())


# ===========================================================================
# Lazy factory for Auto planner
# ===========================================================================


def _make_lazy_auto_factory() -> type[Any]:
    """Create a lazy-loading factory for AutoPlanner.

    AutoPlanner requires a model adapter for LLM-based planning.
    Falls back to SequentialPlanner on failure per AD-016.
    """

    class LazyAutoFactory:
        """Lazy factory for AutoPlanner."""

        def __init__(self, **kwargs: Any) -> None:
            from axis_core.adapters.planners.auto import AutoPlanner

            instance = AutoPlanner(**kwargs)
            self.__dict__.update(instance.__dict__)
            self.__class__ = instance.__class__  # type: ignore[assignment]

    return LazyAutoFactory


planner_registry.register("auto", _make_lazy_auto_factory())


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
