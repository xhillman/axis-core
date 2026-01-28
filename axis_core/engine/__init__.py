"""Engine internals for axis-core.

Provides the core execution engine including lifecycle management,
step execution, and adapter resolution.
"""

from axis_core.engine.lifecycle import LifecycleEngine, Phase

__all__ = [
    "LifecycleEngine",
    "Phase",
]
