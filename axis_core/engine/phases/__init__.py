"""Per-phase modules for the lifecycle engine.

Each module contains the logic for a single lifecycle phase,
extracted from the monolithic lifecycle.py for maintainability.
"""

from axis_core.engine.phases.act import act
from axis_core.engine.phases.evaluate import evaluate
from axis_core.engine.phases.finalize import finalize
from axis_core.engine.phases.initialize import initialize
from axis_core.engine.phases.observe import observe
from axis_core.engine.phases.plan import plan

__all__ = [
    "act",
    "evaluate",
    "finalize",
    "initialize",
    "observe",
    "plan",
]
