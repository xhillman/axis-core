"""Planner protocol and associated dataclasses.

This module defines the Planner protocol interface for execution planning strategies,
along with enums for step types and dataclasses for plans and steps.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from axis_core.config import RetryPolicy


class StepType(Enum):
    """Types of execution steps in a plan.

    Attributes:
        MODEL: Call the LLM model for reasoning/generation
        TOOL: Execute a tool with arguments
        TRANSFORM: Transform data between steps
        TERMINAL: Final step that produces the result
    """

    MODEL = "model"
    TOOL = "tool"
    TRANSFORM = "transform"
    TERMINAL = "terminal"


@dataclass(frozen=True)
class PlanStep:
    """A single step in an execution plan.

    Attributes:
        id: Unique identifier for this step
        type: Type of step (model, tool, transform, terminal)
        payload: Step-specific data (e.g., tool name and args, model prompt)
        dependencies: Tuple of step IDs that must complete before this step
        retry_policy: Retry configuration for this step (None = use default)
    """

    id: str
    type: StepType
    payload: dict[str, Any] = field(default_factory=dict)
    dependencies: tuple[str, ...] | None = None
    retry_policy: RetryPolicy | None = None


@dataclass(frozen=True)
class Plan:
    """An execution plan consisting of multiple steps.

    Attributes:
        id: Unique identifier for this plan
        goal: High-level description of what this plan aims to achieve
        steps: Tuple of steps to execute (order may matter depending on dependencies)
        reasoning: Optional explanation of why this plan was chosen
        confidence: Optional confidence score (0.0-1.0)
        metadata: Additional planner-specific metadata
    """

    id: str
    goal: str
    steps: tuple[PlanStep, ...]
    reasoning: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Planner(Protocol):
    """Protocol for execution planning strategies.

    Planners determine how an agent should achieve its goals by generating sequences
    of steps. Different planners implement different strategies (ReAct, Sequential,
    Tree-of-Thoughts, etc.).

    Implementations must provide:
    - plan() method that generates a Plan from observations and context
    """

    async def plan(
        self,
        observation: Any,  # Observation
        ctx: Any,  # RunContext
    ) -> Plan:
        """Generate an execution plan from an observation.

        Args:
            observation: Current observation from the environment
            ctx: Run context containing history, config, and state

        Returns:
            Plan with steps to execute

        Examples:
            A ReAct planner might generate a plan with:
            1. MODEL step: Reason about what to do
            2. TOOL step: Execute a chosen tool
            3. MODEL step: Reflect on results
            4. TERMINAL step: Return final answer

            A Sequential planner might generate a plan with:
            1. TOOL step: Call tool A
            2. TOOL step: Call tool B with results from A
            3. TERMINAL step: Return combined results
        """
        ...
