"""Sequential planner implementation.

This planner implements a simple sequential execution strategy:
- If the model requests tools, execute them in order
- Add a terminal step to return the final result
- No parallelization or dependency optimization
"""

import uuid

from axis_core.context import Observation, RunContext
from axis_core.protocols.planner import Plan, PlanStep, StepType


class SequentialPlanner:
    """Simple sequential execution planner.

    This planner generates plans that execute tool calls in the order they were
    requested by the model, followed by a terminal step. It's deterministic and
    always succeeds, making it ideal as a fallback strategy.

    Strategy:
        1. If model requested tools: Create TOOL step for each tool call
        2. Create TERMINAL step to return results
        3. All steps execute sequentially (no explicit dependencies needed)

    This is the simplest possible planner - it doesn't optimize, parallelize,
    or reorder steps. It just executes exactly what the model asked for.

    Example:
        >>> planner = SequentialPlanner()
        >>> plan = await planner.plan(observation, ctx)
        >>> print(len(plan.steps))
        3  # 2 tool calls + 1 terminal
    """

    async def plan(
        self,
        observation: Observation,
        ctx: RunContext,
    ) -> Plan:
        """Generate a sequential execution plan from an observation.

        Args:
            observation: Current observation containing tool requests and context
            ctx: Run context with execution history and configuration

        Returns:
            Plan with tool steps (if any) followed by a terminal step

        Examples:
            With tool requests:
                observation.tool_requests = [
                    ToolCall(id="1", name="search", arguments={"q": "test"}),
                    ToolCall(id="2", name="summarize", arguments={}),
                ]
                -> Plan with 3 steps: TOOL(search), TOOL(summarize), TERMINAL

            Without tool requests:
                observation.tool_requests = None
                observation.response = "Here's your answer."
                -> Plan with 1 step: TERMINAL
        """
        steps: list[PlanStep] = []

        # Decision tree for what to do:
        # 1. If model requested tools → execute them
        # 2. If we have a final response (no tools requested) → terminate
        # 3. Otherwise → call model

        if observation.tool_requests:
            # Model requested tools - create TOOL steps
            for i, tool_call in enumerate(observation.tool_requests):
                step = PlanStep(
                    id=f"tool_{i}_{tool_call.id}",
                    type=StepType.TOOL,
                    payload={
                        "tool": tool_call.name,
                        "tool_call_id": tool_call.id,
                        "args": tool_call.arguments,
                    },
                    dependencies=None,  # Sequential - no explicit dependencies
                    retry_policy=None,  # Use default retry policy
                )
                steps.append(step)
            # After tools execute, loop back for next cycle

        elif observation.response is not None:
            # We have a response and no pending tool requests → done!
            terminal_step = PlanStep(
                id="terminal",
                type=StepType.TERMINAL,
                payload={"output": observation.response},
                dependencies=None,
                retry_policy=None,
            )
            steps.append(terminal_step)

        else:
            # No response yet - need to call model
            model_step = PlanStep(
                id="model",
                type=StepType.MODEL,
                payload={},  # Executor will build messages from context
                dependencies=None,
                retry_policy=None,
            )
            steps.append(model_step)

        # Create the plan
        plan = Plan(
            id=f"plan_{ctx.run_id}_{uuid.uuid4().hex[:8]}",
            goal=observation.goal or "Execute requested operations",
            steps=tuple(steps),
            reasoning=(
                f"Sequential execution: {len(steps) - 1} tool call(s) "
                f"followed by terminal output"
                if observation.tool_requests
                else "Direct response with no tool calls"
            ),
            confidence=1.0,  # Deterministic planner is always confident
            metadata={
                "planner": "sequential",
                "tool_count": len(observation.tool_requests) if observation.tool_requests else 0,
            },
        )

        return plan
