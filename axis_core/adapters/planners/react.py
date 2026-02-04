"""ReActPlanner: Reason + Act planning with explicit reasoning traces.

This planner implements the ReAct (Reason + Act) pattern from the paper
"ReAct: Synergizing Reasoning and Acting in Language Models".

Pattern:
    Thought: [Reasoning about what to do]
    Action: [Tool to call]
    Observation: [Tool result]
    ... (repeat until task complete)
    Thought: [Final reasoning]
    Final Answer: [Output]

The planner extracts explicit reasoning from model outputs and structures
execution into MODEL (thought) and TOOL (action) steps.

Architecture Decisions:
- Explicit reasoning captured in plan metadata for observability
- max_iterations limit prevents infinite loops
- Falls back to terminal output when iterations exhausted
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any

from axis_core.context import Observation, RunContext
from axis_core.protocols.planner import Plan, PlanStep, StepType

logger = logging.getLogger(__name__)

# Default max iterations before forcing termination
_DEFAULT_MAX_ITERATIONS = 10

# ReAct prompt template
_REACT_SYSTEM_PROMPT = """\
You are an AI assistant using the ReAct (Reason + Act) framework to solve tasks.

Follow this pattern:

Thought: [Your reasoning about what to do next]
Action: [Tool to use, if needed]
Observation: [Result from tool - this will be provided to you]
... (repeat Thought/Action/Observation as needed)
Thought: [Final reasoning]
Final Answer: [Your complete response to the user]

Rules:
- Always start with "Thought:" to explain your reasoning
- Use "Action:" when you need to use a tool
- After receiving tool results (Observation:), continue with another "Thought:"
- When you have enough information, provide "Final Answer:"
- Be explicit about your reasoning process
- Keep thoughts focused and concise
"""


def _extract_thought(text: str) -> str:
    """Extract the most recent thought from model response.

    Args:
        text: Model response text

    Returns:
        Extracted thought text, or empty string if none found
    """
    # Look for "Thought:" pattern
    # Captures text until the next Action/Observation/Final Answer marker
    pattern = r"Thought:\s*([^\n]+(?:\n(?!(?:Action:|Observation:|Final Answer:))[^\n]+)*)"
    matches: list[Any] = re.findall(pattern, text, re.IGNORECASE)
    if matches:
        # Return the last (most recent) thought
        last_match: str = matches[-1]
        return last_match.strip()
    return ""


def _extract_final_answer(text: str) -> str | None:
    """Extract final answer from model response.

    Args:
        text: Model response text

    Returns:
        Final answer text if present, None otherwise
    """
    match = re.search(r"Final Answer:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _has_final_answer(text: str) -> bool:
    """Check if model response contains a final answer.

    Args:
        text: Model response text

    Returns:
        True if "Final Answer:" is present
    """
    return bool(re.search(r"Final Answer:", text, re.IGNORECASE))


def _build_react_message(
    observation: Observation,
    ctx: RunContext,
) -> list[dict[str, str]]:
    """Build message history for ReAct model call.

    Args:
        observation: Current observation
        ctx: Run context with history

    Returns:
        List of message dicts for model API
    """
    messages: list[dict[str, str]] = []

    # Initial user input
    messages.append({
        "role": "user",
        "content": observation.input.text,
    })

    # If we have a response from previous cycle, add it as assistant message
    if observation.response:
        messages.append({
            "role": "assistant",
            "content": observation.response,
        })

    # If continuing after tool execution, add observation
    if observation.tool_requests and ctx.cycle_count > 0:
        # Tool results are in the response already, formatted by lifecycle engine
        # Just continue the conversation
        messages.append({
            "role": "user",
            "content": "Continue with your reasoning based on the tool results above.",
        })

    return messages


class ReActPlanner:
    """ReAct (Reason + Act) planner with explicit reasoning traces.

    Implements the ReAct pattern: Thought → Action → Observation loop.
    Extracts reasoning from model outputs and structures execution into
    explicit MODEL and TOOL steps.

    Args:
        model: ModelAdapter instance for generating thoughts and actions
        max_iterations: Maximum number of thought-action cycles (default: 10)

    Example:
        >>> from axis_core.adapters.models.anthropic import AnthropicModel
        >>> model = AnthropicModel("claude-sonnet-4-20250514")
        >>> planner = ReActPlanner(model=model, max_iterations=5)
        >>> plan = await planner.plan(observation, ctx)
        >>> print(plan.reasoning)  # Shows extracted thought
    """

    def __init__(self, model: Any, max_iterations: int = _DEFAULT_MAX_ITERATIONS) -> None:
        """Initialize ReActPlanner.

        Args:
            model: ModelAdapter for generating reasoning and actions
            max_iterations: Max thought-action cycles before forcing termination
        """
        self._model = model
        self._max_iterations = max_iterations

    async def plan(
        self,
        observation: Observation,
        ctx: RunContext,
    ) -> Plan:
        """Generate a ReAct-style execution plan.

        Calls the model with ReAct prompt structure, extracts reasoning
        (thoughts), and creates appropriate steps (MODEL for reasoning,
        TOOL for actions, TERMINAL for final answers).

        Enforces max_iterations limit by creating a TERMINAL step when
        the cycle count exceeds the configured maximum.

        Args:
            observation: Current observation with input and context
            ctx: Run context with execution history and cycle count

        Returns:
            Plan with steps following ReAct pattern. Metadata includes:
            - "planner": "react"
            - "cycle": Current cycle number
            - "max_iterations": Configured limit
            - "thought": Extracted reasoning text
            - "iterations_exceeded": True if max iterations hit
        """
        # Check if we've exceeded max iterations
        if ctx.cycle_count >= self._max_iterations:
            return self._create_termination_plan(
                observation=observation,
                ctx=ctx,
                reason=f"Maximum iterations ({self._max_iterations}) reached",
            )

        # If observation already has a final response, terminate
        if observation.response and not observation.tool_requests:
            # Check if the response contains a final answer
            if _has_final_answer(observation.response):
                final_answer = _extract_final_answer(observation.response)
                if final_answer:
                    return self._create_final_answer_plan(
                        observation=observation,
                        ctx=ctx,
                        answer=final_answer,
                        thought=_extract_thought(observation.response),
                    )

        # Call model to get next thought/action
        try:
            response = await self._call_model(observation, ctx)
        except Exception as exc:
            logger.warning("ReActPlanner model call failed: %s", exc)
            # Create terminal plan with error
            return self._create_termination_plan(
                observation=observation,
                ctx=ctx,
                reason=f"Model call failed: {exc}",
            )

        # Extract thought from response
        thought = _extract_thought(response.content)

        # Check for final answer
        if _has_final_answer(response.content):
            final_answer = _extract_final_answer(response.content)
            if final_answer:
                return self._create_final_answer_plan(
                    observation=observation,
                    ctx=ctx,
                    answer=final_answer,
                    thought=thought,
                )

        # Model wants to use tools - create plan with MODEL + TOOL steps
        if response.tool_calls:
            return self._create_action_plan(
                observation=observation,
                ctx=ctx,
                tool_calls=response.tool_calls,
                thought=thought,
            )

        # No tools and no final answer - create MODEL step to continue reasoning
        return self._create_reasoning_plan(
            observation=observation,
            ctx=ctx,
            thought=thought,
        )

    async def _call_model(
        self,
        observation: Observation,
        ctx: RunContext,
    ) -> Any:  # ModelResponse
        """Call model with ReAct prompt.

        Args:
            observation: Current observation
            ctx: Run context

        Returns:
            ModelResponse from model adapter
        """
        messages = _build_react_message(observation, ctx)

        return await self._model.complete(
            messages=messages,
            system=_REACT_SYSTEM_PROMPT,
            temperature=0.0,  # Deterministic reasoning
            max_tokens=2048,
        )

    def _create_action_plan(
        self,
        observation: Observation,
        ctx: RunContext,
        tool_calls: tuple[Any, ...],
        thought: str,
    ) -> Plan:
        """Create plan with thought (MODEL) + actions (TOOL steps).

        Args:
            observation: Current observation
            ctx: Run context
            tool_calls: Tool calls from model response
            thought: Extracted thought text

        Returns:
            Plan with MODEL step followed by TOOL steps
        """
        steps: list[PlanStep] = []

        # First step: MODEL (the thought/reasoning)
        model_step = PlanStep(
            id="thought",
            type=StepType.MODEL,
            payload={"thought": thought},
            dependencies=None,
            retry_policy=None,
        )
        steps.append(model_step)

        # Following steps: TOOL (the actions)
        for i, tool_call in enumerate(tool_calls):
            tool_step = PlanStep(
                id=f"action_{i}_{tool_call.id}",
                type=StepType.TOOL,
                payload={
                    "tool": tool_call.name,
                    "tool_call_id": tool_call.id,
                    "args": tool_call.arguments,
                },
                dependencies=None,  # Sequential execution
                retry_policy=None,
            )
            steps.append(tool_step)

        return Plan(
            id=f"react_{ctx.run_id}_{uuid.uuid4().hex[:8]}",
            goal=observation.goal or "Execute ReAct cycle",
            steps=tuple(steps),
            reasoning=thought or "ReAct: Thought → Action pattern",
            confidence=0.8,  # Moderate confidence for intermediate steps
            metadata={
                "planner": "react",
                "cycle": ctx.cycle_count,
                "max_iterations": self._max_iterations,
                "thought": thought,
                "action_count": len(tool_calls),
            },
        )

    def _create_final_answer_plan(
        self,
        observation: Observation,
        ctx: RunContext,
        answer: str,
        thought: str,
    ) -> Plan:
        """Create plan with final answer (TERMINAL step).

        Args:
            observation: Current observation
            ctx: Run context
            answer: Final answer text
            thought: Extracted thought text

        Returns:
            Plan with single TERMINAL step
        """
        terminal_step = PlanStep(
            id="final_answer",
            type=StepType.TERMINAL,
            payload={"output": answer},
            dependencies=None,
            retry_policy=None,
        )

        return Plan(
            id=f"react_final_{ctx.run_id}_{uuid.uuid4().hex[:8]}",
            goal=observation.goal or "Complete task",
            steps=(terminal_step,),
            reasoning=thought or "ReAct: Final reasoning complete",
            confidence=0.9,  # High confidence for completion
            metadata={
                "planner": "react",
                "cycle": ctx.cycle_count,
                "max_iterations": self._max_iterations,
                "thought": thought,
                "final_answer": True,
            },
        )

    def _create_reasoning_plan(
        self,
        observation: Observation,
        ctx: RunContext,
        thought: str,
    ) -> Plan:
        """Create plan with just MODEL step for continued reasoning.

        Args:
            observation: Current observation
            ctx: Run context
            thought: Extracted thought text

        Returns:
            Plan with single MODEL step
        """
        model_step = PlanStep(
            id="continue_thought",
            type=StepType.MODEL,
            payload={"thought": thought},
            dependencies=None,
            retry_policy=None,
        )

        return Plan(
            id=f"react_think_{ctx.run_id}_{uuid.uuid4().hex[:8]}",
            goal=observation.goal or "Continue reasoning",
            steps=(model_step,),
            reasoning=thought or "ReAct: Continued reasoning",
            confidence=0.7,  # Lower confidence for incomplete reasoning
            metadata={
                "planner": "react",
                "cycle": ctx.cycle_count,
                "max_iterations": self._max_iterations,
                "thought": thought,
            },
        )

    def _create_termination_plan(
        self,
        observation: Observation,
        ctx: RunContext,
        reason: str,
    ) -> Plan:
        """Create plan that forces termination.

        Args:
            observation: Current observation
            ctx: Run context
            reason: Why termination was forced

        Returns:
            Plan with TERMINAL step
        """
        # Use observation response if available, otherwise generic message
        output = observation.response or f"Task terminated: {reason}"

        terminal_step = PlanStep(
            id="forced_termination",
            type=StepType.TERMINAL,
            payload={"output": output},
            dependencies=None,
            retry_policy=None,
        )

        return Plan(
            id=f"react_term_{ctx.run_id}_{uuid.uuid4().hex[:8]}",
            goal=observation.goal or "Terminate task",
            steps=(terminal_step,),
            reasoning=f"ReAct: {reason}",
            confidence=0.5,  # Low confidence for forced termination
            metadata={
                "planner": "react",
                "cycle": ctx.cycle_count,
                "max_iterations": self._max_iterations,
                "iterations_exceeded": True,
                "termination_reason": reason,
            },
        )
