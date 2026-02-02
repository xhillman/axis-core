"""AutoPlanner: LLM-based execution planning with fallback.

This planner uses a model adapter to generate structured execution plans from
observations. It analyzes available tools and the current context to determine
the optimal sequence and dependencies of steps.

On any planning failure (model error, parse error, invalid plan), it falls back
to SequentialPlanner per AD-016.

Architecture Decisions:
- AD-016: AutoPlanner falls back to SequentialPlanner on failure
- AD-006: Plans are validated by the lifecycle engine after generation
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

from axis_core.adapters.planners.sequential import SequentialPlanner
from axis_core.context import Observation, RunContext
from axis_core.protocols.planner import Plan, PlanStep, StepType

logger = logging.getLogger(__name__)

# Default confidence when model doesn't specify one
_DEFAULT_CONFIDENCE = 0.7

# Reduced confidence for fallback plans
_FALLBACK_CONFIDENCE = 0.5

_PLANNING_SYSTEM_PROMPT = """\
You are an execution planner for an AI agent. Your job is to analyze the current \
situation and produce a structured JSON plan.

CRITICAL: Respond with ONLY valid JSON. No explanation before or after. Just the JSON object.

Required format:
{
  "reasoning": "Brief explanation of your planning strategy",
  "confidence": 0.85,
  "steps": [
    {
      "type": "tool",
      "tool": "tool_name",
      "args": {"param": "value"},
      "reason": "Why this step is needed"
    }
  ]
}

Step types:
- "tool": Execute a tool. Requires "tool" and "args" fields.
- "model": Call the LLM for further reasoning. No extra fields needed.
- "terminal": End execution. Include "output" field with final response.

Rules:
- "depends_on" field is optional: array of step IDs like ["step_0", "step_1"]
- "confidence" is 0.0-1.0 (how confident you are in this plan)
- Only use tools that are listed as available
- If the task can be answered without tools, use a single "terminal" step
- Keep JSON properly formatted with escaped quotes in strings
- Minimize the number of steps needed
"""


def _build_planning_message(
    observation: Observation,
    available_tools: dict[str, str],
) -> str:
    """Build the user message for the planning model call.

    Args:
        observation: Current observation with input, tool requests, etc.
        available_tools: Dict mapping tool name to description

    Returns:
        Formatted planning prompt string
    """
    parts: list[str] = []

    parts.append(f"Goal: {observation.goal or observation.input.text}")

    if available_tools:
        tools_desc = "\n".join(
            f"  - {name}: {desc}" for name, desc in available_tools.items()
        )
        parts.append(f"Available tools:\n{tools_desc}")
    else:
        parts.append("Available tools: none")

    if observation.tool_requests:
        pending = ", ".join(tc.name for tc in observation.tool_requests)
        parts.append(f"Pending tool requests from model: {pending}")

    if observation.response:
        parts.append(f"Previous model response: {observation.response}")

    if observation.previous_cycles:
        parts.append(f"Previous cycles completed: {len(observation.previous_cycles)}")

    return "\n\n".join(parts)


def _extract_json(text: str) -> str:
    """Extract JSON from model response, handling markdown code blocks.

    Args:
        text: Raw model response text

    Returns:
        Extracted JSON string

    Raises:
        ValueError: If no JSON object can be found
    """
    # Try extracting from markdown code block first
    code_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()

    # Look for JSON object anywhere in the text
    # Find the first { and try to extract a complete JSON object
    start = text.find("{")
    if start != -1:
        # Try to find matching closing brace
        brace_count = 0
        in_string = False
        escape = False

        for i in range(start, len(text)):
            char = text[i]

            if escape:
                escape = False
                continue

            if char == "\\":
                escape = True
                continue

            if char == '"' and not escape:
                in_string = not in_string
                continue

            if not in_string:
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        # Found complete JSON object
                        return text[start : i + 1]

    raise ValueError(f"No valid JSON object found in response: {text[:200]}")


def _parse_step(raw_step: dict[str, Any], index: int) -> PlanStep:
    """Parse a raw step dict from the model into a PlanStep.

    Args:
        raw_step: Dict from model's JSON response
        index: Zero-based step index for ID generation

    Returns:
        Validated PlanStep

    Raises:
        ValueError: If step type is invalid or required fields are missing
    """
    step_id = f"step_{index}"
    raw_type = raw_step.get("type", "")

    try:
        step_type = StepType(raw_type)
    except ValueError:
        raise ValueError(f"Invalid step type '{raw_type}' at step {index}")

    payload: dict[str, Any] = {}

    if step_type == StepType.TOOL:
        tool_name = raw_step.get("tool")
        if not tool_name:
            raise ValueError(f"TOOL step at index {index} missing 'tool' field")
        payload["tool"] = tool_name
        payload["tool_call_id"] = f"auto_{uuid.uuid4().hex[:8]}"
        payload["args"] = raw_step.get("args", {})

    elif step_type == StepType.TERMINAL:
        payload["output"] = raw_step.get("output", "")

    elif step_type == StepType.MODEL:
        pass  # MODEL steps have no required payload

    elif step_type == StepType.TRANSFORM:
        payload = {k: v for k, v in raw_step.items() if k not in ("type", "reason", "depends_on")}

    # Parse dependencies
    depends_on = raw_step.get("depends_on")
    dependencies: tuple[str, ...] | None = None
    if depends_on and isinstance(depends_on, list):
        dependencies = tuple(str(d) for d in depends_on)

    return PlanStep(
        id=step_id,
        type=step_type,
        payload=payload,
        dependencies=dependencies,
        retry_policy=None,
    )


def _compute_confidence(
    raw_confidence: Any,
    step_count: int,
) -> float:
    """Compute plan confidence score.

    Takes the model's self-reported confidence and adjusts it based on
    plan characteristics.

    Args:
        raw_confidence: Confidence value from model (may be None or out of range)
        step_count: Number of steps in the plan

    Returns:
        Confidence score clamped to [0.0, 1.0]
    """
    if raw_confidence is not None:
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            confidence = _DEFAULT_CONFIDENCE
    else:
        confidence = _DEFAULT_CONFIDENCE

    # Clamp to valid range
    return max(0.0, min(1.0, confidence))


class AutoPlanner:
    """LLM-based execution planner with automatic fallback.

    Uses a model adapter to analyze the current observation and available tools,
    then generates an optimized execution plan. Falls back to SequentialPlanner
    on any failure per AD-016.

    Args:
        model: ModelAdapter instance for planning calls

    Example:
        >>> from axis_core.adapters.models.anthropic import AnthropicModel
        >>> model = AnthropicModel("claude-haiku")
        >>> planner = AutoPlanner(model=model)
        >>> plan = await planner.plan(observation, ctx)
    """

    def __init__(self, model: Any) -> None:
        self._model = model
        self._fallback = SequentialPlanner()

    async def plan(
        self,
        observation: Observation,
        ctx: RunContext,
    ) -> Plan:
        """Generate an execution plan using LLM reasoning.

        Calls the model with a planning prompt describing available tools and
        current context. Parses the structured JSON response into a Plan.
        Falls back to SequentialPlanner on any failure.

        Args:
            observation: Current observation with input and tool requests
            ctx: Run context with execution history and configuration

        Returns:
            Plan with steps to execute. Metadata includes:
            - "planner": "auto" or "sequential" (if fallback)
            - "fallback": True if fallback was used
            - "fallback_reason": Why fallback was triggered
        """
        try:
            return await self._plan_with_model(observation, ctx)
        except Exception as exc:
            logger.warning(
                "AutoPlanner failed, falling back to SequentialPlanner: %s",
                exc,
            )
            return await self._fallback_plan(observation, ctx, reason=str(exc))

    async def _plan_with_model(
        self,
        observation: Observation,
        ctx: RunContext,
    ) -> Plan:
        """Generate plan via model call.

        Raises on any failure so the caller can fall back.
        """
        # Build available tools description from context
        # Tools can be in ctx.context["__tools__"] (set by lifecycle engine)
        # or ctx.config.tools (if config has it)
        available_tools: dict[str, str] = {}

        # Try context dict first (set by lifecycle engine)
        if "__tools__" in ctx.context:
            tools_info = ctx.context["__tools__"]
            if isinstance(tools_info, dict):
                available_tools = {
                    name: (desc if isinstance(desc, str) else str(desc))
                    for name, desc in tools_info.items()
                }
        # Fall back to config if available
        elif hasattr(ctx, "config") and ctx.config and hasattr(ctx.config, "tools"):
            tools_info = ctx.config.tools
            if isinstance(tools_info, dict):
                available_tools = {
                    name: (desc if isinstance(desc, str) else str(desc))
                    for name, desc in tools_info.items()
                }

        # Build planning message
        user_message = _build_planning_message(observation, available_tools)

        # Call model for planning
        messages = [{"role": "user", "content": user_message}]
        response = await self._model.complete(
            messages=messages,
            system=_PLANNING_SYSTEM_PROMPT,
            temperature=0.0,  # Deterministic planning
            max_tokens=2048,
        )

        # Extract and parse JSON
        raw_json = _extract_json(response.content)
        plan_data = json.loads(raw_json)

        # Validate minimum structure
        if not isinstance(plan_data, dict):
            raise ValueError("Model response is not a JSON object")

        raw_steps = plan_data.get("steps")
        if not isinstance(raw_steps, list) or len(raw_steps) == 0:
            raise ValueError("Plan has no steps")

        # Parse steps
        steps: list[PlanStep] = []
        for i, raw_step in enumerate(raw_steps):
            if not isinstance(raw_step, dict):
                raise ValueError(f"Step {i} is not a dict: {raw_step}")
            steps.append(_parse_step(raw_step, i))

        # Extract reasoning and confidence
        reasoning = plan_data.get("reasoning", "LLM-generated plan")
        confidence = _compute_confidence(
            plan_data.get("confidence"),
            len(steps),
        )

        return Plan(
            id=f"auto_{ctx.run_id}_{uuid.uuid4().hex[:8]}",
            goal=observation.goal or "Execute requested operations",
            steps=tuple(steps),
            reasoning=reasoning,
            confidence=confidence,
            metadata={
                "planner": "auto",
                "step_count": len(steps),
            },
        )

    async def _fallback_plan(
        self,
        observation: Observation,
        ctx: RunContext,
        reason: str,
    ) -> Plan:
        """Generate a fallback plan using SequentialPlanner.

        Sets metadata to indicate fallback was used, with reduced confidence.

        Args:
            observation: Current observation
            ctx: Run context
            reason: Why the fallback was triggered

        Returns:
            Plan from SequentialPlanner with fallback metadata
        """
        plan = await self._fallback.plan(observation, ctx)

        # Create a new plan with fallback metadata
        return Plan(
            id=plan.id,
            goal=plan.goal,
            steps=plan.steps,
            reasoning=plan.reasoning,
            confidence=_FALLBACK_CONFIDENCE,
            metadata={
                **plan.metadata,
                "planner": "sequential",
                "fallback": True,
                "fallback_reason": reason,
            },
        )
