"""ReActPlanner example demonstrating explicit reasoning traces.

This example shows how ReActPlanner implements the ReAct (Reason + Act) pattern
from the paper "ReAct: Synergizing Reasoning and Acting in Language Models".

The ReActPlanner makes the agent's reasoning process explicit and observable by:
1. Generating explicit "Thought" steps that explain reasoning
2. Following each thought with an "Action" (tool call or final answer)
3. Observing the results and reasoning about them
4. Repeating the Thought → Action → Observation cycle until task complete

This is ideal for:
- Tasks requiring multi-step reasoning
- Debugging agent decision-making
- Educational/research applications where reasoning visibility matters
- Complex problem-solving where intermediate steps help

Requires ANTHROPIC_API_KEY environment variable to be set.

Usage:
    export ANTHROPIC_API_KEY=your-key-here
    python examples/react_planner_example.py
"""

import asyncio
import os

from axis_core.adapters.models.anthropic import AnthropicModel
from axis_core.adapters.planners.react import ReActPlanner
from axis_core.agent import Agent
from axis_core.budget import Budget
from axis_core.tool import tool


# Define tools for a research/calculation agent
@tool
def search_database(query: str) -> str:
    """Search a knowledge database for information.

    Args:
        query: The search query

    Returns:
        Search results from the database
    """
    # Mock knowledge database
    knowledge = {
        "eiffel tower height": "The Eiffel Tower is 330 meters (1,083 feet) tall.",
        "empire state building": "The Empire State Building is 443 meters (1,454 feet) tall.",
        "burj khalifa": "Burj Khalifa is 828 meters (2,717 feet) tall, the world's tallest.",
        "tokyo tower": "Tokyo Tower is 333 meters (1,093 feet) tall.",
        "cn tower": "CN Tower is 553 meters (1,815 feet) tall.",
        "python release": "Python 3.11 was released in October 2022.",
        "python features": (
            "Python 3.11 includes performance improvements and better error messages."
        ),
        "earth diameter": "Earth's diameter is approximately 12,742 km.",
        "moon diameter": "The Moon's diameter is approximately 3,474 km.",
        "mars diameter": "Mars has a diameter of approximately 6,779 km.",
    }

    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return value

    return f"No information found for: {query}"


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Args:
        expression: Mathematical expression (e.g., "2 + 2", "10 * 5")

    Returns:
        The result of the calculation
    """
    try:
        # Safe eval with limited scope
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {e}"


@tool
def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between units of measurement.

    Args:
        value: The numeric value to convert
        from_unit: Source unit (e.g., "meters", "feet", "km")
        to_unit: Target unit

    Returns:
        Converted value with explanation
    """
    conversions = {
        ("meters", "feet"): 3.28084,
        ("feet", "meters"): 0.3048,
        ("km", "miles"): 0.621371,
        ("miles", "km"): 1.60934,
        ("meters", "km"): 0.001,
        ("km", "meters"): 1000,
    }

    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        result = value * conversions[key]
        return f"{value} {from_unit} = {result:.2f} {to_unit}"

    return f"Conversion from {from_unit} to {to_unit} not supported"


@tool
def compare_values(value1: float, value2: float, metric: str = "difference") -> str:
    """Compare two numeric values.

    Args:
        value1: First value
        value2: Second value
        metric: Comparison metric ("difference", "ratio", "percentage")

    Returns:
        Comparison result
    """
    if metric == "difference":
        diff = abs(value1 - value2)
        return f"Difference: {diff:.2f}"
    elif metric == "ratio":
        if value2 == 0:
            return "Cannot compute ratio (division by zero)"
        ratio = value1 / value2
        return f"Ratio: {ratio:.2f}"
    elif metric == "percentage":
        if value2 == 0:
            return "Cannot compute percentage (division by zero)"
        percentage = ((value1 - value2) / value2) * 100
        return f"Percentage difference: {percentage:.2f}%"

    return f"Unknown metric: {metric}"


async def demonstrate_basic_react():
    """Demonstrate basic ReAct reasoning loop."""

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        print("Please set it before running this example:")
        print("  export ANTHROPIC_API_KEY=your-key-here")
        return

    print("=" * 80)
    print("REACT PLANNER: EXPLICIT REASONING DEMO")
    print("=" * 80)
    print()
    print("ReActPlanner makes the agent's thinking process visible through")
    print("explicit Thought → Action → Observation cycles.")
    print()

    # Create ReActPlanner with a model
    model = AnthropicModel(model_id="claude-sonnet-4-20250514")
    planner = ReActPlanner(model=model, max_iterations=5)

    # Create agent with ReActPlanner
    agent = Agent(
        tools=[search_database, calculate, convert_units, compare_values],
        model="claude-sonnet-4-20250514",
        planner=planner,
        budget=Budget(max_cycles=10, max_cost_usd=1.00),
        system=(
            "You are a research and calculation assistant. Think step-by-step and "
            "explain your reasoning before taking actions. Use the ReAct pattern: "
            "state your thought, then your action, observe results, and continue reasoning."
        ),
    )

    # Example 1: Multi-step reasoning with calculation
    print("Example 1: Multi-Step Reasoning")
    print("-" * 80)
    print("User: How much taller is Burj Khalifa than the Eiffel Tower in feet?")
    print()
    print("Expected reasoning flow:")
    print("  1. Thought: Need to find heights of both buildings")
    print("  2. Action: Search for Burj Khalifa height")
    print("  3. Observation: [Height in meters]")
    print("  4. Thought: Now need Eiffel Tower height")
    print("  5. Action: Search for Eiffel Tower height")
    print("  6. Observation: [Height in meters]")
    print("  7. Thought: Need to convert to feet and calculate difference")
    print("  8. Action: Calculate difference")
    print("  9. Final Answer: [Result]")
    print()

    result = await agent.run_async(
        "How much taller is Burj Khalifa than the Eiffel Tower in feet?"
    )

    print(f"Output: {result.output}")
    print(f"\nCycles: {result.stats.cycles}")
    print(f"Tool calls: {result.stats.tool_calls}")
    print(f"Model calls: {result.stats.model_calls}")
    print(f"Total cost: ${result.stats.cost_usd:.4f}")

    # Show reasoning traces
    if result.state and result.state.cycles:
        print("\n🧠 Reasoning Trace:")
        for i, cycle in enumerate(result.state.cycles, 1):
            if cycle.plan and cycle.plan.reasoning:
                thought = cycle.plan.metadata.get("thought", cycle.plan.reasoning)
                print(f"  Cycle {i} Thought: {thought[:100]}...")

    print("\n" + "=" * 80 + "\n")


async def demonstrate_reasoning_visibility():
    """Demonstrate how ReAct makes reasoning visible."""

    if not os.getenv("ANTHROPIC_API_KEY"):
        return

    print("=" * 80)
    print("REACT PLANNER: REASONING VISIBILITY")
    print("=" * 80)
    print()
    print("This example shows how ReAct captures and exposes reasoning at each step.")
    print()

    model = AnthropicModel(model_id="claude-sonnet-4-20250514")
    planner = ReActPlanner(model=model, max_iterations=8)

    agent = Agent(
        tools=[search_database, calculate, convert_units, compare_values],
        model="claude-sonnet-4-20250514",
        planner=planner,
        budget=Budget(max_cycles=15, max_cost_usd=1.00),
        system=(
            "You are a mathematical reasoning assistant. Always explain your thinking "
            "clearly before performing calculations. Break complex problems into steps."
        ),
    )

    print("Example: Complex Multi-Step Problem")
    print("-" * 80)
    print("User: Compare the diameters of Earth and Mars. What percentage smaller is Mars?")
    print()

    result = await agent.run_async(
        "Compare the diameters of Earth and Mars. What percentage smaller is Mars?"
    )

    print(f"Output: {result.output}")
    print("\nStatistics:")
    print(f"  Cycles: {result.stats.cycles}")
    print(f"  Tool calls: {result.stats.tool_calls}")
    print(f"  Model calls: {result.stats.model_calls}")
    print(f"  Total cost: ${result.stats.cost_usd:.4f}")

    # Detailed reasoning trace
    if result.state and result.state.cycles:
        print("\n🧠 Complete Reasoning Trace:")
        for i, cycle in enumerate(result.state.cycles, 1):
            print(f"\n  Cycle {i}:")
            if cycle.plan:
                thought = cycle.plan.metadata.get("thought", "")
                if thought:
                    print(f"    💭 Thought: {thought}")
                print(f"    📋 Plan steps: {len(cycle.plan.steps)}")
                print(f"    ✓ Confidence: {cycle.plan.confidence:.2f}")

            if cycle.execution:
                tool_count = len(cycle.execution.results)
                if tool_count > 0:
                    print(f"    🔧 Tools executed: {tool_count}")

    print("\n" + "=" * 80 + "\n")


async def demonstrate_max_iterations():
    """Demonstrate max_iterations limit preventing infinite loops."""

    if not os.getenv("ANTHROPIC_API_KEY"):
        return

    print("=" * 80)
    print("REACT PLANNER: MAX ITERATIONS LIMIT")
    print("=" * 80)
    print()
    print("ReActPlanner has a max_iterations parameter to prevent infinite reasoning loops.")
    print("This is useful for controlling cost and ensuring termination.")
    print()

    model = AnthropicModel(model_id="claude-sonnet-4-20250514")

    # Create planner with low max_iterations
    planner = ReActPlanner(model=model, max_iterations=3)

    agent = Agent(
        tools=[search_database, calculate, convert_units],
        model="claude-sonnet-4-20250514",
        planner=planner,
        budget=Budget(max_cycles=10, max_cost_usd=0.50),
        system="You are a helpful assistant.",
    )

    print("Example: Agent with max_iterations=3")
    print("-" * 80)
    print("User: Search for Python, then calculate something, then convert units")
    print("(This would normally take 4+ cycles)")
    print()

    result = await agent.run_async(
        "Tell me about Python 3.11, calculate 100 * 50, and convert 500 meters to feet"
    )

    print(f"Output: {result.output}")
    print(f"\nCycles: {result.stats.cycles} (limited by max_iterations=3)")
    print(f"Tool calls: {result.stats.tool_calls}")
    print(f"Total cost: ${result.stats.cost_usd:.4f}")

    if result.state and result.state.cycles:
        last_cycle = result.state.cycles[-1]
        if last_cycle.plan.metadata.get("iterations_exceeded"):
            print("\n⚠️  Max iterations reached!")
            print(f"   Reason: {last_cycle.plan.metadata.get('termination_reason')}")
            print("   Agent was forced to terminate and provide best available output.")

    print("\n" + "=" * 80 + "\n")


async def demonstrate_comparison():
    """Compare ReActPlanner with SequentialPlanner."""

    if not os.getenv("ANTHROPIC_API_KEY"):
        return

    print("=" * 80)
    print("REACT VS SEQUENTIAL: SIDE-BY-SIDE COMPARISON")
    print("=" * 80)
    print()
    print("Comparing ReActPlanner (explicit reasoning) vs SequentialPlanner (direct execution)")
    print()

    from axis_core.adapters.planners.sequential import SequentialPlanner

    query = "Search for the Eiffel Tower height and convert it to feet"

    # Test with SequentialPlanner
    print("=" * 40)
    print("SEQUENTIAL PLANNER")
    print("=" * 40)

    sequential_agent = Agent(
        tools=[search_database, convert_units],
        model="claude-sonnet-4-20250514",
        planner=SequentialPlanner(),
        budget=Budget(max_cycles=5, max_cost_usd=0.25),
        system="You are a helpful assistant.",
    )

    seq_result = await sequential_agent.run_async(query)
    print(f"Output: {seq_result.output}")
    print(f"Cycles: {seq_result.stats.cycles}")
    print(f"Tool calls: {seq_result.stats.tool_calls}")
    print(f"Cost: ${seq_result.stats.cost_usd:.4f}")

    print("\n" + "=" * 40)
    print("REACT PLANNER")
    print("=" * 40)

    # Test with ReActPlanner
    model = AnthropicModel(model_id="claude-sonnet-4-20250514")
    react_agent = Agent(
        tools=[search_database, convert_units],
        model="claude-sonnet-4-20250514",
        planner=ReActPlanner(model=model, max_iterations=5),
        budget=Budget(max_cycles=5, max_cost_usd=0.25),
        system="You are a helpful assistant. Think through problems step by step.",
    )

    react_result = await react_agent.run_async(query)
    print(f"Output: {react_result.output}")
    print(f"Cycles: {react_result.stats.cycles}")
    print(f"Tool calls: {react_result.stats.tool_calls}")
    print(f"Cost: ${react_result.stats.cost_usd:.4f}")

    # Show reasoning trace for ReAct
    if react_result.state and react_result.state.cycles:
        print("\n🧠 ReAct Reasoning:")
        for i, cycle in enumerate(react_result.state.cycles, 1):
            if cycle.plan:
                thought = cycle.plan.metadata.get("thought", "")
                if thought:
                    print(f"  Cycle {i}: {thought[:80]}...")

    print("\n" + "=" * 80)
    print("\nKey Differences:")
    print("  Sequential: Direct execution, minimal reasoning overhead")
    print("  ReAct: Explicit reasoning, better for complex multi-step problems")
    print("  ReAct: More model calls (thinking + execution) but clearer decision path")
    print()


if __name__ == "__main__":
    # Run all demonstrations
    asyncio.run(demonstrate_basic_react())
    asyncio.run(demonstrate_reasoning_visibility())
    asyncio.run(demonstrate_max_iterations())
    asyncio.run(demonstrate_comparison())

    print("=" * 80)
    print("SUMMARY: When to use ReActPlanner")
    print("=" * 80)
    print()
    print("✓ Use ReActPlanner when:")
    print("  - You need visible reasoning traces for debugging or auditing")
    print("  - Task requires complex multi-step reasoning")
    print("  - Educational/research applications")
    print("  - Building agents that need to explain their thinking")
    print()
    print("✓ Use SequentialPlanner when:")
    print("  - You want minimal overhead and direct execution")
    print("  - Tasks are straightforward tool sequences")
    print("  - Cost optimization is critical")
    print()
    print("✓ Use AutoPlanner when:")
    print("  - You want intelligent tool selection and dependency management")
    print("  - Tasks involve parallel tool execution")
    print("  - You need optimal execution plans")
    print()
