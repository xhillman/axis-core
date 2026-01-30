"""Simple example demonstrating tool-using agent.

This example shows the complete flow of an agent that can use tools.
Requires ANTHROPIC_API_KEY environment variable to be set.

Usage:
    export ANTHROPIC_API_KEY=your-key-here
    python examples/simple_tool_agent.py
"""

import asyncio
import os

# Import adapters to trigger registration for string-based resolution
import axis_core.adapters.models  # noqa: F401 - registers model adapters
import axis_core.adapters.planners  # noqa: F401 - registers planner adapters
from axis_core.agent import Agent
from axis_core.tool import tool


# Define some simple tools
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The name of the city

    Returns:
        Weather description
    """
    # In a real implementation, this would call a weather API
    weather_data = {
        "San Francisco": "sunny and 72°F",
        "New York": "cloudy and 55°F",
        "London": "rainy and 50°F",
        "Tokyo": "clear and 65°F",
    }
    return weather_data.get(city, f"Weather data not available for {city}")


@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression.

    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 2")

    Returns:
        The result of the calculation
    """
    try:
        # WARNING: eval() is dangerous in production! Use a proper math parser.
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error calculating {expression}: {e}"


@tool
def search_docs(query: str, max_results: int = 3) -> str:
    """Search documentation for a query.

    Args:
        query: The search query
        max_results: Maximum number of results to return (default: 3)

    Returns:
        Search results
    """
    # Mock search results
    docs = [
        "axis-core is an AI agent framework",
        "Tools are decorated with @tool",
        "Agents use the observe-plan-act-evaluate loop",
        "Sequential planner executes steps in order",
        "Budget tracking prevents runaway costs",
    ]

    results = [doc for doc in docs if query.lower() in doc.lower()][:max_results]

    if results:
        return "\n".join(f"{i+1}. {result}" for i, result in enumerate(results))
    return f"No results found for: {query}"


async def main():
    """Run the tool-using agent example."""

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        print("Please set it before running this example:")
        print("  export ANTHROPIC_API_KEY=your-key-here")
        return

    # Create agent with tools
    agent = Agent(
        tools=[get_weather, calculate, search_docs],
        model="claude-haiku",
        planner="sequential",
        system="You are a helpful assistant with access to tools. Use them when appropriate.",
    )

    print("=" * 80)
    print("AXIS-CORE TOOL-USING AGENT DEMO")
    print("=" * 80)
    print()

    # Example 1: Weather query
    print("Example 1: Weather Query")
    print("-" * 80)
    result = await agent.run_async("What's the weather like in San Francisco?")
    print(f"Output: {result.output}")
    print(f"Success: {result.success}")
    print(f"Cycles: {result.stats.cycles}")
    print(f"Tool calls: {result.stats.tool_calls}")
    print(f"Model calls: {result.stats.model_calls}")
    print(f"Cost: ${result.stats.cost_usd:.4f}")
    print()

    # Example 2: Calculation
    print("Example 2: Calculation")
    print("-" * 80)
    result = await agent.run_async("What is 12 * 34 + 56?")
    print(f"Output: {result.output}")
    print(f"Tool calls: {result.stats.tool_calls}")
    print(f"Cost: ${result.stats.cost_usd:.4f}")
    print()

    # Example 3: Documentation search
    print("Example 3: Documentation Search")
    print("-" * 80)
    result = await agent.run_async("Tell me about tools in axis-core")
    print(f"Output: {result.output}")
    print(f"Tool calls: {result.stats.tool_calls}")
    print(f"Cost: ${result.stats.cost_usd:.4f}")
    print()

    # Example 4: Multi-tool usage
    print("Example 4: Multi-Tool Query")
    print("-" * 80)
    result = await agent.run_async(
        "What's the weather in Tokyo and New York? Also, what's 15 * 20?"
    )
    print(f"Output: {result.output}")
    print(f"Tool calls: {result.stats.tool_calls}")
    print(f"Cost: ${result.stats.cost_usd:.4f}")
    print()

    print("=" * 80)
    print("Demo complete!")


if __name__ == "__main__":
    asyncio.run(main())
