"""AutoPlanner example demonstrating LLM-based intelligent planning.

This example shows how AutoPlanner analyzes available tools and creates optimized
execution plans with proper dependencies and reasoning. It falls back gracefully
to SequentialPlanner on any planning failures (AD-016).

The AutoPlanner uses a separate model call during the Plan phase to:
1. Analyze the user's request and available tools
2. Determine the optimal sequence of tool calls
3. Set up dependencies between steps when needed
4. Provide reasoning for its planning decisions

Requires ANTHROPIC_API_KEY environment variable to be set.

Usage:
    export ANTHROPIC_API_KEY=your-key-here
    python examples/autoplanner_example.py
"""

import asyncio
import os

from axis_core.adapters.models.anthropic import AnthropicModel
from axis_core.adapters.planners.auto import AutoPlanner
from axis_core.agent import Agent
from axis_core.budget import Budget
from axis_core.tool import tool


# Define tools for a research assistant
@tool
def search_web(query: str) -> str:
    """Search the web for information.

    Args:
        query: The search query

    Returns:
        Search results (simulated)
    """
    # Mock search results
    results = {
        "python type hints": "Type hints in Python use annotations like str, int, list[str]...",
        "async programming": "Async/await in Python enables concurrent I/O operations...",
        "pydantic models": (
            "Pydantic provides runtime validation using Python type annotations..."
        ),
        "json schema": (
            "JSON Schema is a vocabulary for annotating and validating JSON documents..."
        ),
    }

    for key in results:
        if key.lower() in query.lower():
            return results[key]

    return f"No results found for: {query}"


@tool
def fetch_documentation(library: str, topic: str) -> str:
    """Fetch documentation for a specific library and topic.

    Args:
        library: The library name (e.g., "pydantic", "fastapi")
        topic: The specific topic to look up

    Returns:
        Documentation content (simulated)
    """
    docs = {
        ("pydantic", "validation"): "Pydantic models validate data at runtime using type hints...",
        ("pydantic", "models"): "BaseModel is the base class for all Pydantic models...",
        ("fastapi", "routes"): "FastAPI routes are defined using @app.get() decorators...",
        ("python", "asyncio"): "asyncio provides infrastructure for writing async code...",
    }

    key = (library.lower(), topic.lower())
    if key in docs:
        return docs[key]

    return f"No documentation found for {library}/{topic}"


@tool
def summarize_text(text: str, max_sentences: int = 2) -> str:
    """Summarize a text to a specified number of sentences.

    Args:
        text: The text to summarize
        max_sentences: Maximum sentences in summary (default: 2)

    Returns:
        Summarized text
    """
    # Simple mock summarization (just truncate)
    sentences = text.split(".")
    summary = ". ".join(sentences[:max_sentences]).strip()
    if summary and not summary.endswith("."):
        summary += "."
    return summary or text


@tool
def compare_concepts(concept1: str, concept2: str) -> str:
    """Compare two programming concepts and highlight differences.

    Args:
        concept1: First concept to compare
        concept2: Second concept to compare

    Returns:
        Comparison summary
    """
    comparisons = {
        ("async", "sync"): "Async enables concurrent operations while sync blocks execution.",
        ("type hints", "runtime validation"): (
            "Type hints are static checks; runtime validation happens during execution."
        ),
        ("pydantic", "dataclasses"): (
            "Pydantic adds runtime validation; dataclasses are simpler."
        ),
    }

    # Normalize and check
    key1 = (concept1.lower(), concept2.lower())
    key2 = (concept2.lower(), concept1.lower())

    if key1 in comparisons:
        return comparisons[key1]
    if key2 in comparisons:
        return comparisons[key2]

    return f"No comparison available for {concept1} vs {concept2}"


async def demonstrate_autoplanner():
    """Demonstrate AutoPlanner with increasingly complex queries."""

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        print("Please set it before running this example:")
        print("  export ANTHROPIC_API_KEY=your-key-here")
        return

    print("=" * 80)
    print("AUTOPLANNER INTELLIGENT PLANNING DEMO")
    print("=" * 80)
    print()
    print("AutoPlanner uses an LLM during the Plan phase to analyze your request")
    print("and create an optimized execution plan with proper tool sequencing.")
    print()

    # Create a planning model (can be different from execution model)
    # Use a fast, cheap model for planning
    planning_model = AnthropicModel(model_id="claude-haiku-4-5-20251001")

    # Create AutoPlanner with the planning model
    planner = AutoPlanner(model=planning_model)

    # Create agent with AutoPlanner
    # The execution model can be different (e.g., more capable)
    agent = Agent(
        tools=[search_web, fetch_documentation, summarize_text, compare_concepts],
        model="claude-sonnet-4-20250514",  # Execution model (more capable)
        planner=planner,  # AutoPlanner instance
        budget=Budget(max_cycles=10, max_cost_usd=0.50),
        system=(
            "You are a technical research assistant. You have access to tools for "
            "searching, fetching documentation, summarizing, and comparing concepts. "
            "Use them strategically to provide comprehensive answers."
        ),
    )

    # Example 1: Simple single-tool query
    print("Example 1: Simple Query (Single Tool)")
    print("-" * 80)
    print("User: What are Python type hints?")
    print()

    result = await agent.run_async("What are Python type hints?")

    print(f"Output: {result.output}")
    print(f"Cycles: {result.stats.cycles}")
    print(f"Tool calls: {result.stats.tool_calls}")
    print(f"Model calls: {result.stats.model_calls}")
    print(f"Total cost: ${result.stats.cost_usd:.4f}")

    # Check if planner fell back
    if result.state and result.state.cycles:
        last_cycle = result.state.cycles[-1]
        if last_cycle.plan.metadata.get("fallback"):
            print(f"⚠️  Planner fell back: {last_cycle.plan.metadata.get('fallback_reason')}")
        else:
            print(f"✓ Plan reasoning: {last_cycle.plan.reasoning}")
            if last_cycle.plan.confidence:
                print(f"✓ Plan confidence: {last_cycle.plan.confidence:.2f}")

    print()

    # Example 2: Multi-step query requiring sequencing
    print("Example 2: Complex Query (Multiple Tools with Dependencies)")
    print("-" * 80)
    print("User: Search for Pydantic info, get its validation docs, and summarize them")
    print()

    result = await agent.run_async(
        "Search for information about Pydantic, then fetch the validation "
        "documentation, and finally summarize it for me."
    )

    print(f"Output: {result.output}")
    print(f"Cycles: {result.stats.cycles}")
    print(f"Tool calls: {result.stats.tool_calls}")
    print(f"Total cost: ${result.stats.cost_usd:.4f}")

    if result.state and result.state.cycles:
        last_cycle = result.state.cycles[-1]
        if last_cycle.plan.metadata.get("fallback"):
            print(f"⚠️  Planner fell back: {last_cycle.plan.metadata.get('fallback_reason')}")
        else:
            print(f"✓ AutoPlanner plan: {last_cycle.plan.reasoning}")
            if last_cycle.plan.confidence:
                print(f"✓ Confidence: {last_cycle.plan.confidence:.2f}")

    print()

    # Example 3: Parallel tool usage
    print("Example 3: Parallel Query (Independent Tools)")
    print("-" * 80)
    print("User: Compare async vs sync and also search for async programming info")
    print()

    result = await agent.run_async(
        "Compare async and sync programming, and also search for information "
        "about async programming in Python."
    )

    print(f"Output: {result.output}")
    print(f"Cycles: {result.stats.cycles}")
    print(f"Tool calls: {result.stats.tool_calls}")
    print(f"Total cost: ${result.stats.cost_usd:.4f}")

    if result.state and result.state.cycles:
        last_cycle = result.state.cycles[-1]
        if last_cycle.plan.metadata.get("fallback"):
            print(f"⚠️  Planner fell back: {last_cycle.plan.metadata.get('fallback_reason')}")
        else:
            print(f"✓ AutoPlanner plan: {last_cycle.plan.reasoning}")
            if last_cycle.plan.confidence:
                print(f"✓ Confidence: {last_cycle.plan.confidence:.2f}")

    print()

    # Example 4: Direct answer (no tools needed)
    print("Example 4: Direct Answer (No Tools)")
    print("-" * 80)
    print("User: What is 2 + 2?")
    print()

    result = await agent.run_async("What is 2 + 2?")

    print(f"Output: {result.output}")
    print(f"Cycles: {result.stats.cycles}")
    print(f"Tool calls: {result.stats.tool_calls}")
    print(f"Total cost: ${result.stats.cost_usd:.4f}")

    if result.state and result.state.cycles:
        last_cycle = result.state.cycles[-1]
        print(f"✓ Plan used {len(last_cycle.plan.steps)} step(s)")

    print()
    print("=" * 80)
    print("Demo complete!")
    print()
    print("Key AutoPlanner features demonstrated:")
    print("  1. Intelligent tool selection based on user query")
    print("  2. Multi-step planning with proper sequencing")
    print("  3. Recognition when no tools are needed (direct answer)")
    print("  4. Automatic fallback to SequentialPlanner on failures (AD-016)")
    print("  5. Confidence scoring and reasoning explanation")
    print()


async def demonstrate_fallback():
    """Demonstrate AutoPlanner fallback behavior."""

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Skipping fallback demo - ANTHROPIC_API_KEY not set")
        return

    print("=" * 80)
    print("AUTOPLANNER FALLBACK DEMO")
    print("=" * 80)
    print()
    print("This demonstrates AD-016: AutoPlanner gracefully falls back to")
    print("SequentialPlanner when LLM-based planning fails.")
    print()

    # Create a model that will fail (invalid API key simulation)
    # In a real scenario, this could be network errors, malformed responses, etc.
    class FailingModel:
        """Mock model that always fails to simulate planning errors."""

        model_id = "failing-model"

        async def complete(self, **kwargs):
            raise Exception("Simulated model failure")

        def estimate_tokens(self, text: str) -> int:
            return len(text.split())

        def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
            return 0.0

    failing_planner = AutoPlanner(model=FailingModel())

    # Use a working model for execution
    agent = Agent(
        tools=[search_web, summarize_text],
        model="claude-sonnet-4-20250514",
        planner=failing_planner,
        budget=Budget(max_cycles=5, max_cost_usd=0.10),
        system="You are a helpful assistant.",
    )

    print("Sending a query with a planner that will fail...")
    print()

    result = await agent.run_async("Search for Python and summarize the results")

    print(f"Output: {result.output}")
    print(f"Success: {result.success}")
    print()

    # Check the plan
    if result.state and result.state.cycles:
        last_cycle = result.state.cycles[-1]
        if last_cycle.plan.metadata.get("fallback"):
            print("✓ Fallback worked!")
            print(f"  Reason: {last_cycle.plan.metadata.get('fallback_reason')}")
            print(f"  Fallback planner: {last_cycle.plan.metadata.get('planner')}")
            print(f"  Reduced confidence: {last_cycle.plan.confidence}")
        else:
            print("❌ Fallback did not trigger (unexpected)")

    print()
    print("=" * 80)
    print()


if __name__ == "__main__":
    asyncio.run(demonstrate_autoplanner())
    print()
    asyncio.run(demonstrate_fallback())
