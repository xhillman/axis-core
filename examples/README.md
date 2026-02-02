# Axis-Core Examples

This directory contains practical examples demonstrating key features of the axis-core framework.

## Prerequisites

All examples require the `ANTHROPIC_API_KEY` environment variable to be set:

```bash
export ANTHROPIC_API_KEY=your-key-here
```

## Examples

### 1. Simple Tool Agent (`simple_tool_agent.py`)

Demonstrates basic agent functionality with tool usage and the SequentialPlanner.

**Features shown:**
- Defining tools with the `@tool` decorator
- Creating an agent with multiple tools
- Using the SequentialPlanner for deterministic execution
- Basic budget tracking and statistics

**Run it:**
```bash
python examples/simple_tool_agent.py
```

**What it does:**
- Weather queries using mock weather data
- Mathematical calculations with the `calculate` tool
- Documentation search with keyword matching
- Multi-tool queries combining different capabilities

### 2. AutoPlanner Example (`autoplanner_example.py`)

Demonstrates intelligent LLM-based planning with the AutoPlanner.

**Features shown:**
- Using AutoPlanner for intelligent tool selection and sequencing
- Separate planning model vs execution model
- Multi-step planning with dependencies
- Automatic fallback to SequentialPlanner on failures (AD-016)
- Confidence scoring and plan reasoning
- Graceful error handling

**Run it:**
```bash
python examples/autoplanner_example.py
```

**What it does:**

**Part 1: Intelligent Planning Demo**
- Simple single-tool queries
- Complex multi-step queries requiring tool sequencing
- Parallel tool usage for independent operations
- Direct answers when no tools are needed

**Part 2: Fallback Behavior Demo**
- Simulates planning failures to demonstrate AD-016 fallback
- Shows how the system gracefully degrades to SequentialPlanner
- Demonstrates continued operation despite planning errors

**Key differences from simple_tool_agent.py:**
- AutoPlanner uses an LLM during the Plan phase to analyze and optimize execution
- Can create plans with step dependencies
- Provides reasoning and confidence scores
- Falls back automatically on planning failures
- Planning model can be different (cheaper/faster) than execution model

## Architecture Concepts

### Planners

**SequentialPlanner:**
- Deterministic execution in request order
- No planning overhead
- Always succeeds
- Confidence: 1.0 (perfect)
- Best for: Simple, predictable tasks

**AutoPlanner:**
- LLM-based intelligent planning
- Analyzes tools and creates optimized plans
- Can set up step dependencies
- Provides reasoning and confidence scores
- Falls back to SequentialPlanner on failure (AD-016)
- Best for: Complex tasks requiring tool orchestration

### Agent Lifecycle

All agents follow the same lifecycle:
1. **Initialize** - Set up context and validate config
2. **Observe** - Gather input and memory context
3. **Plan** - Generate execution plan (varies by planner)
4. **Act** - Execute plan steps (tools and model calls)
5. **Evaluate** - Check termination conditions
6. **Finalize** - Persist memory and return results

The cycle repeats until a terminal condition is met (terminal plan, budget exhausted, etc.).

## Writing Your Own Examples

When creating new examples:

1. Import adapters to trigger registration:
   ```python
   import axis_core.adapters.models  # noqa: F401
   import axis_core.adapters.planners  # noqa: F401
   ```

2. Check for API keys before proceeding:
   ```python
   if not os.getenv("ANTHROPIC_API_KEY"):
       print("ERROR: ANTHROPIC_API_KEY not set")
       return
   ```

3. Use descriptive tool names and docstrings - they help the planner
4. Set appropriate budgets to prevent runaway costs
5. Print statistics to show agent efficiency

## Need Help?

- Check the main [README.md](../README.md) for full documentation
- See [SPEC.md](../dev/SPEC.md) for architecture decisions
- Review the [task list](../dev/tasks-axis-core-prd.md) for implementation status
