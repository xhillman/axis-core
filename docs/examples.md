# Examples Index

This page points to runnable examples in `/examples`.

## Core Examples

- `examples/simple_tool_agent.py`
  - Shows a tool-using agent with model + planner strings
  - Demonstrates run stats and multiple prompts

- `examples/autoplanner_example.py`
  - Shows `AutoPlanner` with a dedicated planning model
  - Demonstrates plan confidence and fallback behavior

- `examples/react_planner_example.py`
  - Shows `ReActPlanner` with explicit thought/action cycles
  - Demonstrates reasoning visibility patterns

- `examples/synaptic_session_first.py`
  - Shows canonical Synaptic 0.3 session-first API (`Synaptic`)
  - Demonstrates axis-core Synaptic adapter interop (`store/retrieve/search` + sessions)

## How to Run

From repository root:

```bash
# Example: simple tool agent
export ANTHROPIC_API_KEY=your-key
python examples/simple_tool_agent.py

# Example: autoplanner
python examples/autoplanner_example.py

# Example: ReAct planner
python examples/react_planner_example.py

# Example: Synaptic session-first interop
python examples/synaptic_session_first.py
```

## Validation Policy for Docs

When adding doc snippets:

- Prefer linking to a runnable example file instead of duplicating long snippets.
- If duplicating a snippet, keep it synchronized with tests or example scripts.
