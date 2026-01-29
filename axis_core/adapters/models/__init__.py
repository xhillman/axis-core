"""Model adapters for LLM providers.

This module provides built-in model adapter implementations with conditional imports
for optional dependencies (per AD-040).

Available adapters (when dependencies are installed):
- AnthropicModel: Claude models via Anthropic API (requires: pip install axis-core[anthropic])
- OpenAIModel: GPT models via OpenAI API (requires: pip install axis-core[openai])
- OllamaModel: Local models via Ollama (requires: pip install axis-core[ollama])
"""

__all__ = []

# Conditional import for Anthropic adapter
try:
    from axis_core.adapters.models.anthropic import AnthropicModel, MODEL_PRICING

    __all__.extend(["AnthropicModel", "MODEL_PRICING"])
except ImportError:
    pass

# Future adapters will follow the same pattern:
# try:
#     from axis_core.adapters.models.openai import OpenAIModel
#     __all__.append('OpenAIModel')
# except ImportError:
#     pass
