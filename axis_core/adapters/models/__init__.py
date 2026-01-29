"""Model adapters for LLM providers.

This module provides built-in model adapter implementations with conditional imports
for optional dependencies (per AD-040).

Available adapters (when dependencies are installed):
- AnthropicModel: Claude models via Anthropic API (requires: pip install axis-core[anthropic])
- OpenAIModel: GPT models via OpenAI API (requires: pip install axis-core[openai])
- OllamaModel: Local models via Ollama (requires: pip install axis-core[ollama])
"""

from typing import Any

__all__ = []

# Conditional import for Anthropic adapter and register built-in models
try:
    from axis_core.adapters.models.anthropic import AnthropicModel, MODEL_PRICING
    from axis_core.engine.registry import model_registry

    __all__.extend(["AnthropicModel", "MODEL_PRICING"])

    def _make_anthropic_factory(model_id: str) -> type[Any]:
        """Create a factory class for a specific Anthropic model.

        This allows registry resolution to work with model_id pre-set,
        while still accepting other constructor kwargs.
        """

        class ModelFactory(AnthropicModel):
            """Factory for specific model ID."""

            def __init__(self, **kwargs: Any) -> None:
                # Override model_id if provided, otherwise use default
                kwargs.setdefault("model_id", model_id)
                super().__init__(**kwargs)

        return ModelFactory

    # Register built-in Anthropic models (Task 16.1)
    # Each registration creates a factory that pre-fills the model_id
    model_registry.register("claude-3-haiku-20240307", _make_anthropic_factory("claude-3-haiku-20240307"))
    model_registry.register("claude-sonnet-4-20250514", _make_anthropic_factory("claude-sonnet-4-20250514"))
    model_registry.register("claude-opus-4-20250514", _make_anthropic_factory("claude-opus-4-20250514"))
    model_registry.register("claude-opus-4-1-20250805", _make_anthropic_factory("claude-opus-4-1-20250805"))
    model_registry.register("claude-sonnet-4-5-20250929", _make_anthropic_factory("claude-sonnet-4-5-20250929"))
    model_registry.register("claude-haiku-4-5-20251001", _make_anthropic_factory("claude-haiku-4-5-20251001"))
    model_registry.register("claude-opus-4-5-20251101", _make_anthropic_factory("claude-opus-4-5-20251101"))

    # Register convenience aliases for latest versions
    model_registry.register("claude-haiku", _make_anthropic_factory("claude-haiku-4-5-20251001"))
    model_registry.register("claude-sonnet", _make_anthropic_factory("claude-sonnet-4-5-20250929"))
    model_registry.register("claude-opus", _make_anthropic_factory("claude-opus-4-5-20251101"))
except ImportError:
    pass

# Future adapters will follow the same pattern:
# try:
#     from axis_core.adapters.models.openai import OpenAIModel
#     from axis_core.engine.registry import model_registry
#
#     __all__.append('OpenAIModel')
#     model_registry.register("gpt-4", _make_openai_factory("gpt-4"))
#     model_registry.register("gpt-4o", _make_openai_factory("gpt-4o"))
#     model_registry.register("gpt-3.5", _make_openai_factory("gpt-3.5-turbo"))
# except ImportError:
#     pass
