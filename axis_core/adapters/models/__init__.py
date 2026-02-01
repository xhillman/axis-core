"""Model adapters for LLM providers.

This module provides built-in model adapter implementations with lazy loading
for optional dependencies (per AD-040).

Available adapters (when dependencies are installed):
- AnthropicModel: Claude models via Anthropic API (requires: pip install axis-core[anthropic])
- OpenAIModel: GPT models via OpenAI API (requires: pip install axis-core[openai])
- OllamaModel: Local models via Ollama (requires: pip install axis-core[ollama])
"""

from typing import Any

from axis_core.engine.registry import model_registry
from axis_core.errors import ConfigError

__all__: list[str] = []


# ===========================================================================
# Lazy factory for Anthropic models
# ===========================================================================


def _make_lazy_anthropic_factory(model_id: str) -> type[Any]:
    """Create a lazy-loading factory for a specific Anthropic model.

    The import is deferred until the factory is instantiated, avoiding
    upfront dependency on the anthropic package.

    Args:
        model_id: The Anthropic model identifier (e.g., "claude-sonnet-4-20250514")

    Returns:
        A factory class that lazy-loads AnthropicModel on instantiation
    """

    class LazyAnthropicFactory:
        """Lazy factory for Anthropic model that defers import."""

        def __init__(self, **kwargs: Any) -> None:
            # Lazy import - only happens when someone actually uses this model
            try:
                from axis_core.adapters.models.anthropic import AnthropicModel
            except ImportError as e:
                raise ConfigError(
                    f"Model '{model_id}' requires the anthropic package. "
                    f"Install with: pip install 'axis-core[anthropic]'"
                ) from e

            # Set model_id default, allow override
            kwargs.setdefault("model_id", model_id)

            # Create the actual AnthropicModel instance and copy its attributes
            # This makes the factory transparent - it behaves like AnthropicModel
            instance = AnthropicModel(**kwargs)
            self.__dict__.update(instance.__dict__)
            self.__class__ = instance.__class__

    return LazyAnthropicFactory


# Register built-in Anthropic models (Task 16.1)
# These are registered unconditionally - import happens lazily on use
_anthropic_models = [
    "claude-3-haiku-20240307",
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    "claude-opus-4-1-20250805",
    "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5-20251001",
    "claude-opus-4-5-20251101",
]

for model_id in _anthropic_models:
    model_registry.register(model_id, _make_lazy_anthropic_factory(model_id))

# Register convenience aliases for latest versions
model_registry.register("claude-haiku", _make_lazy_anthropic_factory("claude-haiku-4-5-20251001"))
model_registry.register("claude-sonnet", _make_lazy_anthropic_factory("claude-sonnet-4-5-20250929"))
model_registry.register("claude-opus", _make_lazy_anthropic_factory("claude-opus-4-5-20251101"))


# ===========================================================================
# Eager export of AnthropicModel class (for direct use)
# ===========================================================================

# Try to export the actual class for users who want to import it directly
# This is optional - if anthropic isn't installed, just skip the export
try:
    from axis_core.adapters.models.anthropic import (
        MODEL_PRICING,  # noqa: F401 - re-exported
        AnthropicModel,
    )

    __all__.extend(["AnthropicModel", "MODEL_PRICING"])
except ImportError:
    # anthropic package not installed - that's fine, lazy registration still works
    pass


# ===========================================================================
# Future adapters will follow the same pattern
# ===========================================================================

# Example for OpenAI (when implemented):
# def _make_lazy_openai_factory(model_id: str) -> type[Any]:
#     class LazyOpenAIFactory:
#         def __init__(self, **kwargs: Any) -> None:
#             try:
#                 from axis_core.adapters.models.openai import OpenAIModel
#             except ImportError as e:
#                 raise ConfigError(
#                     f"Model '{model_id}' requires the openai package. "
#                     f"Install with: pip install 'axis-core[openai]'"
#                 ) from e
#             kwargs.setdefault("model_id", model_id)
#             instance = OpenAIModel(**kwargs)
#             self.__dict__.update(instance.__dict__)
#             self.__class__ = instance.__class__
#     return LazyOpenAIFactory
#
# model_registry.register("gpt-4", _make_lazy_openai_factory("gpt-4"))
# model_registry.register("gpt-4o", _make_lazy_openai_factory("gpt-4o"))
