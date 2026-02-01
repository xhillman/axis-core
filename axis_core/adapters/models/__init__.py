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
            self.__class__ = instance.__class__  # type: ignore[assignment]

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
        AnthropicModel,  # noqa: F401 - re-exported
    )

    __all__.extend(["AnthropicModel", "MODEL_PRICING"])
except ImportError:
    # anthropic package not installed - that's fine, lazy registration still works
    pass


# ===========================================================================
# Lazy factory for OpenAI models
# ===========================================================================


def _make_lazy_openai_factory(model_id: str) -> type[Any]:
    """Create a lazy-loading factory for a specific OpenAI model.

    The import is deferred until the factory is instantiated, avoiding
    upfront dependency on the openai package.

    Args:
        model_id: The OpenAI model identifier (e.g., "gpt-4", "gpt-4o")

    Returns:
        A factory class that lazy-loads OpenAIModel on instantiation
    """

    class LazyOpenAIFactory:
        """Lazy factory for OpenAI model that defers import."""

        def __init__(self, **kwargs: Any) -> None:
            # Lazy import - only happens when someone actually uses this model
            try:
                from axis_core.adapters.models.openai import OpenAIModel
            except ImportError as e:
                raise ConfigError(
                    f"Model '{model_id}' requires the openai package. "
                    f"Install with: pip install 'axis-core[openai]'"
                ) from e

            # Set model_id default, allow override
            kwargs.setdefault("model_id", model_id)

            # Create the actual OpenAIModel instance and copy its attributes
            # This makes the factory transparent - it behaves like OpenAIModel
            instance = OpenAIModel(**kwargs)
            self.__dict__.update(instance.__dict__)
            self.__class__ = instance.__class__  # type: ignore[assignment]

    return LazyOpenAIFactory


# Register built-in OpenAI models (Task 16.1)
# These are registered unconditionally - import happens lazily on use
_openai_models = [
    # GPT-5 series
    "gpt-5.2",
    "gpt-5.1",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    # GPT-5 chat variants
    "gpt-5.2-chat-latest",
    "gpt-5.1-chat-latest",
    "gpt-5-chat-latest",
    # GPT-5 codex variants
    "gpt-5.2-codex",
    "gpt-5.1-codex-max",
    "gpt-5.1-codex",
    "gpt-5-codex",
    "gpt-5.1-codex-mini",
    "codex-mini-latest",
    # GPT-5 pro
    "gpt-5.2-pro",
    "gpt-5-pro",
    # GPT-4.1 series
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    # GPT-4o series
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-mini",
    # Realtime models
    "gpt-realtime",
    "gpt-realtime-mini",
    "gpt-4o-realtime-preview",
    "gpt-4o-mini-realtime-preview",
    # Audio models
    "gpt-audio",
    "gpt-audio-mini",
    "gpt-4o-audio-preview",
    "gpt-4o-mini-audio-preview",
    # O-series reasoning models
    "o1",
    "o1-pro",
    "o1-mini",
    "o3",
    "o3-pro",
    "o3-mini",
    "o3-deep-research",
    "o4-mini",
    "o4-mini-deep-research",
    # Search models
    "gpt-5-search-api",
    "gpt-4o-mini-search-preview",
    "gpt-4o-search-preview",
    # Computer use
    "computer-use-preview",
    # Image models
    "gpt-image-1.5",
    "chatgpt-image-latest",
    "gpt-image-1",
    "gpt-image-1-mini",
]

for model_id in _openai_models:
    model_registry.register(model_id, _make_lazy_openai_factory(model_id))


# ===========================================================================
# Eager export of OpenAIModel class (for direct use)
# ===========================================================================

# Try to export the actual class for users who want to import it directly
# This is optional - if openai isn't installed, just skip the export
try:
    from axis_core.adapters.models.openai import (
        MODEL_PRICING as OPENAI_PRICING,  # noqa: F401 - re-exported
    )
    from axis_core.adapters.models.openai import (
        OpenAIModel,  # noqa: F401 - re-exported
    )

    __all__.extend(["OpenAIModel", "OPENAI_PRICING"])
except ImportError:
    # openai package not installed - that's fine, lazy registration still works
    pass


# ===========================================================================
# Future adapters will follow the same pattern
# ===========================================================================
