"""Model adapters for LLM providers.

This module provides built-in model adapter implementations with lazy loading
for optional dependencies (per AD-040).

Available adapters (when dependencies are installed):
- AnthropicModel: Claude models via Anthropic API (requires: pip install axis-core[anthropic])
- OpenAIModel: GPT models via OpenAI API (requires: pip install axis-core[openai])
- OpenRouter via OpenAIModel: OpenAI-compatible endpoint
  (requires: pip install axis-core[openrouter])
"""

from axis_core.engine.registry import make_lazy_factory, model_registry

__all__: list[str] = []

_ANTHROPIC_MODULE = "axis_core.adapters.models.anthropic"
_OPENAI_MODULE = "axis_core.adapters.models.openai"


# ===========================================================================
# Register built-in Anthropic models (Task 16.1)
# ===========================================================================

_anthropic_models = [
    "claude-3-haiku-20240307",
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    "claude-opus-4-1-20250805",
    "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5-20251001",
    "claude-opus-4-5-20251101",
    "claude-opus-4-6",
]

for _model_id in _anthropic_models:
    model_registry.register(
        _model_id,
        make_lazy_factory(
            _ANTHROPIC_MODULE,
            "AnthropicModel",
            defaults={"model_id": _model_id},
            missing_dep_message=(
                f"Model '{_model_id}' requires the anthropic package. "
                f"Install with: pip install 'axis-core[anthropic]'"
            ),
        ),
    )

# Convenience aliases for latest versions
_anthropic_aliases = {
    "claude-haiku": "claude-haiku-4-5-20251001",
    "claude-sonnet": "claude-sonnet-4-5-20250929",
    "claude-opus": "claude-opus-4-6",
}

for _alias, _target in _anthropic_aliases.items():
    model_registry.register(
        _alias,
        make_lazy_factory(
            _ANTHROPIC_MODULE,
            "AnthropicModel",
            defaults={"model_id": _target},
            missing_dep_message=(
                f"Model '{_alias}' requires the anthropic package. "
                f"Install with: pip install 'axis-core[anthropic]'"
            ),
        ),
    )


# ===========================================================================
# Eager export of AnthropicModel class (for direct use)
# ===========================================================================

try:
    from axis_core.adapters.models.anthropic import (
        MODEL_PRICING,  # noqa: F401 - re-exported
        AnthropicModel,  # noqa: F401 - re-exported
    )

    __all__.extend(["AnthropicModel", "MODEL_PRICING"])
except ImportError:
    pass


# ===========================================================================
# Register built-in OpenAI models (Task 16.1)
# ===========================================================================

_openai_models = [
    # GPT-5 series (chat completions API)
    "gpt-5.2",
    "gpt-5.1",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5.2-chat-latest",
    "gpt-5.1-chat-latest",
    "gpt-5-chat-latest",
    "gpt-5.2-pro",
    "gpt-5-pro",
    # GPT-4.1 series (chat completions API)
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    # GPT-4o series (chat completions API)
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-mini",
    # O-series reasoning models (chat completions API)
    "o1",
    "o1-pro",
    "o1-mini",
    "o3",
    "o3-pro",
    "o3-mini",
    "o4-mini",
]

for _model_id in _openai_models:
    model_registry.register(
        _model_id,
        make_lazy_factory(
            _OPENAI_MODULE,
            "OpenAIModel",
            defaults={"model_id": _model_id},
            missing_dep_message=(
                f"Model '{_model_id}' requires the openai package. "
                f"Install with: pip install 'axis-core[openai]'"
            ),
        ),
    )

# Responses API models (routed internally by OpenAIModel)
_openai_responses_models = [
    # Codex
    "gpt-5.2-codex",
    "gpt-5.1-codex-max",
    "gpt-5.1-codex",
    "gpt-5-codex",
    "gpt-5.1-codex-mini",
    "codex-mini-latest",
    # Search
    "gpt-5-search",
    "gpt-5-search-api",
    "gpt-4o-search-preview",
    "gpt-4o-mini-search-preview",
    # Deep research
    "o3-deep-research",
    "o4-mini-deep-research",
    # Computer use
    "computer-use-preview",
]

for _model_id in _openai_responses_models:
    model_registry.register(
        _model_id,
        make_lazy_factory(
            _OPENAI_MODULE,
            "OpenAIModel",
            defaults={"model_id": _model_id},
            missing_dep_message=(
                f"Model '{_model_id}' requires the openai package. "
                f"Install with: pip install 'axis-core[openai]'"
            ),
        ),
    )


# ===========================================================================
# Eager export of OpenAIModel class (for direct use)
# ===========================================================================

try:
    from axis_core.adapters.models.openai import (
        MODEL_PRICING as OPENAI_PRICING,  # noqa: F401 - re-exported
    )
    from axis_core.adapters.models.openai import (
        OpenAIModel,  # noqa: F401 - re-exported
    )
    from axis_core.adapters.models.openai_responses import (
        OpenAIResponsesModel,  # noqa: F401 - re-exported
    )

    __all__.extend(["OpenAIModel", "OpenAIResponsesModel", "OPENAI_PRICING"])
except ImportError:
    pass
