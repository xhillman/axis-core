"""Model adapters for LLM providers.

This module provides built-in model adapter implementations with lazy loading
for optional dependencies (per AD-040).

Available adapters (when dependencies are installed):
- AnthropicModel: Claude models via Anthropic API (requires: pip install axis-core[anthropic])
- OpenAIModel: GPT models via OpenAI API (requires: pip install axis-core[openai])
- OpenRouter via OpenAIModel: OpenAI-compatible endpoint
  (requires: pip install axis-core[openrouter])
"""

from axis_core.adapters.models.catalog import (
    ANTHROPIC_MODEL_CATALOG,
    OPENAI_MODEL_CATALOG,
    ModelCatalogEntry,
    iter_registered_model_targets,
)
from axis_core.engine.registry import make_lazy_factory, model_registry

__all__: list[str] = []

_ANTHROPIC_MODULE = "axis_core.adapters.models.anthropic"
_OPENAI_MODULE = "axis_core.adapters.models.openai"


def _register_catalog_models(
    catalog: tuple[ModelCatalogEntry, ...],
    *,
    module_path: str,
    class_name: str,
    dependency_name: str,
) -> None:
    for registered_name, target_model_id in iter_registered_model_targets(catalog):
        model_registry.register(
            registered_name,
            make_lazy_factory(
                module_path,
                class_name,
                defaults={"model_id": target_model_id},
                missing_dep_message=(
                    f"Model '{registered_name}' requires the {dependency_name} package. "
                    f"Install with: pip install 'axis-core[{dependency_name}]'"
                ),
            ),
        )


_register_catalog_models(
    ANTHROPIC_MODEL_CATALOG,
    module_path=_ANTHROPIC_MODULE,
    class_name="AnthropicModel",
    dependency_name="anthropic",
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


_register_catalog_models(
    OPENAI_MODEL_CATALOG,
    module_path=_OPENAI_MODULE,
    class_name="OpenAIModel",
    dependency_name="openai",
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
