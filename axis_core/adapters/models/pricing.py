"""Shared provider pricing metadata for coarse `estimate_cost()` calculations.

Pricing ownership lives in this module. When provider prices change:

1. Update the canonical table for the provider below.
2. Keep the provider source URL and `as of` date adjacent to that table.
3. Run the pricing and adapter cost-estimation tests.

This data is operational metadata for rough cost estimates only. It is not intended to be a
billing source of truth.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Final

from axis_core.adapters.models.catalog import ANTHROPIC_ALIAS_TARGETS

_REQUIRED_PRICING_FIELDS: Final[tuple[str, str]] = ("input_per_mtok", "output_per_mtok")


def _normalize_pricing_entry(
    *,
    provider_name: str,
    model_id: str,
    raw_entry: Mapping[str, float | int],
) -> dict[str, float]:
    if not isinstance(raw_entry, Mapping):
        raise ValueError(
            f"{provider_name} pricing for {model_id!r} must be a mapping of token price fields."
        )

    normalized: dict[str, float] = {}
    for field in _REQUIRED_PRICING_FIELDS:
        if field not in raw_entry:
            raise ValueError(
                f"{provider_name} pricing for {model_id!r} must define {field!r}."
            )

        raw_value = raw_entry[field]
        if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
            raise ValueError(
                f"{provider_name} pricing for {model_id!r} field {field!r} must be numeric."
            )

        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(
                f"{provider_name} pricing for {model_id!r} field {field!r} must be finite."
            )
        if value < 0:
            raise ValueError(
                f"{provider_name} pricing for {model_id!r} field {field!r} must be non-negative."
            )
        normalized[field] = value

    unexpected_fields = set(raw_entry) - set(_REQUIRED_PRICING_FIELDS)
    if unexpected_fields:
        unexpected = ", ".join(sorted(unexpected_fields))
        raise ValueError(
            f"{provider_name} pricing for {model_id!r} has unexpected fields: {unexpected}."
        )

    return normalized


def build_pricing_table(
    *,
    provider_name: str,
    canonical_pricing: Mapping[str, Mapping[str, float | int]],
    aliases: Mapping[str, str] | None = None,
) -> dict[str, dict[str, float]]:
    """Validate and expand provider pricing metadata into a mutable lookup table."""
    table = {
        model_id: _normalize_pricing_entry(
            provider_name=provider_name,
            model_id=model_id,
            raw_entry=pricing,
        )
        for model_id, pricing in canonical_pricing.items()
    }

    for alias, target in (aliases or {}).items():
        if target not in table:
            raise ValueError(
                f"{provider_name} pricing alias {alias!r} references unknown target {target!r}."
            )
        table[alias] = table[target].copy()

    return table


OPENAI_PRICING_SOURCE_URL: Final[str] = "https://platform.openai.com/docs/pricing"
OPENAI_PRICING_AS_OF: Final[str] = "2026-02"
_OPENAI_CANONICAL_MODEL_PRICING: Final[dict[str, dict[str, float]]] = {
    "gpt-5.2": {"input_per_mtok": 1.75, "output_per_mtok": 14.00},
    "gpt-5.1": {"input_per_mtok": 1.25, "output_per_mtok": 10.00},
    "gpt-5": {"input_per_mtok": 1.25, "output_per_mtok": 10.00},
    "gpt-5-mini": {"input_per_mtok": 0.25, "output_per_mtok": 2.00},
    "gpt-5-nano": {"input_per_mtok": 0.05, "output_per_mtok": 0.40},
    "gpt-5.2-chat-latest": {"input_per_mtok": 1.75, "output_per_mtok": 14.00},
    "gpt-5.1-chat-latest": {"input_per_mtok": 1.25, "output_per_mtok": 10.00},
    "gpt-5-chat-latest": {"input_per_mtok": 1.25, "output_per_mtok": 10.00},
    "gpt-5.2-pro": {"input_per_mtok": 21.00, "output_per_mtok": 168.00},
    "gpt-5-pro": {"input_per_mtok": 15.00, "output_per_mtok": 120.00},
    "gpt-5.2-codex": {"input_per_mtok": 1.75, "output_per_mtok": 14.00},
    "gpt-5.1-codex-max": {"input_per_mtok": 1.25, "output_per_mtok": 10.00},
    "gpt-5.1-codex": {"input_per_mtok": 1.25, "output_per_mtok": 10.00},
    "gpt-5-codex": {"input_per_mtok": 1.25, "output_per_mtok": 10.00},
    "gpt-5.1-codex-mini": {"input_per_mtok": 0.25, "output_per_mtok": 2.00},
    "codex-mini-latest": {"input_per_mtok": 1.50, "output_per_mtok": 6.00},
    "gpt-5-search": {"input_per_mtok": 1.25, "output_per_mtok": 10.00},
    "gpt-5-search-api": {"input_per_mtok": 1.25, "output_per_mtok": 10.00},
    "gpt-4o-search-preview": {"input_per_mtok": 2.50, "output_per_mtok": 10.00},
    "gpt-4o-mini-search-preview": {"input_per_mtok": 0.15, "output_per_mtok": 0.60},
    "gpt-4.1": {"input_per_mtok": 2.00, "output_per_mtok": 8.00},
    "gpt-4.1-mini": {"input_per_mtok": 0.40, "output_per_mtok": 1.60},
    "gpt-4.1-nano": {"input_per_mtok": 0.10, "output_per_mtok": 0.40},
    "gpt-4o": {"input_per_mtok": 2.50, "output_per_mtok": 10.00},
    "gpt-4o-2024-05-13": {"input_per_mtok": 5.00, "output_per_mtok": 15.00},
    "gpt-4o-mini": {"input_per_mtok": 0.15, "output_per_mtok": 0.60},
    "o1": {"input_per_mtok": 15.00, "output_per_mtok": 60.00},
    "o1-pro": {"input_per_mtok": 150.00, "output_per_mtok": 600.00},
    "o1-mini": {"input_per_mtok": 1.10, "output_per_mtok": 4.40},
    "o3": {"input_per_mtok": 2.00, "output_per_mtok": 8.00},
    "o3-pro": {"input_per_mtok": 20.00, "output_per_mtok": 80.00},
    "o3-mini": {"input_per_mtok": 1.10, "output_per_mtok": 4.40},
    "o4-mini": {"input_per_mtok": 1.10, "output_per_mtok": 4.40},
    "o3-deep-research": {"input_per_mtok": 10.00, "output_per_mtok": 40.00},
    "o4-mini-deep-research": {"input_per_mtok": 2.00, "output_per_mtok": 8.00},
    "computer-use-preview": {"input_per_mtok": 3.00, "output_per_mtok": 12.00},
}
OPENAI_MODEL_PRICING: Final[dict[str, dict[str, float]]] = build_pricing_table(
    provider_name="OpenAI",
    canonical_pricing=_OPENAI_CANONICAL_MODEL_PRICING,
)


ANTHROPIC_PRICING_SOURCE_URL: Final[str] = (
    "https://docs.anthropic.com/en/docs/about-claude/models/all-models"
)
ANTHROPIC_PRICING_AS_OF: Final[str] = "2026-02"
_ANTHROPIC_CANONICAL_MODEL_PRICING: Final[dict[str, dict[str, float]]] = {
    "claude-opus-4-20250514": {"input_per_mtok": 15.00, "output_per_mtok": 75.00},
    "claude-opus-4-1-20250805": {"input_per_mtok": 15.00, "output_per_mtok": 75.00},
    "claude-opus-4-5-20251101": {"input_per_mtok": 5.00, "output_per_mtok": 25.00},
    "claude-opus-4-6": {"input_per_mtok": 5.00, "output_per_mtok": 25.00},
    "claude-sonnet-4-20250514": {"input_per_mtok": 3.00, "output_per_mtok": 15.00},
    "claude-sonnet-4-5-20250929": {"input_per_mtok": 3.00, "output_per_mtok": 15.00},
    "claude-haiku-4-5-20251001": {"input_per_mtok": 1.00, "output_per_mtok": 5.00},
    "claude-3-haiku-20240307": {"input_per_mtok": 0.25, "output_per_mtok": 1.25},
}
ANTHROPIC_MODEL_PRICING: Final[dict[str, dict[str, float]]] = build_pricing_table(
    provider_name="Anthropic",
    canonical_pricing=_ANTHROPIC_CANONICAL_MODEL_PRICING,
    aliases=ANTHROPIC_ALIAS_TARGETS,
)
