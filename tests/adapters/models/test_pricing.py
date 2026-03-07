"""Tests for shared provider pricing metadata."""

import pytest

from axis_core.adapters.models.catalog import ANTHROPIC_ALIAS_TARGETS
from axis_core.adapters.models.pricing import (
    ANTHROPIC_MODEL_PRICING,
    OPENAI_MODEL_PRICING,
    build_pricing_table,
)


@pytest.mark.unit
class TestProviderPricing:
    """Validate shared pricing metadata behavior."""

    def test_openai_pricing_contains_expected_models(self) -> None:
        """OpenAI pricing should expose known cost-estimation entries."""
        assert "gpt-5" in OPENAI_MODEL_PRICING
        assert "gpt-4o" in OPENAI_MODEL_PRICING
        assert "o1" in OPENAI_MODEL_PRICING
        assert "gpt-4o-search-preview" in OPENAI_MODEL_PRICING

    def test_anthropic_pricing_expands_aliases(self) -> None:
        """Anthropic pricing should include alias entries from the shared catalog."""
        for alias, target in ANTHROPIC_ALIAS_TARGETS.items():
            assert ANTHROPIC_MODEL_PRICING[alias] == ANTHROPIC_MODEL_PRICING[target]

    def test_build_pricing_table_rejects_missing_fields(self) -> None:
        """Pricing validation should reject incomplete table entries."""
        with pytest.raises(ValueError, match="output_per_mtok"):
            build_pricing_table(
                provider_name="test-provider",
                canonical_pricing={"test-model": {"input_per_mtok": 1.0}},
            )

    def test_build_pricing_table_rejects_negative_values(self) -> None:
        """Pricing validation should reject negative token prices."""
        with pytest.raises(ValueError, match="must be non-negative"):
            build_pricing_table(
                provider_name="test-provider",
                canonical_pricing={
                    "test-model": {
                        "input_per_mtok": -1.0,
                        "output_per_mtok": 2.0,
                    }
                },
            )
