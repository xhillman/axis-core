"""Tests for provider usage normalization and budget integration."""

from axis_core.budget import BudgetState
from axis_core.protocols.model import NormalizedUsage


def test_normalized_usage_from_openai_mapping() -> None:
    """OpenAI token fields should normalize to canonical usage keys."""
    usage = NormalizedUsage.from_openai(
        {
            "prompt_tokens": "120",
            "completion_tokens": 30,
            "total_tokens": None,
        }
    )

    assert usage.input_tokens == 120
    assert usage.output_tokens == 30
    assert usage.total_tokens == 150


def test_normalized_usage_from_anthropic_mapping_with_invalid_values() -> None:
    """Invalid numeric values should degrade gracefully to zero."""
    usage = NormalizedUsage.from_anthropic(
        {
            "input_tokens": "bad",
            "output_tokens": "40",
        }
    )

    assert usage.input_tokens == 0
    assert usage.output_tokens == 40
    assert usage.total_tokens == 40


def test_normalized_usage_from_openai_object_with_missing_fields() -> None:
    """Object-style usage payloads with missing attributes should default to zeros."""

    class OpenAIUsagePayload:
        prompt_tokens = None
        completion_tokens = None

    usage = NormalizedUsage.from_openai(OpenAIUsagePayload())
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0
    assert usage.total_tokens == 0


def test_budget_state_records_normalized_usage() -> None:
    """Budget state should accumulate canonical usage and cost values."""
    state = BudgetState()
    usage = NormalizedUsage(input_tokens=11, output_tokens=7, total_tokens=18)

    state.record_model_usage(usage=usage, cost_usd=0.123)

    assert state.model_calls == 1
    assert state.input_tokens == 11
    assert state.output_tokens == 7
    assert state.total_tokens == 18
    assert state.cost_usd == 0.123
