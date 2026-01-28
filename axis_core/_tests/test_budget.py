"""Tests for axis_core.budget module."""

import pytest

from axis_core.budget import Budget, BudgetState


class TestBudget:
    """Tests for Budget dataclass."""

    def test_default_values(self):
        """Budget should have sensible defaults."""
        budget = Budget()
        assert budget.max_cycles == 10
        assert budget.max_tool_calls == 50
        assert budget.max_model_calls == 20
        assert budget.max_cost_usd == 1.00
        assert budget.max_wall_time_seconds == 300.0
        assert budget.max_input_tokens is None
        assert budget.max_output_tokens is None
        assert budget.warn_at_cost_usd == 0.80

    def test_custom_values(self):
        """Budget should accept custom values."""
        budget = Budget(
            max_cycles=5,
            max_tool_calls=25,
            max_model_calls=10,
            max_cost_usd=0.50,
            max_wall_time_seconds=120.0,
            max_input_tokens=1000,
            max_output_tokens=500,
            warn_at_cost_usd=0.40,
        )
        assert budget.max_cycles == 5
        assert budget.max_tool_calls == 25
        assert budget.max_model_calls == 10
        assert budget.max_cost_usd == 0.50
        assert budget.max_wall_time_seconds == 120.0
        assert budget.max_input_tokens == 1000
        assert budget.max_output_tokens == 500
        assert budget.warn_at_cost_usd == 0.40

    def test_frozen(self):
        """Budget should be immutable (frozen)."""
        budget = Budget()
        with pytest.raises(AttributeError):
            budget.max_cycles = 20  # type: ignore

    def test_type_hints(self):
        """Budget should have correct type annotations."""
        # This test is more for documentation; mypy checks actual types
        budget = Budget(max_input_tokens=None, max_output_tokens=None)
        assert budget.max_input_tokens is None
        assert budget.max_output_tokens is None


class TestBudgetState:
    """Tests for BudgetState dataclass."""

    def test_default_values(self):
        """BudgetState should initialize with all zeros."""
        state = BudgetState()
        assert state.cycles == 0
        assert state.tool_calls == 0
        assert state.model_calls == 0
        assert state.input_tokens == 0
        assert state.output_tokens == 0
        assert state.cost_usd == 0.0
        assert state.wall_time_seconds == 0.0

    def test_mutable(self):
        """BudgetState should be mutable to track consumption."""
        state = BudgetState()
        state.cycles = 1
        state.tool_calls = 2
        state.model_calls = 3
        state.input_tokens = 100
        state.output_tokens = 50
        state.cost_usd = 0.25
        state.wall_time_seconds = 10.5

        assert state.cycles == 1
        assert state.tool_calls == 2
        assert state.model_calls == 3
        assert state.input_tokens == 100
        assert state.output_tokens == 50
        assert state.cost_usd == 0.25
        assert state.wall_time_seconds == 10.5

    def test_total_tokens_property(self):
        """total_tokens should return sum of input and output tokens."""
        state = BudgetState()
        state.input_tokens = 100
        state.output_tokens = 50
        assert state.total_tokens == 150

        state.input_tokens = 0
        state.output_tokens = 0
        assert state.total_tokens == 0

    def test_cost_remaining_usd(self):
        """cost_remaining_usd should return remaining budget, minimum 0."""
        budget = Budget(max_cost_usd=1.00)
        state = BudgetState()

        # No consumption yet
        assert state.cost_remaining_usd(budget) == 1.00

        # Some consumption
        state.cost_usd = 0.30
        assert state.cost_remaining_usd(budget) == 0.70

        # Over budget
        state.cost_usd = 1.50
        assert state.cost_remaining_usd(budget) == 0.0

    def test_cycles_remaining(self):
        """cycles_remaining should return remaining cycles, minimum 0."""
        budget = Budget(max_cycles=10)
        state = BudgetState()

        assert state.cycles_remaining(budget) == 10

        state.cycles = 5
        assert state.cycles_remaining(budget) == 5

        state.cycles = 15
        assert state.cycles_remaining(budget) == 0

    def test_tool_calls_remaining(self):
        """tool_calls_remaining should return remaining calls, minimum 0."""
        budget = Budget(max_tool_calls=50)
        state = BudgetState()

        assert state.tool_calls_remaining(budget) == 50

        state.tool_calls = 30
        assert state.tool_calls_remaining(budget) == 20

        state.tool_calls = 60
        assert state.tool_calls_remaining(budget) == 0

    def test_model_calls_remaining(self):
        """model_calls_remaining should return remaining calls, minimum 0."""
        budget = Budget(max_model_calls=20)
        state = BudgetState()

        assert state.model_calls_remaining(budget) == 20

        state.model_calls = 10
        assert state.model_calls_remaining(budget) == 10

        state.model_calls = 25
        assert state.model_calls_remaining(budget) == 0

    def test_is_exhausted_no_limits_hit(self):
        """is_exhausted should return False when within all limits."""
        budget = Budget(
            max_cycles=10,
            max_tool_calls=50,
            max_model_calls=20,
            max_cost_usd=1.00,
            max_wall_time_seconds=300.0,
        )
        state = BudgetState()
        state.cycles = 5
        state.tool_calls = 25
        state.model_calls = 10
        state.cost_usd = 0.50
        state.wall_time_seconds = 150.0

        assert state.is_exhausted(budget) is False

    def test_is_exhausted_cycles(self):
        """is_exhausted should return True when cycles limit hit."""
        budget = Budget(max_cycles=10)
        state = BudgetState()
        state.cycles = 10

        assert state.is_exhausted(budget) is True

    def test_is_exhausted_tool_calls(self):
        """is_exhausted should return True when tool_calls limit hit."""
        budget = Budget(max_tool_calls=50)
        state = BudgetState()
        state.tool_calls = 50

        assert state.is_exhausted(budget) is True

    def test_is_exhausted_model_calls(self):
        """is_exhausted should return True when model_calls limit hit."""
        budget = Budget(max_model_calls=20)
        state = BudgetState()
        state.model_calls = 20

        assert state.is_exhausted(budget) is True

    def test_is_exhausted_cost(self):
        """is_exhausted should return True when cost limit hit."""
        budget = Budget(max_cost_usd=1.00)
        state = BudgetState()
        state.cost_usd = 1.00

        assert state.is_exhausted(budget) is True

    def test_is_exhausted_wall_time(self):
        """is_exhausted should return True when wall time limit hit."""
        budget = Budget(max_wall_time_seconds=300.0)
        state = BudgetState()
        state.wall_time_seconds = 300.0

        assert state.is_exhausted(budget) is True

    def test_is_exhausted_with_token_limits(self):
        """is_exhausted should check token limits when set."""
        budget = Budget(max_input_tokens=1000, max_output_tokens=500)
        state = BudgetState()

        # Within limits
        state.input_tokens = 500
        state.output_tokens = 250
        assert state.is_exhausted(budget) is False

        # Hit input token limit
        state.input_tokens = 1000
        assert state.is_exhausted(budget) is True

        # Reset input, hit output token limit
        state.input_tokens = 500
        state.output_tokens = 500
        assert state.is_exhausted(budget) is True

    def test_should_warn_below_threshold(self):
        """should_warn should return False when below threshold."""
        budget = Budget(max_cost_usd=1.00, warn_at_cost_usd=0.80)
        state = BudgetState()
        state.cost_usd = 0.50

        assert state.should_warn(budget) is False

    def test_should_warn_at_threshold(self):
        """should_warn should return True when at or above threshold."""
        budget = Budget(max_cost_usd=1.00, warn_at_cost_usd=0.80)
        state = BudgetState()
        state.cost_usd = 0.80

        assert state.should_warn(budget) is True

    def test_should_warn_above_threshold(self):
        """should_warn should return True when above threshold."""
        budget = Budget(max_cost_usd=1.00, warn_at_cost_usd=0.80)
        state = BudgetState()
        state.cost_usd = 0.90

        assert state.should_warn(budget) is True
