"""Tests for axis_core error handling system."""

from datetime import datetime

import pytest

from axis_core.errors import (
    AxisError,
    BudgetError,
    CancelledError,
    ConfigError,
    ErrorClass,
    ErrorRecord,
    InputError,
    ModelError,
    PlanError,
    TimeoutError,
    ToolError,
)


class TestErrorClass:
    """Tests for ErrorClass enum."""

    def test_error_class_values(self) -> None:
        """Test that ErrorClass has all 9 values with lowercase string values."""
        assert ErrorClass.INPUT.value == "input"
        assert ErrorClass.CONFIG.value == "config"
        assert ErrorClass.PLAN.value == "plan"
        assert ErrorClass.TOOL.value == "tool"
        assert ErrorClass.MODEL.value == "model"
        assert ErrorClass.BUDGET.value == "budget"
        assert ErrorClass.TIMEOUT.value == "timeout"
        assert ErrorClass.CANCELLED.value == "cancelled"
        assert ErrorClass.RUNTIME.value == "runtime"

    def test_error_class_count(self) -> None:
        """Test that we have exactly 9 error classes."""
        assert len(ErrorClass) == 9


class TestAxisError:
    """Tests for AxisError base class."""

    def test_axis_error_is_exception(self) -> None:
        """Test that AxisError is an Exception."""
        error = AxisError(message="test error", error_class=ErrorClass.RUNTIME)
        assert isinstance(error, Exception)

        # Can be raised and caught
        with pytest.raises(AxisError) as exc_info:
            raise error
        assert exc_info.value.message == "test error"

    def test_axis_error_fields(self) -> None:
        """Test that AxisError has all required fields."""
        error = AxisError(
            message="test error",
            error_class=ErrorClass.RUNTIME,
            phase="act",
            cycle=2,
            step_id="step-123",
            recoverable=True,
            retry_after=5.0,
            details={"key": "value"},
            cause=ValueError("original"),
        )

        assert error.message == "test error"
        assert error.error_class == ErrorClass.RUNTIME
        assert error.phase == "act"
        assert error.cycle == 2
        assert error.step_id == "step-123"
        assert error.recoverable is True
        assert error.retry_after == 5.0
        assert error.details == {"key": "value"}
        assert isinstance(error.cause, ValueError)

    def test_axis_error_defaults(self) -> None:
        """Test that AxisError has correct defaults."""
        error = AxisError(message="test", error_class=ErrorClass.RUNTIME)

        assert error.phase is None
        assert error.cycle is None
        assert error.step_id is None
        assert error.recoverable is False
        assert error.retry_after is None
        assert error.details == {}
        assert error.cause is None

    def test_axis_error_str(self) -> None:
        """Test that str(error) returns the message."""
        error = AxisError(message="test error message", error_class=ErrorClass.RUNTIME)
        assert str(error) == "test error message"


class TestInputError:
    """Tests for InputError."""

    def test_input_error_class(self) -> None:
        """Test that InputError has fixed error_class."""
        error = InputError(message="invalid input")
        assert error.error_class == ErrorClass.INPUT
        assert isinstance(error, AxisError)

    def test_input_error_cannot_override_class(self) -> None:
        """Test that error_class cannot be overridden."""
        error = InputError(message="test")
        assert error.error_class == ErrorClass.INPUT


class TestConfigError:
    """Tests for ConfigError."""

    def test_config_error_class(self) -> None:
        """Test that ConfigError has fixed error_class."""
        error = ConfigError(message="invalid config")
        assert error.error_class == ErrorClass.CONFIG
        assert isinstance(error, AxisError)


class TestPlanError:
    """Tests for PlanError."""

    def test_plan_error_class(self) -> None:
        """Test that PlanError has fixed error_class."""
        error = PlanError(message="planning failed")
        assert error.error_class == ErrorClass.PLAN
        assert isinstance(error, AxisError)


class TestTimeoutError:
    """Tests for TimeoutError."""

    def test_timeout_error_class(self) -> None:
        """Test that TimeoutError has fixed error_class."""
        error = TimeoutError(message="operation timed out")
        assert error.error_class == ErrorClass.TIMEOUT
        assert isinstance(error, AxisError)


class TestCancelledError:
    """Tests for CancelledError."""

    def test_cancelled_error_class(self) -> None:
        """Test that CancelledError has fixed error_class."""
        error = CancelledError(message="operation cancelled")
        assert error.error_class == ErrorClass.CANCELLED
        assert isinstance(error, AxisError)


class TestToolError:
    """Tests for ToolError."""

    def test_tool_error_class(self) -> None:
        """Test that ToolError has fixed error_class."""
        error = ToolError(message="tool failed")
        assert error.error_class == ErrorClass.TOOL
        assert isinstance(error, AxisError)

    def test_tool_error_with_tool_name(self) -> None:
        """Test that ToolError can track tool_name."""
        error = ToolError(message="calculator failed", tool_name="calculator")
        assert error.tool_name == "calculator"
        assert error.error_class == ErrorClass.TOOL

    def test_tool_error_tool_name_optional(self) -> None:
        """Test that tool_name is optional."""
        error = ToolError(message="tool failed")
        assert error.tool_name is None


class TestModelError:
    """Tests for ModelError."""

    def test_model_error_class(self) -> None:
        """Test that ModelError has fixed error_class."""
        error = ModelError(message="model failed")
        assert error.error_class == ErrorClass.MODEL
        assert isinstance(error, AxisError)

    def test_model_error_with_model_id(self) -> None:
        """Test that ModelError can track model_id."""
        error = ModelError(message="model failed", model_id="claude-sonnet-4-20250514")
        assert error.model_id == "claude-sonnet-4-20250514"

    def test_model_error_model_id_optional(self) -> None:
        """Test that model_id is optional."""
        error = ModelError(message="model failed")
        assert error.model_id is None

    def test_from_exception_rate_limit_error(self) -> None:
        """Test that RateLimitError is classified as recoverable."""

        class RateLimitError(Exception):
            pass

        original = RateLimitError("Rate limit exceeded")
        error = ModelError.from_exception(original, model_id="gpt-4")

        assert error.recoverable is True
        assert error.model_id == "gpt-4"
        assert error.cause is original
        assert "Rate limit exceeded" in error.message

    def test_from_exception_timeout_error(self) -> None:
        """Test that TimeoutError is classified as recoverable."""

        class TimeoutError(Exception):
            pass

        original = TimeoutError("Request timed out")
        error = ModelError.from_exception(original, model_id="gpt-4")

        assert error.recoverable is True

    def test_from_exception_connection_errors(self) -> None:
        """Test that connection errors are classified as recoverable."""

        class ConnectionError(Exception):
            pass

        class ConnectError(Exception):
            pass

        class ReadTimeoutError(Exception):
            pass

        class APIConnectionError(Exception):
            pass

        for exc_class in [ConnectionError, ConnectError, ReadTimeoutError, APIConnectionError]:
            original = exc_class("Connection failed")
            error = ModelError.from_exception(original, model_id="gpt-4")
            assert error.recoverable is True, f"{exc_class.__name__} should be recoverable"

    def test_from_exception_validation_error(self) -> None:
        """Test that ValidationError is not recoverable."""

        class ValidationError(Exception):
            pass

        original = ValidationError("Invalid request")
        error = ModelError.from_exception(original, model_id="gpt-4")

        assert error.recoverable is False

    def test_from_exception_auth_error(self) -> None:
        """Test that AuthenticationError is not recoverable."""

        class AuthenticationError(Exception):
            pass

        original = AuthenticationError("Invalid API key")
        error = ModelError.from_exception(original, model_id="gpt-4")

        assert error.recoverable is False

    def test_from_exception_python_errors(self) -> None:
        """Test that Python built-in errors are not recoverable."""
        for exc_class in [TypeError, ValueError, KeyError]:
            original = exc_class("Python error")
            error = ModelError.from_exception(original, model_id="gpt-4")
            assert error.recoverable is False, f"{exc_class.__name__} should not be recoverable"

    def test_from_exception_unknown_error(self) -> None:
        """Test that unknown exceptions default to not recoverable."""

        class UnknownError(Exception):
            pass

        original = UnknownError("Unknown error")
        error = ModelError.from_exception(original, model_id="gpt-4")

        assert error.recoverable is False


class TestBudgetError:
    """Tests for BudgetError."""

    def test_budget_error_class(self) -> None:
        """Test that BudgetError has fixed error_class."""
        error = BudgetError(message="budget exceeded")
        assert error.error_class == ErrorClass.BUDGET
        assert isinstance(error, AxisError)

    def test_budget_error_with_resource_info(self) -> None:
        """Test that BudgetError can track resource, used, and limit."""
        error = BudgetError(
            message="Cost limit exceeded",
            resource="cost_usd",
            used=1.50,
            limit=1.00,
        )

        assert error.resource == "cost_usd"
        assert error.used == 1.50
        assert error.limit == 1.00

    def test_budget_error_optional_fields(self) -> None:
        """Test that resource fields are optional."""
        error = BudgetError(message="budget exceeded")
        assert error.resource is None
        assert error.used is None
        assert error.limit is None

    def test_budget_error_generates_suggestion_cost(self) -> None:
        """Test that BudgetError generates actionable suggestion for cost."""
        error = BudgetError(
            message="Cost limit exceeded",
            resource="cost_usd",
            used=1.50,
            limit=1.00,
        )

        # Check that suggestion is appended to message
        assert "Cost limit exceeded" in error.message
        assert "budget.max_cost_usd" in error.message
        # Should suggest ~1.5x the used value (2.25)
        assert "2.25" in error.message

    def test_budget_error_generates_suggestion_input_tokens(self) -> None:
        """Test that BudgetError generates actionable suggestion for input tokens."""
        error = BudgetError(
            message="Input token limit exceeded",
            resource="input_tokens",
            used=10000,
            limit=8000,
        )

        assert "budget.max_input_tokens" in error.message
        # Should suggest ~1.5x the used value (15000)
        assert "15000" in error.message

    def test_budget_error_generates_suggestion_cycles(self) -> None:
        """Test that BudgetError generates actionable suggestion for cycles."""
        error = BudgetError(
            message="Cycle limit exceeded",
            resource="cycles",
            used=12,
            limit=10,
        )

        assert "budget.max_cycles" in error.message
        # Should suggest ~1.5x the used value (18)
        assert "18" in error.message

    def test_budget_error_no_suggestion_without_resource_info(self) -> None:
        """Test that no suggestion is generated without resource info."""
        error = BudgetError(message="Budget exceeded")
        # Should not have any suggestion text
        assert "budget.max_" not in error.message


class TestErrorRecord:
    """Tests for ErrorRecord dataclass."""

    def test_error_record_creation(self) -> None:
        """Test that ErrorRecord can be created with all fields."""
        error = InputError(message="test error")
        timestamp = datetime.now()

        record = ErrorRecord(
            error=error,
            timestamp=timestamp,
            phase="act",
            cycle=2,
            recovered=True,
        )

        assert record.error is error
        assert record.timestamp == timestamp
        assert record.phase == "act"
        assert record.cycle == 2
        assert record.recovered is True

    def test_error_record_is_frozen(self) -> None:
        """Test that ErrorRecord is immutable (frozen)."""
        error = InputError(message="test error")
        record = ErrorRecord(
            error=error,
            timestamp=datetime.now(),
            phase="act",
            cycle=1,
            recovered=False,
        )

        # Should not be able to modify fields (frozen dataclass raises AttributeError)
        with pytest.raises(AttributeError):
            record.recovered = True  # type: ignore

    def test_error_record_fields_required(self) -> None:
        """Test that all ErrorRecord fields are required."""
        error = InputError(message="test error")

        # Should be able to create with all fields
        record = ErrorRecord(
            error=error,
            timestamp=datetime.now(),
            phase="act",
            cycle=1,
            recovered=False,
        )
        assert record is not None


class TestErrorInheritance:
    """Tests for error inheritance and catching."""

    def test_all_errors_catchable_as_axis_error(self) -> None:
        """Test that all specific errors can be caught as AxisError."""
        errors = [
            InputError(message="test"),
            ConfigError(message="test"),
            PlanError(message="test"),
            ToolError(message="test"),
            ModelError(message="test"),
            BudgetError(message="test"),
            TimeoutError(message="test"),
            CancelledError(message="test"),
        ]

        for error in errors:
            with pytest.raises(AxisError):
                raise error

    def test_specific_error_catching(self) -> None:
        """Test that specific errors can be caught by their type."""
        error = InputError(message="test")

        with pytest.raises(InputError):
            raise error

        # But also catchable as AxisError
        with pytest.raises(AxisError):
            raise error
